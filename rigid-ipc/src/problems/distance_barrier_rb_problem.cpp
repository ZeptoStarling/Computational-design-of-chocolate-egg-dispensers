#include "distance_barrier_rb_problem.hpp"
#include <cmath>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include <ipc/distance/edge_edge.hpp>
#include <ipc/distance/edge_edge_mollifier.hpp>
#include <ipc/distance/point_triangle.hpp>
#include <ipc/ipc.hpp>

#ifdef RIGID_IPC_WITH_DERIVATIVE_CHECK
#include <finitediff.hpp>
#endif

#include <constants.hpp>
#include <geometry/distance.hpp>
#include <solvers/solver_factory.hpp>
#include <utils/not_implemented_error.hpp>

#include <logger.hpp>
#include <profiler.hpp>

namespace ipc::rigid {

DistanceBarrierRBProblem::DistanceBarrierRBProblem()
    : m_barrier_stiffness(1)
    , min_distance(-1)
    , m_had_collisions(false)
    , static_friction_speed_bound(1e-3)
    , friction_iterations(1)
    , body_energy_integration_method(DEFAULT_BODY_ENERGY_INTEGRATION_METHOD)
{
    m_funnel_enabled = false;
}

double DistanceBarrierRBProblem::compute_step_max_speed(
    const Eigen::VectorXd& x, const Eigen::VectorXd& direction) const
{
    // Original rigid-IPC mesh behaviour.
    if (!m_funnel_enabled) {
        Eigen::MatrixXd V_prev = world_vertices(x);

        Eigen::MatrixXd V = world_vertices(x + direction);

        return (V - V_prev).lpNorm<Eigen::Infinity>() / timestep();
    }

    // Analytic 2D-circle behaviour.
    assert(dim() == 2);

    const int ndof = PoseD::dim_to_ndof(dim());

    const int pos_ndof = PoseD::dim_to_pos_ndof(dim());

    const int rot_ndof = PoseD::dim_to_rot_ndof(dim());

    assert(pos_ndof == 2);
    assert(rot_ndof == 1);

    assert(x.size() == direction.size());
    assert(direction.size() == static_cast<int>(num_bodies()) * ndof);

    double max_surface_step = 0.0;

    for (size_t i = 0; i < num_bodies(); ++i) {
        const int offset = static_cast<int>(i) * ndof;

        const double translation_step =
            direction.segment(offset, pos_ndof).lpNorm<Eigen::Infinity>();

        const double dtheta = direction[offset + pos_ndof];

        const double radius = m_assembler[i].radius;

        const double rotation_step =
            2.0 * radius * std::abs(std::sin(0.5 * dtheta));

        const double body_step = translation_step + rotation_step;

        max_surface_step = std::max(max_surface_step, body_step);
    }

    return max_surface_step / timestep();
}

bool DistanceBarrierRBProblem::settings(const nlohmann::json& params)
{
    // Configure the distance-barrier constraint before the base-class
    // initialization. RigidBodyProblem::settings() calls init(), which
    // dispatches to this class's update_constraints().
    m_constraint.settings(params["distance_barrier_constraint"]);

    body_energy_integration_method =
        params["rigid_body_problem"]["time_stepper"]
            .get<BodyEnergyIntegrationMethod>();

    // The funnel is disabled during the base-class initialization because
    // dim() is not known until the rigid bodies have been initialized.
    m_funnel_enabled = false;

    bool success = RigidBodyProblem::settings(params["rigid_body_problem"]);

    if (!success) {
        return false;
    }

    // Now that the rigid bodies have been initialized, dim() is known.
    if (params.contains("spline_funnel")
        && params["spline_funnel"].value("enabled", false)) {

        if (dim() != 2) {
            spdlog::error("Spline funnel is only supported in 2D simulations!");
            return false;
        }

        const auto& sf = params["spline_funnel"];

        m_funnel_enabled = true;
        m_funnel_height = sf.value("height", 6.0);

        m_funnel = rigid_ipc::CubicSpline1D(
            sf.value("r_bottom", 2.5), sf.value("r_top", 5.0),
            sf.value("s_bottom", 0.0), sf.value("s_top", 0.0), m_funnel_height);
    }

    // Select the optimization solver.
    std::string solver_name = params["solver"].get<std::string>();

    m_opt_solver = SolverFactory::factory().get_barrier_solver(solver_name);

    m_opt_solver->settings(params[solver_name]);
    m_opt_solver->set_problem(*this);

    if (m_opt_solver->has_inner_solver()) {
        m_opt_solver->inner_solver().settings(
            params[m_opt_solver->inner_solver().name()]);
    }

    // Friction.
    static_friction_speed_bound =
        params["friction_constraints"]["static_friction_speed_bound"];

    friction_iterations = params["friction_constraints"]["iterations"];

    if (friction_iterations == 0) {
        spdlog::info("Disabling friction because friction iterations is zero");
        coefficient_friction = 0;
    }

    min_distance = compute_min_distance(starting_point());

    if (min_distance < 0) {
        spdlog::info(
            "init_min_distance>d̂+dmin={:.8e}",
            barrier_activation_distance()
                + m_constraint.minimum_separation_distance);
    } else {
        spdlog::info("init_min_distance={:.8e}", min_distance);
    }

    return true;
}

nlohmann::json DistanceBarrierRBProblem::settings() const
{
    nlohmann::json json = RigidBodyProblem::settings();

    json["friction_iterations"] = friction_iterations;
    json["static_friction_speed_bound"] = static_friction_speed_bound;
    json["time_stepper"] = body_energy_integration_method;

    return json;
}

nlohmann::json DistanceBarrierRBProblem::state() const
{
    nlohmann::json json = RigidBodyProblem::state();

    if (min_distance < 0) {
        json["min_distance"] = nullptr;
    } else {
        json["min_distance"] = min_distance;
    }

    return json;
}

Eigen::VectorXi DistanceBarrierRBProblem::free_dof() const
{
    const VectorXb& is_dof_fixed = this->is_dof_fixed();

    std::vector<int> free_dofs;
    free_dofs.reserve(is_dof_fixed.size() - is_dof_fixed.count());

    for (int i = 0; i < is_dof_fixed.size(); i++) {
        if (!is_dof_fixed[i] && !is_dof_satisfied[i]) {
            free_dofs.push_back(i);
        }
    }

    return Eigen::Map<Eigen::VectorXi>(free_dofs.data(), free_dofs.size());
}

////////////////////////////////////////////////////////////
// Rigid Body Problem

void DistanceBarrierRBProblem::simulation_step(
    bool& had_collisions, bool& _has_intersections, bool solve_collisions)
{
    // Advance the poses, but leave the current pose unchanged for now.
    for (size_t i = 0; i < num_bodies(); i++) {
        m_assembler[i].pose_prev = m_assembler[i].pose;

        m_assembler[i].velocity_prev = m_assembler[i].velocity;
    }

    // Update the stored poses and initial value for the solver.
    update_dof();

    // Reset collision state.
    m_had_collisions = false;
    m_num_contacts = 0;

    // Disable barriers if solve_collisions == false.
    this->m_use_barriers = solve_collisions;

    update_constraints();

    opt_result = solve_constraints();

    _has_intersections = take_step(opt_result.x);

    step_kinematic_bodies();

    had_collisions = m_had_collisions;
}

void DistanceBarrierRBProblem::update_constraints()
{
    PROFILE_POINT("DistanceBarrierRBProblem::update_constraints");
    PROFILE_START();

    if (!m_funnel_enabled) {
        // Original mesh path.
        RigidBodyProblem::update_constraints();

        Constraints collision_constraints;
        m_constraint.construct_constraint_set(
            m_assembler, poses_t0, collision_constraints);

        Eigen::SparseMatrix<double> hess;

        compute_barrier_term(
            x0, collision_constraints, grad_barrier_t0, hess,
            /*compute_grad=*/true,
            /*compute_hess=*/false);

        update_friction_constraints(collision_constraints, poses_t0);

        init_augmented_lagrangian();

        PROFILE_END();
        return;
    }

    // Analytic path.
    RigidBodyProblem::update_constraints();

    Constraints collision_constraints;

    Eigen::SparseMatrix<double> hess;

    compute_barrier_term(
        x0, collision_constraints, grad_barrier_t0, hess,
        /*compute_grad=*/true,
        /*compute_hess=*/false);

    update_friction_constraints(collision_constraints, poses_t0);

    init_augmented_lagrangian();

    PROFILE_END();
}

void DistanceBarrierRBProblem::update_friction_constraints(
    const Constraints& collision_constraints, const PosesD& poses)
{
    friction_constraints.clear();
    analytic_friction_constraints.clear();

    if (coefficient_friction <= 0)
        return;

    PROFILE_POINT("DistanceBarrierRBProblem::update_friction_constraints");
    PROFILE_START();

    // Original mesh friction path.
    if (!m_funnel_enabled) {
        Eigen::MatrixXd V0 = m_assembler.world_vertices(poses);
        construct_friction_constraint_set(
            V0, edges(), faces(), collision_constraints,
            barrier_activation_distance(), barrier_stiffness(),
            coefficient_friction, friction_constraints);
        PROFILE_END();
        return;
    }

    // Analytic path: currently only 2D circles.
    assert(dim() == 2);

    const int ndof = PoseD::dim_to_ndof(dim());
    Eigen::VectorXd x_lagged(static_cast<int>(num_bodies()) * ndof);

    for (size_t i = 0; i < num_bodies(); ++i) {
        x_lagged.segment(static_cast<int>(i) * ndof, ndof) = poses[i].dof();
    }

    std::vector<rigid_ipc::FunnelCandidate> funnel_cands;
    std::vector<rigid_ipc::CircleCircleCandidate> circle_cands;
    detect_funnel_candidates(x_lagged, funnel_cands, circle_cands);

    const double dmin = m_constraint.minimum_separation_distance;
    const double dhat = barrier_activation_distance();
    const double z_hat = 2.0 * dmin * dhat + dhat * dhat;

    auto world_to_local = [](const rigid_ipc::RigidCircle& body,
                             const Eigen::Vector2d& p_world) {
        const double c = std::cos(body.theta);
        const double s = std::sin(body.theta);
        const Eigen::Vector2d d = p_world - body.pos;
        return Eigen::Vector2d(c * d.x() + s * d.y(), -s * d.x() + c * d.y());
    };

    // Circle -> funnel.
    for (const auto& cand : funnel_cands) {
        const auto projection = rigid_ipc::project_rigid_circle_to_funnel(
            cand.body_id, cand.body, *cand.spline, cand.height);

        const double gap = projection.distance;
        if (gap <= dmin || gap >= dmin + dhat)
            continue;

        const double z = gap * gap - dmin * dmin;
        const double lambda_n = std::max(
            0.0,
            -barrier_stiffness() * ipc::barrier_gradient(z, z_hat) * 2.0 * gap);

        if (!std::isfinite(lambda_n) || lambda_n <= 0.0)
            continue;

        const Eigen::Vector2d n = projection.normal_spatial;
        const Eigen::Vector2d tangent(-n.y(), n.x());
        const Eigen::Vector2d contact_world =
            cand.body.world_center() - cand.body.radius * n;

        AnalyticFrictionConstraint fc;
        fc.body1_id = cand.body_id;
        fc.body2_id = -1;
        fc.tangent = tangent;
        fc.local_point1 = world_to_local(cand.body, contact_world);
        fc.normal_force = lambda_n;
        analytic_friction_constraints.push_back(fc);
    }

    // Circle -> circle.
    for (const auto& cand : circle_cands) {
        const auto projection = rigid_ipc::project_rigid_circle_to_rigid_circle(
            cand.body1_id, cand.body1, cand.body2_id, cand.body2);

        const double gap = projection.distance;
        if (gap <= dmin || gap >= dmin + dhat)
            continue;

        const double z = gap * gap - dmin * dmin;
        const double lambda_n = std::max(
            0.0,
            -barrier_stiffness() * ipc::barrier_gradient(z, z_hat) * 2.0 * gap);

        if (!std::isfinite(lambda_n) || lambda_n <= 0.0)
            continue;

        const Eigen::Vector2d n = projection.normal_spatial;
        const Eigen::Vector2d tangent(-n.y(), n.x());

        const Eigen::Vector2d contact1_world =
            cand.body1.world_center() - cand.body1.radius * n;
        const Eigen::Vector2d contact2_world =
            cand.body2.world_center() + cand.body2.radius * n;

        AnalyticFrictionConstraint fc;
        fc.body1_id = cand.body1_id;
        fc.body2_id = cand.body2_id;
        fc.tangent = tangent;
        fc.local_point1 = world_to_local(cand.body1, contact1_world);
        fc.local_point2 = world_to_local(cand.body2, contact2_world);
        fc.normal_force = lambda_n;
        analytic_friction_constraints.push_back(fc);
    }

    spdlog::debug(
        "analytic friction constraints={}",
        analytic_friction_constraints.size());
    PROFILE_END();
}

void DistanceBarrierRBProblem::init_augmented_lagrangian()
{
    int ndof = PoseD::dim_to_ndof(dim());

    int pos_ndof = PoseD::dim_to_pos_ndof(dim());

    int rot_ndof = PoseD::dim_to_rot_ndof(dim());

    linear_augmented_lagrangian_penalty = 1e3;
    angular_augmented_lagrangian_penalty = 1e3;

    size_t num_kinematic_bodies = m_assembler.count_kinematic_bodies();

    linear_augmented_lagrangian_multiplier.setZero(
        pos_ndof * num_kinematic_bodies);

    angular_augmented_lagrangian_multiplier.setZero(
        rot_ndof * num_kinematic_bodies, rot_ndof);

    for (int i = 0; i < num_bodies(); i++) {
        if (m_assembler[i].type == RigidBodyType::KINEMATIC
            && m_assembler[i].kinematic_max_time < 0) {
            m_assembler[i].convert_to_static();
        }
    }

    x_pred = x0;

    for (int i = 0; i < num_bodies(); i++) {
        if (m_assembler[i].kinematic_poses.size()) {
            const PoseD& pose = m_assembler[i].kinematic_poses.front();

            x_pred.segment(ndof * i, pos_ndof) = pose.position;

            x_pred.segment(ndof * i + pos_ndof, rot_ndof) = pose.rotation;
        } else {
            x_pred.segment(ndof * i, pos_ndof) +=
                timestep() * m_assembler[i].velocity.position;

            x_pred.segment(ndof * i + pos_ndof, rot_ndof) +=
                timestep() * m_assembler[i].velocity.rotation;
        }
    }

    is_dof_satisfied.setZero(x0.size());

    for (int i = 0; i < num_bodies(); i++) {
        if (m_assembler[i].type == RigidBodyType::STATIC) {
            is_dof_satisfied.segment(ndof * i, ndof).setOnes();
        }
    }
}

void DistanceBarrierRBProblem::step_kinematic_bodies()
{
    for (int i = 0; i < num_bodies(); i++) {
        if (m_assembler[i].type == RigidBodyType::KINEMATIC) {

            if (m_assembler[i].kinematic_max_time < 0) {
                m_assembler[i].convert_to_static();
            } else {
                m_assembler[i].kinematic_max_time -= timestep();

                if (m_assembler[i].kinematic_poses.size()) {
                    m_assembler[i].kinematic_poses.pop_front();
                }
            }
        }
    }
}

inline DiagonalMatrix3d compute_J(const VectorMax3d& I)
{
    return DiagonalMatrix3d(
        0.5 * (-I.x() + I.y() + I.z()), 0.5 * (I.x() - I.y() + I.z()),
        0.5 * (I.x() + I.y() - I.z()));
}

inline DiagonalMatrix3d compute_Jinv(const VectorMax3d& I)
{
    return DiagonalMatrix3d(
        2 / (-I.x() + I.y() + I.z()), 2 / (I.x() - I.y() + I.z()),
        2 / (I.x() + I.y() - I.z()));
}

inline DiagonalMatrix3d compute_Jsqrt(const VectorMax3d& I)
{
    assert(0.5 * (-I.x() + I.y() + I.z()) >= 0);

    assert(0.5 * (I.x() - I.y() + I.z()) >= 0);

    assert(0.5 * (I.x() + I.y() - I.z()) >= 0);

    return DiagonalMatrix3d(
        sqrt(std::max(0.5 * (-I.x() + I.y() + I.z()), 0.0)),
        sqrt(std::max(0.5 * (I.x() - I.y() + I.z()), 0.0)),
        sqrt(std::max(0.5 * (I.x() + I.y() - I.z()), 0.0)));
}

double DistanceBarrierRBProblem::compute_linear_augment_lagrangian_progress(
    const Eigen::VectorXd& x) const
{
    int ndof = PoseD::dim_to_ndof(dim());

    int pos_ndof = PoseD::dim_to_pos_ndof(dim());

    double a = 0;
    double b = 0;

    for (size_t i = 0; i < num_bodies(); i++) {
        if (m_assembler[i].type == RigidBodyType::KINEMATIC) {

            a += (x_pred.segment(ndof * i, pos_ndof)
                  - x.segment(ndof * i, pos_ndof))
                     .squaredNorm();

            b += (x_pred.segment(ndof * i, pos_ndof)
                  - x0.segment(ndof * i, pos_ndof))
                     .squaredNorm();
        }
    }

    if (a == 0 && b == 0) {
        return 1;
    }

    return 1 - sqrt(a / (b == 0 ? 1 : b));
}

double DistanceBarrierRBProblem::compute_angular_augment_lagrangian_progress(
    const Eigen::VectorXd& x) const
{
    int ndof = PoseD::dim_to_ndof(dim());

    int rot_ndof = PoseD::dim_to_rot_ndof(dim());

    int pos_ndof = PoseD::dim_to_pos_ndof(dim());

    double a = 0;
    double b = 0;

    for (size_t i = 0; i < num_bodies(); i++) {
        if (m_assembler[i].type == RigidBodyType::KINEMATIC) {

            size_t ri = ndof * i + pos_ndof;

            if (dim() == 2) {
                a += (x_pred.segment(ri, rot_ndof) - x.segment(ri, rot_ndof))
                         .squaredNorm();

                b += (x_pred.segment(ri, rot_ndof) - x0.segment(ri, rot_ndof))
                         .squaredNorm();
            } else {
                auto Q_pred = construct_rotation_matrix(
                    VectorMax3d(x_pred.segment(ri, rot_ndof)));

                auto Q = construct_rotation_matrix(
                    VectorMax3d(x.segment(ri, rot_ndof)));

                auto Q0 = construct_rotation_matrix(
                    VectorMax3d(x0.segment(ri, rot_ndof)));

                a += (Q - Q_pred).squaredNorm();

                b += (Q0 - Q_pred).squaredNorm();
            }
        }
    }

    if (a == 0 && b == 0) {
        return 1;
    }

    return 1 - sqrt(a / (b == 0 ? 1 : b));
}

void DistanceBarrierRBProblem::update_augmented_lagrangian(
    const Eigen::VectorXd& x)
{
    int ndof = PoseD::dim_to_ndof(dim());

    int pos_ndof = PoseD::dim_to_pos_ndof(dim());

    int rot_ndof = PoseD::dim_to_rot_ndof(dim());

    double eta_q = compute_linear_augment_lagrangian_progress(x);

    double eta_Q = compute_angular_augment_lagrangian_progress(x);

    if (eta_q >= 0.999) {
        for (size_t i = 0; i < num_bodies(); i++) {
            if (m_assembler[i].type == RigidBodyType::KINEMATIC) {

                is_dof_satisfied.segment(ndof * i, pos_ndof).setOnes();
            }
        }
    } else if (eta_q < 0.99 && linear_augmented_lagrangian_penalty < 1e8) {

        linear_augmented_lagrangian_penalty *= 2;
    } else {
        for (size_t i = 0, ki = 0; i < num_bodies(); i++) {

            if (m_assembler[i].type == RigidBodyType::KINEMATIC) {

                linear_augmented_lagrangian_multiplier.segment(
                    ki * pos_ndof, pos_ndof) -=
                    linear_augmented_lagrangian_penalty
                    * sqrt(m_assembler[i].mass)
                    * (x.segment(ndof * i, pos_ndof)
                       - x_pred.segment(ndof * i, pos_ndof));

                ki++;
            }
        }
    }

    if (eta_Q >= 0.999) {
        for (size_t i = 0; i < num_bodies(); i++) {
            if (m_assembler[i].type == RigidBodyType::KINEMATIC) {

                is_dof_satisfied.segment(ndof * i + pos_ndof, rot_ndof)
                    .setOnes();
            }
        }
    } else if (eta_Q < 0.99 && angular_augmented_lagrangian_penalty < 1e8) {

        angular_augmented_lagrangian_penalty *= 2;
    } else {
        for (size_t i = 0, ki = 0; i < num_bodies(); i++) {

            if (m_assembler[i].type == RigidBodyType::KINEMATIC) {

                size_t ri = ndof * i + pos_ndof;

                if (dim() == 2) {
                    angular_augmented_lagrangian_multiplier.middleRows(
                        ki * rot_ndof, rot_ndof) -=
                        angular_augmented_lagrangian_penalty
                        * sqrt(m_assembler[i].moment_of_inertia[0])
                        * (x.segment(ri, rot_ndof)
                           - x_pred.segment(ri, rot_ndof));
                } else {
                    auto Q_pred = construct_rotation_matrix(
                        VectorMax3d(x_pred.segment(ri, rot_ndof)));

                    auto Q = construct_rotation_matrix(
                        VectorMax3d(x.segment(ri, rot_ndof)));

                    angular_augmented_lagrangian_multiplier.middleRows(
                        rot_ndof * ki, rot_ndof) -=
                        angular_augmented_lagrangian_penalty * (Q - Q_pred)
                        * compute_Jsqrt(m_assembler[i].moment_of_inertia);
                }

                ki++;
            }
        }
    }

    if (eta_q < 0.999 || eta_Q < 0.999) {
        spdlog::info(
            "updated augmented Lagrangian "
            "κ_q={:g} κ_Q={:g} ||λ||∞={:g} ||Λ||∞={:g} "
            "η_q={:g} η_Q={:g}",
            linear_augmented_lagrangian_penalty,
            angular_augmented_lagrangian_penalty,
            linear_augmented_lagrangian_multiplier.lpNorm<Eigen::Infinity>(),
            angular_augmented_lagrangian_multiplier.lpNorm<Eigen::Infinity>(),
            eta_q, eta_Q);
    }
}

bool DistanceBarrierRBProblem::are_equality_constraints_satisfied(
    const Eigen::VectorXd& x) const
{
    if (m_assembler.count_kinematic_bodies()) {
        return compute_linear_augment_lagrangian_progress(x) >= 0.999
            && compute_angular_augment_lagrangian_progress(x) >= 0.999;
    }

    return true;
}

OptimizationResults DistanceBarrierRBProblem::solve_constraints()
{
    OptimizationResults opt_result;

    opt_result.x = starting_point();

    double momentum_balance;
    double eps_d = m_funnel_enabled ? 1e-6 : 1e-2 * world_bbox_diagonal();
    // double eps_d =
    // 1e-2 * world_bbox_diagonal();

    int i = 0;
    int total_newton_iterations = 0;

    do {
        opt_result = solver().solve(opt_result.x);

        total_newton_iterations += opt_result.num_iterations;

        if (!opt_result.success) {
            break;
        }

        PosesD poses = this->dofs_to_poses(opt_result.x);

        Constraints collision_constraints;

        if (!m_funnel_enabled) {
            m_constraint.construct_constraint_set(
                m_assembler, poses, collision_constraints);
        }

        update_friction_constraints(collision_constraints, poses);

        Eigen::VectorXd grad_Ex;
        Eigen::VectorXd grad_Bx;
        Eigen::VectorXd grad_Dx;

        compute_energy_term(opt_result.x, grad_Ex);

        compute_barrier_term(opt_result.x, collision_constraints, grad_Bx);

        compute_friction_term(opt_result.x, grad_Dx);

        Eigen::VectorXd tmp = grad_Ex + barrier_stiffness() * grad_Bx + grad_Dx;

        tmp = is_dof_fixed().select(0, tmp);

        momentum_balance = tmp.norm();

        i++;
    } while ((friction_iterations < 0 || i < friction_iterations)
             && momentum_balance > eps_d);

    if (opt_result.success) {
        spdlog::info(
            "Finished friction solve after {:d} "
            "lagging iteration(s) and a momentum "
            "balance error of {:g}",
            i, momentum_balance);
    } else {
        spdlog::error(
            "Ending friction solve early because "
            "newton solve {:d} failed!",
            i);
    }

    opt_result.num_iterations = total_newton_iterations;

    return opt_result;
}

bool DistanceBarrierRBProblem::take_step(const Eigen::VectorXd& x)
{
    min_distance = compute_min_distance(x);

    if (min_distance < 0) {
        spdlog::info("final_step min_distance=N/A");
    } else {
        spdlog::info("final_step min_distance={:.8e}", min_distance);
    }

    const double h = timestep();

    // Update final pose.
    m_assembler.set_rb_poses(this->dofs_to_poses(x));

    PosesD poses_q1 = m_assembler.rb_poses_t1();

    // Update velocities.
    for (RigidBody& rb : m_assembler.m_rbs) {

        if (rb.type != RigidBodyType::DYNAMIC) {
            continue;
        }

        switch (body_energy_integration_method) {
        case IMPLICIT_EULER:
            rb.velocity.position =
                (rb.pose.position - rb.pose_prev.position) / h;
            break;

        case IMPLICIT_NEWMARK:
            rb.velocity.position =
                2 * (rb.pose.position - rb.pose_prev.position) / h
                - rb.velocity.position;

            rb.acceleration.position =
                2 * (rb.velocity.position - rb.velocity_prev.position) / h
                - rb.acceleration.position;
            break;

        case STABILIZED_NEWMARK: {
            rb.velocity.position =
                2 * (rb.pose.position - rb.pose_prev.position) / h
                - rb.velocity.position;

            VectorMax3d pos_tilde = rb.pose_prev.position
                + h
                    * (rb.velocity_prev.position
                       + h / 4.0
                           * (gravity + rb.force.position / rb.mass
                              + rb.acceleration.position));

            rb.acceleration.position =
                4 * (rb.pose.position - pos_tilde) / (h * h) + gravity
                + rb.force.position / rb.mass;
            break;
        }
        }

        if (dim() == 2) {
            switch (body_energy_integration_method) {
            case IMPLICIT_EULER:
                rb.velocity.rotation =
                    (rb.pose.rotation - rb.pose_prev.rotation) / h;
                break;

            case IMPLICIT_NEWMARK:
            case STABILIZED_NEWMARK:
                rb.velocity.rotation =
                    2 * (rb.pose.rotation - rb.pose_prev.rotation) / h
                    - rb.velocity.rotation;

                rb.acceleration.rotation =
                    2 * (rb.velocity.rotation - rb.velocity_prev.rotation) / h
                    - rb.acceleration.rotation;
                break;
            }
        } else {
            Eigen::Matrix3d R = rb.pose.construct_rotation_matrix()
                * rb.pose_prev.construct_rotation_matrix().transpose();

            Eigen::AngleAxisd omega(R);

            rb.velocity.rotation =
                omega.angle() / timestep() * rb.R0.transpose() * omega.axis();

            Eigen::Matrix3d Q = rb.pose.construct_rotation_matrix();

            Eigen::Matrix3d Q_prev = rb.pose_prev.construct_rotation_matrix();

            switch (body_energy_integration_method) {
            case IMPLICIT_EULER:
                rb.Qdot = (Q - Q_prev) / h;
                break;

            case IMPLICIT_NEWMARK: {
                auto Qdot_prev = rb.Qdot;

                rb.Qdot = 2 * (Q - Q_prev) / h - rb.Qdot;

                rb.Qddot = 2 * (rb.Qdot - Qdot_prev) / h - rb.Qddot;
                break;
            }

            case STABILIZED_NEWMARK: {
                auto Qdot_prev = rb.Qdot;

                rb.Qdot = 2 * (Q - Q_prev) / h - rb.Qdot;

                Eigen::Matrix3d Q_tilde =
                    Q_prev + h * (Qdot_prev + h / 4.0 * rb.Qddot);

                rb.Qddot = 4 * (Q - Q_tilde) / (h * h);
                break;
            }
            }
        }

        rb.velocity.zero_dof(rb.is_dof_fixed, rb.R0);

        rb.acceleration.zero_dof(rb.is_dof_fixed, rb.R0);
    }

    if (do_intersection_check) {
        // Original mesh behaviour.
        if (!m_funnel_enabled) {
            return detect_intersections(poses_q1);
        }

        // Analytic circle/funnel behaviour.
        std::vector<rigid_ipc::FunnelCandidate> funnel_cands;

        std::vector<rigid_ipc::CircleCircleCandidate> circle_cands;

        detect_funnel_candidates(x, funnel_cands, circle_cands);

        for (const auto& cand : funnel_cands) {
            if (cand.compute_distance() <= 0.0) {
                return true;
            }
        }

        for (const auto& cand : circle_cands) {
            if (cand.compute_distance() <= 0.0) {
                return true;
            }
        }

        return false;
    }

    return false;
}

////////////////////////////////////////////////////////////
// Barrier Problem

double DistanceBarrierRBProblem::compute_objective(
    const Eigen::VectorXd& x,
    Eigen::VectorXd& grad,
    Eigen::SparseMatrix<double>& hess,
    bool compute_grad,
    bool compute_hess)
{
    double Ex = compute_energy_term(x, grad, hess, compute_grad, compute_hess);

    Ex /= average_mass();

    if (compute_grad) {
        grad /= average_mass();
    }

    if (compute_hess) {
        hess /= average_mass();
    }

    Eigen::VectorXd grad_AL;
    Eigen::SparseMatrix<double> hess_AL;

    double ALx = compute_augmented_lagrangian(
        x, grad_AL, hess_AL, compute_grad, compute_hess);

    Ex += ALx / average_mass();

    if (compute_grad) {
        grad += grad_AL / average_mass();
    }

    if (compute_hess) {
        hess += hess_AL / average_mass();
    }

    if (!m_use_barriers) {
        return Ex;
    }

    Constraints constraints;

    if (!m_funnel_enabled) {
        m_constraint.construct_constraint_set(
            m_assembler, this->dofs_to_poses(x), constraints);
    }

    Eigen::VectorXd grad_Bx;
    Eigen::SparseMatrix<double> hess_Bx;

    double Bx = compute_barrier_term(
        x, constraints, grad_Bx, hess_Bx, compute_grad, compute_hess);

    Eigen::VectorXd grad_Dx;
    Eigen::SparseMatrix<double> hess_Dx;

    double Dx =
        compute_friction_term(x, grad_Dx, hess_Dx, compute_grad, compute_hess);

    double kappa_over_avg_mass = barrier_stiffness() / average_mass();

    if (compute_grad) {
        grad += kappa_over_avg_mass * grad_Bx + grad_Dx / average_mass();
    }

    if (compute_hess) {
        hess += kappa_over_avg_mass * hess_Bx + hess_Dx / average_mass();
    }

    return Ex + kappa_over_avg_mass * Bx + Dx / average_mass();
}

double DistanceBarrierRBProblem::compute_energy_term(
    const Eigen::VectorXd& x,
    Eigen::VectorXd& grad,
    Eigen::SparseMatrix<double>& hess,
    bool compute_grad,
    bool compute_hess)
{
    PROFILE_POINT("DistanceBarrierRBProblem::compute_energy_term");
    PROFILE_START();

    typedef AutodiffType<
        Eigen::Dynamic,
        /*maxN=*/6>
        Diff;

    int ndof = PoseD::dim_to_ndof(dim());

    int pos_ndof = PoseD::dim_to_pos_ndof(dim());

    int rot_ndof = PoseD::dim_to_rot_ndof(dim());

    Eigen::VectorXd energies = Eigen::VectorXd::Zero(num_bodies());

    if (compute_grad) {
        grad.setZero(x.size());
    }

    tbb::concurrent_vector<Eigen::Triplet<double>> hess_triplets;

    if (compute_hess) {
        hess_triplets.reserve(num_bodies() * ndof * ndof);
    }

    const std::vector<PoseD> poses = this->dofs_to_poses(x);

    assert(poses.size() == num_bodies());

    tbb::parallel_for(
        tbb::blocked_range<size_t>(size_t(0), poses.size()),
        [&](const tbb::blocked_range<size_t>& range) {
            Diff::activate(ndof);

            for (long i = range.begin(); i != range.end(); ++i) {

                const PoseD& pose = poses[i];

                const RigidBody& body = m_assembler[i];

                if (body.type != RigidBodyType::DYNAMIC) {
                    continue;
                }

                VectorMax6d gradi;

                if (compute_hess) {
                    Pose<Diff::DDouble2> pose_diff(Diff::d2vars(0, pose.dof()));

                    Diff::DDouble2 dExi = compute_body_energy<Diff::DDouble2>(
                        body, pose_diff,
                        grad_barrier_t0.segment(i * ndof, ndof));

                    energies[i] = dExi.getValue();

                    gradi = dExi.getGradient();

                    MatrixMax6d hessi = dExi.getHessian();

                    hessi.topLeftCorner(pos_ndof, pos_ndof) = project_to_psd(
                        MatrixMax3d(hessi.topLeftCorner(pos_ndof, pos_ndof)));

                    for (int r = 0; r < hessi.rows(); r++) {

                        for (int c = 0; c < hessi.cols(); c++) {

                            hess_triplets.emplace_back(
                                i * ndof + r, i * ndof + c, hessi(r, c));
                        }
                    }
                } else if (compute_grad) {
                    Pose<Diff::DDouble1> pose_diff(Diff::d1vars(0, pose.dof()));

                    Diff::DDouble1 dExi = compute_body_energy<Diff::DDouble1>(
                        body, pose_diff,
                        grad_barrier_t0.segment(i * ndof, ndof));

                    energies[i] = dExi.getValue();

                    gradi = dExi.getGradient();
                } else {
                    energies[i] = compute_body_energy<double>(
                        body, pose, grad_barrier_t0.segment(i * ndof, ndof));
                }

                if (compute_grad) {
                    grad.segment(i * ndof, ndof) = gradi;
                }
            }
        });

    if (compute_hess) {
        NAMED_PROFILE_POINT(
            "DistanceBarrierRBProblem::compute_energy_term:"
            "assemble_hessian",
            ASSEMBLE_ENERGY_HESS);

        PROFILE_START(ASSEMBLE_ENERGY_HESS);

        hess.resize(x.size(), x.size());

        hess.setFromTriplets(hess_triplets.begin(), hess_triplets.end());

        PROFILE_END(ASSEMBLE_ENERGY_HESS);
    }

    PROFILE_END();

#ifdef RIGID_IPC_WITH_DERIVATIVE_CHECK
    if (!is_checking_derivative) {
        is_checking_derivative = true;

        double mass_Linf =
            m_assembler.m_rb_mass_matrix.diagonal().lpNorm<Eigen::Infinity>();

        double tol = std::max(1e-8 * mass_Linf, 1e-4);

        if (compute_grad) {
            Eigen::VectorXd grad_approx = eval_grad_energy_approx(*this, x);

            if (!fd::compare_gradient(grad, grad_approx, tol)) {

                spdlog::error("finite gradient check failed for E(x)");
            }
        }

        if (compute_hess) {
            Eigen::MatrixXd hess_approx = eval_hess_energy_approx(*this, x);

            if (!fd::compare_jacobian(hess, hess_approx, tol)) {

                spdlog::error("finite hessian check failed for E(x)");
            }
        }

        is_checking_derivative = false;
    }
#endif

    return energies.sum();
}

template <typename T>
T DistanceBarrierRBProblem::compute_body_energy(
    const RigidBody& body,
    const Pose<T>& pose,
    const VectorMax6d& /*grad_barrier_t0*/)
{
    double h = timestep();

    T energy(0.0);

    // Linear energy.
    if (!body.is_dof_fixed.head(pose.pos_ndof()).all()) {

        VectorMax3<T> q = pose.position;

        const VectorMax3d& q_t0 = body.pose.position;

        const VectorMax3d& qdot_t0 = body.velocity.position;

        VectorMax3d qddot_t0 = gravity + body.force.position / body.mass;

        switch (body_energy_integration_method) {
        case IMPLICIT_EULER:
            break;

        case IMPLICIT_NEWMARK:
        case STABILIZED_NEWMARK:
            qddot_t0 += body.acceleration.position;

            qddot_t0 *= 0.25;
            break;
        }

        energy += 0.5 * body.mass * q.dot(q)
            - body.mass * q.dot(q_t0 + h * (qdot_t0 + h * qddot_t0));
    }

    // Rotational energy.
    if (!body.is_dof_fixed.tail(pose.rot_ndof()).all()) {

        if (dim() == 3) {
            Matrix3<T> Q = pose.construct_rotation_matrix();

            Eigen::Matrix3d Q_t0 = body.pose.construct_rotation_matrix();

            Eigen::Matrix3d Qdot_t0 = body.Qdot;

            DiagonalMatrix3d J = compute_J(body.moment_of_inertia);

            Eigen::Matrix3d Qddot_t0;

            switch (body_energy_integration_method) {
            case IMPLICIT_EULER:
                Qddot_t0.setZero();
                break;

            case IMPLICIT_NEWMARK:
            case STABILIZED_NEWMARK:
                Qddot_t0 = 0.25 * body.Qddot;
                break;
            }

            energy += 0.5 * (Q * J * Q.transpose()).trace();

            energy -=
                (Q * J * (Q_t0 + h * (Qdot_t0 + h * Qddot_t0)).transpose())
                    .trace();

            Eigen::Matrix3d Tau = Q_t0.transpose() * Hat(body.force.rotation);

            switch (body_energy_integration_method) {
            case IMPLICIT_EULER:
                energy += h * h * (Q * Tau).trace();
                break;

            case IMPLICIT_NEWMARK:
            case STABILIZED_NEWMARK:
                energy += 0.25 * h * h * (Q * Tau).trace();
                break;
            }
        } else {
            assert(pose.rot_ndof() == 1);

            T theta = pose.rotation[0];

            double theta_t0 = body.pose.rotation[0];

            double theta_dot_t0 = body.velocity.rotation[0];

            double I = body.moment_of_inertia[0];

            double theta_ddot_t0 = body.force.rotation[0] / I;

            switch (body_energy_integration_method) {
            case IMPLICIT_EULER:
                break;

            case IMPLICIT_NEWMARK:
            case STABILIZED_NEWMARK:
                theta_ddot_t0 += body.acceleration.rotation[0];

                theta_ddot_t0 *= 0.25;
                break;
            }

            double theta_hat =
                theta_t0 + h * (theta_dot_t0 + h * theta_ddot_t0);

            energy += 0.5 * I * theta * theta - I * theta * theta_hat;
        }
    }

    return energy;
}

double DistanceBarrierRBProblem::compute_augmented_lagrangian(
    const Eigen::VectorXd& x,
    Eigen::VectorXd& grad,
    Eigen::SparseMatrix<double>& hess,
    bool compute_grad,
    bool compute_hess)
{
    int ndof = PoseD::dim_to_ndof(dim());

    int pos_ndof = PoseD::dim_to_pos_ndof(dim());

    int rot_ndof = PoseD::dim_to_rot_ndof(dim());

    size_t num_kinematic_bodies = m_assembler.count_kinematic_bodies();

    double potential = 0;

    if (compute_grad) {
        grad.setZero(x.size());
    }

    std::vector<Eigen::Triplet<double>> hess_triplets;

    if (compute_hess) {
        hess.resize(x.size(), x.size());

        hess_triplets.reserve(num_kinematic_bodies * ndof);
    }

    bool all_kinematic_dof_satisfied = true;

    for (size_t i = 0; i < num_bodies(); i++) {

        if (m_assembler[i].type == RigidBodyType::KINEMATIC
            && !is_dof_satisfied.segment(ndof * i, ndof).all()) {

            all_kinematic_dof_satisfied = false;

            break;
        }
    }

    if (all_kinematic_dof_satisfied) {
        return potential;
    }

    PROFILE_POINT("DistanceBarrierRBProblem::compute_augmented_lagrangian");
    PROFILE_START();

    const double& kappa_q = linear_augmented_lagrangian_penalty;

    for (size_t i = 0, ki = 0; i < num_bodies(); i++) {

        if (m_assembler[i].type != RigidBodyType::KINEMATIC) {
            continue;
        }

        double m = m_assembler[i].mass;

        const auto& lambda = linear_augmented_lagrangian_multiplier.segment(
            ki * pos_ndof, pos_ndof);

        const auto& q = x.segment(i * ndof, pos_ndof);

        const auto& q_pred = x_pred.segment(i * ndof, pos_ndof);

        potential += kappa_q / 2 * m * (q - q_pred).squaredNorm()
            - sqrt(m) * lambda.dot(q - q_pred);

        if (compute_grad) {
            grad.segment(i * ndof, pos_ndof) =
                kappa_q * m * (q - q_pred) - sqrt(m) * lambda;
        }

        if (compute_hess) {
            for (int j = 0; j < pos_ndof; j++) {

                hess_triplets.emplace_back(
                    ndof * i + j, ndof * i + j, kappa_q * m);
            }
        }

        ki++;
    }

    typedef AutodiffType<
        Eigen::Dynamic,
        /*maxN=*/3>
        Diff;

    Diff::activate(rot_ndof);

    const double& kappa_Q = angular_augmented_lagrangian_penalty;

    for (size_t i = 0, ki = 0; i < num_bodies(); i++) {

        if (m_assembler[i].type != RigidBodyType::KINEMATIC) {
            continue;
        }

        const VectorMax3d& moment_of_inertia = m_assembler[i].moment_of_inertia;

        MatrixMax3d lambda = angular_augmented_lagrangian_multiplier.middleRows(
            rot_ndof * ki, rot_ndof);

        VectorMax3d theta = x.segment(i * ndof + pos_ndof, rot_ndof);

        VectorMax3d theta_pred = x_pred.segment(i * ndof + pos_ndof, rot_ndof);

        if (dim() == 2) {
            double I = moment_of_inertia(0);

            double Isqrt = sqrt(I);

            potential += kappa_Q / 2 * I * (theta - theta_pred).squaredNorm()
                - (Isqrt * lambda.transpose() * (theta - theta_pred)).trace();

            if (compute_grad) {
                grad.segment(i * ndof + pos_ndof, rot_ndof) =
                    kappa_Q * I * (theta - theta_pred) - Isqrt * lambda;
            }

            if (compute_hess) {
                for (int j = 0; j < rot_ndof; j++) {

                    hess_triplets.emplace_back(
                        ndof * i + pos_ndof + j, ndof * i + pos_ndof + j,
                        kappa_Q * I);
                }
            }
        } else {
            VectorMax3<Diff::DDouble2> theta_diff = Diff::d2vars(0, theta);

            DiagonalMatrix3d J = compute_J(moment_of_inertia);

            DiagonalMatrix3d Jsqrt = compute_Jsqrt(moment_of_inertia);

            const auto& Q = construct_rotation_matrix(theta_diff);

            const auto& Q_pred = construct_rotation_matrix(theta_pred);

            Diff::DDouble2 dAL = kappa_Q / 2
                    * ((Q - Q_pred) * J * (Q - Q_pred).transpose()).trace()
                - (lambda.transpose() * (Q - Q_pred) * Jsqrt).trace();

            potential += dAL.getValue();

            if (compute_grad) {
                grad.segment(i * ndof + pos_ndof, rot_ndof) = dAL.getGradient();
            }

            if (compute_hess) {
                Eigen::Matrix3d H = dAL.getHessian();

                for (int hi = 0; hi < H.rows(); hi++) {

                    for (int hj = 0; hj < H.cols(); hj++) {

                        hess_triplets.emplace_back(
                            ndof * i + pos_ndof + hi, ndof * i + pos_ndof + hj,
                            H(hi, hj));
                    }
                }
            }
        }

        ki++;
    }

    if (compute_hess) {
        NAMED_PROFILE_POINT(
            "DistanceBarrierRBProblem::compute_augmented_lagrangian:"
            "assemble_hessian",
            ASSEMBLE_AL_HESS);

        PROFILE_START(ASSEMBLE_AL_HESS);

        hess.setFromTriplets(hess_triplets.begin(), hess_triplets.end());

        PROFILE_END(ASSEMBLE_AL_HESS);
    }

    PROFILE_END();

#ifdef RIGID_IPC_WITH_DERIVATIVE_CHECK
    if (!is_checking_derivative) {
        is_checking_derivative = true;

        if (compute_grad) {
            check_augmented_lagrangian_gradient(x, grad);
        }

        if (compute_hess) {
            check_augmented_lagrangian_hessian(x, hess);
        }

        is_checking_derivative = false;
    }
#endif

    return potential;
}

// Compute B(x) = ∑ b(d(x_k)).
double DistanceBarrierRBProblem::compute_barrier_term(
    const Eigen::VectorXd& x,
    Eigen::VectorXd& grad,
    Eigen::SparseMatrix<double>& hess,
    int& num_constraints,
    bool compute_grad,
    bool compute_hess)
{
    PosesD poses = this->dofs_to_poses(x);

    Constraints constraints;

    if (!m_funnel_enabled) {
        m_constraint.construct_constraint_set(m_assembler, poses, constraints);

        num_constraints = constraints.num_constraints();
    } else {

        std::vector<rigid_ipc::FunnelCandidate> funnel_candidates;

        std::vector<rigid_ipc::CircleCircleCandidate> circle_candidates;

        detect_funnel_candidates(x, funnel_candidates, circle_candidates);

        num_constraints = static_cast<int>(
            funnel_candidates.size() + circle_candidates.size());
    }

    m_num_contacts = std::max(m_num_contacts, num_constraints);

    return compute_barrier_term(
        x, constraints, grad, hess, compute_grad, compute_hess);
}

template <typename DerivedLocalGradient>
void local_gradient_to_global(
    const Eigen::MatrixBase<DerivedLocalGradient>& local_gradient,
    const std::array<long, 2>& body_ids,
    int ndof,
    Eigen::VectorXd& grad)
{
    assert(local_gradient.size() == 2 * ndof);

    for (int b_i = 0; b_i < body_ids.size(); b_i++) {

        grad.segment(ndof * body_ids[b_i], ndof) +=
            local_gradient.segment(ndof * b_i, ndof);
    }
}

template <typename DerivedLocalHessian>
void local_hessian_to_global_triplets(
    const Eigen::MatrixBase<DerivedLocalHessian>& local_hessian,
    const std::array<long, 2>& body_ids,
    int ndof,
    std::vector<Eigen::Triplet<double>>& triplets)
{
    assert(local_hessian.rows() == 2 * ndof);

    assert(local_hessian.cols() == 2 * ndof);

    for (int b_i = 0; b_i < body_ids.size(); b_i++) {

        for (int b_j = 0; b_j < body_ids.size(); b_j++) {

            for (int dof_i = 0; dof_i < ndof; dof_i++) {

                for (int dof_j = 0; dof_j < ndof; dof_j++) {

                    double v =
                        local_hessian(ndof * b_i + dof_i, ndof * b_j + dof_j);

                    int r = ndof * body_ids[b_i] + dof_i;

                    int c = ndof * body_ids[b_j] + dof_j;

                    triplets.emplace_back(r, c, v);
                }
            }
        }
    }
}

void apply_chain_rule(
    const VectorMax12d& grad_f,
    const Eigen::MatrixXd& jac_V,
    const MatrixMax12d& hess_f,
    const Eigen::MatrixXd& hess_V,
    const std::vector<long>& vertex_ids,
    const std::vector<uint8_t>& local_body_ids,
    const std::array<long, 2>& body_ids,
    const int dim,
    Eigen::VectorXd& grad,
    std::vector<Eigen::Triplet<double>>& hess_triplets,
    bool compute_grad,
    bool compute_hess)
{
    if (!compute_grad && !compute_hess) {
        return;
    }

    const int rb_ndof = PoseD::dim_to_ndof(dim);

    if (compute_grad) {
        VectorMax12d local_grad = VectorMax12d::Zero(2 * rb_ndof);

        for (int i = 0; i < vertex_ids.size(); i++) {

            local_grad.segment(rb_ndof * local_body_ids[i], rb_ndof) +=
                jac_V.middleRows(vertex_ids[i] * dim, dim).transpose()
                * grad_f.segment(i * dim, dim);
        }

        local_gradient_to_global(local_grad, body_ids, rb_ndof, grad);
    }
    if (compute_hess) {
        MatrixMax12d jac_Vi =
            MatrixMax12d::Zero(vertex_ids.size() * dim, 2 * rb_ndof);

        for (int i = 0; i < vertex_ids.size(); i++) {

            jac_Vi.block(i * dim, local_body_ids[i] * rb_ndof, dim, rb_ndof) =
                jac_V.middleRows(vertex_ids[i] * dim, dim);
        }

        MatrixMax12d hess = jac_Vi.transpose() * hess_f * jac_Vi;

        for (int i = 0; i < vertex_ids.size(); i++) {

            for (int j = 0; j < dim; j++) {

                hess.block(
                    local_body_ids[i] * rb_ndof, local_body_ids[i] * rb_ndof,
                    rb_ndof, rb_ndof) +=
                    hess_V.middleRows(
                        rb_ndof * (vertex_ids[i] * dim + j), rb_ndof)
                    * grad_f[i * dim + j];
            }
        }

        hess = project_to_psd(hess);

        local_hessian_to_global_triplets(
            hess, body_ids, rb_ndof, hess_triplets);
    }
}

struct PotentialStorage {
    PotentialStorage() { }

    PotentialStorage(size_t nvars) { gradient.setZero(nvars); }

    double potential = 0;

    Eigen::VectorXd gradient;

    std::vector<Eigen::Triplet<double>> hessian_triplets;
};

typedef tbb::enumerable_thread_specific<PotentialStorage>
    ThreadSpecificPotentials;

double merge_derivative_storage(
    const ThreadSpecificPotentials& potentials,
    size_t nvars,
    Eigen::VectorXd& grad,
    Eigen::SparseMatrix<double>& hess,
    bool compute_grad,
    bool compute_hess)
{
    PROFILE_POINT("merge_derivative_storage");
    PROFILE_START();

    if (compute_grad) {
        grad.setZero(nvars);
    }

    if (compute_hess) {
        hess.resize(nvars, nvars);
    }

    double potential = 0;

    for (const auto& p : potentials) {
        potential += p.potential;

        if (compute_grad) {
            grad += p.gradient;
        }

        if (compute_hess) {
            Eigen::SparseMatrix<double> p_hess(nvars, nvars);

            p_hess.setFromTriplets(
                p.hessian_triplets.begin(), p.hessian_triplets.end());

            hess += p_hess;
        }
    }

    PROFILE_END();

    return potential;
}

double DistanceBarrierRBProblem::compute_barrier_term(
    const Eigen::VectorXd& x,
    const Constraints& constraints,
    Eigen::VectorXd& grad,
    Eigen::SparseMatrix<double>& hess,
    bool compute_grad,
    bool compute_hess)
{
    if (constraints.size() == 0 && !m_funnel_enabled) {

        if (compute_grad) {
            grad.setZero(x.size());
        }

        if (compute_hess) {
            hess.resize(x.size(), x.size());
        }

        return 0;
    }

    PROFILE_POINT("DistanceBarrierRBProblem::compute_barrier_term");
    PROFILE_START();

    int rb_ndof = PoseD::dim_to_ndof(dim());

    Eigen::MatrixXd jac_V;
    Eigen::MatrixXd hess_V;

    Eigen::MatrixXd V = m_assembler.world_vertices_diff(
        x, jac_V, hess_V, compute_grad || compute_hess, compute_hess);

    double dhat = barrier_activation_distance();

    ThreadSpecificPotentials thread_storage(x.size());

    tbb::parallel_for(
        tbb::blocked_range<size_t>(size_t(0), constraints.size()),
        [&](const tbb::blocked_range<size_t>& range) {
            auto& local_storage = thread_storage.local();

            auto& potential = local_storage.potential;

            auto& local_grad = local_storage.gradient;

            auto& hess_triplets = local_storage.hessian_triplets;

            for (size_t ci = range.begin(); ci != range.end(); ++ci) {

                const auto& constraint = constraints[ci];

                potential +=
                    constraint.compute_potential(V, edges(), faces(), dhat);

                VectorMax12d grad_B;

                if (compute_grad || compute_hess) {
                    grad_B = constraint.compute_potential_gradient(
                        V, edges(), faces(), dhat);
                }

                MatrixMax12d hess_B;

                if (compute_hess) {
                    hess_B = constraint.compute_potential_hessian(
                        V, edges(), faces(), dhat,
                        /*project_hessian_to_psd=*/
                        false);
                }

                apply_chain_rule(
                    grad_B, jac_V, hess_B, hess_V,
                    constraint.vertex_indices(edges(), faces()),
                    vertex_local_body_ids(constraints, ci),
                    body_ids(m_assembler, constraints, ci), dim(), local_grad,
                    hess_triplets, compute_grad, compute_hess);
            }
        });

    double potential = merge_derivative_storage(
        thread_storage, x.size(), grad, hess, compute_grad, compute_hess);
    // Custom analytic funnel and circle-circle barriers.
    if (m_funnel_enabled) {
        std::vector<rigid_ipc::FunnelCandidate> funnel_cands;
        std::vector<rigid_ipc::CircleCircleCandidate> circle_cands;

        detect_funnel_candidates(x, funnel_cands, circle_cands);

        std::vector<Eigen::Triplet<double>> custom_triplets;

        const double dmin = m_constraint.minimum_separation_distance;

        const double dhat = barrier_activation_distance();

        const double z_hat = 2.0 * dmin * dhat + dhat * dhat;

        // Funnel candidates
        for (const auto& cand : funnel_cands) {
            const double gap = cand.compute_distance();

            if (gap >= dmin + dhat) {
                continue;
            }

            if (gap <= dmin) {
                potential = std::numeric_limits<double>::infinity();
                continue;
            }

            const double squared_distance = gap * gap;

            const double z = squared_distance - dmin * dmin;

            potential += ipc::barrier(z, z_hat);

            if (compute_grad || compute_hess) {
                const Eigen::Vector3d grad_gap = cand.compute_gradient();

                const Eigen::Vector3d grad_z = 2.0 * gap * grad_gap;

                const double b_prime = ipc::barrier_gradient(z, z_hat);

                if (compute_grad) {
                    grad.segment(rb_ndof * cand.body_id, rb_ndof) +=
                        b_prime * grad_z;
                }

                if (compute_hess) {
                    const Eigen::Matrix3d H_gap = cand.compute_hessian();

                    const Eigen::Matrix3d H_z =
                        2.0 * (grad_gap * grad_gap.transpose())
                        + 2.0 * gap * H_gap;

                    const double b_pprime = ipc::barrier_hessian(z, z_hat);

                    Eigen::Matrix3d local_H =
                        b_pprime * (grad_z * grad_z.transpose())
                        + b_prime * H_z;

                    local_H = project_to_psd(local_H);

                    const int offset = rb_ndof * cand.body_id;

                    for (int r = 0; r < rb_ndof; ++r) {
                        for (int c = 0; c < rb_ndof; ++c) {
                            custom_triplets.emplace_back(
                                offset + r, offset + c, local_H(r, c));
                        }
                    }
                }
            }
        }

        // Circle-circle candidates
        for (const auto& cand : circle_cands) {
            const double gap = cand.compute_distance();

            if (gap >= dmin + dhat) {
                continue;
            }

            if (gap <= dmin) {
                potential = std::numeric_limits<double>::infinity();
                continue;
            }

            const double squared_distance = gap * gap;

            const double z = squared_distance - dmin * dmin;

            potential += ipc::barrier(z, z_hat);

            if (compute_grad || compute_hess) {
                const Eigen::Matrix<double, 6, 1> grad_gap =
                    cand.compute_gradient();

                const Eigen::Matrix<double, 6, 1> grad_z = 2.0 * gap * grad_gap;

                const double b_prime = ipc::barrier_gradient(z, z_hat);

                if (compute_grad) {
                    grad.segment(rb_ndof * cand.body1_id, rb_ndof) +=
                        b_prime * grad_z.head(rb_ndof);

                    grad.segment(rb_ndof * cand.body2_id, rb_ndof) +=
                        b_prime * grad_z.tail(rb_ndof);
                }

                if (compute_hess) {
                    const Eigen::Matrix<double, 6, 6> H_gap =
                        cand.compute_hessian();

                    const Eigen::Matrix<double, 6, 6> H_z =
                        2.0 * (grad_gap * grad_gap.transpose())
                        + 2.0 * gap * H_gap;

                    const double b_pprime = ipc::barrier_hessian(z, z_hat);

                    Eigen::Matrix<double, 6, 6> local_H =
                        b_pprime * (grad_z * grad_z.transpose())
                        + b_prime * H_z;

                    local_H = project_to_psd(local_H);

                    const int ids[2] = { cand.body1_id, cand.body2_id };

                    for (int bi = 0; bi < 2; ++bi) {
                        for (int bj = 0; bj < 2; ++bj) {
                            const int r_off = rb_ndof * ids[bi];

                            const int c_off = rb_ndof * ids[bj];

                            for (int r = 0; r < rb_ndof; ++r) {
                                for (int c = 0; c < rb_ndof; ++c) {
                                    custom_triplets.emplace_back(
                                        r_off + r, c_off + c,
                                        local_H(
                                            rb_ndof * bi + r,
                                            rb_ndof * bj + c));
                                }
                            }
                        }
                    }
                }
            }
        }

        if (compute_hess && !custom_triplets.empty()) {

            Eigen::SparseMatrix<double> custom_hess(x.size(), x.size());

            custom_hess.setFromTriplets(
                custom_triplets.begin(), custom_triplets.end());

            hess += custom_hess;
        }
    }

    PROFILE_END();

#ifdef RIGID_IPC_WITH_DERIVATIVE_CHECK
    if (!is_checking_derivative) {
        is_checking_derivative = true;

        if (compute_grad) {
            check_barrier_gradient(x, constraints, grad);
        }

        if (compute_hess) {
            check_barrier_hessian(x, constraints, hess);
        }

        is_checking_derivative = false;
    }
#endif

    return potential;
}

template <typename RigidBodyConstraint, typename FrictionConstraint>
double DistanceBarrierRBProblem::compute_friction_potential(
    const Eigen::MatrixXd& U,
    const Eigen::MatrixXd& jac_V,
    const Eigen::MatrixXd& hess_V,
    const FrictionConstraint& constraint,
    Eigen::VectorXd& grad,
    std::vector<Eigen::Triplet<double>>& hess_triplets,
    bool compute_grad,
    bool compute_hess)
{
    int rb_ndof = PoseD::dim_to_ndof(dim());

    double epsv_times_h = static_friction_speed_bound * timestep();

    double Dx = constraint.compute_potential(U, edges(), faces(), epsv_times_h);

    VectorMax12d grad_D;

    if (compute_grad || compute_hess) {
        grad_D = constraint.compute_potential_gradient(
            U, edges(), faces(), epsv_times_h);
    }

    MatrixMax12d hess_D;

    if (compute_hess) {
        hess_D = constraint.compute_potential_hessian(
            U, edges(), faces(), epsv_times_h,
            /*project_hessian_to_psd=*/
            false);
    }

    RigidBodyConstraint rbc(m_assembler, constraint);

    apply_chain_rule(
        grad_D, jac_V, hess_D, hess_V,
        constraint.vertex_indices(edges(), faces()),
        rbc.vertex_local_body_ids(), rbc.body_ids(), dim(), grad, hess_triplets,
        compute_grad, compute_hess);

    return Dx;
}

double DistanceBarrierRBProblem::compute_friction_term(
    const Eigen::VectorXd& x,
    Eigen::VectorXd& grad,
    Eigen::SparseMatrix<double>& hess,
    bool compute_grad,
    bool compute_hess)
{
    if (coefficient_friction <= 0
        || (friction_constraints.size() == 0
            && analytic_friction_constraints.empty())) {
        if (compute_grad)
            grad.setZero(x.size());
        if (compute_hess) {
            hess.resize(x.size(), x.size());
            hess.setZero();
        }
        return 0;
    }

    PROFILE_POINT("DistanceBarrierRBProblem::compute_friction_term");
    PROFILE_START();

    // Analytic circle friction
    if (m_funnel_enabled) {
        assert(dim() == 2);
        const int ndof = PoseD::dim_to_ndof(dim());
        assert(ndof == 3);

        if (compute_grad)
            grad.setZero(x.size());

        std::vector<Eigen::Triplet<double>> triplets;
        if (compute_hess) {
            hess.resize(x.size(), x.size());
            hess.setZero();
            triplets.reserve(36 * analytic_friction_constraints.size());
        }

        const double eps =
            std::max(static_friction_speed_bound * timestep(), 1e-12);

        auto rotate = [](const Eigen::Vector2d& p, double theta) {
            const double c = std::cos(theta), s = std::sin(theta);
            return Eigen::Vector2d(
                c * p.x() - s * p.y(), s * p.x() + c * p.y());
        };

        auto perp = [](const Eigen::Vector2d& p) {
            return Eigen::Vector2d(-p.y(), p.x());
        };

        auto point_x = [&](int body_id,
                           const Eigen::Vector2d& local) -> Eigen::Vector2d {
            const int off = body_id * ndof;
            const Eigen::Vector2d q(x[off], x[off + 1]);
            const Eigen::Vector2d r = rotate(local, x[off + 2]);
            return q + r;
        };

        auto point_t0 = [&](int body_id,
                            const Eigen::Vector2d& local) -> Eigen::Vector2d {
            const PoseD& pose = poses_t0[body_id];
            const Eigen::Vector2d q(pose.position[0], pose.position[1]);
            const Eigen::Vector2d r = rotate(local, pose.rotation[0]);
            return q + r;
        };

        double potential = 0.0;

        for (const auto& fc : analytic_friction_constraints) {
            const bool is_pair = fc.body2_id >= 0;
            const int nlocal = is_pair ? 6 : 3;

            Eigen::VectorXd ds = Eigen::VectorXd::Zero(nlocal);
            Eigen::MatrixXd Hs = Eigen::MatrixXd::Zero(nlocal, nlocal);

            const int off1 = fc.body1_id * ndof;
            const Eigen::Vector2d p1 = point_x(fc.body1_id, fc.local_point1);
            const Eigen::Vector2d p10 = point_t0(fc.body1_id, fc.local_point1);
            const Eigen::Vector2d r1 = rotate(fc.local_point1, x[off1 + 2]);

            double slip = fc.tangent.dot(p1 - p10);

            ds[0] = fc.tangent.x();
            ds[1] = fc.tangent.y();
            ds[2] = fc.tangent.dot(perp(r1));
            Hs(2, 2) = -fc.tangent.dot(r1);

            if (is_pair) {
                const int off2 = fc.body2_id * ndof;
                const Eigen::Vector2d p2 =
                    point_x(fc.body2_id, fc.local_point2);
                const Eigen::Vector2d p20 =
                    point_t0(fc.body2_id, fc.local_point2);
                const Eigen::Vector2d r2 = rotate(fc.local_point2, x[off2 + 2]);

                slip -= fc.tangent.dot(p2 - p20);

                ds[3] = -fc.tangent.x();
                ds[4] = -fc.tangent.y();
                ds[5] = -fc.tangent.dot(perp(r2));
                Hs(5, 5) = fc.tangent.dot(r2);
            }

            const double r = std::abs(slip);
            const double eps2 = eps * eps;

            double f, df, d2f;
            if (r < eps) {
                f = -r * r * r / (3.0 * eps2) + r * r / eps + eps / 3.0;
                df = slip * (2.0 / eps - r / eps2);
                d2f = 2.0 / eps - 2.0 * r / eps2;
            } else {
                f = r;
                df = slip > 0.0 ? 1.0 : (slip < 0.0 ? -1.0 : 0.0);
                d2f = 0.0;
            }

            const double weight = coefficient_friction * fc.normal_force;

            potential += weight * f;

            if (compute_grad) {
                const Eigen::VectorXd local_grad = weight * df * ds;

                grad.segment(off1, ndof) += local_grad.head(ndof);

                if (is_pair) {
                    const int off2 = fc.body2_id * ndof;
                    grad.segment(off2, ndof) += local_grad.tail(ndof);
                }
            }

            if (compute_hess) {
                Eigen::MatrixXd local_H =
                    weight * (d2f * ds * ds.transpose() + df * Hs);

                local_H = project_to_psd(local_H);

                const int ids[2] = { fc.body1_id, fc.body2_id };
                const int nbodies = is_pair ? 2 : 1;

                for (int bi = 0; bi < nbodies; ++bi) {
                    for (int bj = 0; bj < nbodies; ++bj) {
                        const int roff = ids[bi] * ndof;
                        const int coff = ids[bj] * ndof;

                        for (int r_i = 0; r_i < ndof; ++r_i) {
                            for (int c_i = 0; c_i < ndof; ++c_i) {
                                const double v =
                                    local_H(bi * ndof + r_i, bj * ndof + c_i);
                                if (v != 0.0)
                                    triplets.emplace_back(
                                        roff + r_i, coff + c_i, v);
                            }
                        }
                    }
                }
            }
        }

        if (compute_hess)
            hess.setFromTriplets(triplets.begin(), triplets.end());

        PROFILE_END();
        return potential;
    }

    // Original mesh friction path
    Eigen::MatrixXd jac_V, hess_V;
    Eigen::MatrixXd V1 = m_assembler.world_vertices_diff(
        x, jac_V, hess_V, compute_grad || compute_hess, compute_hess);

    NAMED_PROFILE_POINT(
        "DistanceBarrierRBProblem::compute_friction_term:displacement",
        DISPLACEMENT);
    PROFILE_START(DISPLACEMENT);

    Eigen::MatrixXd U = V1 - m_assembler.world_vertices(poses_t0);

    PROFILE_END(DISPLACEMENT);

    ThreadSpecificPotentials thread_storage(x.size());

    tbb::parallel_for(
        tbb::blocked_range<size_t>(size_t(0), friction_constraints.size()),
        [&](const tbb::blocked_range<size_t>& range) {
            auto& local_storage = thread_storage.local();
            auto& potential = local_storage.potential;
            auto& local_grad = local_storage.gradient;
            auto& hess_triplets = local_storage.hessian_triplets;

            for (size_t ci = range.begin(); ci != range.end(); ++ci) {
                size_t local_ci = ci;

                if (local_ci < friction_constraints.vv_constraints.size()) {
                    potential += compute_friction_potential<
                        RigidBodyVertexVertexConstraint>(
                        U, jac_V, hess_V,
                        friction_constraints.vv_constraints[local_ci],
                        local_grad, hess_triplets, compute_grad, compute_hess);
                    continue;
                }

                local_ci -= friction_constraints.vv_constraints.size();

                if (local_ci < friction_constraints.ev_constraints.size()) {
                    potential += compute_friction_potential<
                        RigidBodyEdgeVertexConstraint>(
                        U, jac_V, hess_V,
                        friction_constraints.ev_constraints[local_ci],
                        local_grad, hess_triplets, compute_grad, compute_hess);
                    continue;
                }

                local_ci -= friction_constraints.ev_constraints.size();

                if (local_ci < friction_constraints.ee_constraints.size()) {
                    potential +=
                        compute_friction_potential<RigidBodyEdgeEdgeConstraint>(
                            U, jac_V, hess_V,
                            friction_constraints.ee_constraints[local_ci],
                            local_grad, hess_triplets, compute_grad,
                            compute_hess);
                    continue;
                }

                local_ci -= friction_constraints.ee_constraints.size();
                assert(local_ci < friction_constraints.fv_constraints.size());

                potential +=
                    compute_friction_potential<RigidBodyFaceVertexConstraint>(
                        U, jac_V, hess_V,
                        friction_constraints.fv_constraints[local_ci],
                        local_grad, hess_triplets, compute_grad, compute_hess);
            }
        });

    double potential = merge_derivative_storage(
        thread_storage, x.size(), grad, hess, compute_grad, compute_hess);

    PROFILE_END();

#ifdef RIGID_IPC_WITH_DERIVATIVE_CHECK
    if (!is_checking_derivative) {
        is_checking_derivative = true;
        if (compute_grad)
            check_friction_gradient(x, grad);
        if (compute_hess)
            check_friction_hessian(x, hess);
        is_checking_derivative = false;
    }
#endif

    return potential;
}

double DistanceBarrierRBProblem::compute_min_distance() const
{
    if (!m_funnel_enabled) {
        double min_distance = m_constraint.compute_minimum_distance(
            m_assembler, m_assembler.rb_poses());

        return std::isfinite(min_distance) ? min_distance : -1;
    }

    return compute_min_distance(starting_point());
}

double
DistanceBarrierRBProblem::compute_min_distance(const Eigen::VectorXd& x) const
{
    if (x.size() == 0) {
        return -1.0;
    }

    PosesD poses = this->dofs_to_poses(x);

    double min_d = std::numeric_limits<double>::infinity();

    if (!m_funnel_enabled) {
        min_d = m_constraint.compute_minimum_distance(m_assembler, poses);
    }

    if (m_funnel_enabled) {
        std::vector<rigid_ipc::FunnelCandidate> funnel_cands;

        std::vector<rigid_ipc::CircleCircleCandidate> circle_cands;

        detect_funnel_candidates(x, funnel_cands, circle_cands);

        for (const auto& cand : funnel_cands) {

            min_d = std::min(min_d, cand.compute_distance());
        }

        for (const auto& cand : circle_cands) {

            min_d = std::min(min_d, cand.compute_distance());
        }
    }

    return std::isfinite(min_d) ? min_d : -1;
}

bool DistanceBarrierRBProblem::has_collisions(
    const Eigen::VectorXd& x_i, const Eigen::VectorXd& x_j)
{
    bool collisions = false;

    if (!m_funnel_enabled) {
        // Original mesh behaviour.
        PosesD poses_i = this->dofs_to_poses(x_i);

        PosesD poses_j = this->dofs_to_poses(x_j);

        collisions =
            m_constraint.has_active_collisions(m_assembler, poses_i, poses_j);
    } else {
        // Analytic behaviour.
        collisions = compute_earliest_toi(x_i, x_j) <= 1.0;
    }

    m_had_collisions |= collisions;

    return m_use_barriers ? collisions : false;
}

double DistanceBarrierRBProblem::compute_earliest_toi(
    const Eigen::VectorXd& x_i, const Eigen::VectorXd& x_j)
{
    // Original mesh path.
    if (!m_funnel_enabled) {
        // Preserve original rigid-IPC behaviour when barriers are disabled.
        if (!m_use_barriers) {
            this->has_collisions(x_i, x_j);

            return std::numeric_limits<double>::infinity();
        }

        PosesD poses_i = this->dofs_to_poses(x_i);

        PosesD poses_j = this->dofs_to_poses(x_j);

        double earliest_toi =
            m_constraint.compute_earliest_toi(m_assembler, poses_i, poses_j);

        m_had_collisions |= earliest_toi <= 1.0;

        return earliest_toi;
    }

    // Analytic circle/funnel path.
    double earliest_toi = std::numeric_limits<double>::infinity();

    const int num_substeps = 100;

    for (int step = 1; step <= num_substeps; ++step) {

        const double alpha_start = double(step - 1) / num_substeps;

        const double alpha_end = double(step) / num_substeps;

        const Eigen::VectorXd x_alpha =
            (1.0 - alpha_end) * x_i + alpha_end * x_j;

        std::vector<rigid_ipc::FunnelCandidate> funnel_cands;

        std::vector<rigid_ipc::CircleCircleCandidate> circle_cands;

        detect_funnel_candidates(x_alpha, funnel_cands, circle_cands);

        bool penetrated = false;

        for (const auto& cand : funnel_cands) {
            if (cand.compute_distance() <= 0.0) {
                penetrated = true;
                break;
            }
        }

        if (!penetrated) {
            for (const auto& cand : circle_cands) {
                if (cand.compute_distance() <= 0.0) {
                    penetrated = true;
                    break;
                }
            }
        }

        if (penetrated) {
            earliest_toi = alpha_start;
            break;
        }
    }

    m_had_collisions |= earliest_toi <= 1.0;

    // When collision solving is disabled, record whether a collision
    // occurred but do not restrict the step.
    return m_use_barriers ? earliest_toi
                          : std::numeric_limits<double>::infinity();
}

void DistanceBarrierRBProblem::detect_funnel_candidates(
    const Eigen::VectorXd& x,
    std::vector<rigid_ipc::FunnelCandidate>& funnel_candidates,
    std::vector<rigid_ipc::CircleCircleCandidate>& circle_candidates) const
{
    if (x.size() == 0) {
        return;
    }

    const double activation_gap = m_constraint.minimum_separation_distance
        + barrier_activation_distance();

    detect_custom_funnel_candidates(
        x, m_funnel, m_funnel_height, activation_gap, funnel_candidates,
        circle_candidates);
}

#ifdef RIGID_IPC_WITH_DERIVATIVE_CHECK

void DistanceBarrierRBProblem::check_barrier_gradient(
    const Eigen::VectorXd& x,
    const Constraints& constraints,
    const Eigen::VectorXd& grad)
{
    for (int i = 0; i < grad.size(); i++) {
        if (!std::isfinite(grad(i))) {
            spdlog::error("barrier gradient is not finite");
        }
    }

    auto b = [&](const Eigen::VectorXd& x) {
        Eigen::VectorXd grad_b;
        Eigen::SparseMatrix<double> hess_b;

        return compute_barrier_term(
            x, constraints, grad_b, hess_b,
            /*compute_grad=*/false,
            /*compute_hess=*/false);
    };

    Eigen::VectorXd grad_approx;

    fd::finite_gradient(x, b, grad_approx);

    if (!fd::compare_gradient(grad, grad_approx, 1e-3)) {

        spdlog::error("finite gradient check failed for barrier");
    }
}

void DistanceBarrierRBProblem::check_barrier_hessian(
    const Eigen::VectorXd& x,
    const Constraints& constraints,
    const Eigen::SparseMatrix<double>& hess)
{
    typedef Eigen::SparseMatrix<double>::InnerIterator Iterator;

    for (int k = 0; k < hess.outerSize(); ++k) {
        for (Iterator it(hess, k); it; ++it) {
            if (!std::isfinite(it.value())) {
                spdlog::error("barrier hessian is not finite");

                return;
            }
        }
    }

    auto b = [&](const Eigen::VectorXd& x) {
        Eigen::VectorXd grad_b;
        Eigen::SparseMatrix<double> hess_b;

        compute_barrier_term(
            x, constraints, grad_b, hess_b,
            /*compute_grad=*/true,
            /*compute_hess=*/false);

        return grad_b;
    };

    Eigen::MatrixXd hess_approx;

    fd::finite_jacobian(x, b, hess_approx);

    if (!fd::compare_jacobian(hess, hess_approx, Constants::FINITE_DIFF_TEST)) {

        spdlog::error("finite hessian check failed for barrier");
    }
}

void DistanceBarrierRBProblem::check_friction_gradient(
    const Eigen::VectorXd& x, const Eigen::VectorXd& grad)
{
    for (int i = 0; i < grad.size(); i++) {
        if (!std::isfinite(grad(i))) {
            spdlog::error("friction gradient is not finite");
        }
    }

    auto f = [&](const Eigen::VectorXd& x) { return compute_friction_term(x); };

    Eigen::VectorXd grad_approx;

    fd::finite_gradient(x, f, grad_approx);

    if (!fd::compare_gradient(grad, grad_approx)) {

        spdlog::error("finite gradient check failed for friction");
    }

    typedef AutodiffType<Eigen::Dynamic> Diff;

    Diff::activate(x.size());

    Diff::D1MatrixXd V_diff =
        m_assembler.world_vertices(this->dofs_to_poses(Diff::d1vars(0, x)));

    Eigen::MatrixXd V0 = m_assembler.world_vertices(poses_t0);

    Diff::DDouble1 f_diff = ipc::compute_friction_potential(
        V0, V_diff, edges(), faces(), friction_constraints,
        static_friction_speed_bound * timestep());

    if (std::isfinite(f_diff.getGradient().sum())
        && !fd::compare_gradient(grad, f_diff.getGradient())) {

        spdlog::error("autodiff gradient check failed for friction");
    }
}

void DistanceBarrierRBProblem::check_friction_hessian(
    const Eigen::VectorXd& x, const Eigen::SparseMatrix<double>& hess)
{
    typedef Eigen::SparseMatrix<double>::InnerIterator Iterator;

    for (int k = 0; k < hess.outerSize(); ++k) {
        for (Iterator it(hess, k); it; ++it) {
            if (!std::isfinite(it.value())) {
                spdlog::error("barrier hessian is not finite");
            }
        }
    }

    Eigen::MatrixXd V0 = m_assembler.world_vertices(poses_t0);

    Eigen::MatrixXd V1 = m_assembler.world_vertices(this->dofs_to_poses(x));

    if ((V1 - V0).lpNorm<Eigen::Infinity>() == 0) {

        return;
    }

    Eigen::MatrixXd dense_hess(hess);

    auto f = [&](const Eigen::VectorXd& x) {
        Eigen::VectorXd grad_f;

        compute_friction_term(x, grad_f);

        return grad_f;
    };

    Eigen::MatrixXd hess_approx;

    fd::finite_jacobian(x, f, hess_approx);

    hess_approx = project_to_psd(hess_approx);

    if (!fd::compare_hessian(hess, hess_approx, 1e-2)) {

        spdlog::error(
            "finite hessian check failed for friction "
            "(hess_L_inf_norm={:g} "
            "diff_L_inf_norm={:g})",
            dense_hess.lpNorm<Eigen::Infinity>(),
            (hess_approx - dense_hess).lpNorm<Eigen::Infinity>());
    }

    typedef AutodiffType<Eigen::Dynamic> Diff;

    Diff::activate(x.size());

    Diff::D2MatrixXd V_diff =
        m_assembler.world_vertices(this->dofs_to_poses(Diff::d2vars(0, x)));

    Diff::DDouble2 f_diff = ipc::compute_friction_potential(
        V0, V_diff, edges(), faces(), friction_constraints,
        static_friction_speed_bound * timestep());

    Eigen::MatrixXd hess_autodiff = project_to_psd(f_diff.getHessian());

    if (std::isfinite(hess_autodiff.sum())) {

        if (!fd::compare_hessian(dense_hess, hess_autodiff, 1e-3)) {

            spdlog::error(
                "autodiff hessian check failed for friction "
                "(hess_L_inf_norm={:g} "
                "diff_L_inf_norm={:g})",
                dense_hess.lpNorm<Eigen::Infinity>(),
                (hess_autodiff - dense_hess).lpNorm<Eigen::Infinity>());
        }
    } else {
        spdlog::warn("autodiff hessian failed for friction");
    }
}

void DistanceBarrierRBProblem::check_augmented_lagrangian_gradient(
    const Eigen::VectorXd& x, const Eigen::VectorXd& grad)
{
    for (int i = 0; i < grad.size(); i++) {
        if (!std::isfinite(grad(i))) {
            spdlog::error("augmented lagrangian gradient is not finite");
        }
    }

    auto AL = [&](const Eigen::VectorXd& x) {
        Eigen::VectorXd grad_AL;
        Eigen::SparseMatrix<double> hess_AL;

        return compute_augmented_lagrangian(
            x, grad_AL, hess_AL,
            /*compute_grad=*/false,
            /*compute_hess=*/false);
    };

    Eigen::VectorXd grad_approx;

    fd::finite_gradient(x, AL, grad_approx);

    if (!fd::compare_gradient(grad, grad_approx)) {

        spdlog::error(
            "finite gradient check failed for "
            "augmented lagrangian");
    }
}

void DistanceBarrierRBProblem::check_augmented_lagrangian_hessian(
    const Eigen::VectorXd& x, const Eigen::SparseMatrix<double>& hess)
{
    typedef Eigen::SparseMatrix<double>::InnerIterator Iterator;

    for (int k = 0; k < hess.outerSize(); ++k) {
        for (Iterator it(hess, k); it; ++it) {
            if (!std::isfinite(it.value())) {
                spdlog::error("augmented lagrangian hessian is not finite");

                return;
            }
        }
    }

    auto AL = [&](const Eigen::VectorXd& x) {
        Eigen::VectorXd grad_AL;
        Eigen::SparseMatrix<double> hess_AL;

        compute_augmented_lagrangian(
            x, grad_AL, hess_AL,
            /*compute_grad=*/true,
            /*compute_hess=*/false);

        return grad_AL;
    };

    Eigen::MatrixXd hess_approx;

    fd::finite_jacobian(x, AL, hess_approx);

    if (!fd::compare_jacobian(hess, hess_approx)) {

        spdlog::error(
            "finite hessian check failed for "
            "augmented lagrangian");
    }
}

#endif

} // namespace ipc::rigid