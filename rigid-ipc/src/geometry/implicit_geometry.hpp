#pragma once

#include <cmath>
#include <algorithm>
#include <vector>
#include <limits>

#include <Eigen/Dense>

namespace rigid_ipc {

class CubicSpline1D {
private:
    double a0 = 0.0;
    double a1 = 0.0;
    double a2 = 0.0;
    double a3 = 0.0;
    double height = 1.0;

public:
    CubicSpline1D() = default;

    CubicSpline1D(
        double r_bottom, double r_top, double s_bottom, double s_top, double h)
        : height(h)
    {
        a0 = r_bottom;
        a1 = s_bottom;

        const double delta_r = r_top - r_bottom - s_bottom * height;

        const double delta_s = s_top - s_bottom;

        const double H2 = height * height;

        const double H3 = H2 * height;

        a2 = (3.0 / H2) * delta_r - (1.0 / height) * delta_s;

        a3 = (-2.0 / H3) * delta_r + (1.0 / H2) * delta_s;
    }

    double eval(double y) const
    {
        return a0 + a1 * y + a2 * y * y + a3 * y * y * y;
    }

    double eval_d1(double y) const
    {
        return a1 + 2.0 * a2 * y + 3.0 * a3 * y * y;
    }

    double eval_d2(double y) const { return 2.0 * a2 + 6.0 * a3 * y; }

    double get_height() const { return height; }
};

struct RigidCircle {
    double radius = 0.0;
    Eigen::Vector2d pos = Eigen::Vector2d::Zero();
    double theta = 0.0;

    Eigen::Vector2d world_center() const { return pos; }
};

struct RigidProjectionResult {
    double distance = 0.0;

    Eigen::Vector2d closest_pt = Eigen::Vector2d::Zero();

    // Unit normal pointing from the wall toward the circle center
    // for an interior-domain contact.
    Eigen::Vector2d normal_spatial = Eigen::Vector2d::Zero();

    std::vector<int> body_ids;

    // 3 DOFs for circle/funnel:
    //   [tx, ty, theta]
    Eigen::VectorXd grad_q;

    Eigen::MatrixXd hessian_q;
};

namespace detail {

    inline double funnel_dist_sq(
        double x_sym, double yc, double y, const CubicSpline1D& spline)
    {
        const double r = spline.eval(y);

        const double dx = x_sym - r;

        const double dy = yc - y;

        return dx * dx + dy * dy;
    }

    inline double funnel_projection_g(
        double x_sym, double yc, double y, const CubicSpline1D& spline)
    {
        const double r = spline.eval(y);

        const double dr = spline.eval_d1(y);

        return (x_sym - r) * dr + (yc - y);
    }

    inline double funnel_projection_dg(
        double x_sym, double yc, double y, const CubicSpline1D& spline)
    {
        const double r = spline.eval(y);

        const double dr = spline.eval_d1(y);

        const double ddr = spline.eval_d2(y);

        return -(dr * dr) + (x_sym - r) * ddr - 1.0;
    }

    inline double golden_section_projection(
        double x_sym, double yc, const CubicSpline1D& spline, double height)
    {
        double a = 0.0;
        double b = height;

        constexpr double phi = 0.6180339887498948482;

        double c = b - phi * (b - a);

        double d = a + phi * (b - a);

        double fc = funnel_dist_sq(x_sym, yc, c, spline);

        double fd = funnel_dist_sq(x_sym, yc, d, spline);

        for (int iter = 0; iter < 50; ++iter) {
            if (fc < fd) {
                b = d;
                d = c;
                fd = fc;

                c = b - phi * (b - a);

                fc = funnel_dist_sq(x_sym, yc, c, spline);
            } else {
                a = c;
                c = d;
                fc = fd;

                d = a + phi * (b - a);

                fd = funnel_dist_sq(x_sym, yc, d, spline);
            }
        }

        return 0.5 * (a + b);
    }

    inline double polish_projection_newton(
        double x_sym, double yc, const CubicSpline1D& spline, double height)
    {
        double y = golden_section_projection(x_sym, yc, spline, height);

        constexpr double g_tol = 1e-12;

        constexpr double dg_tol = 1e-12;

        for (int iter = 0; iter < 20; ++iter) {
            const double g = funnel_projection_g(x_sym, yc, y, spline);

            if (std::abs(g) < g_tol) {
                break;
            }

            const double dg = funnel_projection_dg(x_sym, yc, y, spline);

            if (std::abs(dg) < dg_tol) {
                break;
            }

            double step = -g / dg;

            const double max_step = 0.25 * std::max(height, 1.0);

            step = std::clamp(step, -max_step, max_step);

            double y_new = y + step;

            y_new = std::clamp(y_new, 0.0, height);

            const double old_abs_g = std::abs(g);

            double new_abs_g =
                std::abs(funnel_projection_g(x_sym, yc, y_new, spline));

            int damping = 0;

            while (new_abs_g > old_abs_g && damping < 10) {
                y_new = 0.5 * (y + y_new);

                new_abs_g =
                    std::abs(funnel_projection_g(x_sym, yc, y_new, spline));

                ++damping;
            }

            y = y_new;
        }

        return std::clamp(y, 0.0, height);
    }

    struct FunnelProjection {
        double y = 0.0;
        bool interior_stationary = false;
    };

    inline FunnelProjection find_closest_funnel_point(
        double x_sym, double yc, const CubicSpline1D& spline, double height)
    {
        FunnelProjection result;

        const double y_golden =
            golden_section_projection(x_sym, yc, spline, height);

        const double y_newton =
            polish_projection_newton(x_sym, yc, spline, height);

        double best_y = y_golden;

        double best_d2 = funnel_dist_sq(x_sym, yc, best_y, spline);

        const double d2_newton = funnel_dist_sq(x_sym, yc, y_newton, spline);

        if (d2_newton < best_d2) {
            best_d2 = d2_newton;
            best_y = y_newton;
        }

        const double d2_bottom = funnel_dist_sq(x_sym, yc, 0.0, spline);

        if (d2_bottom < best_d2) {
            best_d2 = d2_bottom;
            best_y = 0.0;
        }

        const double d2_top = funnel_dist_sq(x_sym, yc, height, spline);

        if (d2_top < best_d2) {
            best_d2 = d2_top;
            best_y = height;
        }

        result.y = best_y;

        const double interior_tol = 1e-10 * std::max(1.0, height);

        const bool away_from_bottom = best_y > interior_tol;

        const bool away_from_top = best_y < height - interior_tol;

        if (away_from_bottom && away_from_top) {
            const double g = funnel_projection_g(x_sym, yc, best_y, spline);

            result.interior_stationary = std::abs(g) < 1e-8;
        } else {
            result.interior_stationary = false;
        }

        return result;
    }

} // namespace detail

inline RigidProjectionResult project_rigid_circle_to_funnel(
    int body_id,
    const RigidCircle& body,
    const CubicSpline1D& spline,
    double height,
    bool is_interior = true)
{
    RigidProjectionResult res;

    res.body_ids = { body_id };

    const Eigen::Vector2d p_world = body.world_center();

    const double xc = p_world.x();

    const double yc = p_world.y();

    const double x_sym = std::abs(xc);

    const double side = (xc >= 0.0) ? 1.0 : -1.0;

    const detail::FunnelProjection projection =
        detail::find_closest_funnel_point(x_sym, yc, spline, height);

    const double y_star = projection.y;

    const double r_star = spline.eval(y_star);

    const double dr_star = spline.eval_d1(y_star);

    const Eigen::Vector2d closest_pt(side * r_star, y_star);

    res.closest_pt = closest_pt;

    const Eigen::Vector2d disp = p_world - closest_pt;

    const double d_center = disp.norm();

    res.distance = d_center - body.radius;

    Eigen::Vector2d normal_spatial;

    if (d_center > 1e-12) {
        normal_spatial = disp / d_center;
    } else {

        Eigen::Vector2d n(-side, dr_star);

        const double n_norm = n.norm();

        if (n_norm > 1e-12) {
            normal_spatial = n / n_norm;
        } else {
            normal_spatial = Eigen::Vector2d(-side, 0.0);
        }
    }

    if (!is_interior) {
        normal_spatial = -normal_spatial;
    }

    res.normal_spatial = normal_spatial;

    Eigen::Matrix2d H_spatial = Eigen::Matrix2d::Zero();

    if (d_center > 1e-12) {
        const Eigen::Matrix2d I = Eigen::Matrix2d::Identity();

        if (projection.interior_stationary) {
            const Eigen::Vector2d T(side * dr_star, 1.0);

            const double dg_star =
                detail::funnel_projection_dg(x_sym, yc, y_star, spline);

            if (std::abs(dg_star) > 1e-10) {
                const Eigen::Matrix2d projector =
                    I - normal_spatial * normal_spatial.transpose();

                H_spatial = (1.0 / d_center)
                    * (projector + (T * T.transpose()) / dg_star);
            } else {
                H_spatial = (1.0 / d_center)
                    * (I - normal_spatial * normal_spatial.transpose());
            }
        } else {
            H_spatial = (1.0 / d_center)
                * (I - normal_spatial * normal_spatial.transpose());
        }

        H_spatial = 0.5 * (H_spatial + H_spatial.transpose());
    }

    res.grad_q.resize(3);
    res.grad_q << res.normal_spatial.x(), res.normal_spatial.y(), 0.0;

    res.hessian_q = Eigen::MatrixXd::Zero(3, 3);
    res.hessian_q.block<2, 2>(0, 0) = H_spatial;

    if (!res.grad_q.allFinite() || !res.hessian_q.allFinite()
        || !std::isfinite(res.distance)) {
        res.grad_q.setZero();
        res.hessian_q.setZero();

        if (!std::isfinite(res.distance)) {
            res.distance = std::numeric_limits<double>::infinity();
        }
    }
    return res;
}

inline RigidProjectionResult project_rigid_circle_to_rigid_circle(
    int body1_id,
    const RigidCircle& body1,
    int body2_id,
    const RigidCircle& body2)
{
    RigidProjectionResult res;

    res.body_ids = { body1_id, body2_id };

    const Eigen::Vector2d p1 = body1.world_center();

    const Eigen::Vector2d p2 = body2.world_center();

    const Eigen::Vector2d disp = p1 - p2;

    const double d_center = disp.norm();

    res.distance = d_center - body1.radius - body2.radius;

    res.normal_spatial =
        (d_center > 1e-12) ? disp / d_center : Eigen::Vector2d(1.0, 0.0);

    res.closest_pt = p2 + res.normal_spatial * body2.radius;

    Eigen::Matrix2d H11 = Eigen::Matrix2d::Zero();

    if (d_center > 1e-12) {
        const Eigen::Matrix2d I = Eigen::Matrix2d::Identity();

        H11 = (1.0 / d_center)
            * (I - res.normal_spatial * res.normal_spatial.transpose());
    }

    res.grad_q.resize(6);

    res.grad_q << res.normal_spatial.x(), res.normal_spatial.y(), 0.0,
        -res.normal_spatial.x(), -res.normal_spatial.y(), 0.0;

    res.hessian_q = Eigen::MatrixXd::Zero(6, 6);

    res.hessian_q.block<2, 2>(0, 0) = H11;
    res.hessian_q.block<2, 2>(0, 3) = -H11;
    res.hessian_q.block<2, 2>(3, 0) = -H11;
    res.hessian_q.block<2, 2>(3, 3) = H11;

    res.hessian_q = 0.5 * (res.hessian_q + res.hessian_q.transpose());

    return res;
}

struct FunnelCandidate {
    int body_id;

    RigidCircle body;

    const CubicSpline1D* spline = nullptr;

    double height = 0.0;

    double compute_distance() const
    {
        return project_rigid_circle_to_funnel(body_id, body, *spline, height)
            .distance;
    }

    Eigen::Vector3d compute_gradient() const
    {
        return project_rigid_circle_to_funnel(body_id, body, *spline, height)
            .grad_q;
    }

    Eigen::Matrix3d compute_hessian() const
    {
        return project_rigid_circle_to_funnel(body_id, body, *spline, height)
            .hessian_q;
    }
};

struct CircleCircleCandidate {
    int body1_id;

    RigidCircle body1;

    int body2_id;

    RigidCircle body2;

    double compute_distance() const
    {
        return project_rigid_circle_to_rigid_circle(
                   body1_id, body1, body2_id, body2)
            .distance;
    }

    Eigen::Matrix<double, 6, 1> compute_gradient() const
    {
        return project_rigid_circle_to_rigid_circle(
                   body1_id, body1, body2_id, body2)
            .grad_q;
    }

    Eigen::Matrix<double, 6, 6> compute_hessian() const
    {
        return project_rigid_circle_to_rigid_circle(
                   body1_id, body1, body2_id, body2)
            .hessian_q;
    }
};

} // namespace rigid_ipc