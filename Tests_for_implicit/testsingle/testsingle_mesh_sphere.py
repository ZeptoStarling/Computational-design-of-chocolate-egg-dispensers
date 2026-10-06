#!/usr/bin/env python3

import json
import math
from pathlib import Path


# ============================================================
# ANALYTIC FUNNEL PARAMETERS
# ============================================================

HEIGHT = 6.0

R_BOTTOM = 1.5
R_TOP = 5.0

S_BOTTOM = 0.0
S_TOP = 0.0

WALL_THICKNESS = 0.2


# ============================================================
# FIXED FUNNEL RESOLUTION
#
# Keep the funnel fixed so that the only changing
# discretization is the sphere.
# ============================================================

FUNNEL_POINTS = 96


# ============================================================
# SPHERE RESOLUTIONS TO TEST
# ============================================================

SPHERE_RESOLUTIONS = [
    8,
    12,
    16,
    24,
    32,
    48,
    64,
    96,
]


# ============================================================
# SPHERE
# ============================================================

SPHERE_X = 1.2
SPHERE_Y = 5.0
SPHERE_RADIUS = 1.0


# ============================================================
# SIMULATION
# ============================================================

TIMESTEP = 0.0005
MAX_TIME = 10.0

COEFFICIENT_RESTITUTION = 0.2

GRAVITY = [
    0.0,
    -9.81,
]

FRICTION_ITERATIONS = 40


# ============================================================
# SAME CUBIC HERMITE SPLINE AS ANALYTIC TEST
# ============================================================

def cubic_hermite(t, r0, r1, s0, s1):

    h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
    h11 = t**3 - t**2

    return (
        h00 * r0
        + h10 * s0
        + h01 * r1
        + h11 * s1
    )


def funnel_radius(t):

    t = max(
        0.0,
        min(1.0, t),
    )

    return max(
        cubic_hermite(
            t,
            R_BOTTOM,
            R_TOP,
            S_BOTTOM,
            S_TOP,
        ),
        R_BOTTOM,
    )


# ============================================================
# FUNNEL WALLS
# ============================================================

def make_funnel_walls(num_points):

    n = num_points

    left_inner = []
    left_outer = []

    right_inner = []
    right_outer = []

    for i in range(n):

        t = i / (n - 1)

        y = t * HEIGHT
        r = funnel_radius(t)

        left_inner.append([
            -r,
            y,
        ])

        left_outer.append([
            -r - WALL_THICKNESS,
            y,
        ])

        right_inner.append([
            +r,
            y,
        ])

        right_outer.append([
            +r + WALL_THICKNESS,
            y,
        ])

    left_wall = (
        left_inner
        + list(reversed(left_outer))
    )

    right_wall = (
        list(reversed(right_inner))
        + right_outer
    )

    return (
        left_wall,
        right_wall,
    )


# ============================================================
# POLYGON EDGES
# ============================================================

def make_polygon_edges(num_vertices):

    return [
        [
            i,
            (i + 1) % num_vertices,
        ]
        for i in range(num_vertices)
    ]


# ============================================================
# CIRCLE POLYGON
# ============================================================

def make_circle_vertices(
    radius,
    circle_segments,
):

    vertices = []

    for i in range(circle_segments):

        theta = (
            2.0
            * math.pi
            * i
            / circle_segments
        )

        vertices.append([
            radius * math.cos(theta),
            radius * math.sin(theta),
        ])

    return vertices


# ============================================================
# MESH BODY
# ============================================================

def make_mesh_body(
    vertices,
    edges,
    position,
    fixed,
):

    return {
        "vertices": vertices,

        "polygons": [
            vertices
        ],

        "edges": edges,

        "oriented": True,

        "position": [
            position[0],
            position[1],
        ],

        "rotation": [
            0.0
        ],

        "linear_velocity": [
            0.0,
            0.0
        ],

        "angular_velocity": [
            0.0
        ],

        "is_dof_fixed": [
            fixed,
            fixed,
            fixed,
        ],
    }


# ============================================================
# SCENE
# ============================================================

def make_scene(circle_segments):

    rigid_bodies = []

    # --------------------------------------------------------
    # FUNNEL
    #
    # Fixed at FUNNEL_POINTS for every sphere test.
    # --------------------------------------------------------

    left_wall, right_wall = make_funnel_walls(
        FUNNEL_POINTS
    )

    left_edges = make_polygon_edges(
        len(left_wall)
    )

    right_edges = make_polygon_edges(
        len(right_wall)
    )

    rigid_bodies.append(
        make_mesh_body(
            vertices=left_wall,
            edges=left_edges,
            position=(0.0, 0.0),
            fixed=True,
        )
    )

    rigid_bodies.append(
        make_mesh_body(
            vertices=right_wall,
            edges=right_edges,
            position=(0.0, 0.0),
            fixed=True,
        )
    )

    # --------------------------------------------------------
    # SINGLE DYNAMIC SPHERE
    #
    # Sphere discretization changes here.
    # --------------------------------------------------------

    sphere_vertices = make_circle_vertices(
        SPHERE_RADIUS,
        circle_segments,
    )

    sphere_edges = make_polygon_edges(
        circle_segments
    )

    rigid_bodies.append(
        make_mesh_body(
            vertices=sphere_vertices,
            edges=sphere_edges,
            position=(
                SPHERE_X,
                SPHERE_Y,
            ),
            fixed=False,
        )
    )

    # --------------------------------------------------------
    # SCENE
    # --------------------------------------------------------

    return {
        "scene_type": "distance_barrier_rb_problem",

        "solver": "ipc_solver",

        "timestep": TIMESTEP,

        "max_time": MAX_TIME,

        "rigid_body_problem": {

            "coefficient_restitution":
                COEFFICIENT_RESTITUTION,

            "gravity": GRAVITY,

            "rigid_bodies": rigid_bodies,
        },

        "friction_constraints": {

            "iterations":
                FRICTION_ITERATIONS,
        },
    }


# ============================================================
# VALIDATION
# ============================================================

def validate(
    scene,
    circle_segments,
):

    bodies = scene[
        "rigid_body_problem"
    ][
        "rigid_bodies"
    ]

    # Two walls + one sphere.
    assert len(bodies) == 3

    # --------------------------------------------------------
    # LEFT WALL
    # --------------------------------------------------------

    left = bodies[0]

    assert len(left["vertices"]) == 2 * FUNNEL_POINTS
    assert len(left["edges"]) == 2 * FUNNEL_POINTS
    assert len(left["polygons"]) == 1

    assert all(
        len(v) == 2
        for v in left["vertices"]
    )

    assert all(
        left["is_dof_fixed"]
    )

    # --------------------------------------------------------
    # RIGHT WALL
    # --------------------------------------------------------

    right = bodies[1]

    assert len(right["vertices"]) == 2 * FUNNEL_POINTS
    assert len(right["edges"]) == 2 * FUNNEL_POINTS
    assert len(right["polygons"]) == 1

    assert all(
        len(v) == 2
        for v in right["vertices"]
    )

    assert all(
        right["is_dof_fixed"]
    )

    # --------------------------------------------------------
    # SPHERE
    # --------------------------------------------------------

    sphere = bodies[2]

    assert len(sphere["vertices"]) == circle_segments
    assert len(sphere["edges"]) == circle_segments
    assert len(sphere["polygons"]) == 1

    assert sphere["is_dof_fixed"] == [
        False,
        False,
        False,
    ]

    assert sphere["position"] == [
        SPHERE_X,
        SPHERE_Y,
    ]

    assert all(
        len(v) == 2
        for v in sphere["vertices"]
    )


# ============================================================
# GENERATE ONE FILE
# ============================================================

def generate_one(
    output_dir,
    circle_segments,
):

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    scene = make_scene(
        circle_segments
    )

    validate(
        scene,
        circle_segments,
    )

    filename = (
        output_dir
        / (
            f"mesh_sphere_N"
            f"{circle_segments:03d}.json"
        )
    )

    with open(
        filename,
        "w",
    ) as f:

        json.dump(
            scene,
            f,
            indent=2,
        )

    print(
        f"Wrote {filename}"
        f"  "
        f"(funnel={FUNNEL_POINTS}, "
        f"sphere={circle_segments})"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    output_dir = (
        "/home/austeja/Everything/EggDispenser/Implicit/"
        "testsforimplicit/testsingle/meshes_spheres"
    )

    for circle_segments in SPHERE_RESOLUTIONS:

        generate_one(
            output_dir,
            circle_segments,
        )

    print()
    print("Generated:")
    print(
        f"  {len(SPHERE_RESOLUTIONS)} mesh scenes"
    )

    print(
        f"  fixed funnel resolution: "
        f"{FUNNEL_POINTS} points per side"
    )

    print(
        "  sphere resolutions: "
        + ", ".join(
            str(n)
            for n in SPHERE_RESOLUTIONS
        )
    )


if __name__ == "__main__":
    main()