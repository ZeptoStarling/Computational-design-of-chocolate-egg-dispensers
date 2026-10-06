#!/usr/bin/env python3

import json
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,FFMpegWriter
from matplotlib.patches import Polygon


TEST_DIR=Path(__file__).resolve().parent

BUILD_DIR=Path(
    "/home/austeja/Everything/EggDispenser/"
    "rigid-ipc/build"
)

SOLVER=BUILD_DIR/"rigid_ipc_sim"

VIDEO_FILE=TEST_DIR/"inclined_friction_mesh.mp4"
IMAGE_FILE=TEST_DIR/"inclined_friction_mesh.png"

SCENARIOS=[
    ("mu_01",0.10),
    ("mu_03",0.30),
]

CIRCLE_SEGMENTS=128

FRICTION_ITERATIONS=10
STATIC_FRICTION_SPEED_BOUND=0.001

NUM_STEPS=3000
CHECKPOINT_FREQUENCY=10001

ANGLE_DEG=30.0
ANGLE=math.radians(ANGLE_DEG)

GRAVITY=9.81
RADIUS=1.0

R_BOTTOM=1.5
HEIGHT=6.0
WALL_THICKNESS=0.2

TIMESTEP=0.0005
MAX_TIME=2.0

RESTITUTION=0.0

SLOPE=1.0/math.tan(ANGLE)
R_TOP=R_BOTTOM+SLOPE*HEIGHT

DOWNHILL=np.array([
    -math.cos(ANGLE),
    -math.sin(ANGLE),
])

INWARD_NORMAL=np.array([
    -math.sin(ANGLE),
    math.cos(ANGLE),
])

WALL_ORIGIN=np.array([
    R_BOTTOM,
    0.0,
])

START_S=8.0
INITIAL_GAP=0.002

WALL_POINT=(
    WALL_ORIGIN
    +START_S*np.array([
        math.cos(ANGLE),
        math.sin(ANGLE),
    ])
)

INITIAL_CENTER=(
    WALL_POINT
    +(RADIUS+INITIAL_GAP)
    *INWARD_NORMAL
)

FPS=60
VIDEO_SECONDS=6.0

SNAPSHOT_FRACTIONS=[
    0.00,
    0.25,
    0.50,
    0.75,
    1.00,
]


def save_json(path,data):
    with open(path,"w") as f:
        json.dump(data,f,indent=2)


def load_json(path):
    with open(path,"r") as f:
        return json.load(f)


def make_edges(n):
    return [
        [i,(i+1)%n]
        for i in range(n)
    ]


def make_circle():
    return [
        [
            RADIUS*math.cos(
                2.0*math.pi*i/CIRCLE_SEGMENTS
            ),
            RADIUS*math.sin(
                2.0*math.pi*i/CIRCLE_SEGMENTS
            ),
        ]
        for i in range(CIRCLE_SEGMENTS)
    ]


def make_wall():
    p0=np.array([
        R_BOTTOM,
        0.0,
    ])

    p1=np.array([
        R_TOP,
        HEIGHT,
    ])

    outward=-INWARD_NORMAL

    q0=(
        p0
        +WALL_THICKNESS*outward
    )

    q1=(
        p1
        +WALL_THICKNESS*outward
    )

    return [
        p0.tolist(),
        p1.tolist(),
        q1.tolist(),
        q0.tolist(),
    ]


def make_body(
    vertices,
    position,
    fixed,
):
    return {
        "vertices":vertices,
        "polygons":[vertices],
        "edges":make_edges(
            len(vertices)
        ),
        "oriented":True,
        "position":[
            float(position[0]),
            float(position[1]),
        ],
        "rotation":[0.0],
        "linear_velocity":[0.0,0.0],
        "angular_velocity":[0.0],
        "is_dof_fixed":[
            fixed,
            fixed,
            fixed,
        ],
    }


def make_scene(mu):
    wall=make_wall()
    circle=make_circle()

    return {
        "scene_type":
            "distance_barrier_rb_problem",

        "solver":
            "ipc_solver",

        "timestep":
            TIMESTEP,

        "max_time":
            MAX_TIME,

        "rigid_body_problem":{
            "coefficient_restitution":
                RESTITUTION,

            "coefficient_friction":
                mu,

            "gravity":[
                0.0,
                -GRAVITY,
            ],

            "rigid_bodies":[
                make_body(
                    wall,
                    (0.0,0.0),
                    True,
                ),

                make_body(
                    circle,
                    INITIAL_CENTER,
                    False,
                ),
            ],
        },

        "friction_constraints":{
            "static_friction_speed_bound":
                STATIC_FRICTION_SPEED_BOUND,

            "iterations":
                FRICTION_ITERATIONS,
        },

        "ipc_solver":{
            "convergence_criteria":
                "velocity",

            "velocity_conv_tol":
                0.0001,

            "is_velocity_conv_tol_abs":
                True,
        },
    }


def run_case(
    mu,
    tmp_root,
    name,
):
    scene=make_scene(mu)

    scene_file=tmp_root/f"{name}.json"
    output_dir=tmp_root/f"output_{name}"

    save_json(
        scene_file,
        scene,
    )

    command=[
        str(SOLVER),
        "--ngui",
        "--num-steps",
        str(NUM_STEPS),
        "--output-path",
        str(output_dir),
        str(scene_file),
        "--checkpoint-frequency",
        str(CHECKPOINT_FREQUENCY),
    ]

    print(f"Running mesh {name}...")

    subprocess.run(
        command,
        cwd=BUILD_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        check=True,
    )

    return (
        scene,
        load_json(
            output_dir/"sim.json"
        ),
    )


def transform(
    vertices,
    position,
    theta,
):
    V=np.asarray(
        vertices,
        dtype=float,
    )

    c=math.cos(theta)
    s=math.sin(theta)

    R=np.array([
        [c,-s],
        [s,c],
    ])

    return (
        V@R.T
        +np.asarray(
            position,
            dtype=float,
        )
    )


def extract(scene,sim):
    states=(
        sim["animation"]
        ["state_sequence"]
    )

    bodies=(
        scene[
            "rigid_body_problem"
        ]["rigid_bodies"]
    )

    body_id=1

    local_vertices=np.asarray(
        bodies[body_id]["vertices"],
        dtype=float,
    )

    positions=np.zeros(
        (len(states),2)
    )

    rotations=np.zeros(
        len(states)
    )

    for k,state in enumerate(states):
        body=state[
            "rigid_bodies"
        ][body_id]

        positions[k]=[
            float(body["position"][0]),
            float(body["position"][1]),
        ]

        rotations[k]=float(
            body["rotation"][0]
        )

    return (
        local_vertices,
        positions,
        rotations,
    )


def orientation_point(
    center,
    theta,
):
    return (
        center
        +RADIUS*np.array([
            math.cos(theta),
            math.sin(theta),
        ])
    )


def main():

    if not SOLVER.exists():
        raise RuntimeError(
            f"Missing solver:\n{SOLVER}"
        )

    with tempfile.TemporaryDirectory(
        prefix="incline_mesh_"
    ) as tmp:

        tmp_root=Path(tmp)

        data={}

        for name,mu in SCENARIOS:

            scene,sim=run_case(
                mu,
                tmp_root,
                name,
            )

            (
                local_vertices,
                positions,
                rotations,
            )=extract(scene,sim)

            data[name]={
                "mu":mu,
                "vertices":local_vertices,
                "positions":positions,
                "rotations":rotations,
            }

        wall=np.asarray(
            make_wall(),
            dtype=float,
        )

        all_positions=np.vstack([
            data["mu_01"]["positions"],
            data["mu_03"]["positions"],
        ])

        x_min=min(
            np.min(wall[:,0]),
            np.min(all_positions[:,0])
            -RADIUS-0.5,
        )

        x_max=max(
            np.max(wall[:,0]),
            np.max(all_positions[:,0])
            +RADIUS+0.5,
        )

        y_min=min(
            np.min(wall[:,1]),
            np.min(all_positions[:,1])
            -RADIUS-0.5,
        )

        y_max=max(
            np.max(wall[:,1]),
            np.max(all_positions[:,1])
            +RADIUS+0.5,
        )

        # -----------------------------------------
        # VIDEO
        # -----------------------------------------

        frame_count=int(
            FPS*VIDEO_SECONDS
        )

        source_frames=np.linspace(
            0,
            min(
                len(
                    data["mu_01"]
                    ["positions"]
                ),
                len(
                    data["mu_03"]
                    ["positions"]
                ),
            )-1,
            frame_count,
        ).astype(int)

        fig,axes=plt.subplots(
            1,2,
            figsize=(12,6),
        )

        circle_patches=[]
        marker_lines=[]

        for ax,(name,mu) in zip(
            axes,
            SCENARIOS,
        ):

            ax.set_aspect("equal")

            ax.set_xlim(
                x_min,
                x_max,
            )

            ax.set_ylim(
                y_min,
                y_max,
            )

            ax.axis("off")

            ax.add_patch(
                Polygon(
                    wall,
                    closed=True,
                    facecolor="0.88",
                    edgecolor="black",
                    linewidth=1.8,
                )
            )

            ax.text(
                0.05,
                0.92,
                rf"$\mu={mu:.1f}$",
                transform=ax.transAxes,
                fontsize=15,
            )

            center=(
                data[name]
                ["positions"][0]
            )

            theta=(
                data[name]
                ["rotations"][0]
            )

            V=transform(
                data[name]["vertices"],
                center,
                theta,
            )

            patch=Polygon(
                V,
                closed=True,
                facecolor="royalblue",
                edgecolor="black",
                linewidth=1.1,
            )

            ax.add_patch(patch)
            circle_patches.append(patch)

            end=orientation_point(
                center,
                theta,
            )

            line,=ax.plot(
                [center[0],end[0]],
                [center[1],end[1]],
                "k-",
                linewidth=2.0,
            )

            marker_lines.append(line)

        fig.subplots_adjust(
            left=0.02,
            right=0.98,
            bottom=0.02,
            top=0.98,
            wspace=0.04,
        )

        def update(video_frame):

            frame_id=(
                source_frames[
                    video_frame
                ]
            )

            artists=[]

            for i,(name,_) in enumerate(
                SCENARIOS
            ):

                center=(
                    data[name]
                    ["positions"][frame_id]
                )

                theta=(
                    data[name]
                    ["rotations"][frame_id]
                )

                V=transform(
                    data[name]["vertices"],
                    center,
                    theta,
                )

                circle_patches[i].set_xy(
                    V
                )

                end=orientation_point(
                    center,
                    theta,
                )

                marker_lines[i].set_data(
                    [center[0],end[0]],
                    [center[1],end[1]],
                )

                artists.extend([
                    circle_patches[i],
                    marker_lines[i],
                ])

            return artists

        animation=FuncAnimation(
            fig,
            update,
            frames=frame_count,
            blit=False,
        )

        writer=FFMpegWriter(
            fps=FPS,
            codec="libx264",
            extra_args=[
                "-pix_fmt",
                "yuv420p",
            ],
        )

        print("Saving mesh MP4...")

        animation.save(
            VIDEO_FILE,
            writer=writer,
        )

        plt.close(fig)

        # -----------------------------------------
        # PNG
        # -----------------------------------------

        fig,axes=plt.subplots(
            1,2,
            figsize=(12,6),
        )

        for ax,(name,mu) in zip(
            axes,
            SCENARIOS,
        ):

            ax.set_aspect("equal")

            ax.set_xlim(
                x_min,
                x_max,
            )

            ax.set_ylim(
                y_min,
                y_max,
            )

            ax.axis("off")

            ax.add_patch(
                Polygon(
                    wall,
                    closed=True,
                    facecolor="0.88",
                    edgecolor="black",
                    linewidth=1.8,
                )
            )

            ax.text(
                0.05,
                0.92,
                rf"$\mu={mu:.1f}$",
                transform=ax.transAxes,
                fontsize=15,
            )

            positions=(
                data[name]["positions"]
            )

            rotations=(
                data[name]["rotations"]
            )

            ids=[
                int(
                    f*(len(positions)-1)
                )
                for f
                in SNAPSHOT_FRACTIONS
            ]

            for j,frame_id in enumerate(ids):

                center=positions[
                    frame_id
                ]

                theta=rotations[
                    frame_id
                ]

                alpha=0.25+0.75*(
                    j/(len(ids)-1)
                )

                V=transform(
                    data[name]["vertices"],
                    center,
                    theta,
                )

                ax.add_patch(
                    Polygon(
                        V,
                        closed=True,
                        facecolor="royalblue",
                        edgecolor="black",
                        linewidth=0.9,
                        alpha=alpha,
                    )
                )

                end=orientation_point(
                    center,
                    theta,
                )

                ax.plot(
                    [center[0],end[0]],
                    [center[1],end[1]],
                    "k-",
                    linewidth=1.5,
                    alpha=alpha,
                )

        fig.subplots_adjust(
            left=0,
            right=1,
            bottom=0,
            top=1,
            wspace=0.03,
        )

        fig.savefig(
            IMAGE_FILE,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
        )

        plt.close(fig)

    print()
    print(f"Video : {VIDEO_FILE}")
    print(f"Image : {IMAGE_FILE}")


if __name__=="__main__":
    main()