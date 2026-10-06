#!/usr/bin/env python3

import copy
import json
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,FFMpegWriter
from matplotlib.patches import Circle


TEST_DIR=Path(__file__).resolve().parent
BUILD_DIR=Path(
    "/home/austeja/Everything/EggDispenser/"
    "Implicit/rigid-ipc/build"
)

SOLVER=BUILD_DIR/"rigid_ipc_sim"
BASE_JSON=TEST_DIR/"inclined_friction.json"

VIDEO_FILE=TEST_DIR/"inclined_friction_implicit.mp4"
IMAGE_FILE=TEST_DIR/"inclined_friction_implicit.png"

SCENARIOS=[
    ("mu_01",0.10),
    ("mu_03",0.30),
]

NUM_STEPS=3000
CHECKPOINT_FREQUENCY=10001

FPS=60
VIDEO_SECONDS=6.0

SNAPSHOT_FRACTIONS=[
    0.00,
    0.25,
    0.50,
    0.75,
    1.00,
]


def load_json(path):
    with open(path,"r") as f:
        return json.load(f)


def save_json(path,data):
    with open(path,"w") as f:
        json.dump(data,f,indent=2)


def run_case(base,mu,tmp_root,name):
    scene=copy.deepcopy(base)

    scene[
        "rigid_body_problem"
    ]["coefficient_friction"]=mu

    scene_file=tmp_root/f"{name}.json"
    output_dir=tmp_root/f"output_{name}"

    save_json(scene_file,scene)

    command=[
        str(SOLVER),
        "--ngui",
        "--num-steps",str(NUM_STEPS),
        "--output-path",str(output_dir),
        str(scene_file),
        "--checkpoint-frequency",
        str(CHECKPOINT_FREQUENCY),
    ]

    print(f"Running implicit {name}...")

    subprocess.run(
        command,
        cwd=BUILD_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        check=True,
    )

    return load_json(output_dir/"sim.json")


def extract(sim):
    args=sim["args"]
    states=sim["animation"]["state_sequence"]
    bodies=args["rigid_body_problem"]["rigid_bodies"]

    dynamic=[
        i
        for i,b in enumerate(bodies)
        if not all(
            b.get(
                "is_dof_fixed",
                [False,False,False],
            )
        )
    ]

    if len(dynamic)!=1:
        raise RuntimeError(
            f"Expected one dynamic circle, found {len(dynamic)}"
        )

    body_id=dynamic[0]

    radius=float(bodies[body_id]["radius"])

    positions=np.zeros((len(states),2))
    rotations=np.zeros(len(states))

    for k,state in enumerate(states):
        body=state["rigid_bodies"][body_id]

        positions[k]=[
            float(body["position"][0]),
            float(body["position"][1]),
        ]

        rotations[k]=float(
            body["rotation"][0]
        )

    return positions,rotations,radius


def orientation_point(center,theta,radius):
    return center+radius*np.array([
        math.cos(theta),
        math.sin(theta),
    ])


def main():
    if not SOLVER.exists():
        raise RuntimeError(f"Missing solver:\n{SOLVER}")

    if not BASE_JSON.exists():
        raise RuntimeError(f"Missing scene:\n{BASE_JSON}")

    base=load_json(BASE_JSON)

    with tempfile.TemporaryDirectory(
        prefix="incline_implicit_"
    ) as tmp:

        tmp_root=Path(tmp)

        sims={}

        for name,mu in SCENARIOS:
            sims[name]=run_case(
                base,
                mu,
                tmp_root,
                name,
            )

        data={}

        for name,mu in SCENARIOS:
            p,r,R=extract(sims[name])

            data[name]={
                "mu":mu,
                "positions":p,
                "rotations":r,
                "radius":R,
            }

        args=sims["mu_01"]["args"]
        funnel=args["spline_funnel"]

        height=float(funnel["height"])
        r_bottom=float(funnel["r_bottom"])
        r_top=float(funnel["r_top"])
        s_bottom=float(funnel["s_bottom"])
        s_top=float(funnel["s_top"])

        spline=CubicSpline(
            [0.0,height],
            [r_bottom,r_top],
            bc_type=(
                (1,s_bottom),
                (1,s_top),
            ),
        )

        wall_y=np.linspace(
            0.0,
            height,
            500,
        )

        wall_x=spline(wall_y)

        all_positions=np.vstack([
            data["mu_01"]["positions"],
            data["mu_03"]["positions"],
        ])

        x_min=min(
            np.min(wall_x),
            np.min(all_positions[:,0])-1.2,
        )

        x_max=max(
            np.max(wall_x),
            np.max(all_positions[:,0])+1.2,
        )

        y_min=min(
            0.0,
            np.min(all_positions[:,1])-1.2,
        )

        y_max=max(
            height,
            np.max(all_positions[:,1])+1.2,
        )

        # --------------------------------------------
        # VIDEO
        # --------------------------------------------

        frame_count=int(
            FPS*VIDEO_SECONDS
        )

        source_frames=np.linspace(
            0,
            min(
                len(data["mu_01"]["positions"]),
                len(data["mu_03"]["positions"]),
            )-1,
            frame_count,
        ).astype(int)

        fig,axes=plt.subplots(
            1,2,
            figsize=(12,6),
        )

        patches=[]
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

            ax.plot(
                wall_x,
                wall_y,
                "k-",
                linewidth=2.5,
            )

            ax.text(
                0.05,
                0.92,
                rf"$\mu={mu:.1f}$",
                transform=ax.transAxes,
                fontsize=15,
            )

            center=data[name]["positions"][0]
            theta=data[name]["rotations"][0]
            radius=data[name]["radius"]

            patch=Circle(
                center,
                radius,
                facecolor="royalblue",
                edgecolor="black",
                linewidth=1.3,
            )

            ax.add_patch(patch)
            patches.append(patch)

            end=orientation_point(
                center,
                theta,
                radius,
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
            source=source_frames[video_frame]

            artists=[]

            for i,(name,_) in enumerate(
                SCENARIOS
            ):
                center=(
                    data[name]["positions"][source]
                )

                theta=(
                    data[name]["rotations"][source]
                )

                radius=data[name]["radius"]

                patches[i].set_center(
                    center
                )

                end=orientation_point(
                    center,
                    theta,
                    radius,
                )

                marker_lines[i].set_data(
                    [center[0],end[0]],
                    [center[1],end[1]],
                )

                artists.extend([
                    patches[i],
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

        print("Saving implicit MP4...")

        animation.save(
            VIDEO_FILE,
            writer=writer,
        )

        plt.close(fig)

        # --------------------------------------------
        # PNG
        #
        # Multiple transparent snapshots make the
        # different amount of rotation visible.
        # --------------------------------------------

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

            ax.plot(
                wall_x,
                wall_y,
                "k-",
                linewidth=2.5,
            )

            ax.text(
                0.05,
                0.92,
                rf"$\mu={mu:.1f}$",
                transform=ax.transAxes,
                fontsize=15,
            )

            positions=data[name]["positions"]
            rotations=data[name]["rotations"]
            radius=data[name]["radius"]

            ids=[
                int(
                    fraction*
                    (len(positions)-1)
                )
                for fraction
                in SNAPSHOT_FRACTIONS
            ]

            for j,frame_id in enumerate(ids):

                center=positions[frame_id]
                theta=rotations[frame_id]

                alpha=0.25+0.75*(
                    j/(len(ids)-1)
                )

                patch=Circle(
                    center,
                    radius,
                    facecolor="royalblue",
                    edgecolor="black",
                    linewidth=1.0,
                    alpha=alpha,
                )

                ax.add_patch(patch)

                end=orientation_point(
                    center,
                    theta,
                    radius,
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