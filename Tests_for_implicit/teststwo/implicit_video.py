#!/usr/bin/env python3

import json
import shutil
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


TEST_DIR=Path(
    "/home/austeja/Everything/EggDispenser/"
    "Implicit/testsforimplicit/teststwo"
)
BUILD_DIR=Path(
    "/home/austeja/Everything/EggDispenser/"
    "Implicit/rigid-ipc/build"
)

SOLVER=BUILD_DIR/"rigid_ipc_sim"
SCENE_FILE=TEST_DIR/"test_json.json"

VIDEO_FILE=TEST_DIR/"two_circle_implicit.mp4"
IMAGE_FILE=TEST_DIR/"two_circle_implicit_interlocked.png"

NUM_STEPS=5000
FPS=60


def load_json(path):
    with open(path,"r") as f:
        return json.load(f)


def main():
    if not SOLVER.exists():
        raise RuntimeError(f"Solver not found: {SOLVER}")

    if not SCENE_FILE.exists():
        raise RuntimeError(f"Scene not found: {SCENE_FILE}")

    with tempfile.TemporaryDirectory(
        prefix="implicit_two_circle_"
    ) as tmp:
        output=Path(tmp)/"output"

        command=[
            str(SOLVER),
            "--ngui",
            "--num-steps",str(NUM_STEPS),
            "--output-path",str(output),
            str(SCENE_FILE),
            "--checkpoint-frequency","10001",
        ]

        print("Running implicit simulation...")

        subprocess.run(
            command,
            cwd=BUILD_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            check=True,
        )

        sim=load_json(output/"sim.json")

        args=sim["args"]
        bodies=args["rigid_body_problem"]["rigid_bodies"]
        states=sim["animation"]["state_sequence"]

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

        if len(dynamic)!=2:
            raise RuntimeError(
                f"Expected 2 dynamic circles, found {len(dynamic)}"
            )

        dynamic.sort(
            key=lambda i:
            states[0]["rigid_bodies"][i]["position"][0]
        )

        radii=np.array([
            float(bodies[i]["radius"])
            for i in dynamic
        ])

        positions=np.zeros((len(states),2,2))

        for frame_id,state in enumerate(states):
            for j,body_id in enumerate(dynamic):
                p=state["rigid_bodies"][body_id]["position"]
                positions[frame_id,j]=[
                    float(p[0]),
                    float(p[1]),
                ]

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

        y_wall=np.linspace(0.0,height,500)
        r_wall=spline(y_wall)

        x_min=-r_top-0.8
        x_max=r_top+0.8

        y_min=-1.2
        y_max=max(
            height+0.5,
            np.max(
                positions[0,:,1]+radii
            )+0.5,
        )

        # --------------------------------------------------
        # MP4
        # --------------------------------------------------

        fig,ax=plt.subplots(figsize=(7,9))

        ax.set_aspect("equal")
        ax.set_xlim(x_min,x_max)
        ax.set_ylim(y_min,y_max)
        ax.axis("off")

        ax.plot(
            -r_wall,y_wall,
            "k-",
            linewidth=2.2,
        )
        ax.plot(
            r_wall,y_wall,
            "k-",
            linewidth=2.2,
        )

        circles=[]

        for i in range(2):
            patch=Circle(
                tuple(positions[0,i]),
                radii[i],
                facecolor=(
                    "royalblue"
                    if i==0
                    else "coral"
                ),
                edgecolor="black",
                linewidth=1.2,
            )
            ax.add_patch(patch)
            circles.append(patch)

        def update(frame_id):
            for i in range(2):
                circles[i].set_center(
                    tuple(positions[frame_id,i])
                )
            return circles

        animation=FuncAnimation(
            fig,
            update,
            frames=len(states),
            blit=False,
        )

        writer=FFMpegWriter(
            fps=FPS,
            codec="libx264",
            extra_args=[
                "-pix_fmt","yuv420p",
            ],
        )

        print("Saving implicit MP4...")

        animation.save(
            VIDEO_FILE,
            writer=writer,
        )

        plt.close(fig)

        # --------------------------------------------------
        # Final interlocked PNG
        # --------------------------------------------------

        final=positions[-1]

        image_y_max=min(
            height,
            max(
                final[:,1]+radii
            )+0.6,
        )

        image_y_max=max(
            image_y_max,
            3.2,
        )

        image_r=float(spline(image_y_max))

        fig,ax=plt.subplots(figsize=(7,6))

        ax.set_aspect("equal")
        ax.set_xlim(
            -image_r-0.6,
            image_r+0.6,
        )
        ax.set_ylim(
            -0.4,
            image_y_max,
        )
        ax.axis("off")

        mask=y_wall<=image_y_max

        ax.plot(
            -r_wall[mask],
            y_wall[mask],
            "k-",
            linewidth=2.4,
        )
        ax.plot(
            r_wall[mask],
            y_wall[mask],
            "k-",
            linewidth=2.4,
        )

        for i in range(2):
            ax.add_patch(
                Circle(
                    tuple(final[i]),
                    radii[i],
                    facecolor=(
                        "royalblue"
                        if i==0
                        else "coral"
                    ),
                    edgecolor="black",
                    linewidth=1.3,
                )
            )

        fig.subplots_adjust(
            left=0,
            right=1,
            bottom=0,
            top=1,
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