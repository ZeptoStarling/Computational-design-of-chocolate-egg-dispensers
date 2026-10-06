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


TEST_DIR=Path(
    "/home/austeja/Everything/EggDispenser/"
    "Implicit/testsforimplicit/teststwo"
)
BUILD_DIR=Path(
    "/home/austeja/Everything/EggDispenser/"
    "rigid-ipc/build"
)

SOLVER=BUILD_DIR/"rigid_ipc_sim"

SCENE_FILE=(
    TEST_DIR/
    "mesh_arch"/
    "N1024_N1024"/
    "matched_wall_gap.json"
)

VIDEO_FILE=TEST_DIR/"two_circle_mesh.mp4"
IMAGE_FILE=TEST_DIR/"two_circle_mesh_slipping.png"

NUM_STEPS=5000
FPS=60

# The image is taken when the average circle centre is
# closest to this height.
SLIP_IMAGE_Y=0.7


def load_json(path):
    with open(path,"r") as f:
        return json.load(f)


def transform(vertices,position,angle):
    V=np.asarray(vertices,dtype=float)

    c=math.cos(angle)
    s=math.sin(angle)

    R=np.array([
        [c,-s],
        [s,c],
    ])

    return (
        V@R.T
        +np.asarray(position,dtype=float)
    )


def main():
    if not SOLVER.exists():
        raise RuntimeError(f"Solver not found: {SOLVER}")

    if not SCENE_FILE.exists():
        raise RuntimeError(
            "Mesh scene not found:\n"
            f"  {SCENE_FILE}\n\n"
            "Run the mesh arch validation script once first."
        )

    with tempfile.TemporaryDirectory(
        prefix="mesh_two_circle_"
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

        print("Running mesh simulation...")

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

        if len(bodies)!=4:
            raise RuntimeError(
                f"Expected 4 bodies, found {len(bodies)}"
            )

        wall_ids=[0,1]
        circle_ids=[2,3]

        wall_vertices=[
            np.asarray(
                bodies[i]["vertices"],
                dtype=float,
            )
            for i in wall_ids
        ]

        circle_local=[
            np.asarray(
                bodies[i]["vertices"],
                dtype=float,
            )
            for i in circle_ids
        ]

        frames=len(states)

        positions=np.zeros((frames,2,2))
        rotations=np.zeros((frames,2))

        for frame_id,state in enumerate(states):
            for j,body_id in enumerate(circle_ids):
                b=state["rigid_bodies"][body_id]

                positions[frame_id,j]=[
                    float(b["position"][0]),
                    float(b["position"][1]),
                ]

                rotations[frame_id,j]=float(
                    b["rotation"][0]
                )

        all_wall=np.vstack(wall_vertices)

        x_min=float(np.min(all_wall[:,0]))-0.5
        x_max=float(np.max(all_wall[:,0]))+0.5

        initial_circle_y=float(
            np.max(positions[0,:,1])
        )

        y_max=max(
            float(np.max(all_wall[:,1]))+0.5,
            initial_circle_y+1.5,
        )

        y_min=min(
            -1.5,
            float(np.min(positions[:,:,1]))-1.2,
        )

        # --------------------------------------------------
        # MP4
        # --------------------------------------------------

        fig,ax=plt.subplots(figsize=(7,9))

        ax.set_aspect("equal")
        ax.set_xlim(x_min,x_max)
        ax.set_ylim(y_min,y_max)
        ax.axis("off")

        for V in wall_vertices:
            ax.add_patch(
                Polygon(
                    V,
                    closed=True,
                    facecolor="0.90",
                    edgecolor="black",
                    linewidth=1.5,
                )
            )

        circle_patches=[]

        for j in range(2):
            V=transform(
                circle_local[j],
                positions[0,j],
                rotations[0,j],
            )

            patch=Polygon(
                V,
                closed=True,
                facecolor=(
                    "royalblue"
                    if j==0
                    else "coral"
                ),
                edgecolor="black",
                linewidth=1.0,
            )

            ax.add_patch(patch)
            circle_patches.append(patch)

        def update(frame_id):
            for j in range(2):
                V=transform(
                    circle_local[j],
                    positions[frame_id,j],
                    rotations[frame_id,j],
                )

                circle_patches[j].set_xy(V)

            return circle_patches

        animation=FuncAnimation(
            fig,
            update,
            frames=frames,
            blit=False,
        )

        writer=FFMpegWriter(
            fps=FPS,
            codec="libx264",
            extra_args=[
                "-pix_fmt","yuv420p",
            ],
        )

        print("Saving mesh MP4...")

        animation.save(
            VIDEO_FILE,
            writer=writer,
        )

        plt.close(fig)

        # --------------------------------------------------
        # Choose frame where they are visibly slipping out.
        # --------------------------------------------------

        mean_y=np.mean(
            positions[:,:,1],
            axis=1,
        )

        candidates=np.where(
            mean_y<mean_y[0]-0.15
        )[0]

        if len(candidates)>0:
            slip_frame=int(
                candidates[
                    np.argmin(
                        np.abs(
                            mean_y[candidates]
                            -SLIP_IMAGE_Y
                        )
                    )
                ]
            )
        else:
            slip_frame=int(
                np.argmin(mean_y)
            )

        print(
            "Slip image frame:",
            slip_frame,
            "mean y =",
            mean_y[slip_frame],
        )

        # --------------------------------------------------
        # Slipping PNG
        # --------------------------------------------------

        fig,ax=plt.subplots(figsize=(7,6))

        ax.set_aspect("equal")

        ax.set_xlim(
            -3.8,
            3.8,
        )
        ax.set_ylim(
            -1.5,
            3.5,
        )

        ax.axis("off")

        for V in wall_vertices:
            ax.add_patch(
                Polygon(
                    V,
                    closed=True,
                    facecolor="0.90",
                    edgecolor="black",
                    linewidth=1.8,
                )
            )

        for j in range(2):
            V=transform(
                circle_local[j],
                positions[slip_frame,j],
                rotations[slip_frame,j],
            )

            ax.add_patch(
                Polygon(
                    V,
                    closed=True,
                    facecolor=(
                        "royalblue"
                        if j==0
                        else "coral"
                    ),
                    edgecolor="black",
                    linewidth=1.1,
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