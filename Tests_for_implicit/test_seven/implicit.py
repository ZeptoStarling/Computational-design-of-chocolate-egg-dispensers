#!/usr/bin/env python3
import json
import math
import shutil
import subprocess
from itertools import combinations
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
from tqdm import tqdm

TEST_DIR=Path(
    "/home/austeja/Everything/EggDispenser/"
    "Implicit/testsforimplicit/test_seven"
)
BUILD_DIR=Path(
    "/home/austeja/Everything/EggDispenser/"
    "Implicit/rigid-ipc/build"
)
SOLVER=BUILD_DIR/"rigid_ipc_sim"
SCENE_FILE=TEST_DIR/"test_json.json"
OUTPUT_DIR=TEST_DIR/"output"
SIM_FILE=OUTPUT_DIR/"sim.json"
SOLVER_LOG=TEST_DIR/"solver.log"
BENCHMARK_FILE=TEST_DIR/"multi_sphere_benchmarks.json"
VIDEO_FILE=TEST_DIR/"multi_sphere_sim.mp4"
INTERLOCK_IMAGE=TEST_DIR/"multi_sphere_interlocked.png"

NUM_STEPS=5000
CHECKPOINT_FREQUENCY=10001
EXPECTED_BODIES=7
CONTACT_THRESHOLD=1e-2
VELOCITY_TOL=1e-4

SPHERE_NAMES=[
    "bottom_center",
    "lower_left",
    "lower_right",
    "middle_center",
    "upper_left",
    "upper_right",
    "top_center",
]

MIRROR_PAIRS=[
    (1,2),
    (4,5),
]

CENTERLINE_SPHERES=[
    0,3,6,
]

SPHERE_COLORS=[
    "skyblue",
    "royalblue",
    "coral",
    "mediumseagreen",
    "purple",
    "orange",
    "gold",
]


def load_json(path):
    with open(path,"r") as f:
        return json.load(f)


def run_simulation():
    if not SOLVER.exists():
        raise RuntimeError(f"Solver does not exist:\n  {SOLVER}")

    if not SCENE_FILE.exists():
        raise RuntimeError(f"Scene does not exist:\n  {SCENE_FILE}")

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    command=[
        str(SOLVER),
        "--ngui",
        "--num-steps",str(NUM_STEPS),
        "--output-path",str(OUTPUT_DIR),
        str(SCENE_FILE),
        "--checkpoint-frequency",str(CHECKPOINT_FREQUENCY),
    ]

    print("Running seven-circle implicit simulation...")

    with open(SOLVER_LOG,"w") as log:
        subprocess.run(
            command,
            cwd=BUILD_DIR,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )

    if not SIM_FILE.exists():
        raise RuntimeError(f"Simulation did not produce:\n  {SIM_FILE}")


def get_radius(body):
    if "radius" in body:
        return float(body["radius"])

    vertices=np.asarray(body["vertices"],dtype=float)
    return float(np.mean(np.linalg.norm(vertices,axis=1)))


def main():
    run_simulation()

    sim=load_json(SIM_FILE)
    args=sim["args"]
    bodies=args["rigid_body_problem"]["rigid_bodies"]
    states=sim["animation"]["state_sequence"]

    dt=float(args["timestep"])
    frames=len(states)
    times=np.arange(frames,dtype=float)*dt

    dynamic=[]
    for i,body in enumerate(bodies):
        fixed=body.get("is_dof_fixed",[False,False,False])
        if not all(fixed):
            dynamic.append(i)

    if len(dynamic)!=EXPECTED_BODIES:
        raise RuntimeError(
            f"Expected {EXPECTED_BODIES} dynamic circles, "
            f"found {len(dynamic)}."
        )

    initial=[]
    for body_id in dynamic:
        p=states[0]["rigid_bodies"][body_id]["position"]
        initial.append((float(p[1]),float(p[0]),body_id))

    initial.sort(key=lambda v:(v[0],v[1]))
    ids=[v[2] for v in initial]

    funnel=args["spline_funnel"]
    height=float(funnel["height"])
    r_bottom=float(funnel["r_bottom"])
    r_top=float(funnel["r_top"])
    s_bottom=float(funnel["s_bottom"])
    s_top=float(funnel["s_top"])

    spline=CubicSpline(
        [0.0,height],
        [r_bottom,r_top],
        bc_type=((1,s_bottom),(1,s_top)),
    )

    def funnel_radius(y):
        return float(spline(np.clip(y,0.0,height)))

    def wall_distance(x,y):
        sign=1.0 if x>=0.0 else -1.0

        def objective(yy):
            dx=x-sign*funnel_radius(yy)
            dy=y-yy
            return dx*dx+dy*dy

        result=minimize_scalar(
            objective,
            bounds=(0.0,height),
            method="bounded",
            options={"xatol":1e-10},
        )

        return math.sqrt(min(
            result.fun,
            objective(0.0),
            objective(height),
        ))

    radii=np.array([
        get_radius(bodies[body_id])
        for body_id in ids
    ])

    positions=np.zeros((frames,EXPECTED_BODIES,2))
    speeds=np.zeros((frames,EXPECTED_BODIES))

    for frame_id,state in enumerate(states):
        for circle_id,body_id in enumerate(ids):
            body=state["rigid_bodies"][body_id]

            positions[frame_id,circle_id]=[
                float(body["position"][0]),
                float(body["position"][1]),
            ]

            if "linear_velocity" in body:
                v=np.asarray(body["linear_velocity"][:2],dtype=float)
                speeds[frame_id,circle_id]=np.linalg.norm(v)

    wall_gaps=np.zeros((frames,EXPECTED_BODIES))

    print("Computing analytic gaps...")

    for frame_id in tqdm(range(frames),desc="Wall gaps"):
        for circle_id in range(EXPECTED_BODIES):
            x,y=positions[frame_id,circle_id]
            wall_gaps[frame_id,circle_id]=(
                wall_distance(x,y)-radii[circle_id]
            )

    pairs=list(combinations(range(EXPECTED_BODIES),2))
    pair_gaps=np.zeros((frames,len(pairs)))

    for pair_id,(a,b) in enumerate(pairs):
        delta=positions[:,a,:]-positions[:,b,:]
        pair_gaps[:,pair_id]=(
            np.linalg.norm(delta,axis=1)
            -radii[a]
            -radii[b]
        )

    contacted_pairs=[
        pairs[i]
        for i in range(len(pairs))
        if np.min(pair_gaps[:,i])<=CONTACT_THRESHOLD
    ]

    min_wall_gap=float(np.min(wall_gaps))
    min_pair_gap=float(np.min(pair_gaps))

    max_wall_penetration=max(0.0,-min_wall_gap)
    max_pair_penetration=max(0.0,-min_pair_gap)

    mirror_x_errors=[]
    mirror_y_errors=[]

    for left,right in MIRROR_PAIRS:
        mirror_x_errors.append(
            np.max(np.abs(
                positions[:,left,0]+positions[:,right,0]
            ))
        )
        mirror_y_errors.append(
            np.max(np.abs(
                positions[:,left,1]-positions[:,right,1]
            ))
        )

    max_mirror_x_error=float(max(mirror_x_errors))
    max_mirror_y_error=float(max(mirror_y_errors))

    max_centerline_x_error=float(max(
        np.max(np.abs(positions[:,circle_id,0]))
        for circle_id in CENTERLINE_SPHERES
    ))

    bottom_passed=bool(
        positions[-1,0,1]+radii[0]<0.0
    )

    final_pair_gaps=pair_gaps[-1]
    final_wall_gaps=wall_gaps[-1]
    final_contact_counts=np.zeros(EXPECTED_BODIES,dtype=int)

    for circle_id in range(EXPECTED_BODIES):
        if final_wall_gaps[circle_id]<=CONTACT_THRESHOLD:
            final_contact_counts[circle_id]+=1

    for pair_id,(a,b) in enumerate(pairs):
        if final_pair_gaps[pair_id]<=CONTACT_THRESHOLD:
            final_contact_counts[a]+=1
            final_contact_counts[b]+=1

    remaining=range(1,EXPECTED_BODIES)

    max_final_speed_remaining=float(max(
        speeds[-1,circle_id]
        for circle_id in remaining
    ))

    remaining_interlocked=bool(
        bottom_passed
        and max_final_speed_remaining<=VELOCITY_TOL
        and all(
            final_contact_counts[circle_id]>0
            for circle_id in remaining
        )
    )

    benchmark={
        "test":"seven_circle_multibody",
        "simulation":{
            "timestep":dt,
            "frames":frames,
            "simulated_time":float(times[-1]),
        },
        "result":{
            "bottom_center_passed_through":bottom_passed,
            "remaining_circles_interlocked":remaining_interlocked,
            "maximum_final_speed_remaining":max_final_speed_remaining,
            "circle_circle_contacts":len(contacted_pairs),
            "minimum_wall_gap":min_wall_gap,
            "minimum_pair_gap":min_pair_gap,
            "maximum_wall_penetration":max_wall_penetration,
            "maximum_pair_penetration":max_pair_penetration,
            "maximum_mirror_x_error":max_mirror_x_error,
            "maximum_mirror_y_error":max_mirror_y_error,
            "maximum_centerline_x_error":max_centerline_x_error,
        },
    }

    with open(BENCHMARK_FILE,"w") as f:
        json.dump(benchmark,f,indent=2)

    print()
    print("="*60)
    print("SEVEN-CIRCLE MULTI-BODY TEST")
    print("="*60)
    print(f"Bottom centre passed through : {bottom_passed}")
    print(f"Remaining circles interlocked: {remaining_interlocked}")
    print(f"Circle-circle contacts       : {len(contacted_pairs)}")
    print(
        "Max final speed remaining   : "
        f"{max_final_speed_remaining:.12e}"
    )
    print(f"Minimum wall gap             : {min_wall_gap:.12e}")
    print(f"Minimum pair gap             : {min_pair_gap:.12e}")
    print(
        "Maximum wall penetration    : "
        f"{max_wall_penetration:.12e}"
    )
    print(
        "Maximum pair penetration    : "
        f"{max_pair_penetration:.12e}"
    )
    print(
        "Maximum mirror x error      : "
        f"{max_mirror_x_error:.12e}"
    )
    print(
        "Maximum mirror y error      : "
        f"{max_mirror_y_error:.12e}"
    )
    print(
        "Maximum centre-line x error : "
        f"{max_centerline_x_error:.12e}"
    )

    y_vals=np.linspace(0.0,height,500)
    r_vals=np.array([funnel_radius(y) for y in y_vals])

    x_min=-r_top-1.5
    x_max=r_top+1.5
    video_y_min=-0.5
    video_y_max=max(
        height+1.0,
        max(
            positions[0,i,1]+radii[i]
            for i in range(EXPECTED_BODIES)
        )+0.5,
    )

    fig,ax=plt.subplots(figsize=(7,9))
    ax.set_aspect("equal")
    ax.set_xlim(x_min,x_max)
    ax.set_ylim(video_y_min,video_y_max)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.plot(-r_vals,y_vals,"k-",linewidth=2.0)
    ax.plot(r_vals,y_vals,"k-",linewidth=2.0)

    patches=[]

    for circle_id in range(EXPECTED_BODIES):
        patch=Circle(
            tuple(positions[0,circle_id]),
            radii[circle_id],
            fc=SPHERE_COLORS[circle_id],
            ec="black",
            lw=1.0,
        )
        ax.add_patch(patch)
        patches.append(patch)

    def update(frame_id):
        for circle_id in range(EXPECTED_BODIES):
            patches[circle_id].set_center(
                tuple(positions[frame_id,circle_id])
            )
        return patches

    anim=FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=20,
        blit=False,
    )

    pbar=tqdm(
        total=frames,
        desc="Saving MP4",
        unit="frame",
        ncols=100,
    )
    last_frame=0

    def progress_callback(current,total):
        nonlocal last_frame
        if current==total or current-last_frame>=50:
            pbar.update(current-last_frame)
            last_frame=current

    anim.save(
        VIDEO_FILE,
        writer="ffmpeg",
        fps=60,
        progress_callback=progress_callback,
    )

    if pbar.n<frames:
        pbar.update(frames-pbar.n)

    pbar.close()
    plt.close(fig)

    if remaining_interlocked:
        image_y_max=max(
            height+1.0,
            max(
                positions[-1,i,1]+radii[i]
                for i in remaining
            )+0.5,
        )

        fig,ax=plt.subplots(figsize=(7,7))
        ax.set_aspect("equal")
        ax.set_xlim(x_min,x_max)
        ax.set_ylim(-0.5,image_y_max)

        ax.plot(-r_vals,y_vals,"k-",linewidth=2.2)
        ax.plot(r_vals,y_vals,"k-",linewidth=2.2)

        for circle_id in remaining:
            x,y=positions[-1,circle_id]
            patch=Circle(
                (x,y),
                radii[circle_id],
                fc=SPHERE_COLORS[circle_id],
                ec="black",
                lw=1.0,
            )
            ax.add_patch(patch)

        ax.axis("off")
        fig.subplots_adjust(
            left=0,
            right=1,
            bottom=0,
            top=1,
        )
        fig.savefig(
            INTERLOCK_IMAGE,
            dpi=250,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close(fig)

    print()
    print(f"Benchmark: {BENCHMARK_FILE}")
    print(f"Video    : {VIDEO_FILE}")

    if remaining_interlocked:
        print(f"Image    : {INTERLOCK_IMAGE}")

    print(f"Solver log: {SOLVER_LOG}")
    print("="*60)


if __name__=="__main__":
    main()