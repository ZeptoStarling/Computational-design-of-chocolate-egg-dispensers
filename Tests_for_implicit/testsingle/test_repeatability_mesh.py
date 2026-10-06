#!/usr/bin/env python3

import json
import math
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

TEST_DIR=Path("/home/austeja/Everything/EggDispenser/Implicit/testsforimplicit/testsingle")
BUILD_DIR=Path("/home/austeja/Everything/EggDispenser/rigid-ipc/build")
SOLVER=BUILD_DIR/"rigid_ipc_sim"

ROOT=TEST_DIR/"trajectory_repeatability"
IMPLICIT_DIR=ROOT/"implicit"
MESH_DIR=ROOT/"mesh"

CONDITIONS_FILE=ROOT/"initial_conditions.json"
RESULT_FILE=MESH_DIR/"mesh_comparison_results.json"

VARIED_PLOT=MESH_DIR/"mesh_implicit_varied_trajectories.png"
REPEAT_PLOT=MESH_DIR/"mesh_repeated_trajectories.png"

RUNS=10

HEIGHT=6.0
R_BOTTOM=1.5
R_TOP=5.0
S_BOTTOM=0.0
S_TOP=0.0
WALL_THICKNESS=0.2

RADIUS=1.0
FUNNEL_POINTS=96
CIRCLE_SEGMENTS=32

TIMESTEP=0.0005
NUM_STEPS=5000
CHECKPOINT_FREQUENCY=10001


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w") as f:
        json.dump(data,f,indent=2)


def make_edges(n):
    return [[i,(i+1)%n] for i in range(n)]


def funnel_radius(t):
    h00=2*t**3-3*t**2+1
    h10=t**3-2*t**2+t
    h01=-2*t**3+3*t**2
    h11=t**3-t**2

    return max(
        h00*R_BOTTOM
        +h10*S_BOTTOM
        +h01*R_TOP
        +h11*S_TOP,
        R_BOTTOM,
    )


def make_funnel_walls():
    left_inner=[]
    left_outer=[]
    right_inner=[]
    right_outer=[]

    for i in range(FUNNEL_POINTS):
        t=i/(FUNNEL_POINTS-1)
        y=t*HEIGHT
        r=funnel_radius(t)

        left_inner.append([-r,y])
        left_outer.append([-r-WALL_THICKNESS,y])
        right_inner.append([r,y])
        right_outer.append([r+WALL_THICKNESS,y])

    left=left_inner+list(reversed(left_outer))
    right=list(reversed(right_inner))+right_outer

    return left,right


def make_circle_vertices():
    return [
        [
            RADIUS*math.cos(2*math.pi*i/CIRCLE_SEGMENTS),
            RADIUS*math.sin(2*math.pi*i/CIRCLE_SEGMENTS),
        ]
        for i in range(CIRCLE_SEGMENTS)
    ]


def make_body(vertices,position,fixed):
    return {
        "vertices":vertices,
        "polygons":[vertices],
        "edges":make_edges(len(vertices)),
        "oriented":True,
        "position":[float(position[0]),float(position[1])],
        "rotation":[0.0],
        "linear_velocity":[0.0,0.0],
        "angular_velocity":[0.0],
        "is_dof_fixed":[fixed,fixed,fixed],
    }


def make_scene(position):
    left,right=make_funnel_walls()
    circle=make_circle_vertices()

    return {
        "scene_type":"distance_barrier_rb_problem",
        "solver":"ipc_solver",
        "timestep":TIMESTEP,
        "max_time":10.0,
        "rigid_body_problem":{
            "coefficient_restitution":0.2,
            "gravity":[0.0,-9.81],
            "rigid_bodies":[
                make_body(left,(0.0,0.0),True),
                make_body(right,(0.0,0.0),True),
                make_body(circle,position,False),
            ],
        },
        "friction_constraints":{
            "iterations":40,
        },
    }


def run_case(case_dir,position):
    case_dir.mkdir(parents=True,exist_ok=True)

    scene_file=case_dir/"input.json"
    output_dir=case_dir/"output"
    sim_file=output_dir/"sim.json"

    if sim_file.exists():
        try:
            if load_json(sim_file)["animation"]["state_sequence"]:
                return sim_file
        except Exception:
            pass

    if output_dir.exists():
        shutil.rmtree(output_dir)

    save_json(scene_file,make_scene(position))

    command=[
        str(SOLVER),
        "--ngui",
        "--num-steps",str(NUM_STEPS),
        "--output-path",str(output_dir),
        str(scene_file),
        "--checkpoint-frequency",str(CHECKPOINT_FREQUENCY),
    ]

    print(" ".join(command))

    with open(case_dir/"solver.log","w") as log:
        subprocess.run(
            command,
            cwd=BUILD_DIR,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )

    if not sim_file.exists():
        raise RuntimeError(f"Missing {sim_file}")

    return sim_file


def dynamic_body_index(sim):
    dynamic=[
        i
        for i,b in enumerate(
            sim["args"]["rigid_body_problem"]["rigid_bodies"]
        )
        if not all(b.get("is_dof_fixed",[False,False,False]))
    ]

    if len(dynamic)!=1:
        raise RuntimeError(
            f"Expected one dynamic body, found {len(dynamic)}"
        )

    return dynamic[0]


def trajectory(sim_file):
    sim=load_json(sim_file)
    body_id=dynamic_body_index(sim)
    states=sim["animation"]["state_sequence"]
    dt=float(sim["args"]["timestep"])

    x=np.array([
        float(s["rigid_bodies"][body_id]["position"][0])
        for s in states
    ])

    y=np.array([
        float(s["rigid_bodies"][body_id]["position"][1])
        for s in states
    ])

    t=np.arange(len(states),dtype=float)*dt
    return t,x,y


def trajectory_error(reference,test):
    t0,x0,y0=reference
    t,x,y=test

    xr=np.interp(t,t0,x0)
    yr=np.interp(t,t0,y0)

    error=np.sqrt((x-xr)**2+(y-yr)**2)

    return {
        "maximum":float(np.max(error)),
        "rms":float(np.sqrt(np.mean(error**2))),
        "final_dx":float(x[-1]-xr[-1]),
        "final_dy":float(y[-1]-yr[-1]),
    }


def implicit_sim_file(kind,run):
    return (
        IMPLICIT_DIR
        /kind
        /f"run{run:02d}"
        /"output"
        /"sim.json"
    )


def save_varied_plot(mesh_trajectories,implicit_trajectories):
    spline=CubicSpline(
        [0.0,HEIGHT],
        [R_BOTTOM,R_TOP],
        bc_type=((1,S_BOTTOM),(1,S_TOP)),
    )

    ys=np.linspace(0.0,HEIGHT,500)
    rs=spline(ys)

    fig,ax=plt.subplots(figsize=(7,8))

    ax.plot(-rs,ys,"k-",linewidth=1.5)
    ax.plot(rs,ys,"k-",linewidth=1.5)

    for i,(implicit,mesh) in enumerate(
        zip(implicit_trajectories,mesh_trajectories)
    ):
        _,xi,yi=implicit
        _,xm,ym=mesh

        ax.plot(
            xi,yi,
            linewidth=1.3,
            alpha=0.75,
            label="implicit" if i==0 else None,
        )

        ax.plot(
            xm,ym,
            "--",
            linewidth=1.1,
            alpha=0.75,
            label="mesh" if i==0 else None,
        )

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(VARIED_PLOT,dpi=200)
    plt.close(fig)


def save_repeat_plot(trajectories):
    spline=CubicSpline(
        [0.0,HEIGHT],
        [R_BOTTOM,R_TOP],
        bc_type=((1,S_BOTTOM),(1,S_TOP)),
    )

    ys=np.linspace(0.0,HEIGHT,500)
    rs=spline(ys)

    fig,ax=plt.subplots(figsize=(7,8))

    ax.plot(-rs,ys,"k-",linewidth=1.5)
    ax.plot(rs,ys,"k-",linewidth=1.5)

    for _,x,y in trajectories:
        ax.plot(x,y,linewidth=1.2,alpha=0.8)

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPEAT_PLOT,dpi=200)
    plt.close(fig)


def main():
    if not SOLVER.exists():
        raise RuntimeError(f"Missing solver: {SOLVER}")

    if not CONDITIONS_FILE.exists():
        raise RuntimeError(
            "Run test_repeatability_implicit.py first.\n"
            f"Missing: {CONDITIONS_FILE}"
        )

    conditions=load_json(CONDITIONS_FILE)
    MESH_DIR.mkdir(parents=True,exist_ok=True)

    mesh_varied=[]
    implicit_varied=[]
    paired_results=[]

    print("\nPAIRED VARIED-INITIAL-CONDITION TEST")

    for item in conditions["varied"]:
        run=item["run"]
        position=item["position"]

        print(f"Run {run}: position={position}")

        mesh_file=run_case(
            MESH_DIR/"varied"/f"run{run:02d}",
            position,
        )

        implicit_file=implicit_sim_file(
            "varied",
            run,
        )

        if not implicit_file.exists():
            raise RuntimeError(
                f"Missing implicit run:\n{implicit_file}"
            )

        mesh_traj=trajectory(mesh_file)
        implicit_traj=trajectory(implicit_file)

        mesh_varied.append(mesh_traj)
        implicit_varied.append(implicit_traj)

        error=trajectory_error(
            implicit_traj,
            mesh_traj,
        )

        paired_results.append({
            "run":run,
            "initial_position":position,
            **error,
        })

        print(
            f"  max={error['maximum']:.8f}  "
            f"RMS={error['rms']:.8f}  "
            f"final dx={error['final_dx']:+.8f}"
        )

    mesh_repeated=[]

    print("\nIDENTICAL MESH REPEATABILITY TEST")

    repeated_position=conditions["repeated_position"]

    for run in range(1,RUNS+1):
        print(f"Run {run}: position={repeated_position}")

        mesh_file=run_case(
            MESH_DIR/"repeated"/f"run{run:02d}",
            repeated_position,
        )

        mesh_repeated.append(
            trajectory(mesh_file)
        )

    mesh_reference=mesh_repeated[0]

    mesh_repeat_errors=[
        trajectory_error(mesh_reference,traj)
        for traj in mesh_repeated
    ]

    implicit_result_file=(
        IMPLICIT_DIR
        /"implicit_repeatability_results.json"
    )

    implicit_result=load_json(
        implicit_result_file
    )

    signed_dx=np.array([
        r["final_dx"]
        for r in paired_results
    ])

    paired_max=np.array([
        r["maximum"]
        for r in paired_results
    ])

    paired_rms=np.array([
        r["rms"]
        for r in paired_results
    ])

    mesh_repeat_max=np.array([
        r["maximum"]
        for r in mesh_repeat_errors
    ])

    mesh_repeat_rms=np.array([
        r["rms"]
        for r in mesh_repeat_errors
    ])

    results={
        "mesh_resolution":{
            "funnel_points_per_side":FUNNEL_POINTS,
            "circle_segments":CIRCLE_SEGMENTS,
        },
        "paired_varied_initial_conditions":{
            "runs":paired_results,
            "mean_maximum_error":float(np.mean(paired_max)),
            "mean_rms_error":float(np.mean(paired_rms)),
            "mean_signed_final_dx":float(np.mean(signed_dx)),
            "std_signed_final_dx":float(np.std(signed_dx)),
            "positive_final_dx_count":int(np.sum(signed_dx>0)),
            "negative_final_dx_count":int(np.sum(signed_dx<0)),
        },
        "mesh_repeatability":{
            "runs":mesh_repeat_errors,
            "maximum_of_maximum_errors":float(np.max(mesh_repeat_max)),
            "mean_maximum_error":float(np.mean(mesh_repeat_max)),
            "mean_rms_error":float(np.mean(mesh_repeat_rms)),
        },
        "implicit_repeatability":implicit_result["repeatability"],
    }

    save_json(RESULT_FILE,results)
    save_varied_plot(mesh_varied,implicit_varied)
    save_repeat_plot(mesh_repeated)

    print("\n====================================================")
    print("PAIRED MESH / IMPLICIT TEST")
    print("====================================================")
    print(
        "Mean max mesh-implicit error : "
        f"{np.mean(paired_max):.8f}"
    )
    print(
        "Mean RMS mesh-implicit error : "
        f"{np.mean(paired_rms):.8f}"
    )
    print(
        "Mean signed final dx         : "
        f"{np.mean(signed_dx):+.8f}"
    )
    print(
        "Std signed final dx          : "
        f"{np.std(signed_dx):.8f}"
    )
    print(
        "Signs of final dx            : "
        f"{np.sum(signed_dx>0)} positive, "
        f"{np.sum(signed_dx<0)} negative"
    )

    print("\n====================================================")
    print("IDENTICAL-RUN REPEATABILITY")
    print("====================================================")
    print(
        "Mesh largest deviation       : "
        f"{np.max(mesh_repeat_max):.12e}"
    )
    print(
        "Implicit largest deviation   : "
        f"{results['implicit_repeatability']['maximum_of_maximum_errors']:.12e}"
    )
    print(f"\nComparison plot : {VARIED_PLOT}")
    print(f"Mesh repeats    : {REPEAT_PLOT}")
    print(f"Results         : {RESULT_FILE}")


if __name__=="__main__":
    main()