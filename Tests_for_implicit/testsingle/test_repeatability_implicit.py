#!/usr/bin/env python3

import json
import random
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TEST_DIR=Path("/home/austeja/Everything/EggDispenser/Implicit/testsforimplicit/testsingle")
BUILD_DIR=Path("/home/austeja/Everything/EggDispenser/Implicit/rigid-ipc/build")
SOLVER=BUILD_DIR/"rigid_ipc_sim"

ROOT=TEST_DIR/"trajectory_repeatability"
IMPLICIT_DIR=ROOT/"implicit"
CONDITIONS_FILE=ROOT/"initial_conditions.json"
RESULT_FILE=IMPLICIT_DIR/"implicit_repeatability_results.json"

VARIED_PLOT=IMPLICIT_DIR/"implicit_varied_trajectories.png"
REPEAT_PLOT=IMPLICIT_DIR/"implicit_repeated_trajectories.png"

RUNS=10
POSITION_SEED=12345

HEIGHT=6.0
R_BOTTOM=1.5
R_TOP=5.0
S_BOTTOM=0.0
S_TOP=0.0

RADIUS=1.0
BASE_X=1.2
BASE_Y=5.0

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


def create_conditions():
    ROOT.mkdir(parents=True,exist_ok=True)

    if CONDITIONS_FILE.exists():
        return load_json(CONDITIONS_FILE)

    rng=random.Random(POSITION_SEED)

    # Small horizontal changes around the original x=1.2 case.
    varied=[
        {
            "run":i+1,
            "position":[rng.uniform(0.9,1.5),BASE_Y],
        }
        for i in range(RUNS)
    ]

    conditions={
        "position_seed":POSITION_SEED,
        "varied":varied,
        "repeated_position":[BASE_X,BASE_Y],
    }

    save_json(CONDITIONS_FILE,conditions)
    return conditions


def make_scene(position):
    return {
        "scene_type":"distance_barrier_rb_problem",
        "solver":"ipc_solver",
        "timestep":TIMESTEP,
        "max_time":10.0,
        "spline_funnel":{
            "enabled":True,
            "r_bottom":R_BOTTOM,
            "r_top":R_TOP,
            "s_bottom":S_BOTTOM,
            "s_top":S_TOP,
            "height":HEIGHT,
        },
        "rigid_body_problem":{
            "coefficient_restitution":0.2,
            "gravity":[0.0,-9.81],
            "rigid_bodies":[
                {
                    "position":[float(position[0]),float(position[1])],
                    "rotation":[0.0],
                    "linear_velocity":[0.0,0.0],
                    "angular_velocity":[0.0],
                    "is_dof_fixed":[False,False,False],
                    "radius":RADIUS,
                }
            ],
        },
        "friction_constraints":{
            "static_friction_speed_bound":0.001,
            "iterations":1,
        },
        "ipc_solver":{
            "convergence_criteria":"velocity",
            "velocity_conv_tol":0.0001,
            "is_velocity_conv_tol_abs":True,
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


def trajectory(sim_file):
    sim=load_json(sim_file)
    states=sim["animation"]["state_sequence"]

    bodies=sim["args"]["rigid_body_problem"]["rigid_bodies"]
    dynamic=[
        i for i,b in enumerate(bodies)
        if not all(b.get("is_dof_fixed",[False,False,False]))
    ]

    if len(dynamic)!=1:
        raise RuntimeError(f"Expected one dynamic body, found {len(dynamic)}")

    body_id=dynamic[0]
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


def save_plot(trajectories,path):
    fig,ax=plt.subplots(figsize=(7,8))

    ys=np.linspace(0.0,HEIGHT,500)

    from scipy.interpolate import CubicSpline

    spline=CubicSpline(
        [0.0,HEIGHT],
        [R_BOTTOM,R_TOP],
        bc_type=((1,S_BOTTOM),(1,S_TOP)),
    )

    rs=spline(ys)

    ax.plot(-rs,ys,"k-",linewidth=1.5)
    ax.plot(rs,ys,"k-",linewidth=1.5)

    for _,x,y in trajectories:
        ax.plot(x,y,linewidth=1.2,alpha=0.8)

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path,dpi=200)
    plt.close(fig)


def main():
    if not SOLVER.exists():
        raise RuntimeError(f"Missing solver: {SOLVER}")

    conditions=create_conditions()
    IMPLICIT_DIR.mkdir(parents=True,exist_ok=True)

    varied=[]
    repeated=[]

    print("\nVARIED INITIAL POSITIONS")

    for item in conditions["varied"]:
        run=item["run"]
        position=item["position"]

        print(f"Run {run}: position={position}")

        sim_file=run_case(
            IMPLICIT_DIR/"varied"/f"run{run:02d}",
            position,
        )

        varied.append(trajectory(sim_file))

    print("\nIDENTICAL REPEATED RUNS")

    repeated_position=conditions["repeated_position"]

    for run in range(1,RUNS+1):
        print(f"Run {run}: position={repeated_position}")

        sim_file=run_case(
            IMPLICIT_DIR/"repeated"/f"run{run:02d}",
            repeated_position,
        )

        repeated.append(trajectory(sim_file))

    reference=repeated[0]

    repeat_errors=[
        trajectory_error(reference,traj)
        for traj in repeated
    ]

    max_errors=[
        r["maximum"]
        for r in repeat_errors
    ]

    rms_errors=[
        r["rms"]
        for r in repeat_errors
    ]

    result={
        "representation":"implicit",
        "varied_initial_positions":conditions["varied"],
        "repeated_position":repeated_position,
        "repeatability":{
            "runs":repeat_errors,
            "maximum_of_maximum_errors":float(max(max_errors)),
            "mean_maximum_error":float(np.mean(max_errors)),
            "mean_rms_error":float(np.mean(rms_errors)),
        },
    }

    save_json(RESULT_FILE,result)
    save_plot(varied,VARIED_PLOT)
    save_plot(repeated,REPEAT_PLOT)

    print("\n==============================================")
    print("IMPLICIT REPEATABILITY")
    print("==============================================")
    print(
        "Largest repeat-vs-run1 trajectory error : "
        f"{max(max_errors):.12e}"
    )
    print(
        "Mean maximum trajectory error           : "
        f"{np.mean(max_errors):.12e}"
    )
    print(
        "Mean RMS trajectory error               : "
        f"{np.mean(rms_errors):.12e}"
    )
    print(f"Varied trajectories   : {VARIED_PLOT}")
    print(f"Repeated trajectories : {REPEAT_PLOT}")
    print(f"Results               : {RESULT_FILE}")


if __name__=="__main__":
    main()