import json
import re
import subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

TEST_DIR=Path("/home/austeja/Everything/EggDispenser/Implicit/testsforimplicit/testsingle")
BUILD_DIR=Path("/home/austeja/Everything/EggDispenser/rigid-ipc/build")
SOLVER=BUILD_DIR/"rigid_ipc_sim"
MESH_DIR=TEST_DIR/"meshes_spheres"
REFERENCE_SIM=TEST_DIR/"output"/"sim.json"
OUTPUT_DIR=TEST_DIR/"mesh_sphere_outputs"
BENCHMARK_FILE=TEST_DIR/"sphere_convergence_benchmark.json"
TRAJECTORY_PLOT=TEST_DIR/"sphere_convergence_trajectories.png"

RUN_SOLVER=True
NUM_STEPS=5000
CHECKPOINT_FREQUENCY=10001
FIXED_FUNNEL_POINTS=96

HEIGHT=6.0
R_BOTTOM=1.5
R_TOP=5.0
S_BOTTOM=0.0
S_TOP=0.0

def load_json(path):
    with open(path) as f:
        return json.load(f)

def dynamic_body_index(sim):
    for i,body in enumerate(sim["args"]["rigid_body_problem"]["rigid_bodies"]):
        if not all(body.get("is_dof_fixed",[False,False,False])):
            return i
    raise RuntimeError("No dynamic body found.")

def trajectory(sim):
    body_id=dynamic_body_index(sim)
    states=sim["animation"]["state_sequence"]
    dt=float(sim["args"]["timestep"])
    x=np.array([float(s["rigid_bodies"][body_id]["position"][0]) for s in states])
    y=np.array([float(s["rigid_bodies"][body_id]["position"][1]) for s in states])
    t=np.arange(len(states))*dt
    return t,x,y

def resolution(path):
    match=re.search(r"mesh_sphere_N(\d+)",path.name)
    if match is None:
        raise RuntimeError(f"Cannot determine sphere resolution from {path.name}")
    return int(match.group(1))

def run_mesh(mesh_file,n):
    case_dir=OUTPUT_DIR/f"sphere_N{n:03d}"
    case_dir.mkdir(parents=True,exist_ok=True)
    sim_file=case_dir/"sim.json"

    if RUN_SOLVER:
        command=[
            str(SOLVER),"--ngui",
            "--num-steps",str(NUM_STEPS),
            "--output-path",str(case_dir),
            str(mesh_file),
            "--checkpoint-frequency",str(CHECKPOINT_FREQUENCY)
        ]
        print(f"Running sphere N={n}, funnel N={FIXED_FUNNEL_POINTS}")
        subprocess.run(command,check=True,cwd=BUILD_DIR)

    if not sim_file.exists():
        raise RuntimeError(f"Missing {sim_file}")

    return sim_file

def main():
    if not REFERENCE_SIM.exists():
        raise RuntimeError(f"Missing implicit reference: {REFERENCE_SIM}")

    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

    mesh_files=sorted(
        MESH_DIR.glob("mesh_sphere_N*.json"),
        key=resolution
    )

    if not mesh_files:
        raise RuntimeError(f"No sphere sweep scenes found in {MESH_DIR}")

    reference=load_json(REFERENCE_SIM)
    t_ref,x_ref,y_ref=trajectory(reference)

    results=[]
    trajectories=[]

    for mesh_file in mesh_files:
        n=resolution(mesh_file)
        sim_file=run_mesh(mesh_file,n)
        sim=load_json(sim_file)
        t,x,y=trajectory(sim)

        xr=np.interp(t,t_ref,x_ref)
        yr=np.interp(t,t_ref,y_ref)

        error=np.sqrt((x-xr)**2+(y-yr)**2)

        maximum=float(np.max(error))
        rms=float(np.sqrt(np.mean(error**2)))

        results.append({
            "sphere_resolution":n,
            "maximum_trajectory_error":maximum,
            "rms_trajectory_error":rms
        })

        trajectories.append((n,x,y))

        print(f"N={n:3d}  max={maximum:.8f}  RMS={rms:.8f}")

    with open(BENCHMARK_FILE,"w") as f:
        json.dump({
            "fixed_funnel_points_per_side":FIXED_FUNNEL_POINTS,
            "reference":str(REFERENCE_SIM),
            "sphere_results":results
        },f,indent=2)

    spline=CubicSpline(
        [0.0,HEIGHT],
        [R_BOTTOM,R_TOP],
        bc_type=((1,S_BOTTOM),(1,S_TOP))
    )

    ys=np.linspace(0.0,HEIGHT,500)
    rs=spline(ys)

    fig,ax=plt.subplots(figsize=(8,8))

    ax.plot(
        x_ref,
        y_ref,
        linewidth=2.5,
        label="implicit"
    )

    for n,x,y in trajectories:
        ax.plot(
            x,
            y,
            linewidth=1.0,
            label=f"N={n}"
        )

    ax.plot(
        -rs,
        ys,
        "k-",
        linewidth=1.5
    )

    ax.plot(
        rs,
        ys,
        "k-",
        linewidth=1.5
    )

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True,alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(TRAJECTORY_PLOT,dpi=150)
    plt.close(fig)

    print(f"\nBenchmark: {BENCHMARK_FILE}")
    print(f"Trajectory plot: {TRAJECTORY_PLOT}")

if __name__=="__main__":
    main()