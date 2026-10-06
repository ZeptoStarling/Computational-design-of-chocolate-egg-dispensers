#!/usr/bin/env python3
import json
import math
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

TEST_DIR=Path("/home/austeja/Everything/EggDispenser/Implicit/testsforimplicit/testsingle")
SIM_FILE=TEST_DIR/"output"/"sim.json"
TRAJECTORY_FILE=TEST_DIR/"sphere.png"
VIDEO_FILE=TEST_DIR/"single_implicit_sim.mp4"
BENCHMARK_FILE=TEST_DIR/"single_implicit_benchmark.json"
CONTACT_THRESHOLD=1e-2

def load_json(path):
    with open(path,"r") as f:
        return json.load(f)

def find_dynamic_body_index(sim_data):
    bodies=sim_data["args"]["rigid_body_problem"]["rigid_bodies"]
    dynamic=[]
    for i,body in enumerate(bodies):
        if not all(body.get("is_dof_fixed",[False,False,False])):
            dynamic.append(i)
    if len(dynamic)!=1:
        raise RuntimeError(f"Expected exactly one dynamic circle, found {len(dynamic)}.")
    return dynamic[0]

def get_radius(body):
    if "radius" in body:
        return float(body["radius"])
    vertices=np.asarray(body["vertices"],dtype=float)
    return float(np.mean(np.linalg.norm(vertices,axis=1)))

def extract_trajectory(sim_data,body_id):
    states=sim_data["animation"]["state_sequence"]
    dt=float(sim_data["args"].get("timestep",0.0005))
    frames=len(states)
    x=np.zeros(frames)
    y=np.zeros(frames)
    vx=np.zeros(frames)
    vy=np.zeros(frames)
    have_velocity=True
    for i,state in enumerate(states):
        body=state["rigid_bodies"][body_id]
        x[i]=float(body["position"][0])
        y[i]=float(body["position"][1])
        if "linear_velocity" in body:
            vx[i]=float(body["linear_velocity"][0])
            vy[i]=float(body["linear_velocity"][1])
        else:
            have_velocity=False
    if not have_velocity:
        vx=np.gradient(x,dt)
        vy=np.gradient(y,dt)
    speed=np.sqrt(vx*vx+vy*vy)
    return {
        "times":np.arange(frames,dtype=float)*dt,
        "x":x,
        "y":y,
        "vx":vx,
        "vy":vy,
        "speed":speed,
        "dt":dt,
        "frames":frames,
    }

def get_funnel(sim_data):
    return sim_data["args"].get(
        "spline_funnel",
        {
            "height":6.0,
            "r_bottom":1.5,
            "r_top":5.0,
            "s_bottom":0.0,
            "s_top":0.0,
        },
    )

def make_spline(funnel):
    return CubicSpline(
        [0.0,float(funnel["height"])],
        [float(funnel["r_bottom"]),float(funnel["r_top"])],
        bc_type=(
            (1,float(funnel["s_bottom"])),
            (1,float(funnel["s_top"])),
        ),
    )

def analytic_wall_distance(x,y,funnel,spline):
    h=float(funnel["height"])
    def objective(yy):
        r=float(spline(np.clip(yy,0.0,h)))
        dx=x-r if x>=0.0 else x+r
        dy=y-yy
        return dx*dx+dy*dy
    result=minimize_scalar(
        objective,
        bounds=(0.0,h),
        method="bounded",
        options={"xatol":1e-10},
    )
    return math.sqrt(min(result.fun,objective(0.0),objective(h)))

def compute_wall_gap(trajectory,radius,funnel,spline):
    gaps=np.zeros(trajectory["frames"])
    for i in tqdm(range(trajectory["frames"]),desc="Wall gap",ncols=100):
        gaps[i]=analytic_wall_distance(
            trajectory["x"][i],
            trajectory["y"][i],
            funnel,
            spline,
        )-radius
    return gaps

def first_contact_time(times,gap):
    indices=np.where(gap<=CONTACT_THRESHOLD)[0]
    return None if len(indices)==0 else float(times[indices[0]])

def save_trajectory(trajectory,radius,funnel,spline):
    h=float(funnel["height"])
    y_wall=np.linspace(0.0,h,500)
    r_wall=np.asarray([float(spline(y)) for y in y_wall])
    fig,ax=plt.subplots(figsize=(7,8))
    ax.plot(-r_wall,y_wall,"k-",linewidth=2.0)
    ax.plot(r_wall,y_wall,"k-",linewidth=2.0)
    ax.plot(trajectory["x"],trajectory["y"],linewidth=2.0,label="implicit trajectory")
    ax.add_patch(Circle(
        (trajectory["x"][0],trajectory["y"][0]),
        radius,
        fill=False,
        linewidth=1.2,
    ))
    margin=radius+0.75
    ax.set_xlim(
        min(-6.0,float(np.min(trajectory["x"]))-margin),
        max(6.0,float(np.max(trajectory["x"]))+margin),
    )
    ax.set_ylim(
        min(-1.0,float(np.min(trajectory["y"]))-margin),
        max(h+1.0,float(np.max(trajectory["y"]))+margin),
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Implicit Circle Trajectory")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(TRAJECTORY_FILE,dpi=150)
    plt.close(fig)

def save_animation(trajectory,radius,funnel,spline):
    h=float(funnel["height"])
    y_wall=np.linspace(0.0,h,500)
    r_wall=np.asarray([float(spline(y)) for y in y_wall])
    fig,ax=plt.subplots(figsize=(7,8))
    ax.plot(-r_wall,y_wall,"k-",linewidth=2.0)
    ax.plot(r_wall,y_wall,"k-",linewidth=2.0)
    margin=radius+0.75
    ax.set_xlim(
        min(-6.0,float(np.min(trajectory["x"]))-margin),
        max(6.0,float(np.max(trajectory["x"]))+margin),
    )
    ax.set_ylim(
        min(-1.0,float(np.min(trajectory["y"]))-margin),
        max(h+1.0,float(np.max(trajectory["y"]))+margin),
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Implicit Circle in Spline Funnel")
    ax.grid(alpha=0.3)
    trail,=ax.plot([],[],linewidth=1.5)
    circle=Circle(
        (trajectory["x"][0],trajectory["y"][0]),
        radius,
        facecolor="C0",
        edgecolor="black",
        linewidth=1.2,
    )
    ax.add_patch(circle)

    def update(frame):
        trail.set_data(
            trajectory["x"][:frame+1],
            trajectory["y"][:frame+1],
        )
        circle.center=(
            trajectory["x"][frame],
            trajectory["y"][frame],
        )
        return trail,circle

    animation=FuncAnimation(
        fig,
        update,
        frames=trajectory["frames"],
        interval=20,
        blit=False,
    )
    pbar=tqdm(
        total=trajectory["frames"],
        desc="Saving MP4",
        unit="frame",
        ncols=100,
    )
    last_frame=0
    def progress(current,total):
        nonlocal last_frame
        if current==total or current-last_frame>=50:
            pbar.update(current-last_frame)
            last_frame=current
    animation.save(
        VIDEO_FILE,
        writer="ffmpeg",
        fps=60,
        progress_callback=progress,
    )
    if pbar.n<trajectory["frames"]:
        pbar.update(trajectory["frames"]-pbar.n)
    pbar.close()
    plt.close(fig)

def main():
    if not SIM_FILE.exists():
        raise RuntimeError(f"Simulation does not exist:\n  {SIM_FILE}")
    sim_data=load_json(SIM_FILE)
    body_id=find_dynamic_body_index(sim_data)
    body=sim_data["args"]["rigid_body_problem"]["rigid_bodies"][body_id]
    radius=get_radius(body)
    trajectory=extract_trajectory(sim_data,body_id)
    funnel=get_funnel(sim_data)
    spline=make_spline(funnel)
    wall_gap=compute_wall_gap(trajectory,radius,funnel,spline)
    contact_time=first_contact_time(trajectory["times"],wall_gap)
    minimum_gap=float(np.min(wall_gap))
    maximum_penetration=float(max(0.0,-minimum_gap))

    benchmark={
        "test":"single_sphere_frictionless_implicit",
        "simulation":{
            "timestep":trajectory["dt"],
            "frames":trajectory["frames"],
            "simulated_time":float(trajectory["times"][-1]),
        },
        "circle":{
            "body_index":body_id,
            "radius":radius,
            "initial_position":[
                float(trajectory["x"][0]),
                float(trajectory["y"][0]),
            ],
            "initial_velocity":[
                float(trajectory["vx"][0]),
                float(trajectory["vy"][0]),
            ],
            "final_position":[
                float(trajectory["x"][-1]),
                float(trajectory["y"][-1]),
            ],
            "final_velocity":[
                float(trajectory["vx"][-1]),
                float(trajectory["vy"][-1]),
            ],
            "maximum_speed":float(np.max(trajectory["speed"])),
            "final_speed":float(trajectory["speed"][-1]),
        },
        "wall_contact":{
            "first_contact_time":contact_time,
            "minimum_gap":minimum_gap,
            "maximum_penetration":maximum_penetration,
        },
    }

    with open(BENCHMARK_FILE,"w") as f:
        json.dump(benchmark,f,indent=2)

    save_trajectory(trajectory,radius,funnel,spline)
    save_animation(trajectory,radius,funnel,spline)

    print()
    print("========================================")
    print("SINGLE-SPHERE IMPLICIT TEST")
    print("========================================")
    print(f"Timestep              : {trajectory['dt']:.8f}")
    print(f"Frames                : {trajectory['frames']}")
    print(f"Simulated time        : {trajectory['times'][-1]:.6f} s")
    print(f"Radius                : {radius:.8f}")
    print(f"Initial position      : ({trajectory['x'][0]:.8f}, {trajectory['y'][0]:.8f})")
    print(f"Initial velocity      : ({trajectory['vx'][0]:.8f}, {trajectory['vy'][0]:.8f})")
    print(f"Final position        : ({trajectory['x'][-1]:.8f}, {trajectory['y'][-1]:.8f})")
    print(f"Final velocity        : ({trajectory['vx'][-1]:.8f}, {trajectory['vy'][-1]:.8f})")
    print(f"Maximum speed         : {np.max(trajectory['speed']):.8f}")
    print(f"Final speed           : {trajectory['speed'][-1]:.8f}")
    print(f"First wall contact    : {contact_time}")
    print(f"Minimum wall gap      : {minimum_gap:.12e}")
    print(f"Maximum penetration   : {maximum_penetration:.12e}")
    print(f"Benchmark             : {BENCHMARK_FILE}")
    print(f"Trajectory            : {TRAJECTORY_FILE}")
    print(f"Animation             : {VIDEO_FILE}")

if __name__=="__main__":
    main()