#!/usr/bin/env python3
import json
import math
import subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
from tqdm import tqdm

TEST_DIR=Path(__file__).resolve().parent
BUILD_DIR=Path("/home/austeja/Everything/EggDispenser/rigid-ipc/build")
SOLVER=BUILD_DIR/"rigid_ipc_sim"
OUTPUT_DIR=TEST_DIR/"mesh_runs"
SUMMARY_FILE=TEST_DIR/"mesh_friction_comparison.json"

SCENARIOS=[
    ("no_friction",None),
    ("mu_0",0.0),
    ("mu_005",0.05),
    ("mu_01",0.1),
    ("mu_02",0.2),
    ("mu_03",0.3),
    ("mu_04",0.4),
    ("mu_05",0.5),
]

FUNNEL_POINTS=128
CIRCLE_SEGMENTS=128
FRICTION_ITERATIONS=10
STATIC_FRICTION_SPEED_BOUND=0.001

RUN_SOLVER=True
MAKE_VIDEO=False
NUM_STEPS=5000
CHECKPOINT_FREQUENCY=10001
CONTACT_THRESHOLD=1e-2

HEIGHT=6.0
R_BOTTOM=1.5
R_TOP=5.0
S_BOTTOM=0.0
S_TOP=0.0
WALL_THICKNESS=0.2

LEFT_X=-1.4
LEFT_Y=5.6
RIGHT_X=1.0
RIGHT_Y=4.5
RADIUS=1.0

TIMESTEP=0.0005
MAX_TIME=10.0
RESTITUTION=0.2
GRAVITY=[0.0,-9.81]

IPC_SOLVER={
    "convergence_criteria":"velocity",
    "velocity_conv_tol":0.0001,
    "is_velocity_conv_tol_abs":True,
}

def load_json(path):
    with open(path) as f:
        return json.load(f)

def save_json(path,data):
    with open(path,"w") as f:
        json.dump(data,f,indent=2)

def parse_time_file(path):
    timing={}
    if not path.exists():
        return timing
    with open(path) as f:
        for line in f:
            parts=line.strip().split()
            if len(parts)!=2:
                continue
            if parts[0]=="real":
                timing["real_seconds"]=float(parts[1])
            elif parts[0]=="user":
                timing["user_seconds"]=float(parts[1])
            elif parts[0]=="sys":
                timing["sys_seconds"]=float(parts[1])
    return timing

def cubic_hermite(t,r0,r1,s0,s1):
    h00=2.0*t**3-3.0*t**2+1.0
    h10=t**3-2.0*t**2+t
    h01=-2.0*t**3+3.0*t**2
    h11=t**3-t**2
    return h00*r0+h10*s0+h01*r1+h11*s1

def funnel_radius(t):
    t=max(0.0,min(1.0,t))
    return max(cubic_hermite(t,R_BOTTOM,R_TOP,S_BOTTOM,S_TOP),R_BOTTOM)

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

def make_edges(n):
    return [[i,(i+1)%n] for i in range(n)]

def make_circle():
    return [
        [
            RADIUS*math.cos(2.0*math.pi*i/CIRCLE_SEGMENTS),
            RADIUS*math.sin(2.0*math.pi*i/CIRCLE_SEGMENTS),
        ]
        for i in range(CIRCLE_SEGMENTS)
    ]

def make_body(vertices,position,fixed):
    return {
        "vertices":vertices,
        "polygons":[vertices],
        "edges":make_edges(len(vertices)),
        "oriented":True,
        "position":[position[0],position[1]],
        "rotation":[0.0],
        "linear_velocity":[0.0,0.0],
        "angular_velocity":[0.0],
        "is_dof_fixed":[fixed,fixed,fixed],
    }

def make_scene(name,mu):
    left_wall,right_wall=make_funnel_walls()
    circle=make_circle()

    bodies=[
        make_body(left_wall,(0.0,0.0),True),
        make_body(right_wall,(0.0,0.0),True),
        make_body(circle,(LEFT_X,LEFT_Y),False),
        make_body(circle,(RIGHT_X,RIGHT_Y),False),
    ]

    scene={
        "scene_type":"distance_barrier_rb_problem",
        "solver":"ipc_solver",
        "timestep":TIMESTEP,
        "max_time":MAX_TIME,
        "rigid_body_problem":{
            "coefficient_restitution":RESTITUTION,
            "gravity":GRAVITY,
            "rigid_bodies":bodies,
        },
        "ipc_solver":IPC_SOLVER,
    }

    if mu is not None:
        scene["rigid_body_problem"]["coefficient_friction"]=mu
        scene["friction_constraints"]={
            "static_friction_speed_bound":STATIC_FRICTION_SPEED_BOUND,
            "iterations":FRICTION_ITERATIONS,
        }

    case_dir=OUTPUT_DIR/name
    case_dir.mkdir(parents=True,exist_ok=True)
    scene_file=case_dir/"input.json"
    save_json(scene_file,scene)
    return scene_file,case_dir

def run_solver(scene_file,case_dir):
    sim_file=case_dir/"sim.json"
    timing_file=case_dir/"timing.txt"

    if not RUN_SOLVER:
        if not sim_file.exists():
            raise RuntimeError(f"Missing {sim_file}")
        return sim_file,parse_time_file(timing_file)

    cmd=[
        "time",
        "-p",
        "-o",str(timing_file),
        str(SOLVER),
        "--ngui",
        "--num-steps",str(NUM_STEPS),
        "--output-path",str(case_dir),
        str(scene_file),
        "--checkpoint-frequency",str(CHECKPOINT_FREQUENCY),
    ]

    print("\nRUNNING MESH:")
    print(" ".join(cmd),flush=True)

    subprocess.run(cmd,cwd=BUILD_DIR,check=True)

    if not sim_file.exists():
        raise RuntimeError(f"Solver did not create {sim_file}")

    timing=parse_time_file(timing_file)

    print(
        f"MESH TIMING: real={timing.get('real_seconds',float('nan')):.3f}s "
        f"user={timing.get('user_seconds',float('nan')):.3f}s "
        f"sys={timing.get('sys_seconds',float('nan')):.3f}s"
    )

    return sim_file,timing

def get_radius(body):
    v=np.asarray(body["vertices"],dtype=float)
    return float(np.mean(np.linalg.norm(v,axis=1)))

def dynamic_indices(sim):
    bodies=sim["args"]["rigid_body_problem"]["rigid_bodies"]
    ids=[
        i for i,b in enumerate(bodies)
        if not all(b.get("is_dof_fixed",[False,False,False]))
    ]
    if len(ids)!=2:
        raise RuntimeError(f"Expected two dynamic mesh circles, found {len(ids)}")
    states=sim["animation"]["state_sequence"]
    ids.sort(key=lambda i:float(states[0]["rigid_bodies"][i]["position"][0]))
    return ids

def extract(sim):
    args=sim["args"]
    states=sim["animation"]["state_sequence"]
    ids=dynamic_indices(sim)
    dt=float(args.get("timestep",TIMESTEP))
    n=len(states)
    t=np.arange(n,dtype=float)*dt

    x=np.zeros((n,2))
    y=np.zeros((n,2))
    theta=np.zeros((n,2))
    vx=np.full((n,2),np.nan)
    vy=np.full((n,2),np.nan)
    omega=np.full((n,2),np.nan)

    for k,state in enumerate(states):
        for j,body_id in enumerate(ids):
            b=state["rigid_bodies"][body_id]
            x[k,j]=float(b["position"][0])
            y[k,j]=float(b["position"][1])
            theta[k,j]=float(b["rotation"][0])
            if "linear_velocity" in b:
                vx[k,j]=float(b["linear_velocity"][0])
                vy[k,j]=float(b["linear_velocity"][1])
            if "angular_velocity" in b:
                omega[k,j]=float(b["angular_velocity"][0])

    for j in range(2):
        if not np.all(np.isfinite(vx[:,j])):
            vx[:,j]=np.gradient(x[:,j],dt)
            vy[:,j]=np.gradient(y[:,j],dt)
        if not np.all(np.isfinite(omega[:,j])):
            omega[:,j]=np.gradient(np.unwrap(theta[:,j]),dt)

    radii=np.array([
        get_radius(args["rigid_body_problem"]["rigid_bodies"][i])
        for i in ids
    ])

    return {
        "args":args,
        "ids":ids,
        "t":t,
        "dt":dt,
        "n":n,
        "x":x,
        "y":y,
        "theta":theta,
        "vx":vx,
        "vy":vy,
        "omega":omega,
        "speed":np.hypot(vx,vy),
        "radii":radii,
    }

def make_spline():
    spline=CubicSpline(
        [0.0,HEIGHT],
        [R_BOTTOM,R_TOP],
        bc_type=((1,S_BOTTOM),(1,S_TOP)),
    )
    return spline

def project_funnel(x,y,spline):
    side=1.0 if x>=0.0 else -1.0

    def dist2(yy):
        r=float(spline(np.clip(yy,0.0,HEIGHT)))
        dx=x-side*r
        dy=y-yy
        return dx*dx+dy*dy

    r=minimize_scalar(
        dist2,
        bounds=(0.0,HEIGHT),
        method="bounded",
        options={"xatol":1e-10},
    )

    yy,d2=min(
        [
            (float(r.x),float(r.fun)),
            (0.0,dist2(0.0)),
            (HEIGHT,dist2(HEIGHT)),
        ],
        key=lambda q:q[1],
    )

    wall=np.array([side*float(spline(yy)),yy])
    c=np.array([x,y])
    d=math.sqrt(max(d2,0.0))
    normal=(c-wall)/d if d>1e-12 else np.array([-side,0.0])
    tangent=np.array([-normal[1],normal[0]])

    return d,normal,tangent

def rot90(v):
    return np.array([-v[1],v[0]])

def first_contact(t,gap):
    ids=np.where(gap<=CONTACT_THRESHOLD)[0]
    return None if len(ids)==0 else float(t[ids[0]])

def abs_stats(values,mask):
    a=np.abs(values[mask])
    if len(a)==0:
        return {
            "samples":0,
            "mean_abs":None,
            "max_abs":None,
            "final_abs":None,
        }
    return {
        "samples":int(len(a)),
        "mean_abs":float(np.mean(a)),
        "max_abs":float(np.max(a)),
        "final_abs":float(a[-1]),
    }

def analyse(sim,requested_mu):
    d=extract(sim)
    spline=make_spline()
    n=d["n"]

    gaps=np.zeros((n,2))
    wall_slip=np.zeros((n,2))

    for k in tqdm(
        range(n),
        desc=f"Mesh wall projection mu={requested_mu}",
        ncols=90,
    ):
        for j in range(2):
            dist,norm,tan=project_funnel(
                d["x"][k,j],
                d["y"][k,j],
                spline,
            )

            gaps[k,j]=dist-d["radii"][j]

            center_velocity=np.array([
                d["vx"][k,j],
                d["vy"][k,j],
            ])

            offset=-d["radii"][j]*norm

            contact_velocity=(
                center_velocity
                +d["omega"][k,j]*rot90(offset)
            )

            wall_slip[k,j]=tan.dot(contact_velocity)

    dx=d["x"][:,0]-d["x"][:,1]
    dy=d["y"][:,0]-d["y"][:,1]

    pair_gap=np.hypot(dx,dy)-d["radii"].sum()
    pair_slip=np.zeros(n)

    for k in range(n):
        p1=np.array([d["x"][k,0],d["y"][k,0]])
        p2=np.array([d["x"][k,1],d["y"][k,1]])

        delta=p1-p2
        dist=np.linalg.norm(delta)

        norm=(
            delta/dist
            if dist>1e-12
            else np.array([1.0,0.0])
        )

        tangent=np.array([-norm[1],norm[0]])

        r1=-d["radii"][0]*norm
        r2=d["radii"][1]*norm

        v1=(
            np.array([d["vx"][k,0],d["vy"][k,0]])
            +d["omega"][k,0]*rot90(r1)
        )

        v2=(
            np.array([d["vx"][k,1],d["vy"][k,1]])
            +d["omega"][k,1]*rot90(r2)
        )

        pair_slip[k]=tangent.dot(v1-v2)

    wall_mask=gaps<=CONTACT_THRESHOLD
    pair_mask=pair_gap<=CONTACT_THRESHOLD

    rotations=np.unwrap(d["theta"],axis=0)
    rotation_change=rotations[-1]-rotations[0]

    rb=d["args"].get("rigid_body_problem",{})
    reported_mu=rb.get(
        "coefficient_friction",
        d["args"].get("coefficient_friction",None),
    )

    friction_args=d["args"].get("friction_constraints",{})

    result={
        "representation":"mesh",
        "funnel_points_per_side":FUNNEL_POINTS,
        "circle_segments":CIRCLE_SEGMENTS,
        "requested_coefficient_friction":requested_mu,
        "reported_coefficient_friction":reported_mu,
        "simulation":{
            "timestep":d["dt"],
            "frames":n,
            "simulated_time":float(d["t"][-1]),
            "friction_iterations":friction_args.get("iterations",None),
            "static_friction_speed_bound":
                friction_args.get("static_friction_speed_bound",None),
        },
        "left":{
            "radius":float(d["radii"][0]),
            "initial_position":[
                float(d["x"][0,0]),
                float(d["y"][0,0]),
            ],
            "final_position":[
                float(d["x"][-1,0]),
                float(d["y"][-1,0]),
            ],
            "final_speed":float(d["speed"][-1,0]),
            "maximum_speed":float(np.max(d["speed"][:,0])),
            "minimum_smooth_reference_wall_gap":
                float(np.min(gaps[:,0])),
            "first_smooth_reference_wall_contact":
                first_contact(d["t"],gaps[:,0]),
            "maximum_absolute_angular_velocity":
                float(np.max(np.abs(d["omega"][:,0]))),
            "final_angular_velocity":
                float(d["omega"][-1,0]),
            "total_rotation_change":
                float(rotation_change[0]),
            "wall_tangential_speed":
                abs_stats(wall_slip[:,0],wall_mask[:,0]),
        },
        "right":{
            "radius":float(d["radii"][1]),
            "initial_position":[
                float(d["x"][0,1]),
                float(d["y"][0,1]),
            ],
            "final_position":[
                float(d["x"][-1,1]),
                float(d["y"][-1,1]),
            ],
            "final_speed":float(d["speed"][-1,1]),
            "maximum_speed":float(np.max(d["speed"][:,1])),
            "minimum_smooth_reference_wall_gap":
                float(np.min(gaps[:,1])),
            "first_smooth_reference_wall_contact":
                first_contact(d["t"],gaps[:,1]),
            "maximum_absolute_angular_velocity":
                float(np.max(np.abs(d["omega"][:,1]))),
            "final_angular_velocity":
                float(d["omega"][-1,1]),
            "total_rotation_change":
                float(rotation_change[1]),
            "wall_tangential_speed":
                abs_stats(wall_slip[:,1],wall_mask[:,1]),
        },
        "pair":{
            "first_ideal_circle_contact":
                first_contact(d["t"],pair_gap),
            "minimum_ideal_circle_gap":
                float(np.min(pair_gap)),
            "final_ideal_circle_gap":
                float(pair_gap[-1]),
            "relative_tangential_speed":
                abs_stats(pair_slip,pair_mask),
        },
    }

    return d,spline,gaps,wall_slip,pair_gap,pair_slip,result

def save_plots(case_dir,d,spline,gaps,wall_slip,pair_gap,pair_slip):
    ys=np.linspace(0.0,HEIGHT,500)
    rs=spline(ys)

    fig,ax=plt.subplots(figsize=(8,9))
    ax.plot(-rs,ys,"k-",lw=2)
    ax.plot(rs,ys,"k-",lw=2)
    ax.plot(d["x"][:,0],d["y"][:,0],label="left circle")
    ax.plot(d["x"][:,1],d["y"][:,1],label="right circle")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(case_dir/"mesh_trajectory.png",dpi=150)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,4.5))
    ax.plot(d["t"],gaps[:,0],label="left circle")
    ax.plot(d["t"],gaps[:,1],label="right circle")
    ax.axhline(0,ls="--",lw=1)
    ax.axhline(CONTACT_THRESHOLD,ls=":",lw=1)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("smooth-reference wall gap [m]")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(case_dir/"mesh_wall_gap.png",dpi=150)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,4.5))
    ax.plot(d["t"],d["omega"][:,0],label="left circle")
    ax.plot(d["t"],d["omega"][:,1],label="right circle")
    ax.axhline(0,ls="--",lw=1)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("angular velocity [rad/s]")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(case_dir/"mesh_angular_velocity.png",dpi=150)
    plt.close(fig)

    wall_mask=gaps<=CONTACT_THRESHOLD

    fig,ax=plt.subplots(figsize=(9,4.5))
    ax.plot(
        d["t"],
        np.where(wall_mask[:,0],wall_slip[:,0],np.nan),
        label="left circle",
    )
    ax.plot(
        d["t"],
        np.where(wall_mask[:,1],wall_slip[:,1],np.nan),
        label="right circle",
    )
    ax.axhline(0,ls="--",lw=1)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("tangential contact speed [m/s]")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(case_dir/"mesh_wall_slip.png",dpi=150)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,4.5))
    ax.plot(d["t"],pair_gap)
    ax.axhline(0,ls="--",lw=1)
    ax.axhline(CONTACT_THRESHOLD,ls=":",lw=1)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("ideal-circle gap [m]")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(case_dir/"mesh_pair_gap.png",dpi=150)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,4.5))
    ax.plot(
        d["t"],
        np.where(
            pair_gap<=CONTACT_THRESHOLD,
            pair_slip,
            np.nan,
        ),
    )
    ax.axhline(0,ls="--",lw=1)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("relative tangential speed [m/s]")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(case_dir/"mesh_pair_slip.png",dpi=150)
    plt.close(fig)

def transform(vertices,position,angle):
    c=math.cos(angle)
    s=math.sin(angle)
    R=np.array([[c,-s],[s,c]])
    return vertices@R.T+np.asarray(position)

def save_video(case_dir,d):
    scene=load_json(case_dir/"input.json")
    bodies=scene["rigid_body_problem"]["rigid_bodies"]

    left_wall=np.asarray(bodies[0]["vertices"],dtype=float)
    right_wall=np.asarray(bodies[1]["vertices"],dtype=float)
    circle_vertices=[
        np.asarray(
            bodies[d["ids"][j]]["vertices"],
            dtype=float,
        )
        for j in range(2)
    ]

    all_x=d["x"].ravel()
    all_y=d["y"].ravel()

    fig,ax=plt.subplots(figsize=(7,8))
    ax.set_aspect("equal")
    ax.set_xlim(np.min(all_x)-2.0,np.max(all_x)+2.0)
    ax.set_ylim(np.min(all_y)-2.0,max(HEIGHT+1.0,np.max(all_y)+2.0))
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.add_patch(
        Polygon(
            left_wall,
            closed=True,
            fill=False,
            edgecolor="black",
        )
    )
    ax.add_patch(
        Polygon(
            right_wall,
            closed=True,
            fill=False,
            edgecolor="black",
        )
    )

    patches=[
        Polygon(
            transform(
                circle_vertices[j],
                [d["x"][0,j],d["y"][0,j]],
                d["theta"][0,j],
            ),
            closed=True,
        )
        for j in range(2)
    ]

    for p in patches:
        ax.add_patch(p)

    trails=[
        ax.plot([],[],lw=1,alpha=0.5)[0]
        for _ in range(2)
    ]

    time_text=ax.text(
        0.02,
        0.97,
        "",
        transform=ax.transAxes,
        va="top",
    )

    def update(k):
        for j in range(2):
            verts=transform(
                circle_vertices[j],
                [d["x"][k,j],d["y"][k,j]],
                d["theta"][k,j],
            )
            patches[j].set_xy(verts)
            trails[j].set_data(
                d["x"][:k+1,j],
                d["y"][:k+1,j],
            )

        time_text.set_text(
            f"t = {d['t'][k]:.3f} s\n"
            f"ωL = {d['omega'][k,0]:.3f} rad/s\n"
            f"ωR = {d['omega'][k,1]:.3f} rad/s"
        )

        return (*patches,*trails,time_text)

    anim=FuncAnimation(
        fig,
        update,
        frames=d["n"],
        interval=20,
        blit=False,
    )

    pbar=tqdm(
        total=d["n"],
        desc=f"Saving MESH {case_dir.name} MP4",
        unit="frame",
        ncols=100,
    )

    last=0

    def progress(current,total):
        nonlocal last
        if current==total or current-last>=50:
            pbar.update(current-last)
            last=current

    anim.save(
        case_dir/"mesh_simulation.mp4",
        writer="ffmpeg",
        fps=60,
        progress_callback=progress,
    )

    if pbar.n<d["n"]:
        pbar.update(d["n"]-pbar.n)

    pbar.close()
    plt.close(fig)

def print_summary(name,result):
    print(f"\nMESH CASE: {name}")
    print(f"  requested mu       : {result['requested_coefficient_friction']}")
    print(f"  reported mu        : {result['reported_coefficient_friction']}")
    print(f"  left final         : {result['left']['final_position']}")
    print(f"  right final        : {result['right']['final_position']}")
    print(f"  left max |omega|   : {result['left']['maximum_absolute_angular_velocity']:.6e}")
    print(f"  right max |omega|  : {result['right']['maximum_absolute_angular_velocity']:.6e}")
    print(f"  left rotation      : {result['left']['total_rotation_change']:.6e}")
    print(f"  right rotation     : {result['right']['total_rotation_change']:.6e}")
    print(f"  left ref wall gap  : {result['left']['minimum_smooth_reference_wall_gap']:.6e}")
    print(f"  right ref wall gap : {result['right']['minimum_smooth_reference_wall_gap']:.6e}")
    print(f"  pair ref min gap   : {result['pair']['minimum_ideal_circle_gap']:.6e}")
    print(f"  pair ref final gap : {result['pair']['final_ideal_circle_gap']:.6e}")

    timing=result.get("timing",{})

    print(f"  real time          : {timing.get('real_seconds',float('nan')):.3f} s")
    print(f"  user time          : {timing.get('user_seconds',float('nan')):.3f} s")
    print(f"  sys time           : {timing.get('sys_seconds',float('nan')):.3f} s")

def main():
    if not SOLVER.exists():
        raise RuntimeError(f"Missing solver {SOLVER}")

    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    comparison={}

    for name,mu in SCENARIOS:
        scene_file,case_dir=make_scene(name,mu)

        print(f"\nMESH CASE: {name}")
        print(f"mesh resolution = N{FUNNEL_POINTS}/N{CIRCLE_SEGMENTS}")
        print(
            "coefficient_friction = "
            f"{mu if mu is not None else 'not specified'}"
        )
        print(
            "friction iterations = "
            f"{0 if mu is None else FRICTION_ITERATIONS}"
        )

        sim_file,timing=run_solver(scene_file,case_dir)
        sim=load_json(sim_file)

        d,spline,gaps,wall_slip,pair_gap,pair_slip,result=analyse(
            sim,
            mu,
        )

        result["timing"]=timing

        save_json(
            case_dir/"mesh_benchmark.json",
            result,
        )

        save_plots(
            case_dir,
            d,
            spline,
            gaps,
            wall_slip,
            pair_gap,
            pair_slip,
        )

        if MAKE_VIDEO:
            save_video(case_dir,d)

        comparison[name]=result
        print_summary(name,result)

    save_json(SUMMARY_FILE,comparison)

    print(f"\nMESH comparison saved to:")
    print(f"  {SUMMARY_FILE}")

if __name__=="__main__":
    main()