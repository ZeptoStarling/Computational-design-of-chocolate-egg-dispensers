#!/usr/bin/env python3
import copy
import json
import math
import subprocess
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

TEST_DIR=Path(__file__).resolve().parent
BUILD_DIR=Path("/home/austeja/Everything/EggDispenser/Implicit/rigid-ipc/build")
SOLVER=BUILD_DIR/"rigid_ipc_sim"
BASE_JSON=TEST_DIR/"test_json.json"
OUTPUT_DIR=TEST_DIR/"runs"
SUMMARY_FILE=TEST_DIR/"friction_comparison.json"

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
FRICTION_ITERATIONS=10
STATIC_FRICTION_SPEED_BOUND=0.001
RUN_SOLVER=True
MAKE_VIDEO=False
NUM_STEPS=5000
CHECKPOINT_FREQUENCY=10001
CONTACT_THRESHOLD=1e-2

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

def make_scene(base,name,mu):
    scene=copy.deepcopy(base)
    rb=scene["rigid_body_problem"]

    rb.pop("coefficient_friction",None)
    scene.pop("friction_constraints",None)

    if mu is not None:
        rb["coefficient_friction"]=mu
        scene["friction_constraints"]={
            "static_friction_speed_bound":STATIC_FRICTION_SPEED_BOUND,
            "iterations":FRICTION_ITERATIONS
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

    print("\nRUNNING:")
    print(" ".join(cmd),flush=True)

    subprocess.run(
        cmd,
        cwd=BUILD_DIR,
        check=True,
    )

    if not sim_file.exists():
        raise RuntimeError(f"Solver did not create {sim_file}")

    timing=parse_time_file(timing_file)

    print(
        f"TIMING: real={timing.get('real_seconds',float('nan')):.3f}s "
        f"user={timing.get('user_seconds',float('nan')):.3f}s "
        f"sys={timing.get('sys_seconds',float('nan')):.3f}s"
    )

    return sim_file,timing
def get_radius(body):
    if "radius" in body:
        return float(body["radius"])
    v=np.asarray(body["vertices"],dtype=float)
    return float(np.mean(np.linalg.norm(v,axis=1)))

def dynamic_indices(sim):
    bodies=sim["args"]["rigid_body_problem"]["rigid_bodies"]
    ids=[i for i,b in enumerate(bodies) if not all(b.get("is_dof_fixed",[False,False,False]))]
    if len(ids)!=2:
        raise RuntimeError(f"Expected two dynamic circles, found {len(ids)}")
    states=sim["animation"]["state_sequence"]
    ids.sort(key=lambda i:float(states[0]["rigid_bodies"][i]["position"][0]))
    return ids

def extract(sim):
    args=sim["args"]
    states=sim["animation"]["state_sequence"]
    ids=dynamic_indices(sim)
    dt=float(args.get("timestep",0.0005))
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
    radii=np.array([get_radius(args["rigid_body_problem"]["rigid_bodies"][i]) for i in ids])
    return {
        "args":args,"ids":ids,"t":t,"dt":dt,"n":n,
        "x":x,"y":y,"theta":theta,"vx":vx,"vy":vy,
        "omega":omega,"speed":np.hypot(vx,vy),"radii":radii,
    }

def make_funnel(args):
    f=args.get("spline_funnel",{
        "height":6.0,"r_bottom":1.5,"r_top":5.0,
        "s_bottom":0.0,"s_top":0.0
    })
    h=float(f["height"])
    spline=CubicSpline(
        [0.0,h],
        [float(f["r_bottom"]),float(f["r_top"])],
        bc_type=((1,float(f["s_bottom"])),(1,float(f["s_top"]))),
    )
    return f,h,spline

def project_funnel(x,y,h,spline):
    side=1.0 if x>=0.0 else -1.0
    def dist2(yy):
        dx=x-side*float(spline(np.clip(yy,0.0,h)))
        dy=y-yy
        return dx*dx+dy*dy
    r=minimize_scalar(dist2,bounds=(0.0,h),method="bounded",options={"xatol":1e-10})
    yy,d2=min([(float(r.x),float(r.fun)),(0.0,dist2(0.0)),(h,dist2(h))],key=lambda q:q[1])
    wall=np.array([side*float(spline(yy)),yy])
    c=np.array([x,y])
    d=math.sqrt(max(d2,0.0))
    n=(c-wall)/d if d>1e-12 else np.array([-side,0.0])
    return d,n,np.array([-n[1],n[0]])

def rot90(v):
    return np.array([-v[1],v[0]])

def first_contact(t,gap):
    ids=np.where(gap<=CONTACT_THRESHOLD)[0]
    return None if len(ids)==0 else float(t[ids[0]])

def abs_stats(values,mask):
    a=np.abs(values[mask])
    if len(a)==0:
        return {"samples":0,"mean_abs":None,"max_abs":None,"final_abs":None}
    return {
        "samples":int(len(a)),
        "mean_abs":float(np.mean(a)),
        "max_abs":float(np.max(a)),
        "final_abs":float(a[-1]),
    }

def analyse(sim,requested_mu):
    d=extract(sim)
    f,h,spline=make_funnel(d["args"])
    n=d["n"]
    gaps=np.zeros((n,2))
    wall_slip=np.zeros((n,2))
    for k in tqdm(range(n),desc=f"Wall projection μ={requested_mu}",ncols=90):
        for j in range(2):
            dist,norm,tan=project_funnel(d["x"][k,j],d["y"][k,j],h,spline)
            gaps[k,j]=dist-d["radii"][j]
            vc=np.array([d["vx"][k,j],d["vy"][k,j]])
            offset=-d["radii"][j]*norm
            vp=vc+d["omega"][k,j]*rot90(offset)
            wall_slip[k,j]=tan.dot(vp)

    dx=d["x"][:,0]-d["x"][:,1]
    dy=d["y"][:,0]-d["y"][:,1]
    pair_gap=np.hypot(dx,dy)-d["radii"].sum()
    pair_slip=np.zeros(n)

    for k in range(n):
        p1=np.array([d["x"][k,0],d["y"][k,0]])
        p2=np.array([d["x"][k,1],d["y"][k,1]])
        delta=p1-p2
        dist=np.linalg.norm(delta)
        norm=delta/dist if dist>1e-12 else np.array([1.0,0.0])
        tan=np.array([-norm[1],norm[0]])
        r1=-d["radii"][0]*norm
        r2=d["radii"][1]*norm
        v1=np.array([d["vx"][k,0],d["vy"][k,0]])+d["omega"][k,0]*rot90(r1)
        v2=np.array([d["vx"][k,1],d["vy"][k,1]])+d["omega"][k,1]*rot90(r2)
        pair_slip[k]=tan.dot(v1-v2)

    wall_mask=gaps<=CONTACT_THRESHOLD
    pair_mask=pair_gap<=CONTACT_THRESHOLD
    rotations=np.unwrap(d["theta"],axis=0)
    rotation_change=rotations[-1]-rotations[0]

    rb=d["args"].get("rigid_body_problem",{})
    reported_mu=rb.get("coefficient_friction",d["args"].get("coefficient_friction",None))
    friction_args=d["args"].get("friction_constraints",{})

    result={
        "requested_coefficient_friction":requested_mu,
        "reported_coefficient_friction":reported_mu,
        "simulation":{
            "timestep":d["dt"],
            "frames":n,
            "simulated_time":float(d["t"][-1]),
            "friction_iterations":friction_args.get("iterations",None),
            "static_friction_speed_bound":friction_args.get("static_friction_speed_bound",None),
        },
        "left":{
            "radius":float(d["radii"][0]),
            "initial_position":[float(d["x"][0,0]),float(d["y"][0,0])],
            "final_position":[float(d["x"][-1,0]),float(d["y"][-1,0])],
            "final_speed":float(d["speed"][-1,0]),
            "maximum_speed":float(np.max(d["speed"][:,0])),
            "minimum_wall_gap":float(np.min(gaps[:,0])),
            "maximum_penetration":float(max(0.0,-np.min(gaps[:,0]))),
            "first_wall_contact":first_contact(d["t"],gaps[:,0]),
            "maximum_absolute_angular_velocity":float(np.max(np.abs(d["omega"][:,0]))),
            "final_angular_velocity":float(d["omega"][-1,0]),
            "total_rotation_change":float(rotation_change[0]),
            "wall_tangential_speed":abs_stats(wall_slip[:,0],wall_mask[:,0]),
        },
        "right":{
            "radius":float(d["radii"][1]),
            "initial_position":[float(d["x"][0,1]),float(d["y"][0,1])],
            "final_position":[float(d["x"][-1,1]),float(d["y"][-1,1])],
            "final_speed":float(d["speed"][-1,1]),
            "maximum_speed":float(np.max(d["speed"][:,1])),
            "minimum_wall_gap":float(np.min(gaps[:,1])),
            "maximum_penetration":float(max(0.0,-np.min(gaps[:,1]))),
            "first_wall_contact":first_contact(d["t"],gaps[:,1]),
            "maximum_absolute_angular_velocity":float(np.max(np.abs(d["omega"][:,1]))),
            "final_angular_velocity":float(d["omega"][-1,1]),
            "total_rotation_change":float(rotation_change[1]),
            "wall_tangential_speed":abs_stats(wall_slip[:,1],wall_mask[:,1]),
        },
        "pair":{
            "first_contact":first_contact(d["t"],pair_gap),
            "minimum_gap":float(np.min(pair_gap)),
            "final_gap":float(pair_gap[-1]),
            "maximum_penetration":float(max(0.0,-np.min(pair_gap))),
            "relative_tangential_speed":abs_stats(pair_slip,pair_mask),
        },
    }
    return d,f,h,spline,gaps,wall_slip,pair_gap,pair_slip,result

def save_plots(case_dir,d,h,spline,gaps,wall_slip,pair_gap,pair_slip):
    ys=np.linspace(0.0,h,500)
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
    fig.savefig(case_dir/"trajectory.png",dpi=150)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,4.5))
    ax.plot(d["t"],gaps[:,0],label="left circle")
    ax.plot(d["t"],gaps[:,1],label="right circle")
    ax.axhline(0,ls="--",lw=1)
    ax.axhline(CONTACT_THRESHOLD,ls=":",lw=1)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("circle-wall gap [m]")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(case_dir/"wall_gap.png",dpi=150)
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
    fig.savefig(case_dir/"angular_velocity.png",dpi=150)
    plt.close(fig)

    wall_mask=gaps<=CONTACT_THRESHOLD
    fig,ax=plt.subplots(figsize=(9,4.5))
    ax.plot(d["t"],np.where(wall_mask[:,0],wall_slip[:,0],np.nan),label="left circle")
    ax.plot(d["t"],np.where(wall_mask[:,1],wall_slip[:,1],np.nan),label="right circle")
    ax.axhline(0,ls="--",lw=1)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("tangential contact speed [m/s]")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(case_dir/"wall_slip.png",dpi=150)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,4.5))
    ax.plot(d["t"],pair_gap)
    ax.axhline(0,ls="--",lw=1)
    ax.axhline(CONTACT_THRESHOLD,ls=":",lw=1)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("circle-circle gap [m]")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(case_dir/"pair_gap.png",dpi=150)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,4.5))
    ax.plot(d["t"],np.where(pair_gap<=CONTACT_THRESHOLD,pair_slip,np.nan))
    ax.axhline(0,ls="--",lw=1)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("relative tangential speed [m/s]")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(case_dir/"pair_slip.png",dpi=150)
    plt.close(fig)

def save_video(case_dir,d,h,spline):
    ys=np.linspace(0.0,h,500)
    rs=spline(ys)
    fig,ax=plt.subplots(figsize=(7,8))
    ax.plot(-rs,ys,"k-",lw=2)
    ax.plot(rs,ys,"k-",lw=2)
    all_x=d["x"].ravel()
    all_y=d["y"].ravel()
    margin=max(d["radii"])+1.0
    ax.set_xlim(float(np.min(all_x))-margin,float(np.max(all_x))+margin)
    ax.set_ylim(float(np.min(all_y))-margin,max(h+1.0,float(np.max(all_y))+margin))
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    balls=[
        Circle((d["x"][0,j],d["y"][0,j]),d["radii"][j],
               fc=("skyblue" if j==0 else "coral"),
               ec=("navy" if j==0 else "darkred"),lw=1.5)
        for j in range(2)
    ]
    for b in balls:
        ax.add_patch(b)

    trails=[ax.plot([],[],lw=1,alpha=0.5)[0] for _ in range(2)]
    rot_lines=[
        ax.plot([],[],lw=2,color=("navy" if j==0 else "darkred"))[0]
        for j in range(2)
    ]
    time_text=ax.text(0.02,0.97,"",transform=ax.transAxes,va="top")

    def update(k):
        for j in range(2):
            c=np.array([d["x"][k,j],d["y"][k,j]])
            balls[j].center=c
            trails[j].set_data(d["x"][:k+1,j],d["y"][:k+1,j])
            v=np.array([math.cos(d["theta"][k,j]),math.sin(d["theta"][k,j])])
            tip=c+0.9*d["radii"][j]*v
            rot_lines[j].set_data([c[0],tip[0]],[c[1],tip[1]])
        time_text.set_text(
            f"t = {d['t'][k]:.3f} s\n"
            f"ωL = {d['omega'][k,0]:.3f} rad/s\n"
            f"ωR = {d['omega'][k,1]:.3f} rad/s"
        )
        return (*balls,*trails,*rot_lines,time_text)

    anim=FuncAnimation(fig,update,frames=d["n"],interval=20,blit=False)
    pbar=tqdm(total=d["n"],desc=f"Saving {case_dir.name} MP4",unit="frame",ncols=100)
    last=0
    def progress(current,total):
        nonlocal last
        if current==total or current-last>=50:
            pbar.update(current-last)
            last=current
    anim.save(case_dir/"simulation.mp4",writer="ffmpeg",fps=60,progress_callback=progress)
    if pbar.n<d["n"]:
        pbar.update(d["n"]-pbar.n)
    pbar.close()
    plt.close(fig)

def print_summary(name,result):
    print(f"\n{name}")
    print(f"  requested mu       : {result['requested_coefficient_friction']}")
    print(f"  reported mu        : {result['reported_coefficient_friction']}")
    print(f"  left final         : {result['left']['final_position']}")
    print(f"  right final        : {result['right']['final_position']}")
    print(f"  left max |omega|   : {result['left']['maximum_absolute_angular_velocity']:.6e}")
    print(f"  right max |omega|  : {result['right']['maximum_absolute_angular_velocity']:.6e}")
    print(f"  left rotation      : {result['left']['total_rotation_change']:.6e}")
    print(f"  right rotation     : {result['right']['total_rotation_change']:.6e}")
    print(f"  left min wall gap  : {result['left']['minimum_wall_gap']:.6e}")
    print(f"  right min wall gap : {result['right']['minimum_wall_gap']:.6e}")
    print(f"  pair min gap       : {result['pair']['minimum_gap']:.6e}")
    print(f"  pair final gap     : {result['pair']['final_gap']:.6e}")
    timing=result.get("timing",{})
    print(f"  real time          : {timing.get('real_seconds',float('nan')):.3f} s")
    print(f"  user time          : {timing.get('user_seconds',float('nan')):.3f} s")
    print(f"  sys time           : {timing.get('sys_seconds',float('nan')):.3f} s")

def main():
    if not BASE_JSON.exists():
        raise RuntimeError(f"Missing {BASE_JSON}")
    if not SOLVER.exists():
        raise RuntimeError(f"Missing solver {SOLVER}")

    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    base=load_json(BASE_JSON)
    comparison={}

    for name,mu in SCENARIOS:
        scene_file,case_dir=make_scene(base,name,mu)
        sim_file,timing=run_solver(scene_file,case_dir)
        sim=load_json(sim_file)

        d,f,h,spline,gaps,wall_slip,pair_gap,pair_slip,result=analyse(sim,mu)

        result["timing"]=timing

        save_json(
            case_dir/"benchmark.json",
            result
        )

        save_plots(
            case_dir,
            d,
            h,
            spline,
            gaps,
            wall_slip,
            pair_gap,
            pair_slip
        )

        if MAKE_VIDEO:
            save_video(
                case_dir,
                d,
                h,
                spline
            )

        comparison[name]=result
        print_summary(name,result)

    save_json(
        SUMMARY_FILE,
        comparison
    )

    print(
        f"\nComparison saved to: {SUMMARY_FILE}"
    )

if __name__=="__main__":
    main()

if __name__=="__main__":
    main()