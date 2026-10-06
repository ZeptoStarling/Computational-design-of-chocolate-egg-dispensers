#!/usr/bin/env python3
import json,math,shutil,subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEST_DIR=Path(__file__).resolve().parent
BUILD_DIR=Path("/home/austeja/Everything/EggDispenser/rigid-ipc/build")
SOLVER=BUILD_DIR/"rigid_ipc_sim"
OUTPUT_DIR=TEST_DIR/"inclined_friction_mesh_runs"
SUMMARY_FILE=TEST_DIR/"inclined_friction_mesh_comparison.json"

SCENARIOS=[("mu_01",0.10),("mu_03",0.30)]
CIRCLE_SEGMENTS=128
FRICTION_ITERATIONS=10
STATIC_FRICTION_SPEED_BOUND=0.001
NUM_STEPS=3000
CHECKPOINT_FREQUENCY=10001
RUN_SOLVER=True

ANGLE_DEG=30.0
ANGLE=math.radians(ANGLE_DEG)
GRAVITY=9.81
RADIUS=1.0
R_BOTTOM=1.5
HEIGHT=6.0
WALL_THICKNESS=0.2
TIMESTEP=0.0005
MAX_TIME=2.0
RESTITUTION=0.0
CONTACT_THRESHOLD=1e-2
FIT_DELAY=0.10
FIT_DURATION=1.00

SLOPE=1.0/math.tan(ANGLE)
R_TOP=R_BOTTOM+SLOPE*HEIGHT
MU_CRITICAL=(1.0/3.0)*math.tan(ANGLE)
DOWNHILL=np.array([-math.cos(ANGLE),-math.sin(ANGLE)])
INWARD_NORMAL=np.array([-math.sin(ANGLE),math.cos(ANGLE)])
WALL_ORIGIN=np.array([R_BOTTOM,0.0])

START_S=8.0
INITIAL_GAP=0.002
WALL_POINT=WALL_ORIGIN+START_S*np.array([math.cos(ANGLE),math.sin(ANGLE)])
INITIAL_CENTER=WALL_POINT+(RADIUS+INITIAL_GAP)*INWARD_NORMAL

IPC_SOLVER={
    "convergence_criteria":"velocity",
    "velocity_conv_tol":0.0001,
    "is_velocity_conv_tol_abs":True,
}

def load_json(path):
    with open(path) as f:return json.load(f)

def save_json(path,data):
    with open(path,"w") as f:json.dump(data,f,indent=2)

def make_edges(n):
    return [[i,(i+1)%n] for i in range(n)]

def make_circle():
    return [[RADIUS*math.cos(2*math.pi*i/CIRCLE_SEGMENTS),RADIUS*math.sin(2*math.pi*i/CIRCLE_SEGMENTS)] for i in range(CIRCLE_SEGMENTS)]

def make_wall():
    p0=np.array([R_BOTTOM,0.0])
    p1=np.array([R_TOP,HEIGHT])
    outward=-INWARD_NORMAL
    q0=p0+WALL_THICKNESS*outward
    q1=p1+WALL_THICKNESS*outward
    return [p0.tolist(),p1.tolist(),q1.tolist(),q0.tolist()]

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

def make_scene(name,mu):
    scene={
        "scene_type":"distance_barrier_rb_problem",
        "solver":"ipc_solver",
        "timestep":TIMESTEP,
        "max_time":MAX_TIME,
        "rigid_body_problem":{
            "coefficient_restitution":RESTITUTION,
            "coefficient_friction":mu,
            "gravity":[0.0,-GRAVITY],
            "rigid_bodies":[
                make_body(make_wall(),(0.0,0.0),True),
                make_body(make_circle(),INITIAL_CENTER,False),
            ],
        },
        "friction_constraints":{
            "static_friction_speed_bound":STATIC_FRICTION_SPEED_BOUND,
            "iterations":FRICTION_ITERATIONS,
        },
        "ipc_solver":IPC_SOLVER,
    }
    case_dir=OUTPUT_DIR/name
    if case_dir.exists():shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True,exist_ok=True)
    scene_file=case_dir/"input.json"
    save_json(scene_file,scene)
    return scene_file,case_dir

def run_solver(scene_file,case_dir):
    sim_file=case_dir/"sim.json"
    if not RUN_SOLVER:
        if not sim_file.exists():raise RuntimeError(f"Missing {sim_file}")
        return sim_file
    cmd=[str(SOLVER),"--ngui","--num-steps",str(NUM_STEPS),"--output-path",str(case_dir),str(scene_file),"--checkpoint-frequency",str(CHECKPOINT_FREQUENCY)]
    print("\nRUNNING MESH:")
    print(" ".join(cmd),flush=True)
    subprocess.run(cmd,cwd=BUILD_DIR,check=True)
    if not sim_file.exists():raise RuntimeError(f"Solver did not create {sim_file}")
    return sim_file

def extract(sim):
    args=sim["args"]
    states=sim["animation"]["state_sequence"]
    bodies=args["rigid_body_problem"]["rigid_bodies"]
    dynamic=[i for i,b in enumerate(bodies) if not all(b.get("is_dof_fixed",[False,False,False]))]
    if len(dynamic)!=1:raise RuntimeError(f"Expected one dynamic circle, found {len(dynamic)}")
    body_id=dynamic[0]
    dt=float(args["timestep"])
    n=len(states)
    t=np.arange(n)*dt
    x=np.zeros(n);y=np.zeros(n)
    vx=np.full(n,np.nan);vy=np.full(n,np.nan);omega=np.full(n,np.nan)
    for k,state in enumerate(states):
        b=state["rigid_bodies"][body_id]
        x[k]=float(b["position"][0]);y[k]=float(b["position"][1])
        if "linear_velocity" in b:
            vx[k]=float(b["linear_velocity"][0]);vy[k]=float(b["linear_velocity"][1])
        if "angular_velocity" in b:omega[k]=float(b["angular_velocity"][0])
    if not np.all(np.isfinite(vx)):
        vx=np.gradient(x,dt);vy=np.gradient(y,dt)
    if not np.all(np.isfinite(omega)):
        theta=np.array([float(s["rigid_bodies"][body_id]["rotation"][0]) for s in states])
        omega=np.gradient(np.unwrap(theta),dt)
    return t,x,y,vx,vy,omega

def predictions(mu):
    if mu>=MU_CRITICAL:
        a=(2.0/3.0)*GRAVITY*math.sin(ANGLE)
        return "rolling",a,a/RADIUS
    a=GRAVITY*(math.sin(ANGLE)-mu*math.cos(ANGLE))
    alpha=2.0*mu*GRAVITY*math.cos(ANGLE)/RADIUS
    return "sliding",a,alpha

def analyse(sim,mu):
    t,x,y,vx,vy,omega=extract(sim)
    position=np.column_stack((x,y))
    velocity=np.column_stack((vx,vy))
    gap=(position-WALL_ORIGIN)@INWARD_NORMAL-RADIUS
    downhill_velocity=velocity@DOWNHILL
    slip_velocity=downhill_velocity-RADIUS*omega
    contact=np.where(gap<=CONTACT_THRESHOLD)[0]
    if len(contact)==0:raise RuntimeError("Circle never contacted the incline")
    first_contact=int(contact[0])
    first_contact_time=float(t[first_contact])
    fit_start=first_contact_time+FIT_DELAY
    fit_end=min(fit_start+FIT_DURATION,float(t[-1]))
    fit_mask=(t>=fit_start)&(t<=fit_end)&(gap<=CONTACT_THRESHOLD)
    if np.count_nonzero(fit_mask)<10:raise RuntimeError("Not enough contact samples for acceleration fit")
    acceleration_fit=np.polyfit(t[fit_mask],downhill_velocity[fit_mask],1)[0]
    angular_acceleration_fit=np.polyfit(t[fit_mask],omega[fit_mask],1)[0]
    regime,a_expected,alpha_expected=predictions(mu)
    result={
        "representation":"mesh",
        "circle_segments":CIRCLE_SEGMENTS,
        "angle_degrees":ANGLE_DEG,
        "coefficient_friction":mu,
        "critical_friction_coefficient":MU_CRITICAL,
        "expected_regime":regime,
        "first_contact_time":first_contact_time,
        "minimum_smooth_reference_gap":float(np.min(gap)),
        "fit_interval":[float(fit_start),float(fit_end)],
        "downhill_acceleration":{
            "predicted":float(a_expected),
            "measured":float(acceleration_fit),
            "absolute_error":float(abs(acceleration_fit-a_expected)),
            "relative_error":float(abs(acceleration_fit-a_expected)/abs(a_expected)),
        },
        "angular_acceleration":{
            "predicted":float(alpha_expected),
            "measured":float(angular_acceleration_fit),
            "absolute_error":float(abs(angular_acceleration_fit-alpha_expected)),
            "relative_error":float(abs(angular_acceleration_fit-alpha_expected)/abs(alpha_expected)),
        },
        "tangential_contact_speed":{
            "mean_abs":float(np.mean(np.abs(slip_velocity[fit_mask]))),
            "max_abs":float(np.max(np.abs(slip_velocity[fit_mask]))),
            "final_abs":float(abs(slip_velocity[-1])),
        },
    }
    return t,gap,downhill_velocity,omega,slip_velocity,result

def save_plots(case_dir,t,gap,downhill_velocity,omega,slip_velocity):
    fig,ax=plt.subplots(figsize=(9,4.5))
    ax.plot(t,downhill_velocity,label=r"$v_{\parallel}$")
    ax.plot(t,RADIUS*omega,label=r"$R\omega$")
    ax.set_xlabel("time [s]");ax.set_ylabel("speed [m/s]")
    ax.grid(alpha=0.3);ax.legend();fig.tight_layout()
    fig.savefig(case_dir/"mesh_rolling_velocity.png",dpi=150);plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,4.5))
    ax.plot(t,slip_velocity);ax.axhline(0,ls="--",lw=1)
    ax.set_xlabel("time [s]");ax.set_ylabel("tangential contact speed [m/s]")
    ax.grid(alpha=0.3);fig.tight_layout()
    fig.savefig(case_dir/"mesh_contact_slip.png",dpi=150);plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,4.5))
    ax.plot(t,gap);ax.axhline(0,ls="--",lw=1)
    ax.set_xlabel("time [s]");ax.set_ylabel("smooth-reference wall gap [m]")
    ax.grid(alpha=0.3);fig.tight_layout()
    fig.savefig(case_dir/"mesh_wall_gap.png",dpi=150);plt.close(fig)

def print_result(name,result):
    a=result["downhill_acceleration"]
    w=result["angular_acceleration"]
    s=result["tangential_contact_speed"]
    print("\n============================================================")
    print(f"MESH INCLINED FRICTION TEST: {name}")
    print("============================================================")
    print(f"circle segments           : {CIRCLE_SEGMENTS}")
    print(f"incline angle             : {ANGLE_DEG:.2f} deg")
    print(f"mu                        : {result['coefficient_friction']:.5f}")
    print(f"critical mu               : {MU_CRITICAL:.8f}")
    print(f"expected regime           : {result['expected_regime']}")
    print(f"predicted acceleration    : {a['predicted']:.8f} m/s^2")
    print(f"measured acceleration     : {a['measured']:.8f} m/s^2")
    print(f"relative error            : {100*a['relative_error']:.3f} %")
    print(f"predicted angular accel.  : {w['predicted']:.8f} rad/s^2")
    print(f"measured angular accel.   : {w['measured']:.8f} rad/s^2")
    print(f"relative error            : {100*w['relative_error']:.3f} %")
    print(f"mean |contact slip|       : {s['mean_abs']:.8e} m/s")
    print(f"max |contact slip|        : {s['max_abs']:.8e} m/s")
    print(f"minimum wall gap          : {result['minimum_smooth_reference_gap']:.8e} m")
    print("============================================================")

def main():
    if not SOLVER.exists():raise RuntimeError(f"Missing solver {SOLVER}")
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    comparison={}
    print(f"Theoretical critical friction: mu = {MU_CRITICAL:.8f}")
    print(f"Initial center: ({INITIAL_CENTER[0]:.12f}, {INITIAL_CENTER[1]:.12f})")
    for name,mu in SCENARIOS:
        scene_file,case_dir=make_scene(name,mu)
        sim_file=run_solver(scene_file,case_dir)
        sim=load_json(sim_file)
        t,gap,downhill_velocity,omega,slip_velocity,result=analyse(sim,mu)
        save_json(case_dir/"mesh_benchmark.json",result)
        save_plots(case_dir,t,gap,downhill_velocity,omega,slip_velocity)
        comparison[name]=result
        print_result(name,result)
    save_json(SUMMARY_FILE,comparison)
    print(f"\nMesh comparison saved to:\n  {SUMMARY_FILE}")

if __name__=="__main__":
    main()