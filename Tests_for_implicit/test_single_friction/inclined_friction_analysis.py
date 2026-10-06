#!/usr/bin/env python3
import copy,json,math,shutil,subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEST_DIR=Path(__file__).resolve().parent
BUILD_DIR=Path("/home/austeja/Everything/EggDispenser/Implicit/rigid-ipc/build")
SOLVER=BUILD_DIR/"rigid_ipc_sim"
BASE_JSON=TEST_DIR/"inclined_friction.json"
OUTPUT_DIR=TEST_DIR/"inclined_friction_runs"
SUMMARY_FILE=TEST_DIR/"inclined_friction_comparison.json"

SCENARIOS=[("mu_01",0.10),("mu_03",0.30)]
NUM_STEPS=3000
CHECKPOINT_FREQUENCY=10001
RUN_SOLVER=True
ANGLE_DEG=30.0
ANGLE=math.radians(ANGLE_DEG)
GRAVITY=9.81
RADIUS=1.0
R_BOTTOM=1.5
CONTACT_THRESHOLD=1e-2
FIT_DELAY=0.10
FIT_DURATION=1.00
MU_CRITICAL=(1.0/3.0)*math.tan(ANGLE)
DOWNHILL=np.array([-math.cos(ANGLE),-math.sin(ANGLE)])
INWARD_NORMAL=np.array([-math.sin(ANGLE),math.cos(ANGLE)])
WALL_ORIGIN=np.array([R_BOTTOM,0.0])

def load_json(path):
    with open(path) as f:return json.load(f)

def save_json(path,data):
    with open(path,"w") as f:json.dump(data,f,indent=2)

def make_scene(base,name,mu):
    scene=copy.deepcopy(base)
    scene["rigid_body_problem"]["coefficient_friction"]=mu
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
    cmd=[
        str(SOLVER),"--ngui",
        "--num-steps",str(NUM_STEPS),
        "--output-path",str(case_dir),
        str(scene_file),
        "--checkpoint-frequency",str(CHECKPOINT_FREQUENCY),
    ]
    print("\nRUNNING:")
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
        if "angular_velocity" in b:
            omega[k]=float(b["angular_velocity"][0])
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
    center_distance=(position-WALL_ORIGIN)@INWARD_NORMAL
    gap=center_distance-RADIUS
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
    mean_slip=float(np.mean(np.abs(slip_velocity[fit_mask])))
    max_slip=float(np.max(np.abs(slip_velocity[fit_mask])))
    result={
        "angle_degrees":ANGLE_DEG,
        "coefficient_friction":mu,
        "critical_friction_coefficient":MU_CRITICAL,
        "expected_regime":regime,
        "first_contact_time":first_contact_time,
        "minimum_gap":float(np.min(gap)),
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
            "mean_abs":mean_slip,
            "max_abs":max_slip,
            "final_abs":float(abs(slip_velocity[-1])),
        },
    }
    return t,gap,downhill_velocity,omega,slip_velocity,fit_mask,result

def save_plots(case_dir,t,gap,downhill_velocity,omega,slip_velocity,fit_mask,result):
    fig,ax=plt.subplots(figsize=(9,4.5))
    ax.plot(t,downhill_velocity,label=r"$v_{\parallel}$")
    ax.plot(t,RADIUS*omega,label=r"$R\omega$")
    ax.set_xlabel("time [s]");ax.set_ylabel("speed [m/s]")
    ax.grid(alpha=0.3);ax.legend();fig.tight_layout()
    fig.savefig(case_dir/"rolling_velocity.png",dpi=150);plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,4.5))
    ax.plot(t,slip_velocity)
    ax.axhline(0.0,ls="--",lw=1)
    ax.set_xlabel("time [s]");ax.set_ylabel("tangential contact speed [m/s]")
    ax.grid(alpha=0.3);fig.tight_layout()
    fig.savefig(case_dir/"contact_slip.png",dpi=150);plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,4.5))
    ax.plot(t,gap)
    ax.axhline(0.0,ls="--",lw=1)
    ax.set_xlabel("time [s]");ax.set_ylabel("circle-wall gap [m]")
    ax.grid(alpha=0.3);fig.tight_layout()
    fig.savefig(case_dir/"wall_gap.png",dpi=150);plt.close(fig)

def print_result(name,result):
    a=result["downhill_acceleration"]
    w=result["angular_acceleration"]
    slip=result["tangential_contact_speed"]
    print("\n============================================================")
    print(f"INCLINED FRICTION TEST: {name}")
    print("============================================================")
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
    print(f"mean |contact slip|       : {slip['mean_abs']:.8e} m/s")
    print(f"max |contact slip|        : {slip['max_abs']:.8e} m/s")
    print(f"minimum wall gap          : {result['minimum_gap']:.8e} m")
    print("============================================================")

def main():
    if not BASE_JSON.exists():raise RuntimeError(f"Missing {BASE_JSON}")
    if not SOLVER.exists():raise RuntimeError(f"Missing solver {SOLVER}")
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    base=load_json(BASE_JSON)
    comparison={}
    print(f"Theoretical critical friction: mu = {MU_CRITICAL:.8f}")
    for name,mu in SCENARIOS:
        scene_file,case_dir=make_scene(base,name,mu)
        sim_file=run_solver(scene_file,case_dir)
        sim=load_json(sim_file)
        t,gap,downhill_velocity,omega,slip_velocity,fit_mask,result=analyse(sim,mu)
        save_json(case_dir/"benchmark.json",result)
        save_plots(case_dir,t,gap,downhill_velocity,omega,slip_velocity,fit_mask,result)
        comparison[name]=result
        print_result(name,result)
    save_json(SUMMARY_FILE,comparison)
    print(f"\nComparison saved to:\n  {SUMMARY_FILE}")

if __name__=="__main__":
    main()