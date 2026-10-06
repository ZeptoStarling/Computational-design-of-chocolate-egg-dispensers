#!/usr/bin/env python3

import copy
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import lsq_linear, minimize_scalar

TEST_DIR=Path(
    "/home/austeja/Everything/EggDispenser/"
    "Implicit/testsforimplicit/teststwo"
)
BUILD_DIR=Path(
    "/home/austeja/Everything/EggDispenser/"
    "Implicit/rigid-ipc/build"
)
EXECUTABLE=BUILD_DIR/"rigid_ipc_sim"
BASE_SCENE=TEST_DIR/"test_json.json"

DROP_SCENE=BASE_SCENE
ARCH_SCENE=TEST_DIR/"test_json_from_arch.json"
PERTURBED_SCENE=TEST_DIR/"test_json_from_arch_perturbed.json"

DROP_OUTPUT=TEST_DIR/"output_drop"
ARCH_OUTPUT=TEST_DIR/"output_from_arch"
PERTURBED_OUTPUT=TEST_DIR/"output_from_arch_perturbed"

RESULT_FILE=TEST_DIR/"implicit_arch_results.json"

NUM_STEPS=5000
PERTURBATION=1e-6


def run_simulation(scene,output):
    if output.exists():
        shutil.rmtree(output)

    cmd=[
        str(EXECUTABLE),
        "--ngui",
        "--num-steps",str(NUM_STEPS),
        "--output-path",str(output),
        str(scene),
        "--checkpoint-frequency","10001",
    ]

    print("\nRunning:")
    print(" ".join(cmd))
    subprocess.run(cmd,cwd=BUILD_DIR,check=True)


def load_sim(output):
    with open(output/"sim.json","r") as f:
        return json.load(f)


def dynamic_body_indices(scene):
    bodies=scene["rigid_body_problem"]["rigid_bodies"]
    ids=[]

    for i,body in enumerate(bodies):
        fixed=body.get("is_dof_fixed",[False,False,False])
        if not all(fixed):
            ids.append(i)

    if len(ids)!=2:
        raise RuntimeError(
            f"Expected exactly two dynamic bodies, found {len(ids)}."
        )

    return sorted(ids,key=lambda i:scene["rigid_body_problem"]
                  ["rigid_bodies"][i]["position"][0])


def final_body_state(sim,body_id):
    return sim["animation"]["state_sequence"][-1]["rigid_bodies"][body_id]


def position(state):
    return np.asarray(state["position"][:2],dtype=float)


def rotation(state):
    return float(state["rotation"][0])


def final_speed(sim,body_id,dt):
    state=final_body_state(sim,body_id)

    if "linear_velocity" in state:
        return float(np.linalg.norm(
            np.asarray(state["linear_velocity"][:2],dtype=float)
        ))

    states=sim["animation"]["state_sequence"]
    p0=position(states[-2]["rigid_bodies"][body_id])
    p1=position(states[-1]["rigid_bodies"][body_id])
    return float(np.linalg.norm((p1-p0)/dt))


def body_radius(scene,body_id):
    return float(
        scene["rigid_body_problem"]["rigid_bodies"][body_id]["radius"]
    )


def make_spline(scene):
    f=scene["spline_funnel"]
    return CubicSpline(
        [0.0,float(f["height"])],
        [float(f["r_bottom"]),float(f["r_top"])],
        bc_type=(
            (1,float(f["s_bottom"])),
            (1,float(f["s_top"])),
        ),
    )


def closest_wall_point(center,spline,height):
    x,y=center
    sign=1.0 if x>=0.0 else -1.0

    def objective(yy):
        p=np.array([sign*float(spline(yy)),yy])
        d=center-p
        return float(d@d)

    result=minimize_scalar(
        objective,
        bounds=(0.0,height),
        method="bounded",
        options={"xatol":1e-12},
    )

    candidates=[result.x,0.0,height]
    yy=min(candidates,key=objective)

    return np.array([
        sign*float(spline(yy)),
        yy,
    ])


def contact_geometry(center,radius,spline,height):
    wall_point=closest_wall_point(center,spline,height)
    delta=center-wall_point
    distance=float(np.linalg.norm(delta))

    return {
        "point":wall_point,
        "normal":delta/distance,
        "gap":distance-radius,
    }


def arch_state(sim,scene,left_id,right_id):
    dt=float(sim["args"].get("timestep",scene["timestep"]))

    left=final_body_state(sim,left_id)
    right=final_body_state(sim,right_id)

    p_left=position(left)
    p_right=position(right)

    speed_left=final_speed(sim,left_id,dt)
    speed_right=final_speed(sim,right_id,dt)

    spline=make_spline(scene)
    height=float(scene["spline_funnel"]["height"])

    left_contact=contact_geometry(
        p_left,body_radius(scene,left_id),spline,height
    )
    right_contact=contact_geometry(
        p_right,body_radius(scene,right_id),spline,height
    )

    pair_gap=float(
        np.linalg.norm(p_left-p_right)
        -body_radius(scene,left_id)
        -body_radius(scene,right_id)
    )

    velocity_tol=float(
        scene["ipc_solver"].get("velocity_conv_tol",1e-4)
    )

    remained=(
        speed_left<=velocity_tol
        and speed_right<=velocity_tol
        and left_contact["gap"]<=1e-2
        and right_contact["gap"]<=1e-2
        and pair_gap<=1e-2
    )

    return {
        "left_center":[float(v) for v in p_left],
        "right_center":[float(v) for v in p_right],
        "left_speed":speed_left,
        "right_speed":speed_right,
        "stable_arch":bool(remained),
    }


def force_balance(sim,scene,left_id,right_id):
    left=position(final_body_state(sim,left_id))
    right=position(final_body_state(sim,right_id))

    spline=make_spline(scene)
    height=float(scene["spline_funnel"]["height"])

    left_contact=contact_geometry(
        left,body_radius(scene,left_id),spline,height
    )
    right_contact=contact_geometry(
        right,body_radius(scene,right_id),spline,height
    )

    n_left=left_contact["normal"]
    n_right=right_contact["normal"]

    delta=left-right
    n_pair_left=delta/np.linalg.norm(delta)
    n_pair_right=-n_pair_left

    # Forces are solved directly in units of mg:
    #
    # lambda_L n_L + lambda_P n_PL = (0,1)
    # lambda_R n_R + lambda_P n_PR = (0,1)
    A=np.array([
        [n_left[0],0.0,n_pair_left[0]],
        [n_left[1],0.0,n_pair_left[1]],
        [0.0,n_right[0],n_pair_right[0]],
        [0.0,n_right[1],n_pair_right[1]],
    ])

    b=np.array([0.0,1.0,0.0,1.0])

    solution=lsq_linear(
        A,b,bounds=(0.0,np.inf),
        method="trf",
        tol=1e-14,
    )

    forces=solution.x
    residual=A@forces-b

    return {
        "left_wall_force_over_mg":float(forces[0]),
        "right_wall_force_over_mg":float(forces[1]),
        "pair_force_over_mg":float(forces[2]),
        "equilibrium_residual":float(np.linalg.norm(residual)),
        "nonnegative_solution":bool(np.all(forces>=0.0)),
    }


def write_restart_scene(
    base_scene,
    source_sim,
    left_id,
    right_id,
    path,
    perturb_left_y=0.0,
):
    scene=copy.deepcopy(base_scene)

    source_left=final_body_state(source_sim,left_id)
    source_right=final_body_state(source_sim,right_id)

    bodies=scene["rigid_body_problem"]["rigid_bodies"]

    for body_id,source in (
        (left_id,source_left),
        (right_id,source_right),
    ):
        bodies[body_id]["position"]=[
            float(source["position"][0]),
            float(source["position"][1]),
        ]
        bodies[body_id]["rotation"]=[
            float(source["rotation"][0])
        ]
        bodies[body_id]["linear_velocity"]=[0.0,0.0]
        bodies[body_id]["angular_velocity"]=[0.0]

    bodies[left_id]["position"][1]+=perturb_left_y

    with open(path,"w") as f:
        json.dump(scene,f,indent=2)


with open(BASE_SCENE,"r") as f:
    base_scene=json.load(f)

left_id,right_id=dynamic_body_indices(base_scene)

# 1. Original drop.
run_simulation(DROP_SCENE,DROP_OUTPUT)
drop_sim=load_sim(DROP_OUTPUT)

# 2. Restart exactly from the resulting arch.
write_restart_scene(
    base_scene,
    drop_sim,
    left_id,
    right_id,
    ARCH_SCENE,
)
run_simulation(ARCH_SCENE,ARCH_OUTPUT)
arch_sim=load_sim(ARCH_OUTPUT)

# 3. Same arch, but raise the left circle by 1e-6 m.
write_restart_scene(
    base_scene,
    drop_sim,
    left_id,
    right_id,
    PERTURBED_SCENE,
    perturb_left_y=PERTURBATION,
)
run_simulation(PERTURBED_SCENE,PERTURBED_OUTPUT)
perturbed_sim=load_sim(PERTURBED_OUTPUT)

results={
    "drop":arch_state(
        drop_sim,base_scene,left_id,right_id
    ),
    "restart_from_arch":arch_state(
        arch_sim,base_scene,left_id,right_id
    ),
    "perturbed_restart":{
        "vertical_perturbation":PERTURBATION,
        **arch_state(
            perturbed_sim,base_scene,left_id,right_id
        ),
    },
    "force_balance":force_balance(
        drop_sim,base_scene,left_id,right_id
    ),
}

with open(RESULT_FILE,"w") as f:
    json.dump(results,f,indent=2)

print("\n==============================================")
print("TWO-CIRCLE FRICTIONLESS ARCH TEST")
print("==============================================")

for name,label in (
    ("drop","Original drop"),
    ("restart_from_arch","Restart from arch"),
    ("perturbed_restart","Perturbed restart"),
):
    r=results[name]

    print(f"\n{label}")
    print(
        "  left centre  : "
        f"({r['left_center'][0]:.12f}, "
        f"{r['left_center'][1]:.12f})"
    )
    print(
        "  right centre : "
        f"({r['right_center'][0]:.12f}, "
        f"{r['right_center'][1]:.12f})"
    )
    print(
        "  left speed   : "
        f"{r['left_speed']:.12e}"
    )
    print(
        "  right speed  : "
        f"{r['right_speed']:.12e}"
    )
    print(
        "  stable arch  : "
        f"{r['stable_arch']}"
    )

fb=results["force_balance"]

print("\nSTATIC FORCE BALANCE")
print(
    "  left wall force / mg  : "
    f"{fb['left_wall_force_over_mg']:.12f}"
)
print(
    "  right wall force / mg : "
    f"{fb['right_wall_force_over_mg']:.12f}"
)
print(
    "  pair force / mg       : "
    f"{fb['pair_force_over_mg']:.12f}"
)
print(
    "  equilibrium residual  : "
    f"{fb['equilibrium_residual']:.12e}"
)
print(
    "  nonnegative solution  : "
    f"{fb['nonnegative_solution']}"
)

print("\nResults saved to:")
print(f"  {RESULT_FILE}")
print("==============================================")