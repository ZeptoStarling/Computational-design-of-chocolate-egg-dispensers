#!/usr/bin/env python3
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar,brentq
TEST_DIR=Path(
    "/home/austeja/Everything/EggDispenser/"
    "Implicit/testsforimplicit/teststwo")
BUILD_DIR=Path(
    "/home/austeja/Everything/EggDispenser/"
    "rigid-ipc/build"
)
SOLVER=BUILD_DIR/"rigid_ipc_sim"
IMPLICIT_RESULTS=TEST_DIR/"implicit_arch_results.json"
MESH_DIR=TEST_DIR/"mesh_arch"
RESULT_FILE=MESH_DIR/"mesh_arch_results.json"

RESOLUTIONS=[
    ("N096_N032",96,32),
    ("N1024_N1024",1024,1024),
]

HEIGHT=6.0
R_BOTTOM=1.5
R_TOP=5.0
S_BOTTOM=0.0
S_TOP=0.0
WALL_THICKNESS=0.2

RADIUS=1.0
DROP_LEFT=(-1.2,5.0)
DROP_RIGHT=(1.2,5.0)

TIMESTEP=0.0005
MAX_TIME=10.0
NUM_STEPS=5000
CHECKPOINT_FREQUENCY=10001
VELOCITY_TOL=1e-4
CONTACT_THRESHOLD=1e-2

spline=CubicSpline(
    [0.0,HEIGHT],
    [R_BOTTOM,R_TOP],
    bc_type=((1,S_BOTTOM),(1,S_TOP)),
)


def funnel_radius(y):
    return float(spline(np.clip(y,0.0,HEIGHT)))


def make_funnel_walls(n):
    left_inner=[]
    left_outer=[]
    right_inner=[]
    right_outer=[]

    for i in range(n):
        y=HEIGHT*i/(n-1)
        r=funnel_radius(y)
        left_inner.append([-r,y])
        left_outer.append([-r-WALL_THICKNESS,y])
        right_inner.append([r,y])
        right_outer.append([r+WALL_THICKNESS,y])

    left=left_inner+list(reversed(left_outer))
    right=list(reversed(right_inner))+right_outer
    return left,right,left_inner,right_inner


def make_edges(n):
    return [[i,(i+1)%n] for i in range(n)]


def make_circle_vertices(n):
    return [
        [
            RADIUS*math.cos(2.0*math.pi*i/n),
            RADIUS*math.sin(2.0*math.pi*i/n),
        ]
        for i in range(n)
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


def make_scene(n_funnel,n_circle,left_pos,right_pos):
    left_wall,right_wall,_,_=make_funnel_walls(n_funnel)
    circle=make_circle_vertices(n_circle)

    return {
        "scene_type":"distance_barrier_rb_problem",
        "solver":"ipc_solver",
        "timestep":TIMESTEP,
        "max_time":MAX_TIME,
        "rigid_body_problem":{
            "coefficient_restitution":0.2,
            "coefficient_friction":0.0,
            "gravity":[0.0,-9.81],
            "rigid_bodies":[
                make_body(left_wall,(0.0,0.0),True),
                make_body(right_wall,(0.0,0.0),True),
                make_body(circle,left_pos,False),
                make_body(circle,right_pos,False),
            ],
        },
        "ipc_solver":{
            "convergence_criteria":"velocity",
            "velocity_conv_tol":VELOCITY_TOL,
            "is_velocity_conv_tol_abs":True,
        },
        "friction_constraints":{
            "iterations":1,
        },
    }


def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w") as f:
        json.dump(data,f,indent=2)


def load_json(path):
    with open(path,"r") as f:
        return json.load(f)


def transform_vertices(vertices,position,angle):
    V=np.asarray(vertices,dtype=float)
    c=math.cos(angle)
    s=math.sin(angle)
    R=np.array([[c,-s],[s,c]])
    return V@R.T+np.asarray(position,dtype=float)


def point_segment_min(points,A,B):
    P=np.asarray(points,dtype=float)
    A=np.asarray(A,dtype=float)
    B=np.asarray(B,dtype=float)

    AB=B-A
    denom=np.sum(AB*AB,axis=1)
    denom=np.maximum(denom,1e-30)

    AP=P[:,None,:]-A[None,:,:]
    t=np.sum(AP*AB[None,:,:],axis=2)/denom[None,:]
    t=np.clip(t,0.0,1.0)

    Q=A[None,:,:]+t[:,:,None]*AB[None,:,:]
    D=P[:,None,:]-Q
    return float(np.sqrt(np.min(np.sum(D*D,axis=2))))


def polygon_distance(V0,E0,V1,E1):
    V0=np.asarray(V0,dtype=float)
    V1=np.asarray(V1,dtype=float)
    E0=np.asarray(E0,dtype=int)
    E1=np.asarray(E1,dtype=int)

    A0=V0[E0[:,0]]
    B0=V0[E0[:,1]]
    A1=V1[E1[:,0]]
    B1=V1[E1[:,1]]

    d01=point_segment_min(V0,A1,B1)
    d10=point_segment_min(V1,A0,B0)
    return min(d01,d10)


def polygon_polyline_distance(poly,poly_edges,line):
    line=np.asarray(line,dtype=float)
    line_edges=np.column_stack((
        np.arange(len(line)-1),
        np.arange(1,len(line)),
    ))
    return polygon_distance(poly,poly_edges,line,line_edges)


def analytic_wall_gap(center):
    x,y=center
    sign=1.0 if x>=0.0 else -1.0

    def objective(yy):
        p=np.array([sign*funnel_radius(yy),yy])
        d=np.asarray(center,dtype=float)-p
        return float(d@d)

    result=minimize_scalar(
        objective,
        bounds=(0.0,HEIGHT),
        method="bounded",
        options={"xatol":1e-12},
    )

    distance=math.sqrt(min(
        result.fun,
        objective(0.0),
        objective(HEIGHT),
    ))
    return distance-RADIUS


def mesh_wall_gap(center,n_funnel,n_circle):
    x,y=center
    left_wall,right_wall,_,_=make_funnel_walls(n_funnel)
    wall=left_wall if x<0.0 else right_wall

    local=np.asarray(make_circle_vertices(n_circle),dtype=float)
    world=local+np.asarray([x,y],dtype=float)

    circle_edges=np.asarray(make_edges(n_circle),dtype=int)
    wall_edges=np.asarray(make_edges(len(wall)),dtype=int)

    return polygon_distance(
        world,circle_edges,
        np.asarray(wall,dtype=float),wall_edges,
    )


def find_matched_y(n_funnel,n_circle,left_center,right_center,
                   left_target,right_target):
    y0=0.5*(left_center[1]+right_center[1])

    def residual(y):
        gl=mesh_wall_gap(
            (left_center[0],y),n_funnel,n_circle
        )
        gr=mesh_wall_gap(
            (right_center[0],y),n_funnel,n_circle
        )

        return 0.5*(
            (gl-left_target)
            +(gr-right_target)
        )

    r0=residual(y0)

    if abs(r0)<=1e-12:
        return y0

    step=1e-4
    max_shift=0.05
    nsteps=int(max_shift/step)

    prev_down_y=y0
    prev_down_r=r0
    prev_up_y=y0
    prev_up_r=r0

    for i in range(1,nsteps+1):
        down_y=y0-i*step
        down_r=residual(down_y)

        if down_r*prev_down_r<=0.0:
            return float(brentq(
                residual,
                down_y,
                prev_down_y,
                xtol=1e-14,
            ))

        up_y=y0+i*step
        up_r=residual(up_y)

        if up_r*prev_up_r<=0.0:
            return float(brentq(
                residual,
                prev_up_y,
                up_y,
                xtol=1e-14,
            ))

        prev_down_y=down_y
        prev_down_r=down_r
        prev_up_y=up_y
        prev_up_r=up_r

    raise RuntimeError(
        "Could not find a matched wall gap within "
        f"{max_shift} m of y={y0:.12f}"
    )


def run(scene_file,output_dir):
    if output_dir.exists():
        shutil.rmtree(output_dir)

    command=[
        str(SOLVER),
        "--ngui",
        "--num-steps",str(NUM_STEPS),
        "--output-path",str(output_dir),
        str(scene_file),
        "--checkpoint-frequency",str(CHECKPOINT_FREQUENCY),
    ]

    print("\nRunning:")
    print(" ".join(command))
    subprocess.run(command,cwd=BUILD_DIR,check=True)

    sim_file=output_dir/"sim.json"
    if not sim_file.exists():
        raise RuntimeError(f"Simulation did not produce {sim_file}")

    return load_json(sim_file)


def body_speed(state):
    v=np.asarray(state.get("linear_velocity",[0.0,0.0]),dtype=float)
    return float(np.linalg.norm(v[:2]))


def scene_initial_geometry(scene):
    bodies=scene["rigid_body_problem"]["rigid_bodies"]

    walls=[]
    for body in bodies[:2]:
        walls.append(np.asarray(body["vertices"],dtype=float))

    left=bodies[2]
    right=bodies[3]

    VL=transform_vertices(
        left["vertices"],left["position"],left["rotation"][0]
    )
    VR=transform_vertices(
        right["vertices"],right["position"],right["rotation"][0]
    )

    EL=np.asarray(left["edges"],dtype=int)
    ER=np.asarray(right["edges"],dtype=int)
    EW0=np.asarray(bodies[0]["edges"],dtype=int)
    EW1=np.asarray(bodies[1]["edges"],dtype=int)

    left_gap=min(
        polygon_distance(VL,EL,walls[0],EW0),
        polygon_distance(VL,EL,walls[1],EW1),
    )
    right_gap=min(
        polygon_distance(VR,ER,walls[0],EW0),
        polygon_distance(VR,ER,walls[1],EW1),
    )
    pair_gap=polygon_distance(VL,EL,VR,ER)

    return left_gap,right_gap,pair_gap


def analyze(scene,sim):
    defs=scene["rigid_body_problem"]["rigid_bodies"]
    states=sim["animation"]["state_sequence"]
    final=states[-1]["rigid_bodies"]

    left=final[2]
    right=final[3]

    left_pos=np.asarray(left["position"][:2],dtype=float)
    right_pos=np.asarray(right["position"][:2],dtype=float)

    VL=transform_vertices(
        defs[2]["vertices"],left_pos,float(left["rotation"][0])
    )
    VR=transform_vertices(
        defs[3]["vertices"],right_pos,float(right["rotation"][0])
    )

    EL=np.asarray(defs[2]["edges"],dtype=int)
    ER=np.asarray(defs[3]["edges"],dtype=int)

    walls=[
        np.asarray(defs[0]["vertices"],dtype=float),
        np.asarray(defs[1]["vertices"],dtype=float),
    ]
    wall_edges=[
        np.asarray(defs[0]["edges"],dtype=int),
        np.asarray(defs[1]["edges"],dtype=int),
    ]

    left_wall_id=0 if left_pos[0]<0.0 else 1
    right_wall_id=0 if right_pos[0]<0.0 else 1

    left_gap=polygon_distance(
        VL,EL,walls[left_wall_id],wall_edges[left_wall_id]
    )
    right_gap=polygon_distance(
        VR,ER,walls[right_wall_id],wall_edges[right_wall_id]
    )
    pair_gap=polygon_distance(VL,EL,VR,ER)

    left_speed=body_speed(left)
    right_speed=body_speed(right)

    initial_left_gap,initial_right_gap,initial_pair_gap=(
        scene_initial_geometry(scene)
    )

    stable=(
        left_speed<=VELOCITY_TOL
        and right_speed<=VELOCITY_TOL
        and left_gap<=CONTACT_THRESHOLD
        and right_gap<=CONTACT_THRESHOLD
        and pair_gap<=CONTACT_THRESHOLD
    )

    return {
        "initial_left_wall_gap":initial_left_gap,
        "initial_right_wall_gap":initial_right_gap,
        "initial_pair_gap":initial_pair_gap,
        "final_left_center":[float(v) for v in left_pos],
        "final_right_center":[float(v) for v in right_pos],
        "final_left_speed":left_speed,
        "final_right_speed":right_speed,
        "final_left_wall_gap":left_gap,
        "final_right_wall_gap":right_gap,
        "final_pair_gap":pair_gap,
        "stable_arch":bool(stable),
    }


def print_result(label,result):
    print(f"\n{label}")
    print(
        "  initial wall gaps : "
        f"{result['initial_left_wall_gap']:.12e}, "
        f"{result['initial_right_wall_gap']:.12e}"
    )
    print(
        "  initial pair gap  : "
        f"{result['initial_pair_gap']:.12e}"
    )
    print(
        "  final left centre : "
        f"({result['final_left_center'][0]:.12f}, "
        f"{result['final_left_center'][1]:.12f})"
    )
    print(
        "  final right centre: "
        f"({result['final_right_center'][0]:.12f}, "
        f"{result['final_right_center'][1]:.12f})"
    )
    print(
        "  final speeds      : "
        f"{result['final_left_speed']:.12e}, "
        f"{result['final_right_speed']:.12e}"
    )
    print(
        "  final wall gaps   : "
        f"{result['final_left_wall_gap']:.12e}, "
        f"{result['final_right_wall_gap']:.12e}"
    )
    print(
        "  final pair gap    : "
        f"{result['final_pair_gap']:.12e}"
    )
    print(
        "  stable arch       : "
        f"{result['stable_arch']}"
    )


def main():
    if not SOLVER.exists():
        raise RuntimeError(f"Solver does not exist:\n  {SOLVER}")

    if not IMPLICIT_RESULTS.exists():
        raise RuntimeError(
            "Implicit arch results do not exist:\n"
            f"  {IMPLICIT_RESULTS}"
        )

    implicit=load_json(IMPLICIT_RESULTS)
    drop=implicit["drop"]

    left_center=np.asarray(drop["left_center"],dtype=float)
    right_center=np.asarray(drop["right_center"],dtype=float)

    left_target_gap=analytic_wall_gap(left_center)
    right_target_gap=analytic_wall_gap(right_center)

    print()
    print("="*60)
    print("IMPLICIT ARCH REFERENCE")
    print("="*60)
    print(
        f"left centre        : "
        f"({left_center[0]:.12f}, {left_center[1]:.12f})"
    )
    print(
        f"right centre       : "
        f"({right_center[0]:.12f}, {right_center[1]:.12f})"
    )
    print(
        f"left wall gap      : {left_target_gap:.12e}"
    )
    print(
        f"right wall gap     : {right_target_gap:.12e}"
    )

    results={
        "implicit_reference":{
            "left_center":[float(v) for v in left_center],
            "right_center":[float(v) for v in right_center],
            "left_wall_gap":left_target_gap,
            "right_wall_gap":right_target_gap,
        }
    }

    for name,n_funnel,n_circle in RESOLUTIONS:
        root=MESH_DIR/name
        root.mkdir(parents=True,exist_ok=True)

        print()
        print("="*60)
        print(
            f"{name}: funnel N={n_funnel}, circle N={n_circle}"
        )
        print("="*60)

        # ----------------------------------------------------
        # Find the nearest vertical position whose actual
        # polygon-wall gap matches the analytic wall gap.
        # ----------------------------------------------------

        matched_y=find_matched_y(
            n_funnel,
            n_circle,
            left_center,
            right_center,
            left_target_gap,
            right_target_gap,
        )

        matched_left_gap=mesh_wall_gap(
            (left_center[0],matched_y),
            n_funnel,
            n_circle,
        )

        matched_right_gap=mesh_wall_gap(
            (right_center[0],matched_y),
            n_funnel,
            n_circle,
        )

        gap_error=max(
            abs(matched_left_gap-left_target_gap),
            abs(matched_right_gap-right_target_gap),
        )

        print()
        print("MATCHED-GAP GEOMETRY")
        print(f"  implicit y          : {left_center[1]:.12f}")
        print(f"  matched mesh y      : {matched_y:.12f}")
        print(
            f"  vertical adjustment : "
            f"{matched_y-left_center[1]:+.12e}"
        )
        print(f"  left matched gap    : {matched_left_gap:.12e}")
        print(f"  left target gap     : {left_target_gap:.12e}")
        print(f"  right matched gap   : {matched_right_gap:.12e}")
        print(f"  right target gap    : {right_target_gap:.12e}")
        print(f"  maximum gap error   : {gap_error:.12e}")

        if gap_error>1e-9:
            raise RuntimeError(
                f"Matched wall gap failed for {name}: "
                f"error={gap_error:.12e}"
            )

        scenarios={
            "drop":(
                DROP_LEFT,
                DROP_RIGHT,
            ),
            "implicit_centers":(
                tuple(left_center),
                tuple(right_center),
            ),
            "matched_wall_gap":(
                (float(left_center[0]),matched_y),
                (float(right_center[0]),matched_y),
            ),
        }

        resolution_results={
            "funnel_points_per_side":n_funnel,
            "circle_segments":n_circle,
            "matched_gap":{
                "implicit_y":float(left_center[1]),
                "matched_y":matched_y,
                "vertical_adjustment":float(
                    matched_y-left_center[1]
                ),
                "target_left_wall_gap":left_target_gap,
                "target_right_wall_gap":right_target_gap,
                "matched_left_wall_gap":matched_left_gap,
                "matched_right_wall_gap":matched_right_gap,
                "maximum_gap_error":gap_error,
            },
            "runs":{},
        }

        for scenario_name,(left_pos,right_pos) in scenarios.items():
            print()
            print("-"*60)
            print(f"RUNNING {name} / {scenario_name}")
            print("-"*60)
            print(
                f"left initial centre  : "
                f"({left_pos[0]:.12f}, {left_pos[1]:.12f})"
            )
            print(
                f"right initial centre : "
                f"({right_pos[0]:.12f}, {right_pos[1]:.12f})"
            )

            scene=make_scene(
                n_funnel,
                n_circle,
                left_pos,
                right_pos,
            )

            scene_file=root/f"{scenario_name}.json"
            output_dir=root/f"output_{scenario_name}"

            write_json(scene_file,scene)
            sim=run(scene_file,output_dir)
            result=analyze(scene,sim)

            resolution_results["runs"][scenario_name]=result

            print_result(
                f"{name} / {scenario_name}",
                result,
            )

        results[name]=resolution_results

    write_json(RESULT_FILE,results)

    print()
    print("="*60)
    print("FINAL SUMMARY")
    print("="*60)

    for name,_,_ in RESOLUTIONS:
        resolution=results[name]

        print()
        print(name)
        print(
            "  matched y          : "
            f"{resolution['matched_gap']['matched_y']:.12f}"
        )
        print(
            "  vertical adjustment: "
            f"{resolution['matched_gap']['vertical_adjustment']:+.12e}"
        )

        for scenario_name,result in resolution["runs"].items():
            print(
                f"  {scenario_name:20s}: "
                f"stable_arch={result['stable_arch']}, "
                f"speeds=("
                f"{result['final_left_speed']:.6e}, "
                f"{result['final_right_speed']:.6e}), "
                f"pair_gap={result['final_pair_gap']:.6e}"
            )

    print()
    print("Saved combined results:")
    print(f"  {RESULT_FILE}")
    print("="*60)


if __name__=="__main__":
    main()