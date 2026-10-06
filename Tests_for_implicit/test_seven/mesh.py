#!/usr/bin/env python3
import json
import math
import shutil
import subprocess
from itertools import combinations
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon

from scipy.interpolate import CubicSpline
from tqdm import tqdm

TEST_DIR=Path(
    "/home/austeja/Everything/EggDispenser/"
    "Implicit/testsforimplicit/test_seven"
)
BUILD_DIR=Path(
    "/home/austeja/Everything/EggDispenser/"
    "rigid-ipc/build"
)
SOLVER=BUILD_DIR/"rigid_ipc_sim"
ROOT=TEST_DIR/"mesh_multiple"

RESULT_FILE=ROOT/"mesh_multibody_results.json"

FUNNEL_POINTS=96
CIRCLE_SEGMENTS=32

HEIGHT=6.0
R_BOTTOM=1.5
R_TOP=5.0
S_BOTTOM=0.0
S_TOP=0.0
WALL_THICKNESS=0.2

TIMESTEP=0.0005
MAX_TIME=10.0
NUM_STEPS=5000
CHECKPOINT_FREQUENCY=10001
VELOCITY_TOL=1e-4
CONTACT_THRESHOLD=1e-2

BALLS=[
    (0.0,5.0,1.0),
    (-2.2,7.0,1.0),
    (2.2,7.0,1.0),
    (0.0,9.2,1.0),
    (-1.8,11.3,0.9),
    (1.8,11.3,0.9),
    (0.0,13.3,0.8),
]

SPHERE_NAMES=[
    "bottom_center",
    "lower_left",
    "lower_right",
    "middle_center",
    "upper_left",
    "upper_right",
    "top_center",
]

MIRROR_PAIRS=[
    (1,2),
    (4,5),
]

CENTERLINE=[
    0,3,6,
]

SPHERE_COLORS=[
    "skyblue",
    "royalblue",
    "coral",
    "mediumseagreen",
    "purple",
    "orange",
    "gold",
]

CASES={
    "free_rotation":{
        "rotation_fixed":False,
        "scene":ROOT/"testmultiple_mesh_free_rotation.json",
        "output":ROOT/"output_free_rotation",
        "log":ROOT/"solver_free_rotation.log",
        "video":ROOT/"mesh_free_rotation.mp4",
        "image":ROOT/"mesh_free_rotation_final.png",
    },
    "fixed_rotation":{
        "rotation_fixed":True,
        "scene":ROOT/"testmultiple_mesh_fixed_rotation.json",
        "output":ROOT/"output_fixed_rotation",
        "log":ROOT/"solver_fixed_rotation.log",
        "video":ROOT/"mesh_fixed_rotation.mp4",
        "image":ROOT/"mesh_fixed_rotation_final.png",
    },
}

spline=CubicSpline(
    [0.0,HEIGHT],
    [R_BOTTOM,R_TOP],
    bc_type=((1,S_BOTTOM),(1,S_TOP)),
)


def funnel_radius(y):
    return float(spline(np.clip(y,0.0,HEIGHT)))


def make_funnel_walls():
    left_inner=[]
    left_outer=[]
    right_inner=[]
    right_outer=[]

    for i in range(FUNNEL_POINTS):
        y=HEIGHT*i/(FUNNEL_POINTS-1)
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


def make_circle_vertices(radius):
    return [
        [
            radius*math.cos(2.0*math.pi*i/CIRCLE_SEGMENTS),
            radius*math.sin(2.0*math.pi*i/CIRCLE_SEGMENTS),
        ]
        for i in range(CIRCLE_SEGMENTS)
    ]


def make_body(vertices,position,fixed,rotation_fixed=False):
    if fixed:
        dof=[True,True,True]
    else:
        dof=[False,False,rotation_fixed]

    return {
        "vertices":vertices,
        "polygons":[vertices],
        "edges":make_edges(len(vertices)),
        "oriented":True,
        "position":[float(position[0]),float(position[1])],
        "rotation":[0.0],
        "linear_velocity":[0.0,0.0],
        "angular_velocity":[0.0],
        "is_dof_fixed":dof,
    }


def make_scene(rotation_fixed):
    left_wall,right_wall,_,_=make_funnel_walls()

    bodies=[
        make_body(left_wall,(0.0,0.0),True),
        make_body(right_wall,(0.0,0.0),True),
    ]

    for x,y,radius in BALLS:
        bodies.append(
            make_body(
                make_circle_vertices(radius),
                (x,y),
                False,
                rotation_fixed,
            )
        )

    return {
        "scene_type":"distance_barrier_rb_problem",
        "solver":"ipc_solver",
        "timestep":TIMESTEP,
        "max_time":MAX_TIME,
        "rigid_body_problem":{
            "coefficient_restitution":0.2,
            "gravity":[0.0,-9.81],
            "rigid_bodies":bodies,
        },
        "ipc_solver":{
            "convergence_criteria":"velocity",
            "velocity_conv_tol":VELOCITY_TOL,
            "is_velocity_conv_tol_abs":True,
        },
    }


def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w") as f:
        json.dump(data,f,indent=2)


def load_json(path):
    with open(path,"r") as f:
        return json.load(f)


def validate_scene(scene,rotation_fixed):
    bodies=scene["rigid_body_problem"]["rigid_bodies"]

    if len(bodies)!=9:
        raise RuntimeError(f"Expected 9 bodies, found {len(bodies)}.")

    if len(bodies[0]["vertices"])!=2*FUNNEL_POINTS:
        raise RuntimeError("Incorrect left funnel resolution.")

    if len(bodies[1]["vertices"])!=2*FUNNEL_POINTS:
        raise RuntimeError("Incorrect right funnel resolution.")

    expected=[
        False,
        False,
        rotation_fixed,
    ]

    for body in bodies[2:]:
        if len(body["vertices"])!=CIRCLE_SEGMENTS:
            raise RuntimeError("Incorrect circle resolution.")

        if body["is_dof_fixed"]!=expected:
            raise RuntimeError(
                "Unexpected dynamic-body DOFs: "
                f"{body['is_dof_fixed']}"
            )


def run_scene(scene_file,output_dir,log_file):
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

    print(" ".join(command))

    with open(log_file,"w") as log:
        subprocess.run(
            command,
            cwd=BUILD_DIR,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )

    sim_file=output_dir/"sim.json"

    if not sim_file.exists():
        raise RuntimeError(f"Simulation did not produce {sim_file}")

    return load_json(sim_file)


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

    return min(
        point_segment_min(V0,A1,B1),
        point_segment_min(V1,A0,B0),
    )


def polygon_polyline_distance(poly,poly_edges,line):
    line=np.asarray(line,dtype=float)

    line_edges=np.column_stack((
        np.arange(len(line)-1),
        np.arange(1,len(line)),
    ))

    return polygon_distance(
        poly,
        poly_edges,
        line,
        line_edges,
    )


def get_dynamic_ids(sim):
    bodies=sim["args"]["rigid_body_problem"]["rigid_bodies"]
    states=sim["animation"]["state_sequence"]

    dynamic=[]

    for body_id,body in enumerate(bodies):
        fixed=body.get("is_dof_fixed",[False,False,False])
        if not all(fixed):
            p=states[0]["rigid_bodies"][body_id]["position"]
            dynamic.append((float(p[1]),float(p[0]),body_id))

    dynamic.sort(key=lambda x:(x[0],x[1]))

    return [x[2] for x in dynamic]


def extract_simulation(sim):
    bodies=sim["args"]["rigid_body_problem"]["rigid_bodies"]
    states=sim["animation"]["state_sequence"]
    ids=get_dynamic_ids(sim)

    if len(ids)!=7:
        raise RuntimeError(f"Expected 7 dynamic circles, found {len(ids)}.")

    frames=len(states)

    positions=np.zeros((frames,7,2))
    angles=np.zeros((frames,7))
    speeds=np.zeros((frames,7))
    angular_velocities=np.zeros((frames,7))

    radii=np.array([
        float(np.mean(np.linalg.norm(
            np.asarray(bodies[body_id]["vertices"],dtype=float),
            axis=1,
        )))
        for body_id in ids
    ])

    for frame_id,state in enumerate(states):
        for circle_id,body_id in enumerate(ids):
            body=state["rigid_bodies"][body_id]

            positions[frame_id,circle_id]=np.asarray(
                body["position"][:2],
                dtype=float,
            )

            angles[frame_id,circle_id]=float(
                body["rotation"][0]
            )

            v=np.asarray(
                body.get("linear_velocity",[0.0,0.0])[:2],
                dtype=float,
            )

            speeds[frame_id,circle_id]=np.linalg.norm(v)

            angular_velocities[frame_id,circle_id]=float(
                body.get("angular_velocity",[0.0])[0]
            )

    return {
        "bodies":bodies,
        "states":states,
        "ids":ids,
        "frames":frames,
        "positions":positions,
        "angles":angles,
        "speeds":speeds,
        "angular_velocities":angular_velocities,
        "radii":radii,
    }


def compute_gaps(data):
    _,_,left_inner,right_inner=make_funnel_walls()

    left_inner=np.asarray(left_inner,dtype=float)
    right_inner=np.asarray(right_inner,dtype=float)

    frames=data["frames"]
    ids=data["ids"]
    bodies=data["bodies"]
    states=data["states"]

    pairs=list(combinations(range(7),2))

    wall_gaps=np.zeros((frames,7))
    pair_gaps=np.zeros((frames,len(pairs)))

    print("Computing polygon gaps...")

    for frame_id in tqdm(range(frames),desc="Mesh gaps"):
        world=[]

        for circle_id,body_id in enumerate(ids):
            state=states[frame_id]["rigid_bodies"][body_id]
            body=bodies[body_id]

            V=transform_vertices(
                body["vertices"],
                state["position"],
                float(state["rotation"][0]),
            )

            E=np.asarray(body["edges"],dtype=int)
            world.append((V,E))

            x=float(state["position"][0])
            wall=left_inner if x<0.0 else right_inner

            wall_gaps[frame_id,circle_id]=(
                polygon_polyline_distance(V,E,wall)
            )

        for pair_id,(a,b) in enumerate(pairs):
            VA,EA=world[a]
            VB,EB=world[b]

            pair_gaps[frame_id,pair_id]=polygon_distance(
                VA,EA,VB,EB
            )

    return wall_gaps,pair_gaps,pairs


def analyze(sim,rotation_fixed):
    data=extract_simulation(sim)
    wall_gaps,pair_gaps,pairs=compute_gaps(data)

    positions=data["positions"]
    speeds=data["speeds"]
    angles=data["angles"]
    angular_velocities=data["angular_velocities"]
    radii=data["radii"]

    contacted_pairs=[
        pairs[i]
        for i in range(len(pairs))
        if np.min(pair_gaps[:,i])<=CONTACT_THRESHOLD
    ]

    mirror_x=[]
    mirror_y=[]

    for left,right in MIRROR_PAIRS:
        mirror_x.append(
            np.max(np.abs(
                positions[:,left,0]+positions[:,right,0]
            ))
        )

        mirror_y.append(
            np.max(np.abs(
                positions[:,left,1]-positions[:,right,1]
            ))
        )

    max_centerline_x=max(
        np.max(np.abs(positions[:,i,0]))
        for i in CENTERLINE
    )

    bottom_passed=bool(
        positions[-1,0,1]+radii[0]<0.0
    )

    final_contacts=np.zeros(7,dtype=int)

    for circle_id in range(7):
        if wall_gaps[-1,circle_id]<=CONTACT_THRESHOLD:
            final_contacts[circle_id]+=1

    for pair_id,(a,b) in enumerate(pairs):
        if pair_gaps[-1,pair_id]<=CONTACT_THRESHOLD:
            final_contacts[a]+=1
            final_contacts[b]+=1

    remaining=range(1,7)

    max_final_speed=max(
        speeds[-1,i]
        for i in remaining
    )

    remaining_interlocked=bool(
        bottom_passed
        and max_final_speed<=VELOCITY_TOL
        and all(
            final_contacts[i]>0
            for i in remaining
        )
    )

    result={
        "rotation_fixed":rotation_fixed,
        "bottom_center_passed_through":bottom_passed,
        "remaining_circles_interlocked":remaining_interlocked,
        "circle_circle_contacts":len(contacted_pairs),
        "minimum_polygon_wall_gap":float(np.min(wall_gaps)),
        "minimum_polygon_pair_gap":float(np.min(pair_gaps)),
        "maximum_final_speed_remaining":float(max_final_speed),
        "maximum_mirror_x_error":float(max(mirror_x)),
        "maximum_mirror_y_error":float(max(mirror_y)),
        "maximum_centerline_x_error":float(max_centerline_x),
        "maximum_abs_rotation":float(np.max(np.abs(angles))),
        "maximum_abs_angular_velocity":float(
            np.max(np.abs(angular_velocities))
        ),
        "final_positions":{
            SPHERE_NAMES[i]:[
                float(positions[-1,i,0]),
                float(positions[-1,i,1]),
            ]
            for i in range(7)
        },
    }

    return result,data


def fixed_plot_limits():
    x_min=-R_TOP-1.5
    x_max=R_TOP+1.5

    y_max=max(
        y+r
        for _,y,r in BALLS
    )+0.5

    return x_min,x_max,-0.5,y_max


def add_funnel(ax):
    _,_,left_inner,right_inner=make_funnel_walls()

    left=np.asarray(left_inner,dtype=float)
    right=np.asarray(right_inner,dtype=float)

    ax.plot(
        left[:,0],
        left[:,1],
        "k-",
        linewidth=2.0,
    )

    ax.plot(
        right[:,0],
        right[:,1],
        "k-",
        linewidth=2.0,
    )


def save_final_png(data,path):
    bodies=data["bodies"]
    states=data["states"]
    ids=data["ids"]

    x0,x1,y0,y1=fixed_plot_limits()

    fig,ax=plt.subplots(figsize=(7,9))
    ax.set_aspect("equal")
    ax.set_xlim(x0,x1)
    ax.set_ylim(y0,y1)

    add_funnel(ax)

    final=states[-1]["rigid_bodies"]

    for circle_id,body_id in enumerate(ids):
        body=bodies[body_id]
        state=final[body_id]

        V=transform_vertices(
            body["vertices"],
            state["position"],
            float(state["rotation"][0]),
        )

        patch=Polygon(
            V,
            closed=True,
            facecolor=SPHERE_COLORS[circle_id],
            edgecolor="black",
            linewidth=1.0,
        )

        ax.add_patch(patch)

    ax.axis("off")

    fig.subplots_adjust(
        left=0,
        right=1,
        bottom=0,
        top=1,
    )

    fig.savefig(
        path,
        dpi=250,
        bbox_inches="tight",
        pad_inches=0,
    )

    plt.close(fig)


def save_video(data,path):
    bodies=data["bodies"]
    states=data["states"]
    ids=data["ids"]

    x0,x1,y0,y1=fixed_plot_limits()

    fig,ax=plt.subplots(figsize=(7,9))
    ax.set_aspect("equal")
    ax.set_xlim(x0,x1)
    ax.set_ylim(y0,y1)

    add_funnel(ax)

    patches={}

    first=states[0]["rigid_bodies"]

    for circle_id,body_id in enumerate(ids):
        body=bodies[body_id]
        state=first[body_id]

        V=transform_vertices(
            body["vertices"],
            state["position"],
            float(state["rotation"][0]),
        )

        patch=Polygon(
            V,
            closed=True,
            facecolor=SPHERE_COLORS[circle_id],
            edgecolor="black",
            linewidth=1.0,
        )

        ax.add_patch(patch)
        patches[body_id]=patch

    ax.axis("off")

    def update(frame_id):
        state_bodies=states[frame_id]["rigid_bodies"]

        for body_id in ids:
            body=bodies[body_id]
            state=state_bodies[body_id]

            V=transform_vertices(
                body["vertices"],
                state["position"],
                float(state["rotation"][0]),
            )

            patches[body_id].set_xy(V)

        return list(patches.values())

    anim=FuncAnimation(
        fig,
        update,
        frames=len(states),
        interval=20,
        blit=False,
    )

    pbar=tqdm(
        total=len(states),
        desc=f"Saving {path.name}",
        unit="frame",
        ncols=100,
    )

    last_frame=0

    def progress_callback(current,total):
        nonlocal last_frame

        if current==total or current-last_frame>=50:
            pbar.update(current-last_frame)
            last_frame=current

    anim.save(
        path,
        writer="ffmpeg",
        fps=60,
        progress_callback=progress_callback,
    )

    if pbar.n<len(states):
        pbar.update(len(states)-pbar.n)

    pbar.close()
    plt.close(fig)


def print_result(name,result):
    print()
    print("="*60)
    print(name)
    print("="*60)
    print(
        "Rotation fixed              : "
        f"{result['rotation_fixed']}"
    )
    print(
        "Bottom centre passed through: "
        f"{result['bottom_center_passed_through']}"
    )
    print(
        "Remaining circles interlocked: "
        f"{result['remaining_circles_interlocked']}"
    )
    print(
        "Circle-circle contacts      : "
        f"{result['circle_circle_contacts']}"
    )
    print(
        "Minimum polygon wall gap    : "
        f"{result['minimum_polygon_wall_gap']:.12e}"
    )
    print(
        "Minimum polygon pair gap    : "
        f"{result['minimum_polygon_pair_gap']:.12e}"
    )
    print(
        "Max final speed remaining   : "
        f"{result['maximum_final_speed_remaining']:.12e}"
    )
    print(
        "Maximum mirror x error      : "
        f"{result['maximum_mirror_x_error']:.12e}"
    )
    print(
        "Maximum mirror y error      : "
        f"{result['maximum_mirror_y_error']:.12e}"
    )
    print(
        "Maximum centre-line x error : "
        f"{result['maximum_centerline_x_error']:.12e}"
    )
    print(
        "Maximum abs rotation        : "
        f"{result['maximum_abs_rotation']:.12e}"
    )
    print(
        "Maximum angular velocity    : "
        f"{result['maximum_abs_angular_velocity']:.12e}"
    )


def main():
    if not SOLVER.exists():
        raise RuntimeError(f"Solver does not exist:\n  {SOLVER}")

    ROOT.mkdir(parents=True,exist_ok=True)

    results={
        "mesh":{
            "funnel_points_per_side":FUNNEL_POINTS,
            "circle_segments":CIRCLE_SEGMENTS,
        },
        "cases":{},
    }

    for name,case in CASES.items():
        print()
        print("="*60)
        print(f"RUNNING {name}")
        print("="*60)

        scene=make_scene(case["rotation_fixed"])
        validate_scene(scene,case["rotation_fixed"])
        write_json(case["scene"],scene)

        sim=run_scene(
            case["scene"],
            case["output"],
            case["log"],
        )

        result,data=analyze(
            sim,
            case["rotation_fixed"],
        )

        save_final_png(
            data,
            case["image"],
        )

        save_video(
            data,
            case["video"],
        )

        results["cases"][name]=result
        print_result(name,result)

    write_json(RESULT_FILE,results)

    print()
    print("="*60)
    print("COMPARISON")
    print("="*60)

    free=results["cases"]["free_rotation"]
    fixed=results["cases"]["fixed_rotation"]

    print(
        "Free rotation interlocked : "
        f"{free['remaining_circles_interlocked']}"
    )
    print(
        "Fixed rotation interlocked: "
        f"{fixed['remaining_circles_interlocked']}"
    )

    print()
    print(f"Results: {RESULT_FILE}")

    for name,case in CASES.items():
        print()
        print(name)
        print(f"  scene : {case['scene']}")
        print(f"  sim   : {case['output']/'sim.json'}")
        print(f"  image : {case['image']}")
        print(f"  video : {case['video']}")
        print(f"  log   : {case['log']}")

    print("="*60)


if __name__=="__main__":
    main()