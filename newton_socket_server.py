#!/usr/bin/env python3
"""
Newton Physics Server - Cable Bundle Hysteresis
Socket-based IPC version for Blender visualization
"""

import socket
import json
import struct
import math
import numpy as np


def send_msg(sock, msg_dict):
    """Send JSON message with length prefix"""
    msg_bytes = json.dumps(msg_dict).encode('utf-8')
    msg_len = struct.pack('>I', len(msg_bytes))
    sock.sendall(msg_len + msg_bytes)


def recv_msg(sock):
    """Receive JSON message with length prefix"""
    raw_msglen = sock.recv(4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]

    chunks = []
    bytes_recd = 0
    while bytes_recd < msglen:
        chunk = sock.recv(min(msglen - bytes_recd, 4096))
        if not chunk:
            raise RuntimeError("socket connection broken")
        chunks.append(chunk)
        bytes_recd += len(chunk)

    return json.loads(b''.join(chunks).decode('utf-8'))


def run_newton_server(sock, usd_path):
    """Run Newton simulation with USD file"""
    try:
        import newton
        import warp as wp

        print(f"[Newton Server] Warp {wp.__version__ if hasattr(wp, '__version__') else 'unknown'}", flush=True)
        print(f"[Newton Server] Newton {newton.__version__ if hasattr(newton, '__version__') else 'unknown'}", flush=True)
        print(f"[Newton Server] Loading USD file: {usd_path}", flush=True)

        # Build model from USD
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5)
        builder.default_shape_cfg.ke = 2.0e3
        builder.default_shape_cfg.kd = 1.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75

        # Load USD file
        builder.add_usd(
            usd_path,
            xform=wp.transform(wp.vec3(0, 0, 0.8)),
            collapse_fixed_joints=True,
            enable_self_collisions=False,
            hide_collision_shapes=True,
            skip_mesh_approximation=True,
        )

        # Configure actuators for position control
        for i in range(6, builder.joint_dof_count):
            builder.joint_target_ke[i] = 1000.0
            builder.joint_target_kd[i] = 5.0
            builder.joint_act_mode[i] = int(newton.ActuatorMode.POSITION)

        # Approximate meshes for faster collision detection
        builder.approximate_meshes("bounding_box")

        # Add ground plane
        ground_cfg = newton.ModelBuilder.ShapeConfig(
            density=builder.default_shape_cfg.density,
            ke=1.0e3,
            kd=1.0e2,
            kf=builder.default_shape_cfg.kf,
            ka=builder.default_shape_cfg.ka,
            mu=0.75,
            restitution=builder.default_shape_cfg.restitution,
        )
        builder.add_ground_plane(cfg=ground_cfg)

        # Color and finalize
        builder.color()
        model = builder.finalize()

        print(f"[Newton Server] Model finalized", flush=True)
        print(f"[Newton Server]   Total bodies: {model.body_count}", flush=True)

        # Create MuJoCo solver
        solver = newton.solvers.SolverMuJoCo(
            model,
            use_mujoco_cpu=False,
            solver="newton",
            integrator="implicitfast",
            njmax=300,
            nconmax=150,
            cone="elliptic",
            impratio=100,
            iterations=100,
            ls_iterations=50,
        )

        # Create states
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()

        # Evaluate forward kinematics for collision detection
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        # Create collision pipeline
        collision_pipeline = newton.CollisionPipelineUnified.from_model(
            model,
            broad_phase_mode=newton.BroadPhaseMode.EXPLICIT,
        )
        contacts = model.collide(state_0, collision_pipeline=collision_pipeline)

        # Send ready signal with model info including all body names
        all_body_names = [model.body_key[i] for i in range(model.body_count)]
        send_msg(sock, {
            'status': 'ready',
            'total_bodies': model.body_count,
            'usd_path': usd_path,
            'body_names': all_body_names
        })

        print(f"[Newton Server] Entering simulation loop...", flush=True)

        # Simulation parameters
        sim_dt = 1.0 / 60.0 / 10  # 60 FPS, 10 substeps
        frame_count = 0
        sim_time = 0.0

        # Main loop
        while True:
            msg = recv_msg(sock)
            if not msg:
                break

            if msg.get('cmd') == 'stop':
                print(f"[Newton Server] Received stop command", flush=True)
                break

            elif msg.get('cmd') == 'restart':
                print(f"[Newton Server] Restarting simulation...", flush=True)
                state_0 = model.state()
                state_1 = model.state()
                frame_count = 0
                sim_time = 0.0
                print(f"[Newton Server] Simulation restarted!", flush=True)
                send_msg(sock, {'status': 'restarted'})

            elif msg.get('cmd') == 'step':
                # Run substeps
                contacts = model.collide(state_0, collision_pipeline=collision_pipeline)
                for substep in range(10):
                    state_0.clear_forces()

                    solver.step(state_0, state_1, control, contacts, sim_dt)
                    state_0, state_1 = state_1, state_0

                    sim_time += sim_dt

                # Extract body transforms
                body_q_np = state_0.body_q.numpy()

                # Send all body transforms with names
                body_transforms = []
                for body_idx in range(model.body_count):
                    transform = body_q_np[body_idx]
                    pos = transform[:3].tolist()
                    quat = transform[3:].tolist()  # [qw, qx, qy, qz]
                    body_name = model.body_key[body_idx]
                    body_transforms.append({
                        'name': body_name,
                        'pos': pos,
                        'quat': quat
                    })

                frame_count += 1
                if frame_count % 60 == 0:
                    print(f"[Newton] Frame {frame_count}, time {sim_time:.2f}s", flush=True)

                # Send state
                send_msg(sock, {
                    'status': 'state',
                    'frame': frame_count,
                    'bodies': body_transforms
                })

        send_msg(sock, {'status': 'stopped'})
        print(f"[Newton Server] Shutdown complete", flush=True)

    except Exception as e:
        import traceback
        error_msg = f"[Newton Server] Error: {e}\n{traceback.format_exc()}"
        print(error_msg, flush=True)
        try:
            send_msg(sock, {'status': 'error', 'message': str(e), 'traceback': traceback.format_exc()})
        except:
            pass


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python newton_socket_server.py <usd_file_path>", flush=True)
        sys.exit(1)

    usd_path = sys.argv[1]

    HOST = 'localhost'
    PORT = 9999

    print(f"[Newton Server] Starting socket server on {HOST}:{PORT}...", flush=True)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((HOST, PORT))
        server_sock.listen(1)

        print(f"[Newton Server] Waiting for Blender to connect...", flush=True)

        conn, addr = server_sock.accept()
        with conn:
            print(f"[Newton Server] Connected by {addr}", flush=True)
            run_newton_server(conn, usd_path)
