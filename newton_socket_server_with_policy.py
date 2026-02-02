#!/usr/bin/env python3
"""
Newton Physics Server with RL Policy
Socket-based IPC version for Blender visualization
"""

import socket
import json
import struct
import yaml
import torch
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


@torch.jit.script
def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by the inverse of a quaternion."""
    q_w = q[..., 3]
    q_vec = q[..., :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    if q_vec.dim() == 2:
        c = q_vec * torch.bmm(q_vec.view(q.shape[0], 1, 3), v.view(q.shape[0], 3, 1)).squeeze(-1) * 2.0
    else:
        c = q_vec * torch.einsum("...i,...i->...", q_vec, v).unsqueeze(-1) * 2.0
    return a - b + c


def compute_obs(
    actions: torch.Tensor,
    state,
    joint_pos_initial: torch.Tensor,
    device: str,
    indices: torch.Tensor,
    gravity_vec: torch.Tensor,
    command: torch.Tensor,
) -> torch.Tensor:
    """Compute observation for robot policy."""
    joint_q = state.joint_q if state.joint_q is not None else []
    joint_qd = state.joint_qd if state.joint_qd is not None else []

    root_quat_w = torch.tensor(joint_q[3:7], device=device, dtype=torch.float32).unsqueeze(0)
    root_lin_vel_w = torch.tensor(joint_qd[:3], device=device, dtype=torch.float32).unsqueeze(0)
    root_ang_vel_w = torch.tensor(joint_qd[3:6], device=device, dtype=torch.float32).unsqueeze(0)
    joint_pos_current = torch.tensor(joint_q[7:], device=device, dtype=torch.float32).unsqueeze(0)
    joint_vel_current = torch.tensor(joint_qd[6:], device=device, dtype=torch.float32).unsqueeze(0)

    vel_b = quat_rotate_inverse(root_quat_w, root_lin_vel_w)
    a_vel_b = quat_rotate_inverse(root_quat_w, root_ang_vel_w)
    grav = quat_rotate_inverse(root_quat_w, gravity_vec)
    joint_pos_rel = joint_pos_current - joint_pos_initial
    joint_vel_rel = joint_vel_current
    rearranged_joint_pos_rel = torch.index_select(joint_pos_rel, 1, indices)
    rearranged_joint_vel_rel = torch.index_select(joint_vel_rel, 1, indices)
    obs = torch.cat([vel_b, a_vel_b, grav, command, rearranged_joint_pos_rel, rearranged_joint_vel_rel, actions], dim=1)

    return obs


def run_newton_server(sock, usd_path):
    """Run Newton simulation with USD file and RL policy"""
    try:
        import newton
        import warp as wp
        import newton.utils

        print(f"[Newton Server] Warp {wp.__version__ if hasattr(wp, '__version__') else 'unknown'}", flush=True)
        print(f"[Newton Server] Newton {newton.__version__ if hasattr(newton, '__version__') else 'unknown'}", flush=True)
        print(f"[Newton Server] Loading USD file: {usd_path}", flush=True)

        # Download and load policy assets
        asset_directory = str(newton.utils.download_asset('unitree_g1'))
        policy_path = f"{asset_directory}/rl_policies/mjw_g1_23DOF.pt"
        yaml_path = f"{asset_directory}/rl_policies/g1_23dof.yaml"

        print(f"[Newton Server] Loading policy config from: {yaml_path}", flush=True)
        with open(yaml_path, encoding='utf-8') as f:
            config = yaml.safe_load(f)

        num_dofs = config['num_dofs']
        print(f"[Newton Server] Policy config loaded: {num_dofs} DOFs", flush=True)

        # Setup PyTorch device
        torch_device = "cuda" if wp.get_device().is_cuda else "cpu"
        print(f"[Newton Server] PyTorch device: {torch_device}", flush=True)

        # Build model from USD
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

        # Use config values for joint settings
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.1,
            limit_ke=1.0e2,
            limit_kd=1.0e0,
        )
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75

        # Load USD file
        builder.add_usd(
            usd_path,
            xform=wp.transform(wp.vec3(0, 0, 0.8)),
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            hide_collision_shapes=True,
            skip_mesh_approximation=True,
        )

        # Set initial joint positions from config
        builder.joint_q[:3] = [0.0, 0.0, 0.76]
        builder.joint_q[3:7] = [0.0, 0.0, 0.7071, 0.7071]  # Quaternion for upright
        builder.joint_q[7:] = config["mjw_joint_pos"]

        # Configure joint actuators with stiffness and damping from config
        for i in range(len(config["mjw_joint_stiffness"])):
            builder.joint_target_ke[i + 6] = config["mjw_joint_stiffness"][i]
            builder.joint_target_kd[i + 6] = config["mjw_joint_damping"][i]
            builder.joint_armature[i + 6] = config["mjw_joint_armature"][i]

        # Approximate meshes for faster collision detection
        builder.approximate_meshes("convex_hull")

        # Add ground plane
        builder.add_ground_plane()

        # Finalize model
        model = builder.finalize()
        model.set_gravity((0.0, 0.0, -9.81))

        print(f"[Newton Server] Model finalized", flush=True)
        print(f"[Newton Server]   Total bodies: {model.body_count}", flush=True)

        # Create MuJoCo solver
        solver = newton.solvers.SolverMuJoCo(
            model,
            use_mujoco_cpu=False,
            solver="newton",
            integrator="implicitfast",
            nconmax=30,
            njmax=100,
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

        # Load policy
        print(f"[Newton Server] Loading policy from: {policy_path}", flush=True)
        policy = torch.jit.load(policy_path, map_location=torch_device)
        print(f"[Newton Server] Policy loaded successfully", flush=True)

        # Setup policy tensors
        joint_q = state_0.joint_q if state_0.joint_q is not None else []
        joint_pos_initial = torch.tensor(joint_q[7:], device=torch_device, dtype=torch.float32).unsqueeze(0)
        actions = torch.zeros(1, num_dofs, device=torch_device, dtype=torch.float32)

        # Physx to MJC mapping (identity for now since we're using MJWarp policy)
        physx_to_mjc_indices = torch.tensor(list(range(num_dofs)), device=torch_device, dtype=torch.long)
        mjc_to_physx_indices = torch.tensor(list(range(num_dofs)), device=torch_device, dtype=torch.long)

        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=torch_device, dtype=torch.float32).unsqueeze(0)
        command = torch.zeros((1, 3), device=torch_device, dtype=torch.float32)  # [forward, lateral, yaw]

        # Send ready signal with model info including all body names
        all_body_names = [model.body_key[i] for i in range(model.body_count)]
        send_msg(sock, {
            'status': 'ready',
            'total_bodies': model.body_count,
            'usd_path': usd_path,
            'body_names': all_body_names
        })

        print(f"[Newton Server] Entering simulation loop with RL policy...", flush=True)

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
                newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
                frame_count = 0
                sim_time = 0.0
                actions = torch.zeros(1, num_dofs, device=torch_device, dtype=torch.float32)
                print(f"[Newton Server] Simulation restarted!", flush=True)
                send_msg(sock, {'status': 'restarted'})

            elif msg.get('cmd') == 'step':
                # Compute observation for policy
                obs = compute_obs(
                    actions,
                    state_0,
                    joint_pos_initial,
                    torch_device,
                    physx_to_mjc_indices,
                    gravity_vec,
                    command,
                )

                # Run policy to get actions
                with torch.no_grad():
                    actions = policy(obs)
                    rearranged_actions = torch.index_select(actions, 1, mjc_to_physx_indices)

                    # Convert actions to joint targets
                    joint_targets = joint_pos_initial + config["action_scale"] * rearranged_actions
                    joint_targets_with_base = torch.cat([
                        torch.zeros(6, device=torch_device, dtype=torch.float32),
                        joint_targets.squeeze(0)
                    ])

                    # Set control targets
                    joint_targets_wp = wp.from_torch(joint_targets_with_base, dtype=wp.float32, requires_grad=False)
                    wp.copy(control.joint_target_pos, joint_targets_wp)

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
                    # Newton quaternion is [qx, qy, qz, qw], reorder to [qw, qx, qy, qz] for Blender
                    quat_xyzw = transform[3:7].tolist()
                    quat = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]  # [qw, qx, qy, qz]
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
        print("Usage: python newton_socket_server_with_policy.py <usd_file_path>", flush=True)
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
