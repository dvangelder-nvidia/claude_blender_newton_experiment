import bpy
import socket
import json
import struct
import subprocess
import time
import atexit
from pathlib import Path

# Newton server process
newton_process = None
newton_socket = None
newton_ready = False
simulation_running = False

# USD path
usd_path = None

# Body objects from USD
body_objects = []

# Newton body names (from ready message)
newton_body_names = []

# Rest detection
is_at_rest = False
rest_start_time = None
REST_DURATION = 1.0  # 10 seconds before restart


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


def start_newton_server(usd_file_path):
    """Start Newton server as separate subprocess"""
    global newton_process, newton_socket, newton_ready

    if newton_process and newton_process.poll() is None:
        print("[Blender] Newton server already running", flush=True)
        return newton_ready

    print("[Blender] Starting Newton server subprocess...", flush=True)

    newton_repo = Path.home() / "newton"
    newton_server_script = Path.home() / "blender_claude/newton_socket_server.py"
    newton_log_path = "/tmp/newton_server.log"

    newton_log_file = open(newton_log_path, 'w')
    newton_process = subprocess.Popen(
        ["uv", "run", "python", str(newton_server_script), usd_file_path],
        cwd=str(newton_repo),
        stdout=newton_log_file,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    print(f"[Blender] Newton server logging to: {newton_log_path}", flush=True)
    print(f"[Blender] Newton server process started (PID: {newton_process.pid})", flush=True)

    if not hasattr(start_newton_server, '_cleanup_registered'):
        atexit.register(stop_newton_server)
        start_newton_server._cleanup_registered = True
        print("[Blender] Registered cleanup handler", flush=True)

    time.sleep(3)

    # Connect to Newton server
    HOST = 'localhost'
    PORT = 9999

    for attempt in range(10):
        try:
            newton_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            newton_socket.connect((HOST, PORT))
            print(f"[Blender] Connected to Newton server on {HOST}:{PORT}", flush=True)
            break
        except ConnectionRefusedError:
            if attempt < 9:
                print(f"[Blender] Connection attempt {attempt + 1}/10...", flush=True)
                time.sleep(1)
            else:
                print(f"✗ Failed to connect to Newton server", flush=True)
                return False

    # Wait for ready message
    global newton_body_names

    try:
        msg = recv_msg(newton_socket)
        if msg and msg.get('status') == 'ready':
            print(f"✓ Newton ready: {msg.get('total_bodies')} bodies loaded from USD", flush=True)
            print(f"  USD path: {msg.get('usd_path')}", flush=True)
            newton_body_names = msg.get('body_names', [])
            newton_ready = True
            return True
        elif msg and msg.get('status') == 'error':
            print(f"✗ Newton server error: {msg.get('message')}", flush=True)
            return False
    except Exception as e:
        print(f"✗ Error receiving ready message: {e}", flush=True)
        return False

    return False


def stop_newton_server():
    """Stop Newton server and clean up resources"""
    global newton_process, newton_socket, newton_ready, simulation_running

    print("[Blender] Shutting down Newton server...", flush=True)

    simulation_running = False

    if newton_socket:
        try:
            send_msg(newton_socket, {'cmd': 'stop'})
            msg = recv_msg(newton_socket)
            if msg:
                print(f"[Blender] Newton acknowledged stop: {msg.get('status')}", flush=True)
        except Exception as e:
            print(f"[Blender] Error sending stop: {e}", flush=True)

        try:
            newton_socket.close()
        except:
            pass
        newton_socket = None

    if newton_process:
        pid = newton_process.pid
        print(f"[Blender] Terminating Newton process (PID: {pid})...", flush=True)

        try:
            newton_process.terminate()
            newton_process.wait(timeout=5)
            print(f"[Blender] Newton process terminated", flush=True)
        except subprocess.TimeoutExpired:
            print(f"[Blender] Forcing kill...", flush=True)
            newton_process.kill()
            newton_process.wait()
            print(f"[Blender] Newton process killed", flush=True)
        except Exception as e:
            print(f"[Blender] Error terminating: {e}", flush=True)

        newton_process = None

    newton_ready = False
    print("[Blender] ✓ Cleanup complete", flush=True)


def load_usd_into_blender(usd_file_path):
    """Load USD file into Blender"""
    global body_objects

    print(f"[Blender] Loading USD file: {usd_file_path}", flush=True)

    # Import USD
    bpy.ops.wm.usd_import(filepath=usd_file_path)

    # Get all imported objects (they should be selected after import)
    imported_objects = list(bpy.context.selected_objects)

    print(f"[Blender] Imported {len(imported_objects)} objects from USD", flush=True)

    # Build a dictionary mapping object names to objects
    # This allows us to match Newton bodies by name
    body_objects = {obj.name: obj for obj in imported_objects}

    return body_objects


def update_bodies(body_transforms):
    """Update body transforms from Newton simulation"""
    global body_objects

    if not body_transforms or not body_objects:
        return

    # Debug: Print data on first update
    if not hasattr(update_bodies, '_debug_printed'):
        print(f"[Blender] Updating {len(body_objects)} Blender objects from {len(body_transforms)} Newton bodies", flush=True)
        # Print Newton body names for palm/hand
        palm_bodies = [t['name'] for t in body_transforms if 'palm' in t['name'].lower() or 'hand' in t['name'].lower()]
        print(f"[Blender] Newton palm/hand bodies: {palm_bodies}", flush=True)
        # Find palm/hand related object names in Blender
        palm_objs = [name for name in body_objects.keys() if 'palm' in name.lower() or 'hand' in name.lower()]
        print(f"[Blender] Blender palm/hand objects: {palm_objs}", flush=True)
        update_bodies._debug_printed = True

    # Track how many bodies we successfully matched
    matched_count = 0
    unmatched_bodies = []

    for transform in body_transforms:
        body_name = transform['name']
        pos = transform['pos']
        quat = transform['quat']  # [qw, qx, qy, qz]

        # Try to find matching Blender object by name using multiple strategies
        obj = None

        # Strategy 1: Exact match
        obj = body_objects.get(body_name)

        # Strategy 2: Remove leading slashes and try again (e.g., "/g1/pelvis" -> "g1/pelvis")
        if obj is None and body_name.startswith('/'):
            obj = body_objects.get(body_name[1:])

        # Strategy 3: Get just the base name after last slash (e.g., "/g1/pelvis" -> "pelvis")
        if obj is None:
            base_name = body_name.split('/')[-1]
            obj = body_objects.get(base_name)

        # Strategy 4: Try with "_link" suffix removed (common in USD)
        if obj is None and base_name.endswith('_link'):
            name_no_link = base_name[:-5]  # Remove "_link"
            obj = body_objects.get(name_no_link)

        # Strategy 5: Try partial matching - find any object containing the base name
        if obj is None:
            for obj_name, obj_candidate in body_objects.items():
                if base_name in obj_name or obj_name in base_name:
                    obj = obj_candidate
                    break

        if obj is not None:
            matched_count += 1
            # Set position
            obj.location = (pos[0], pos[1], pos[2])

            # Set rotation (Blender quaternion is [w, x, y, z])
            obj.rotation_mode = 'QUATERNION'
            obj.rotation_quaternion = (quat[0], quat[1], quat[2], quat[3])
        else:
            unmatched_bodies.append(body_name)

    # Debug unmatched bodies on first frame
    if not hasattr(update_bodies, '_mismatch_printed'):
        print(f"[Blender] Successfully matched: {matched_count}/{len(body_transforms)} bodies", flush=True)
        if unmatched_bodies:
            print(f"[Blender] Warning: {len(unmatched_bodies)} bodies could not be matched", flush=True)
            print(f"[Blender] Unmatched body names: {unmatched_bodies}", flush=True)
        update_bodies._mismatch_printed = True


def restart_simulation():
    """Send restart command to Newton"""
    global newton_socket, is_at_rest, rest_start_time

    if not newton_socket:
        return False

    try:
        print("[Blender] Restarting simulation...", flush=True)
        send_msg(newton_socket, {'cmd': 'restart'})
        msg = recv_msg(newton_socket)

        if msg and msg.get('status') == 'restarted':
            print("[Blender] Simulation restarted!", flush=True)
            is_at_rest = False
            rest_start_time = None
            return True
    except Exception as e:
        print(f"Error restarting: {e}", flush=True)

    return False


def step_simulation():
    """Send step command and receive state"""
    global newton_socket

    if not newton_socket:
        return None

    try:
        send_msg(newton_socket, {'cmd': 'step'})
        msg = recv_msg(newton_socket)
        if msg and msg.get('status') == 'state':
            return msg
    except Exception as e:
        print(f"Error stepping: {e}", flush=True)

    return None


class NewtonSimulationTimer(bpy.types.Operator):
    """Modal operator to run Newton simulation"""
    bl_idname = "wm.newton_simulation_timer"
    bl_label = "Newton Simulation Timer"

    _timer = None
    _frame_count = 0

    def modal(self, context, event):
        global simulation_running, newton_ready, is_at_rest, rest_start_time

        if event.type == 'TIMER':
            if newton_ready and simulation_running:
                state = step_simulation()
                if state:
                    self._frame_count += 1

                    # Get simulation data
                    body_transforms = state.get('bodies', [])

                    # Update visualization
                    update_bodies(body_transforms)

                    # Restart periodically (but not at frame 0)
                    restart_frame = int(60 * REST_DURATION)
                    if self._frame_count > 0 and self._frame_count % restart_frame == 0:
                        print(f"[Blender] Restarting after {REST_DURATION}s (frame {self._frame_count})...", flush=True)
                        restart_simulation()
                        self._frame_count = 0

                    # Print info every 60 frames
                    if self._frame_count % 60 == 0:
                        print(f"[Blender] Frame {self._frame_count} ({self._frame_count/60.0:.1f}s)", flush=True)

                    # Force viewport update
                    for area in context.screen.areas:
                        if area.type == 'VIEW_3D':
                            area.tag_redraw()

        if not simulation_running:
            self.cancel(context)
            return {'CANCELLED'}

        return {'PASS_THROUGH'}

    def execute(self, context):
        wm = context.window_manager
        self._timer = wm.event_timer_add(1.0/60.0, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)


def main(usd_file_path):
    """Main function to set up and run the simulation"""
    global simulation_running, usd_path

    usd_path = usd_file_path

    print("="*60, flush=True)
    print("BLENDER + NEWTON USD SIMULATION", flush=True)
    print("="*60, flush=True)

    # Clear existing scene
    print("[Blender] Clearing scene...", flush=True)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Start Newton server with USD path
    print(f"[Blender] Launching Newton server with USD: {usd_file_path}", flush=True)
    newton_loaded = start_newton_server(usd_file_path)

    if not newton_loaded:
        print("[Blender] ✗ Failed to start Newton server", flush=True)
        return

    # Load USD into Blender
    print("[Blender] Loading USD into Blender...", flush=True)
    imported_objects = load_usd_into_blender(usd_file_path)

    # Print diagnostic name mapping report
    print("\n" + "="*60, flush=True)
    print("BODY NAME MAPPING DIAGNOSTIC", flush=True)
    print("="*60, flush=True)
    print(f"Newton has {len(newton_body_names)} bodies", flush=True)
    print(f"Blender has {len(imported_objects)} objects", flush=True)

    print("\nALL Newton body names:", flush=True)
    for i, name in enumerate(newton_body_names):
        print(f"  {i}: {name}", flush=True)

    print("\nSearching for specific bodies in Blender:", flush=True)
    test_names = ['palm', 'head', 'chest', 'torso', 'neck']
    for test in test_names:
        matches = [name for name in imported_objects.keys() if test.lower() in name.lower()]
        print(f"  '{test}' in Blender: {matches[:5]}", flush=True)

    print("="*60 + "\n", flush=True)

    # Add ground plane for reference
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "Ground"
    ground_mat = bpy.data.materials.new(name="GroundMat")
    ground_mat.use_nodes = True
    ground_mat.node_tree.nodes["Principled BSDF"].inputs['Base Color'].default_value = (0.3, 0.3, 0.3, 1.0)
    ground.data.materials.append(ground_mat)

    # Add camera
    bpy.ops.object.camera_add(location=(3.0, -3.0, 2.0))
    camera = bpy.context.active_object
    camera.rotation_euler = (1.1, 0.0, 0.785)
    bpy.context.scene.camera = camera

    # Add light
    bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
    light = bpy.context.active_object
    light.data.energy = 2.0

    # Summary
    print("\n" + "="*60, flush=True)
    print(f"Newton Server:   ✓ Running (PID: {newton_process.pid})", flush=True)
    print(f"USD Loaded:      ✓ {len(imported_objects)} objects in Blender", flush=True)
    print(f"USD Path:        {usd_file_path}", flush=True)
    print(f"Auto-Restart:    Every {REST_DURATION} seconds", flush=True)
    print("="*60 + "\n", flush=True)

    # Register and start simulation
    bpy.utils.register_class(NewtonSimulationTimer)
    simulation_running = True
    bpy.ops.wm.newton_simulation_timer()
    print("[Blender] ✓ Simulation started!", flush=True)
    print("[Blender] Newton server will auto-cleanup on exit", flush=True)


# Run main
if __name__ == "__main__":
    # USD file path
    USD_FILE = "/Users/dvangelder/blender_claude/unitree_g1/usd/g1_minimal.usd"
    main(USD_FILE)
