# Claude Blender Newton Experiment

NOTE: all code and documentation in this project generated with AI, through Claude. A fun experiment

A real-time physics simulation and visualization system that combines Newton physics engine with Blender for rendering and RL policy control for the Unitree G1 humanoid robot.

![Unitree G1 Simulation](Blender_g1.gif)

## Overview

This project demonstrates a socket-based IPC architecture connecting:
- **Newton Physics Engine**: Running physics simulation with RL policy control
- **Blender**: Real-time visualization of the USD robot model
- **Unitree G1 Robot**: 29-DOF humanoid robot model with RL policies

The system automatically manages the physics simulation, applies learned control policies, and visualizes the results in Blender in real-time.

## Features

- Real-time physics simulation with Newton engine + MuJoCo solver
- RL policy-based robot control (37-DOF policy with 23-DOF model)
- Socket-based IPC between physics engine and Blender
- Automatic simulation restart every 10 seconds
- USD-based robot model with full visual fidelity
- DOF padding to bridge policy/model mismatch
- 60 FPS visualization with 10 physics substeps per frame

## Prerequisites

- **Python 3.10+**
- **Blender 4.0+** (macOS: `/Applications/Blender.app`)
- **Newton Physics Engine** (development install required)
- **UV Package Manager** (for Python dependencies)
- **X Server** (Linux only, for display)

### Python Dependencies

- PyTorch
- NumPy
- PyYAML
- trimesh
- scipy
- Newton Physics Engine with MuJoCo support
- Warp

## Setup

### macOS Setup (Recommended)

1. **Install UV package manager**:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Clone and setup Newton repository** (required for MuJoCo solver):
```bash
cd ~
git clone https://github.com/NVIDIA/newton.git
cd newton
uv sync
```

3. **Clone this project**:
```bash
cd ~
git clone <repository-url>
cd claude_blender_newton_experiment
```

4. **Install dependencies with UV**:
```bash
uv pip install torch numpy pyyaml trimesh scipy
```

5. **Verify Blender installation**:
```bash
/Applications/Blender.app/Contents/MacOS/Blender --version
```

### Linux Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd claude_blender_newton_experiment
```

2. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install required Python packages:
```bash
pip install torch numpy pyyaml trimesh scipy
# Install Newton physics engine (follow Newton's installation instructions)
```

4. Verify Blender installation:
```bash
blender --version
```

## Project Structure

```
.
├── blender_socket_client.py           # Blender client for visualization
├── newton_socket_server_with_policy.py # Physics server with RL policy
├── run_simulation.sh                   # Launch script
├── unitree_g1/                        # Robot model and assets
│   ├── usd/                           # USD robot models
│   ├── meshes/                        # Robot mesh files
│   ├── urdf/                          # URDF robot descriptions
│   ├── mjcf/                          # MuJoCo XML descriptions
│   └── rl_policies/                   # RL policy configurations
│       ├── g1_23dof.yaml              # 23-DOF policy config
│       └── g1_29dof.yaml              # 29-DOF policy config
└── README.md                          # This file
```

## Running the Simulation

### Quick Start

Simply run the provided shell script:

```bash
./run_simulation.sh
```

This will:
1. Set the display environment variable
2. Launch Blender with the simulation client
3. Automatically start the Newton physics server
4. Begin real-time visualization

### Manual Execution

If you need more control, you can run components separately:

1. Start the physics server:
```bash
python newton_socket_server_with_policy.py
```

2. In another terminal, launch Blender with the client:
```bash
export DISPLAY=:1  # Adjust as needed
blender --python blender_socket_client.py
```

### Configuration

The simulation uses the Unitree G1 robot with the RL policy configuration:
- **37-DOF Policy**: Full model with dexterous hands (14 finger DOFs)
- **23-DOF MJCF Model**: Actual physical model (without individual finger joints)
- The code automatically pads missing DOFs with zeros to match policy expectations

RL policy configurations are located in `unitree_g1/rl_policies/`:
- `g1_23dof.yaml` - Policy config (expects 37 DOFs with fingers)
- `mjw_g1_23DOF.pt` - Trained policy weights

## How It Works

1. **Blender Client** (`blender_socket_client.py`):
   - Launches Newton physics server as subprocess
   - Connects via socket on localhost:9999
   - Loads USD robot model into Blender
   - Receives transform updates from physics engine
   - Updates Blender scene in real-time

2. **Newton Server** (`newton_socket_server_with_policy.py`):
   - Initializes physics simulation
   - Loads RL policy for robot control
   - Runs physics steps and applies policy actions
   - Sends body transforms to Blender client
   - Detects rest state and restarts simulation

3. **Communication Protocol**:
   - Length-prefixed JSON messages over TCP socket
   - Message types: ready, request_step, update, reset

## Rest Detection

The simulation automatically detects when the robot comes to rest and restarts after 10 seconds to ensure continuous operation.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

The Unitree G1 robot models are provided by Unitree Robotics and are subject to their own licensing terms. See `unitree_g1/LICENSE` for details.

## macOS-Specific Notes

### Newton Repository Dependency

The simulation requires the Newton physics repository at `~/newton` because:
- It provides the MuJoCo solver backend (more stable than XPBD for humanoid robots)
- The RL policy requires 37 DOFs with full dexterous hands
- Uses `uv run python` to access Newton's environment with MuJoCo installed

### Architecture

On macOS, the setup is:
1. Blender runs the visualization client (`blender_socket_client.py`)
2. Client spawns Newton server using: `uv run python` from `~/newton` directory
3. Newton server runs in Newton's environment (has MuJoCo + all dependencies)
4. Socket communication on localhost:9999 connects them

### Key Differences from Linux

- Uses Blender.app from `/Applications/` instead of system `blender`
- Newton server runs via `uv run` from `~/newton` (not local venv)
- MuJoCo solver required (XPBD produces NaN values for humanoid)
- No X Server needed (native macOS display)

## Troubleshooting

### "MuJoCo backend not installed" Error
Ensure you have the Newton repository at `~/newton` with dependencies installed:
```bash
cd ~/newton
uv sync
```

### NaN Position Values / Robot Not Moving
This indicates solver instability. Solution:
- Use MuJoCo solver (not XPBD) by ensuring Newton repo is properly set up
- Verify `blender_socket_client.py` uses `uv run python` from `~/newton` directory

### Display Issues (Linux only)
If you encounter display errors, ensure your `DISPLAY` environment variable is set correctly:
```bash
export DISPLAY=:0  # Or :1, depending on your system
```

### Socket Connection Errors
- Ensure port 9999 is not in use by another application
- Check firewall settings if running across different machines
- Kill any hung processes: `pkill -f newton_socket_server`

### Blender Python Module Issues
Make sure you're using Blender's built-in Python or that your Python environment is compatible with Blender's Python API.

### Robot Falls/Collapses
- Verify RL policy file matches DOF configuration
- Check that joint stiffness and damping values are appropriate
- Ensure MuJoCo solver is being used (check Newton server log)

## Credits

- **Unitree G1 Robot Model**: [Unitree Robotics](https://www.unitree.com/)
- **Newton Physics Engine**: Project Newton
- **Blender**: Open source 3D creation suite

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
