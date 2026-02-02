#!/bin/bash
# Launch Blender with the USD simulation
export DISPLAY=:1
blender --python /home/dvangelder/claude_blender_newton_experiment/blender_socket_client.py
