import sys
import os
import glob
from pathlib import Path

import numpy as np
from PIL import Image
import torch

# Add project path
project_root = Path('../../')
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src/diffusion-policy'))

from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent

class Args:
    def __init__(self):
        self.device = "cuda:0"
        self.model_path = "/home/user/codes/Intern/InternNav/checkpoints/InternVLA-N1-DualVLN"
        self.resize_w = 384
        self.resize_h = 384
        self.num_history = 8
        self.camera_intrinsic = np.array([
            [386.5, 0.0, 328.9, 0.0],
            [0.0, 386.5, 244.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        self.plan_step_gap = 4

args = Args()
print(f"Model path: {args.model_path}")
print(f"Device: {args.device}")
print(f"Image size: {args.resize_w}x{args.resize_h}")
print(f"History frames: {args.num_history}")

print("Loading model...")
agent = InternVLAN1AsyncAgent(args)

# Warm up model
print("Warming up model...")
dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
dummy_depth = np.zeros((480, 640), dtype=np.float32)
dummy_pose = np.eye(4)
agent.reset()
agent.step(dummy_rgb, dummy_depth, dummy_pose, "hello", intrinsic=args.camera_intrinsic)
print("Model loaded successfully!")