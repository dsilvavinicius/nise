#!/usr/bin/env python
# coding: utf-8

"""
Simple script to run multiple SDF reconstructions given a base log directory
and a set of checkpoints.
"""

import argparse
import os
import os.path as osp
import torch
from i4d.util import from_pth, reconstruct_at_times


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run marching cubes using a trained model."
    )
    parser.add_argument(
        "model_path",
        help="Path to the PyTorch weights file"
    )
    parser.add_argument(
        "output_path",
        help="Path to the output mesh file"
    )
    parser.add_argument(
        "--omega0", "-w", type=int, default=1,
        help="Value for \\omega_0. Default is 1"
    )
    parser.add_argument(
        "--resolution", "-r", default=128, type=int,
        help="Resolution to use on marching cubes. Default is 128"
    )
    parser.add_argument(
        "--times", "-t", nargs='+', default=[-1, 0, 1],
        help="Parameter values to run inference on. Default is [-1, 0, 1]."
    )

    args = parser.parse_args()
    out_dir = osp.split(args.output_path)[0]
    if out_dir and not osp.exists(out_dir):
        os.makedirs(out_dir)

    devstr = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(devstr)

    model = from_pth(args.model_path, w0=args.omega0, device=device).eval()
    model = model.to(device)
    print(model)
    print(f"Running marching cubes running with resolution {args.resolution}")

    times = [float(t) for t in args.times]
    reconstruct_at_times(model, times, out_dir, device=device)

    print("Done")
