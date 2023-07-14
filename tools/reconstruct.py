#!/usr/bin/env python
# coding: utf-8

"""
Simple script to run multiple SDF reconstructions given a base log directory
and a set of checkpoints.
"""

import argparse
import os
import os.path as osp
from meshing import create_mesh
from util import from_pth, reconstruct_at_times


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
        "w0", type=int, default=30,
        help="Value for \\omega_0."
    )
    parser.add_argument(
        "--resolution", "-r", default=128, type=int,
        help="Resolution to use on marching cubes."
    )
    parser.add_argument(
        "--times", "-t", nargs='+', default=[-1, 0, 1],
        help="Parameter values to run inference on. Default is [-1, 0, 1]."
    )

    args = parser.parse_args()
    out_dir = osp.split(args.output_path)[0]
    if out_dir and not osp.exists(out_dir):
        os.makedirs(out_dir)

    model = from_pth(args.model_path, w0=args.w0, device="cuda:0").eval().to("cuda:0")
    print(model)
    print(f"Running marching cubes running with resolution {args.resolution}")

    times = [-0.4, 0.0, 0.5]
    reconstruct_at_times(model, times, out_dir, device="cuda:0")

    # create_mesh(
    #     model,
    #     args.output_path,
    #     N=args.resolution
    # )

    print("Done")
