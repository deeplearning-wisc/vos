import torch
import argparse
from collections import OrderedDict

import torch


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Model Converter")
    parser.add_argument(
        "--model",
        required=True,
        metavar="FILE",
        help="path to model weights",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="FILE",
        help="path to model weights",
    )
    return parser


def convert_weight():
    args = get_parser().parse_args()
    ckpt = torch.load(args.model, map_location="cpu")
    if "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    # breakpoint()
    state_dict = state_dict['model_state']
    model = {"model": state_dict,"__author__": "custom", "matching_heuristics": True}

    torch.save(model, args.output)

if __name__ == "__main__":
    convert_weight()
