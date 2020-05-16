import random
import cv2

from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog

from humans.dataset import register_humans_voc
from humans.config import setup_cfg


def parse_args():
    import argparse

    # Parse command line arguments
    ap = argparse.ArgumentParser(description="humans prediction")
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--scale", type=float, default=1.0)

    # Detectron settings
    ap.add_argument("--config-file",
                    required=True,
                    help="path to config file")
    ap.add_argument("--confidence-threshold", type=float, default=0.5,
                    help="minimum score for instance predictions to be shown (default: 0.5)")
    ap.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                    help="modify model config options using the command-line")

    return ap.parse_args()


def main(args):
    if args.confidence_threshold is not None:
        # Set score_threshold for builtin models
        args.opts.append('MODEL.ROI_HEADS.SCORE_THRESH_TEST')
        args.opts.append(str(args.confidence_threshold))
        args.opts.append('MODEL.RETINANET.SCORE_THRESH_TEST')
        args.opts.append(str(args.confidence_threshold))
        print("args", type(args), args)


if __name__ == "__main__":
    args = parse_args()
    print("Command Line Args:", args)

    main(args)
