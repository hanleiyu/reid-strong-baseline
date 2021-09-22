# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
from os import mkdir

import torch
from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from data import make_data_loader_prcc, make_data_loader_vc, make_data_loader_ltcc
from engine.inference import prcc_inference
from modeling import build_part_model, build_model
from utils.logger import setup_logger


def mean(lists):
    return sum(lists)/len(lists)

def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    model = build_model(cfg, 71)
    # model = build_part_model(cfg, 71)
    model.load_param(cfg.TEST.WEIGHT)
    map = [0 for _ in range(10)]
    r1 = [0 for _ in range(10)]
    r5 = [0 for _ in range(10)]
    r10 = [0 for _ in range(10)]
    if cfg.DATASETS.NAMES == "prcc":
        for i in range(10):
            val_loader, num_query, val_set = make_data_loader_prcc(cfg, trial=i)
            r1[i], r5[i], r10[i], map[i] = prcc_inference(cfg, model, val_loader, num_query, val_set, index=i)
        logger.info(
            "r1: {:.1%}, r5: {:.1%}, r10: {:.1%}, map: {:.1%}".format(mean(r1), mean(r5), mean(r10), mean(map)))
    elif cfg.DATASETS.NAMES == "vc":
        val_loader, num_query, val_set = make_data_loader_vc(cfg)
        r1, r5, r10, map = prcc_inference(cfg, model, val_loader, num_query, val_set)
        logger.info("r1: {:.1%}, r5: {:.1%}, r10: {:.1%}, map: {:.1%}".format(r1, r5, r10, map))
    elif cfg.DATASETS.NAMES == "ltcc":
        val_loader, num_query, val_set = make_data_loader_ltcc(cfg)
        r1, r5, r10, map = prcc_inference(cfg, model, val_loader, num_query, val_set)
        logger.info("r1: {:.1%}, r5: {:.1%}, r10: {:.1%}, map: {:.1%}".format(r1, r5, r10, map))

    # for i in range(25, 35):
    #     val_loader, num_query, val_set = make_data_loader_prcc(cfg, trial=i)
    #     r1[i], r5[i], r10[i], map[i] = prcc_inference(cfg, model, val_loader, num_query, val_set, index=i)
    # logger.info("map: {:.1%}, r1: {:.1%}, r5: {:.1%}, r10: {:.1%}".format(mean(map), mean(r1), mean(r5), mean(r10)))
    # for i in range(25, 35):
    #     val_loader, num_query, val_set = make_data_loader_prcc(cfg, trial=i)
    #     r1[i-25], r5[i-25], r10[i-25], map[i-25] = prcc_inference(cfg, model, val_loader, num_query, val_set, index=i)
    # logger.info("map: {:.1%}, r1: {:.1%}, r5: {:.1%}, r10: {:.1%}".format(mean(map), mean(r1), mean(r5), mean(r10)))

if __name__ == '__main__':
    main()
