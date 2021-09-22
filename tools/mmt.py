# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
import torch

from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from data import make_data_loader_mmt
from engine.trainer import do_train_mmt
from modeling.baseline import Baseline
from layers import make_loss_mmt
from solver import make_optimizer, WarmupMultiStepLR

from utils.logger import setup_logger
import random
import numpy as np
import datetime
from torch.backends import cudnn

from torch import nn


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def create_model(cfg, num_classes):
    model_1 = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK,
                     cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE)
    model_2 = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK,
                     cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE)

    model_1_ema = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK,
                     cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE)
    model_2_ema = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK,
                     cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE)

    model_1.load_param(cfg.MODEL.init_1)
    model_2.load_param(cfg.MODEL.init_2)
    model_1_ema.load_param(cfg.MODEL.init_1)
    model_2_ema.load_param(cfg.MODEL.init_2)

    for param in model_1_ema.parameters():
        param.detach_()
    for param in model_2_ema.parameters():
        param.detach_()

    return model_1, model_2, model_1_ema, model_2_ema


def create_optimizer(cfg, model_1, model_2):
    optimizer_1 = make_optimizer(cfg, model_1)
    optimizer_2 = make_optimizer(cfg, model_2)

    # path_to_optimizer = cfg.MODEL.init_1.replace('model', 'optimizer')
    # print('Path to the checkpoint of optimizer1:', path_to_optimizer)
    # optimizer_1.load_state_dict(torch.load(path_to_optimizer))
    #
    # path_to_optimizer = cfg.MODEL.init_2.replace('model', 'optimizer')
    # print('Path to the checkpoint of optimizer2:', path_to_optimizer)
    # optimizer_2.load_state_dict(torch.load(path_to_optimizer))

    return optimizer_1, optimizer_2


def train(cfg):
    # prepare dataset
    train_loader, val_loader, num_query, num_classes = make_data_loader_mmt(cfg)

    # prepare model
    model_1, model_2, model_1_ema, model_2_ema = create_model(cfg, num_classes)

    print('Train without center loss, the loss type is', cfg.MODEL.METRIC_LOSS_TYPE)

    start_epoch = eval(cfg.MODEL.init_1.split('/')[-1].split('.')[0].split('_')[-1])
    print('Start epoch:', start_epoch)
    optimizer_1, optimizer_2 = create_optimizer(cfg, model_1, model_2)
    scheduler = WarmupMultiStepLR(optimizer_1, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                  cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD, start_epoch)

    loss_func = make_loss_mmt(cfg, num_classes)     # modified by gu

    do_train_mmt(
        cfg,
        model_1,
        model_2,
        model_1_ema,
        model_2_ema,
        train_loader,
        val_loader,
        optimizer_1,
        optimizer_2,
        scheduler,      # modify for using self trained model
        loss_func,
        num_query,
        start_epoch     # add for using self trained model
    )


def main():
    setup_seed(0)

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
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

    output_dir = os.path.join(cfg.OUTPUT_DIR, datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(output_dir + '/code_backup')

    os.system('cp -r /home/yhl/project/reid-strong-baseline/ ' + output_dir + '/code_backup')
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
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID    # new add by gu
    # cudnn.benchmark = True
    train(cfg)


if __name__ == '__main__':
    main()
