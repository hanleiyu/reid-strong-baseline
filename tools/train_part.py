# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
import torch
import numpy as np
from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from data import make_data_loader_part
from engine.trainer import do_train_part, do_train_with_center_part
from modeling import build_part_model
from layers import make_loss, make_loss_with_center
from solver import make_optimizer, make_optimizer_with_center, WarmupMultiStepLR

from utils.logger import setup_logger
import random
import datetime
from torch.backends import cudnn

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def train(cfg):
    # prepare dataset
    train_loader, val_loader, num_query, num_classes = make_data_loader_part(cfg)

    # prepare model
    model = build_part_model(cfg, num_classes)

    if cfg.MODEL.IF_WITH_CENTER == 'no':
        print('Train without center loss, the loss type is', cfg.MODEL.METRIC_LOSS_TYPE)
        if cfg.MODEL.IF_UNCENTAINTY == 'on':
            log_var = torch.zeros(4, requires_grad=True)
            optimizer = make_optimizer(cfg, model, log_var)
        else:
            optimizer = make_optimizer(cfg, model)
        # scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
        #                               cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

        loss_func = make_loss(cfg, num_classes)     # modified by gu

        # Add for using self trained model
        if cfg.MODEL.PRETRAIN_CHOICE == 'self':
            start_epoch = eval(cfg.MODEL.PRETRAIN_PATH.split('/')[-1].split('.')[0].split('_')[-1])
            print('Start epoch:', start_epoch)
            path_to_optimizer = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer')
            print('Path to the checkpoint of optimizer:', path_to_optimizer)
            model.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_PATH))
            optimizer.load_state_dict(torch.load(path_to_optimizer))
            scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                          cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD, start_epoch)
        elif cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
            start_epoch = 0
            scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                          cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
        else:
            print('Only support pretrain_choice for imagenet and self, but got {}'.format(cfg.MODEL.PRETRAIN_CHOICE))

        arguments = {}
        if cfg.MODEL.IF_UNCENTAINTY == 'on':
            do_train_part(
                cfg,
                model,
                train_loader,
                val_loader,
                optimizer,
                scheduler,      # modify for using self trained model
                loss_func,
                num_query,
                start_epoch,     # add for using self trained model
                log_var
            )
        else:
            do_train_part(
                cfg,
                model,
                train_loader,
                val_loader,
                optimizer,
                scheduler,      # modify for using self trained model
                loss_func,
                num_query,
                start_epoch    # add for using self trained model
            )
    elif cfg.MODEL.IF_WITH_CENTER == 'yes':
        print('Train with center loss, the loss type is', cfg.MODEL.METRIC_LOSS_TYPE)
        log_var = torch.zeros(4, requires_grad=True)

        # loss_func, center_criterion1, center_criterion2, center_criterion3 = make_loss_with_center(
        #     cfg, num_classes)
        # optimizer, optimizer_center1, optimizer_center2, optimizer_center3 = \
        #     make_optimizer_with_center(cfg, model, center_criterion1, center_criterion2, center_criterion3, log_var)

        loss_func, center_criterion1, center_criterion2, center_criterion3, center_criterion4 = make_loss_with_center(cfg, num_classes)
        optimizer, optimizer_center1, optimizer_center2, optimizer_center3, optimizer_center4 = \
            make_optimizer_with_center(cfg, model, center_criterion1, center_criterion2, center_criterion3, center_criterion4, log_var)

        # Add for using self trained model
        if cfg.MODEL.PRETRAIN_CHOICE == 'self':
            start_epoch = eval(cfg.MODEL.PRETRAIN_PATH.split('/')[-1].split('.')[0].split('_')[-1])
            print('Start epoch:', start_epoch)
            path_to_optimizer = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer')
            print('Path to the checkpoint of optimizer:', path_to_optimizer)
            path_to_center_param1 = cfg.MODEL.PRETRAIN_PATH.replace('model', 'center_param1')
            path_to_center_param2 = cfg.MODEL.PRETRAIN_PATH.replace('model', 'center_param2')
            path_to_center_param3 = cfg.MODEL.PRETRAIN_PATH.replace('model', 'center_param3')
            path_to_center_param4 = cfg.MODEL.PRETRAIN_PATH.replace('model', 'center_param4')
            print('Path to the checkpoint of center_param:', path_to_center_param1)
            path_to_optimizer_center1 = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer_center1')
            path_to_optimizer_center2 = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer_center2')
            path_to_optimizer_center3 = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer_center3')
            path_to_optimizer_center4 = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer_center4')
            print('Path to the checkpoint of optimizer_center:', path_to_optimizer_center1)
            model.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_PATH))
            optimizer.load_state_dict(torch.load(path_to_optimizer))
            center_criterion1.load_state_dict(torch.load(path_to_center_param1))
            center_criterion2.load_state_dict(torch.load(path_to_center_param2))
            center_criterion3.load_state_dict(torch.load(path_to_center_param3))
            center_criterion4.load_state_dict(torch.load(path_to_center_param4))
            optimizer_center1.load_state_dict(torch.load(path_to_optimizer_center1))
            optimizer_center2.load_state_dict(torch.load(path_to_optimizer_center2))
            optimizer_center3.load_state_dict(torch.load(path_to_optimizer_center3))
            optimizer_center4.load_state_dict(torch.load(path_to_optimizer_center4))
            scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                          cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD, start_epoch)
        elif cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
            start_epoch = 0
            scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                          cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
        else:
            print('Only support pretrain_choice for imagenet and self, but got {}'.format(cfg.MODEL.PRETRAIN_CHOICE))

        do_train_with_center_part(
            cfg,
            model,
            center_criterion1,
            center_criterion2,
            center_criterion3,
            center_criterion4,
            train_loader,
            val_loader,
            optimizer,
            optimizer_center1,
            optimizer_center2,
            optimizer_center3,
            optimizer_center4,
            scheduler,  # modify for using self trained model
            loss_func,
            num_query,
            start_epoch,
            log_var# add for using self trained model
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
    cudnn.enabled = False
    train(cfg)


if __name__ == '__main__':
    main()
