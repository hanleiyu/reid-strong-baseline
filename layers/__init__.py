# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F

from .triplet_loss import *
from .center_loss import CenterLoss
import torch


def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
        if cfg.MODEL.IF_UNCENTAINTY == 'on':
            triplet = TripletLossUncertainty(cfg.SOLVER.MARGIN)
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        if cfg.MODEL.IF_UNCENTAINTY == 'on':
            xent = CrossEntropyLabelSmoothUncertainty(num_classes=num_classes)  # new add by Yu
            print("label smooth on, numclasses:", num_classes)
        else:
            xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
            print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)
    elif cfg.DATALOADER.SAMPLER == 'triplet':
        def loss_func(score, feat, target):
            return triplet(feat, target)[0]
    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target, log_var=None):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if cfg.MODEL.IF_UNCENTAINTY == 'on':
                        return [xent(score, target, log_var) + triplet(feat, target, log_var)[0], xent(score, target, log_var)]
                    else:
                        return xent(score, target) + triplet(feat, target)[0]
                        # return [xent(score, target) + triplet(feat, target)[0], xent(score, target)]
                else:
                    if cfg.MODEL.IF_UNCENTAINTY == 'on':
                        return [torch.exp(-log_var).cuda() * F.cross_entropy(score, target) + log_var.cuda() + triplet(feat, target, log_var)[0], torch.exp(-log_var).cuda() * F.cross_entropy(score, target) + log_var.cuda()]
                    else:
                        # return [F.cross_entropy(score, target) + triplet(feat, target)[0], F.cross_entropy(score, target)]
                        # baseline
                        return F.cross_entropy(score, target) + triplet(feat, target)[0]
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    else:
        print('expected sampler should be softmax, triplet or softmax_triplet, '
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func


def make_loss_with_center(cfg, num_classes):    # modified by gu
    if cfg.MODEL.NAME == 'resnet18' or cfg.MODEL.NAME == 'resnet34':
        feat_dim = 512
    else:
        feat_dim = 2048

    if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
        center_criterion1 = CenterLoss(num_classes=num_classes, feat_dim=2048, use_gpu=True)  # center loss
        center_criterion2 = CenterLoss(num_classes=num_classes, feat_dim=2176, use_gpu=True)  # center loss
        center_criterion3 = CenterLoss(num_classes=num_classes, feat_dim=1024, use_gpu=True)  # center loss
        # center_criterion4 = CenterLoss(num_classes=num_classes, feat_dim=1024, use_gpu=True)  # center loss

    else:
        print('expected METRIC_LOSS_TYPE with center should be center, triplet_center'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)

    # feat_loss = FeatLoss()

    def loss_func(score, feat, target, log_var=None, i=0):
        if i == 0:
            center_criterion = center_criterion1
        elif i == 1:
            center_criterion = center_criterion2
        elif i == 2:
            center_criterion = center_criterion3
        # elif i == 3:
        #     center_criterion = center_criterion4

        if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                return xent(score, target) + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
            else:
                return F.cross_entropy(score, target) + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)

        elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                return xent(score, target) + \
                        triplet(feat, target)[0] + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
            else:
                if cfg.MODEL.IF_UNCENTAINTY == 'on':
                    return torch.exp(-log_var).cuda() * \
                           (F.cross_entropy(score, target) + triplet(feat, target)[0] +
                            cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)) \
                           + log_var.cuda()
                else:
                    return F.cross_entropy(score, target) + \
                            triplet(feat, target)[0] + \
                            cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)

        else:
            print('expected METRIC_LOSS_TYPE with center should be center, triplet_center'
                  'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    return loss_func, center_criterion1, center_criterion2, center_criterion3
    # return loss_func, center_criterion1, center_criterion2, center_criterion3, center_criterion4


def make_loss_mmt(cfg, num_classes, ce_soft_weight=0.5, tri_soft_weight=0.8):
    triplet = TripletLoss(cfg.SOLVER.MARGIN)
    triplet_soft = SoftTripletLoss(cfg.SOLVER.MARGIN)
    ce_soft = SoftEntropy()

    def loss_func(score_1, score_2, feat_1, feat_2, score_1_ema, score_2_ema,
                  feat_1_ema, feat_2_ema, target):
        loss_tri_1 = triplet(feat_1, target)[0]
        loss_tri_2 = triplet(feat_2, target)[0]

        loss_ce_1 = F.cross_entropy(score_1, target)
        loss_ce_2 = F.cross_entropy(score_2, target)

        loss_tri_soft = triplet_soft(feat_1, feat_2_ema, target) + triplet_soft(feat_2, feat_1_ema, target)
        loss_ce_soft = ce_soft(score_1, score_2_ema) + ce_soft(score_2, score_1_ema)

        loss = (loss_ce_1 + loss_ce_2)*(1-ce_soft_weight) + \
                     (loss_tri_1 + loss_tri_2)*(1-tri_soft_weight) + \
                     loss_ce_soft*ce_soft_weight + loss_tri_soft*tri_soft_weight

        return loss, [loss_tri_1.item(), loss_tri_2.item(), loss_ce_1.item(), loss_ce_2.item(), loss_tri_soft.item(), loss_ce_soft.item()]

    return loss_func
