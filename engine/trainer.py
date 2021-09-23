# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging

import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage

from utils.reid_metric import R1_mAP
import numpy
import copy

global ITER
ITER = 0


def _update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def create_supervised_trainer(model, optimizer, loss_fn,
                              device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        img, target = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        score, feat = model(img)
        loss = loss_fn(score, feat, target)
        loss.backward()
        optimizer.step()
        # compute acc
        acc = (score.max(1)[1] == target).float().mean()
        return loss.item(), acc.item()

    return Engine(_update)


def part_trainer(model, optimizer, loss_fn, log_var=None, device=None):
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        img, target, _, _, pc = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        score, feat = model(img, pc)
        loss_part = [0 for _ in range(len(feat))]
        acc = [0 for _ in range(len(feat))]
        ten = [torch.tensor(1.0).cuda() for _ in range(len(feat))]
        for i in range(len(feat)):
            if log_var is not None:
                loss_part[i] = loss_fn(score[i], feat[i], target, log_var[i])[0]
                # if i == len(feat) - 1:
                #     loss_part[i] = loss_fn(score[i], feat[i], target, log_var[i])[0]
                # else:
                #     loss_part[i] = loss_fn(score[i], feat[i], target, log_var[i])[1]
            else:
                loss_part[i] = loss_fn(score[i], feat[i], target)[0]
                # if i == len(feat) - 1:
                #     loss_part[i] = loss_fn(score[i], feat[i], target)[0]
                # else:
                #     loss_part[i] = loss_fn(score[i], feat[i], target)[1]
            acc[i] = (score[i].max(1)[1] == target).float().mean()
        torch.autograd.backward(loss_part, ten)
        optimizer.step()
        return loss_part, acc

    return Engine(_update)


def part_trainer_with_center(model, center_criterion1, center_criterion2, center_criterion3,
                             optimizer,
                             optimizer_center1, optimizer_center2, optimizer_center3,
                             loss_fn, cetner_loss_weight, device=None, log_var=None):
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        optimizer_center1.zero_grad()
        optimizer_center2.zero_grad()
        optimizer_center3.zero_grad()

        img, target, _, _, pc = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        score, feat = model(img, pc)

        loss_part = [0 for _ in range(len(feat))]
        acc = [0 for _ in range(len(feat))]
        ten = [torch.tensor(1.0).cuda() for _ in range(len(feat))]
        for i in range(len(feat)):
            if log_var is not None:
                loss_part[i] = loss_fn(score[i], feat[i], target, log_var[i], i)
                # if i == len(feat) - 1:
                #     loss_part[i] = loss_fn(score[i], feat[i], target, log_var[i])[0]
                # else:
                #     loss_part[i] = loss_fn(score[i], feat[i], target, log_var[i])[1]
            else:
                loss_part[i] = loss_fn(score[i], feat[i], target)[0]
                # if i == len(feat) - 1:
                #     loss_part[i] = loss_fn(score[i], feat[i], target)[0]
                # else:
                #     loss_part[i] = loss_fn(score[i], feat[i], target)[1]
            acc[i] = (score[i].max(1)[1] == target).float().mean()
        torch.autograd.backward(loss_part, ten)
        optimizer.step()
        for param in center_criterion1.parameters():
            param.grad.data *= (1. / cetner_loss_weight)
        optimizer_center1.step()
        for param in center_criterion2.parameters():
            param.grad.data *= (1. / cetner_loss_weight)
        optimizer_center2.step()
        for param in center_criterion3.parameters():
            param.grad.data *= (1. / cetner_loss_weight)
        optimizer_center3.step()
        return loss_part, acc

    return Engine(_update)


def create_supervised_trainer_with_center(model, center_criterion, optimizer, optimizer_center, loss_fn,
                                          cetner_loss_weight,
                                          device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        optimizer_center.zero_grad()
        img, target = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        score, feat = model(img)
        loss = loss_fn(score, feat, target)
        # print("Total loss is {}, center loss is {}".format(loss, center_criterion(feat, target)))
        loss.backward()
        optimizer.step()
        for param in center_criterion.parameters():
            param.grad.data *= (1. / cetner_loss_weight)
        optimizer_center.step()

        # compute acc
        acc = (score.max(1)[1] == target).float().mean()
        return loss.item(), acc.item()

    return Engine(_update)


def create_supervised_trainer_mmt(model_1, model_2, model_1_ema, model_2_ema,
                                  optimizer_1, optimizer_2, loss_fn, device=None):
    if device:
        if torch.cuda.device_count() > 1:
            model_1 = nn.DataParallel(model_1)
            model_2 = nn.DataParallel(model_2)
            model_1_ema = nn.DataParallel(model_1_ema)
            model_2_ema = nn.DataParallel(model_2_ema)
        model_1.to(device)
        model_2.to(device)
        model_1_ema.to(device)
        model_2_ema.to(device)

    def _update(engine, batch):
        model_1.train()
        model_2.train()
        model_1_ema.train()
        model_2_ema.train()

        # for k,v in model_1.state_dict().items():
        #     if v.equal(model_1_ema.state_dict()[k]):
        #         print("yes")
        #     else:
        #         print(k)

        optimizer_1.zero_grad()
        optimizer_2.zero_grad()
        img, img2, target = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        img2 = img2.to(device) if torch.cuda.device_count() >= 1 else img2
        target = target.to(device) if torch.cuda.device_count() >= 1 else target

        score_1, feat_1 = model_1(img)
        score_2, feat_2 = model_2(img2)

        score_1_ema, feat_1_ema = model_1_ema(img)
        score_2_ema, feat_2_ema = model_2_ema(img2)

        loss, all_loss = loss_fn(score_1, score_2, feat_1, feat_2, score_1_ema, score_2_ema,
                       feat_1_ema, feat_2_ema, target)
        loss.backward()
        optimizer_1.step()
        optimizer_2.step()

        #len(train_loader) = 346
        _update_ema_variables(model_1, model_1_ema, 0.999, engine.state.epoch * 346 + ITER)
        _update_ema_variables(model_2, model_2_ema, 0.999, engine.state.epoch * 346 + ITER)

        # compute acc
        acc_1 = (score_1_ema.max(1)[1] == target).float().mean()
        acc_2 = (score_2_ema.max(1)[1] == target).float().mean()
        acc_3 = (score_1.max(1)[1] == target).float().mean()
        acc_4 = (score_2.max(1)[1] == target).float().mean()
        return loss.item(), all_loss, acc_1.item(), acc_2.item()

    return Engine(_update)


def create_supervised_evaluator(model, metrics,
                                device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, clothids, camids = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            feat = model(data)
            return feat, pids, clothids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def part_evaluator(model, metrics, device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, clothids, camids, pc = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            # feat = model(data)
            feat = model(data, pc)
            return feat, pids, clothids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        num_query,
        start_epoch
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    evaluator = create_supervised_evaluator(model, metrics={
        'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)

    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=10, require_empty=False)
    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model, 'optimizer': optimizer})
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1
        if ITER % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, "
                        "Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, ITER, len(train_loader),
                                engine.state.metrics['avg_loss'],
                                engine.state.metrics['avg_acc'],
                                scheduler.get_lr()[0]))
        if len(train_loader) == ITER:
            ITER = 0

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    trainer.run(train_loader, max_epochs=epochs)


def do_train_part(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        num_query,
        start_epoch,
        log_var=None
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")
    if cfg.MODEL.IF_UNCENTAINTY == 'on':
        trainer = part_trainer(model, optimizer, loss_fn, log_var, device=device)
    else:
        trainer = part_trainer(model, optimizer, loss_fn, device=device)
    evaluator = part_evaluator(model, metrics={
        'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)

    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=10, require_empty=False)
    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model, 'optimizer': optimizer})
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer

    # RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    # RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')

    RunningAverage(output_transform=lambda x: x[0][0]).attach(trainer, 'avg_loss1')
    RunningAverage(output_transform=lambda x: x[0][1]).attach(trainer, 'avg_loss2')
    RunningAverage(output_transform=lambda x: x[0][2]).attach(trainer, 'avg_loss3')
    # RunningAverage(output_transform=lambda x: x[0][3]).attach(trainer, 'avg_loss4')
    # RunningAverage(output_transform=lambda x: x[0][4]).attach(trainer, 'avg_loss5')
    # RunningAverage(output_transform=lambda x: x[0][5]).attach(trainer, 'avg_loss6')
    # # RunningAverage(output_transform=lambda x: x[0][6]).attach(trainer, 'avg_loss7')
    RunningAverage(output_transform=lambda x: x[1][0]).attach(trainer, 'avg_acc1')
    RunningAverage(output_transform=lambda x: x[1][1]).attach(trainer, 'avg_acc2')
    RunningAverage(output_transform=lambda x: x[1][2]).attach(trainer, 'avg_acc3')

    # RunningAverage(output_transform=lambda x: x[1][3]).attach(trainer, 'avg_acc4')
    # RunningAverage(output_transform=lambda x: x[1][4]).attach(trainer, 'avg_acc5')
    # RunningAverage(output_transform=lambda x: x[1][5]).attach(trainer, 'avg_acc6')
    # # RunningAverage(output_transform=lambda x: x[1][6]).attach(trainer, 'avg_acc7')

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1
        # if ITER % log_period == 0:
        #     logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, "
        #                 "Acc: {:.3f},  Base Lr: {:.2e}"
        #                 .format(engine.state.epoch, ITER, len(train_loader),
        #                         engine.state.metrics['avg_loss'],
        #                         engine.state.metrics['avg_acc'],
        #                         scheduler.get_lr()[0]))
        #
        # if ITER % log_period == 0:
        #     logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Loss1: {:.3f}, "
        #                 "Acc: {:.3f}, Acc1: {:.3f},  Base Lr: {:.2e}, var: {:.3f}, var1: {:.3f}"
        #                 .format(engine.state.epoch, ITER, len(train_loader),
        #                         engine.state.metrics['avg_loss1'], engine.state.metrics['avg_loss2'],
        #                         engine.state.metrics['avg_acc1'], engine.state.metrics['avg_acc2'],
        #                         scheduler.get_lr()[0], log_var[0], log_var[1]))
        if ITER % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Loss1: {:.3f}, Loss2: {:.3f},"
                        "Acc: {:.3f}, Acc1: {:.3f}, Acc2: {:.3f}, Base Lr: {:.2e}, var: {:.3f}, var1: {:.3f}, var2: {:.3f}"
                        .format(engine.state.epoch, ITER, len(train_loader),
                                engine.state.metrics['avg_loss1'], engine.state.metrics['avg_loss2'],
                                engine.state.metrics['avg_loss3'],
                                engine.state.metrics['avg_acc1'], engine.state.metrics['avg_acc2'],
                                engine.state.metrics['avg_acc3'],
                                scheduler.get_lr()[0], log_var[0], log_var[1], log_var[2]))
        # if ITER % log_period == 0:
        #     logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Loss1: {:.3f}, Loss2: {:.3f}, Loss3: {:.3f},"
        #                 "Acc: {:.3f}, Acc1: {:.3f}, Acc2: {:.3f}, Acc3: {:.3f}, Base Lr: {:.2e}, var: {:.3f}, "
        #                 "var1: {:.3f}, var2: {:.3f}, var3: {:.3f}"
        #                 .format(engine.state.epoch, ITER, len(train_loader),
        #                         engine.state.metrics['avg_loss1'], engine.state.metrics['avg_loss2'],
        #                         engine.state.metrics['avg_loss3'], engine.state.metrics['avg_loss4'],
        #                         engine.state.metrics['avg_acc1'], engine.state.metrics['avg_acc2'],
        #                         engine.state.metrics['avg_acc3'], engine.state.metrics['avg_acc4'],
        #                         scheduler.get_lr()[0], log_var[0], log_var[1], log_var[2], log_var[3]))
        # if ITER % log_period == 0:
        #     logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Loss1: {:.3f}, Loss2: {:.3f}, Loss3: {:.3f}, Loss4: {:.3f},"
        #                 "Acc: {:.3f}, Acc1: {:.3f}, Acc2: {:.3f}, Acc3: {:.3f}, Acc4: {:.3f},"
        #                 "Base Lr: {:.2e}, var: {:.3f}, var1: {:.3f}, var2: {:.3f}, var3: {:.3f}, var4: {:.3f}"
        #                 .format(engine.state.epoch, ITER, len(train_loader),
        #                         engine.state.metrics['avg_loss'], engine.state.metrics['avg_loss1'],
        #                         engine.state.metrics['avg_loss2'], engine.state.metrics['avg_loss3'],
        #                         engine.state.metrics['avg_loss4'],
        #                         engine.state.metrics['avg_acc'], engine.state.metrics['avg_acc1'],
        #                         engine.state.metrics['avg_acc2'], engine.state.metrics['avg_acc3'],
        #                         engine.state.metrics['avg_acc4'],
        #                         scheduler.get_lr()[0], log_var[0], log_var[1], log_var[2], log_var[3], log_var[4]))
        # if ITER % log_period == 0:
        #     logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Loss1: {:.3f}, Loss2: {:.3f}, Loss3: {:.3f}, Loss4: {:.3f}, Loss5: {:.3f},"
        #                 "Acc: {:.3f}, Acc1: {:.3f}, Acc2: {:.3f}, Acc3: {:.3f}, Acc4: {:.3f},  Acc5: {:.3f},"
        #                 "Base Lr: {:.2e}, var: {:.3f}, var1: {:.3f}, var2: {:.3f}, var3: {:.3f}, var4: {:.3f}, var5: {:.3f}"
        #                 .format(engine.state.epoch, ITER, len(train_loader),
        #                          engine.state.metrics['avg_loss1'],
        #                         engine.state.metrics['avg_loss2'], engine.state.metrics['avg_loss3'],
        #                         engine.state.metrics['avg_loss4'], engine.state.metrics['avg_loss5'],engine.state.metrics['avg_loss6'],
        #                          engine.state.metrics['avg_acc1'],
        #                         engine.state.metrics['avg_acc2'], engine.state.metrics['avg_acc3'],
        #                         engine.state.metrics['avg_acc4'], engine.state.metrics['avg_acc5'],engine.state.metrics['avg_acc6'],
        #                         scheduler.get_lr()[0], log_var[0], log_var[1], log_var[2], log_var[3]
        #                                         , log_var[4], log_var[5]))
        # if ITER % log_period == 0:
        #     logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Loss1: {:.3f}, Loss2: {:.3f}, "
        #                 "Loss3: {:.3f}, Loss4: {:.3f}, Loss5: {:.3f}, Loss6: {:.3f},"
        #                 "Acc: {:.3f}, Acc1: {:.3f}, Acc2: {:.3f}, Acc3: {:.3f}, Acc4: {:.3f},"
        #                 " Acc5: {:.3f}, Acc6: {:.3f}, Base Lr: {:.2e}, var: {:.3f}, "
        #                         "var1: {:.3f}, var2: {:.3f}, var3: {:.3f}, var4: {:.3f}, var5: {:.3f}, "
        #                         "var6: {:.3f}"
        #                 .format(engine.state.epoch, ITER, len(train_loader),
        #                         engine.state.metrics['avg_loss'], engine.state.metrics['avg_loss1'],
        #                         engine.state.metrics['avg_loss2'], engine.state.metrics['avg_loss3'],
        #                         engine.state.metrics['avg_loss4'], engine.state.metrics['avg_loss5'],
        #                         engine.state.metrics['avg_loss6'],
        #                         engine.state.metrics['avg_acc'], engine.state.metrics['avg_acc1'],
        #                         engine.state.metrics['avg_acc2'], engine.state.metrics['avg_acc3'],
        #                         engine.state.metrics['avg_acc4'], engine.state.metrics['avg_acc5'],
        #                         engine.state.metrics['avg_acc6'],
        #                         scheduler.get_lr()[0], log_var[0], log_var[1], log_var[2], log_var[3]
        #                                 , log_var[4], log_var[5], log_var[6]))
        # if ITER % log_period == 0:
        #     logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Loss1: {:.3f}, Loss2: {:.3f}, "
        #                 "Loss3: {:.3f}, Loss4: {:.3f}, Loss5: {:.3f}, Loss6: {:.3f}, Loss7: {:.3f},"
        #                 "Acc: {:.3f}, Acc1: {:.3f}, Acc2: {:.3f}, Acc3: {:.3f}, Acc4: {:.3f},"
        #                 " Acc5: {:.3f}, Acc6: {:.3f}, Acc7: {:.3f}, Base Lr: {:.2e}, var: {:.3f}, "
        #                 "var1: {:.3f}, var2: {:.3f}, var3: {:.3f}, var4: {:.3f}, var5: {:.3f}, "
        #                 "var6: {:.3f}, var7: {:.3f}"
        #                 .format(engine.state.epoch, ITER, len(train_loader),
        #                         engine.state.metrics['avg_loss'], engine.state.metrics['avg_loss1'],
        #                         engine.state.metrics['avg_loss2'], engine.state.metrics['avg_loss3'],
        #                         engine.state.metrics['avg_loss4'], engine.state.metrics['avg_loss5'],
        #                         engine.state.metrics['avg_loss6'], engine.state.metrics['avg_loss7'],
        #                         engine.state.metrics['avg_acc'], engine.state.metrics['avg_acc1'],
        #                         engine.state.metrics['avg_acc2'], engine.state.metrics['avg_acc3'],
        #                         engine.state.metrics['avg_acc4'], engine.state.metrics['avg_acc5'],
        #                         engine.state.metrics['avg_acc6'], engine.state.metrics['avg_acc7'],
        #                         scheduler.get_lr()[0], log_var[0], log_var[1], log_var[2], log_var[3]
        #                         , log_var[4], log_var[5], log_var[6], log_var[7]))
        # if ITER % log_period == 0:
        #     logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Loss1: {:.3f}, "
        #                 "Acc: {:.3f}, Acc1: {:.3f},  Base Lr: {:.2e}"
        #                 .format(engine.state.epoch, ITER, len(train_loader),
        #                         engine.state.metrics['avg_loss'], engine.state.metrics['avg_loss1'],
        #                         engine.state.metrics['avg_acc'], engine.state.metrics['avg_acc1'],
        #                         scheduler.get_lr()[0]))
        # if ITER % log_period == 0:
        #     logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Loss1: {:.3f}, Loss2: {:.3f},"
        #                 "Acc: {:.3f}, Acc1: {:.3f}, Acc2: {:.3f}, Base Lr: {:.2e}"
        #                 .format(engine.state.epoch, ITER, len(train_loader),
        #                         engine.state.metrics['avg_loss'], engine.state.metrics['avg_loss1'],
        #                         engine.state.metrics['avg_loss2'],
        #                         engine.state.metrics['avg_acc'], engine.state.metrics['avg_acc1'],
        #                         engine.state.metrics['avg_acc2'],
        #                         scheduler.get_lr()[0]))
        # if ITER % log_period == 0:
        #     logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Loss1: {:.3f}, Loss2: {:.3f}, Loss3: {:.3f}, Loss4: {:.3f},"
        #                 "Acc: {:.3f}, Acc1: {:.3f}, Acc2: {:.3f}, Acc3: {:.3f}, Acc4: {:.3f},"
        #                 "Base Lr: {:.2e}"
        #                 .format(engine.state.epoch, ITER, len(train_loader),
        #                         engine.state.metrics['avg_loss'], engine.state.metrics['avg_loss1'],
        #                         engine.state.metrics['avg_loss2'], engine.state.metrics['avg_loss3'],
        #                         engine.state.metrics['avg_loss4'],
        #                         engine.state.metrics['avg_acc'], engine.state.metrics['avg_acc1'],
        #                         engine.state.metrics['avg_acc2'], engine.state.metrics['avg_acc3'],
        #                         engine.state.metrics['avg_acc4'],
        #                         scheduler.get_lr()[0]))
        # if ITER % log_period == 0:
        #     logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Loss1: {:.3f}, Loss2: {:.3f}, Loss3: {:.3f}, Loss4: {:.3f}, Loss5: {:.3f},"
        #                 "Acc: {:.3f}, Acc1: {:.3f}, Acc2: {:.3f}, Acc3: {:.3f}, Acc4: {:.3f},  Acc5: {:.3f},"
        #                 "Base Lr: {:.2e}"
        #                 .format(engine.state.epoch, ITER, len(train_loader),
        #                         engine.state.metrics['avg_loss'], engine.state.metrics['avg_loss1'],
        #                         engine.state.metrics['avg_loss2'], engine.state.metrics['avg_loss3'],
        #                         engine.state.metrics['avg_loss4'], engine.state.metrics['avg_loss5'],
        #                         engine.state.metrics['avg_acc'], engine.state.metrics['avg_acc1'],
        #                         engine.state.metrics['avg_acc2'], engine.state.metrics['avg_acc3'],
        #                         engine.state.metrics['avg_acc4'], engine.state.metrics['avg_acc5'],
        #                         scheduler.get_lr()[0]))
        # if ITER % log_period == 0:
        #     logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Loss1: {:.3f}, Loss2: {:.3f}, "
        #                 "Loss3: {:.3f}, Loss4: {:.3f}, Loss5: {:.3f}, Loss6: {:.3f},"
        #                 "Acc: {:.3f}, Acc1: {:.3f}, Acc2: {:.3f}, Acc3: {:.3f}, Acc4: {:.3f},"
        #                 " Acc5: {:.3f}, Acc6: {:.3f}, Base Lr: {:.2e}"
        #                 .format(engine.state.epoch, ITER, len(train_loader),
        #                         engine.state.metrics['avg_loss'], engine.state.metrics['avg_loss1'],
        #                         engine.state.metrics['avg_loss2'], engine.state.metrics['avg_loss3'],
        #                         engine.state.metrics['avg_loss4'], engine.state.metrics['avg_loss5'],
        #                         engine.state.metrics['avg_loss6'],
        #                         engine.state.metrics['avg_acc'], engine.state.metrics['avg_acc1'],
        #                         engine.state.metrics['avg_acc2'], engine.state.metrics['avg_acc3'],
        #                         engine.state.metrics['avg_acc4'], engine.state.metrics['avg_acc5'],
        #                         engine.state.metrics['avg_acc6'],
        #                         scheduler.get_lr()[0]))
        # if ITER % log_period == 0:
        #     logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Loss1: {:.3f}, Loss2: {:.3f}, "
        #                 "Loss3: {:.3f}, Loss4: {:.3f}, Loss5: {:.3f}, Loss6: {:.3f}, Loss7: {:.3f},"
        #                 "Acc: {:.3f}, Acc1: {:.3f}, Acc2: {:.3f}, Acc3: {:.3f}, Acc4: {:.3f},"
        #                 " Acc5: {:.3f}, Acc6: {:.3f}, Acc7: {:.3f}, Base Lr: {:.2e}"
        #                 .format(engine.state.epoch, ITER, len(train_loader),
        #                         engine.state.metrics['avg_loss'], engine.state.metrics['avg_loss1'],
        #                         engine.state.metrics['avg_loss2'], engine.state.metrics['avg_loss3'],
        #                         engine.state.metrics['avg_loss4'], engine.state.metrics['avg_loss5'],
        #                         engine.state.metrics['avg_loss6'], engine.state.metrics['avg_loss7'],
        #                         engine.state.metrics['avg_acc'], engine.state.metrics['avg_acc1'],
        #                         engine.state.metrics['avg_acc2'], engine.state.metrics['avg_acc3'],
        #                         engine.state.metrics['avg_acc4'], engine.state.metrics['avg_acc5'],
        #                         engine.state.metrics['avg_acc6'], engine.state.metrics['avg_acc7'],
        #                         scheduler.get_lr()[0]))
        if len(train_loader) == ITER:
            ITER = 0

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    torch.multiprocessing.set_sharing_strategy('file_system')
    trainer.run(train_loader, max_epochs=epochs)


def do_train_with_center(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_fn,
        num_query,
        start_epoch
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")
    trainer = create_supervised_trainer_with_center(model, center_criterion, optimizer, optimizer_center, loss_fn,
                                                    cfg.SOLVER.CENTER_LOSS_WEIGHT, device=device)
    evaluator = create_supervised_evaluator(model, metrics={
        'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=10, require_empty=False)
    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
                                                                     'optimizer': optimizer,
                                                                     'center_param': center_criterion,
                                                                     'optimizer_center': optimizer_center})

    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1

        if ITER % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, ITER, len(train_loader),
                                engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc'],
                                scheduler.get_lr()[0]))
        if len(train_loader) == ITER:
            ITER = 0

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    trainer.run(train_loader, max_epochs=epochs)


def do_train_with_center_part(
        cfg,
        model,
        center_criterion1,
        center_criterion2,
        center_criterion3,

        train_loader,
        val_loader,
        optimizer,
        optimizer_center1,
        optimizer_center2,
        optimizer_center3,

        scheduler,
        loss_fn,
        num_query,
        start_epoch,
        log_var
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")
    trainer = part_trainer_with_center(model, center_criterion1, center_criterion2, center_criterion3,
                                       optimizer, optimizer_center1, optimizer_center2, optimizer_center3,
                                       loss_fn, cfg.SOLVER.CENTER_LOSS_WEIGHT, device=device, log_var=log_var)
    evaluator = part_evaluator(model, metrics={
        'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=10, require_empty=False)
    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
                                                                     'optimizer': optimizer,
                                                                     'center_param1': center_criterion1,
                                                                     'center_param2': center_criterion2,
                                                                     'center_param3': center_criterion3,

                                                                     'optimizer_center1': optimizer_center1,
                                                                     'optimizer_center2': optimizer_center2,
                                                                     'optimizer_center3': optimizer_center3,
                                                                     })

    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0][0]).attach(trainer, 'avg_loss1')
    RunningAverage(output_transform=lambda x: x[0][1]).attach(trainer, 'avg_loss2')
    RunningAverage(output_transform=lambda x: x[0][2]).attach(trainer, 'avg_loss3')
    # RunningAverage(output_transform=lambda x: x[0][3]).attach(trainer, 'avg_loss4')
    # RunningAverage(output_transform=lambda x: x[0][4]).attach(trainer, 'avg_loss5')
    # RunningAverage(output_transform=lambda x: x[0][5]).attach(trainer, 'avg_loss6')
    # # RunningAverage(output_transform=lambda x: x[0][6]).attach(trainer, 'avg_loss7')
    RunningAverage(output_transform=lambda x: x[1][0]).attach(trainer, 'avg_acc1')
    RunningAverage(output_transform=lambda x: x[1][1]).attach(trainer, 'avg_acc2')
    RunningAverage(output_transform=lambda x: x[1][2]).attach(trainer, 'avg_acc3')

    # RunningAverage(output_transform=lambda x: x[1][3]).attach(trainer, 'avg_acc4')
    # RunningAverage(output_transform=lambda x: x[1][4]).attach(trainer, 'avg_acc5')
    # RunningAverage(output_transform=lambda x: x[1][5]).attach(trainer, 'avg_acc6')
    # # RunningAverage(output_transform=lambda x: x[1][6]).attach(trainer, 'avg_acc7')

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1

        if ITER % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Loss1: {:.3f}, Loss2: {:.3f},"
                        "Acc: {:.3f}, Acc1: {:.3f}, Acc2: {:.3f}, Base Lr: {:.2e}, "
                        "var: {:.3f}, var1: {:.3f}, var2: {:.3f}"
                        .format(engine.state.epoch, ITER, len(train_loader),
                                engine.state.metrics['avg_loss1'], engine.state.metrics['avg_loss2'],
                                engine.state.metrics['avg_loss3'],
                                engine.state.metrics['avg_acc1'], engine.state.metrics['avg_acc2'],
                                engine.state.metrics['avg_acc3'],
                                scheduler.get_lr()[0], log_var[0], log_var[1], log_var[2], ))
        if len(train_loader) == ITER:
            ITER = 0

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    trainer.run(train_loader, max_epochs=epochs)


def do_train_mmt(
        cfg,
        model_1,
        model_2,
        model_1_ema,
        model_2_ema,
        train_loader,
        val_loader,
        optimizer_1,
        optimizer_2,
        scheduler,
        loss_fn,
        num_query,
        start_epoch
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")
    trainer = create_supervised_trainer_mmt(model_1, model_2, model_1_ema, model_2_ema,
                                            optimizer_1, optimizer_2, loss_fn, device=device)

    evaluator_1_ema = create_supervised_evaluator(model_1_ema, metrics={
        'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    evaluator_2_ema = create_supervised_evaluator(model_2_ema, metrics={
        'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)

    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=10, require_empty=False)
    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model_1_ema': model_1_ema,
                                                                     'model_2_ema': model_2_ema,
                                                                     'optimizer_1': optimizer_1,
                                                                     'optimizer_2': optimizer_2})
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1][0]).attach(trainer, 'loss_tri_1')
    RunningAverage(output_transform=lambda x: x[1][1]).attach(trainer, 'loss_tri_2')
    RunningAverage(output_transform=lambda x: x[1][2]).attach(trainer, 'loss_ce_1')
    RunningAverage(output_transform=lambda x: x[1][3]).attach(trainer, 'loss_ce_2')
    RunningAverage(output_transform=lambda x: x[1][4]).attach(trainer, 'loss_tri_soft')
    RunningAverage(output_transform=lambda x: x[1][5]).attach(trainer, 'loss_ce_soft')
    RunningAverage(output_transform=lambda x: x[2]).attach(trainer, 'avg_acc_1')
    RunningAverage(output_transform=lambda x: x[3]).attach(trainer, 'avg_acc_2')

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1
        if ITER % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, loss_tri_1: {:.3f}, loss_tri_2: {:.3f}, "
                        "loss_ce_1: {:.3f}, loss_ce_2: {:.3f}, loss_tri_soft: {:.3f}, loss_ce_soft: {:.3f}, "
                        "Acc1: {:.3f}, Acc2: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, ITER, len(train_loader),
                                engine.state.metrics['avg_loss'],
                                engine.state.metrics['loss_tri_1'],
                                engine.state.metrics['loss_tri_2'],
                                engine.state.metrics['loss_ce_1'],
                                engine.state.metrics['loss_ce_2'],
                                engine.state.metrics['loss_tri_soft'],
                                engine.state.metrics['loss_ce_soft'],
                                engine.state.metrics['avg_acc_1'],
                                engine.state.metrics['avg_acc_2'],
                                scheduler.get_lr()[0]))
        if len(train_loader) == ITER:
            ITER = 0

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator_1_ema.run(val_loader)
            cmc, mAP = evaluator_1_ema.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("evaluator_1_ema CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

            evaluator_2_ema.run(val_loader)
            cmc, mAP = evaluator_2_ema.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("evaluator_2_ema CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    trainer.run(train_loader, max_epochs=epochs)
