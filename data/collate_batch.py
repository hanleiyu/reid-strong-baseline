# encoding: utf-8

import torch


def train_collate_fn(batch):
    imgs, pids, _, _, = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids


def train_collate_fn_mmt(batch):
    imgs, img2s, pids, _, _, = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), torch.stack(img2s, dim=0), pids


def val_collate_fn(batch):
    imgs, pids, clothids, camids= zip(*batch)
    return torch.stack(imgs, dim=0), pids, clothids, camids


def val_collate_fn_mmt(batch):
    imgs, pids, clothids, camids= zip(*batch)
    return torch.stack(imgs, dim=0), pids, clothids, camids


def part_train_collate_fn(batch):
    imgs, pids, clothids, camids, pc = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    clothids = torch.tensor(clothids, dtype=torch.int64)
    # camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, clothids, camids, pc


def part_val_collate_fn(batch):
    imgs, pids, clothids, camids, _ = zip(*batch)
    return torch.stack(imgs, dim=0), pids, clothids, camids
