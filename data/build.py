# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import DataLoader

from .collate_batch import part_train_collate_fn, part_val_collate_fn, train_collate_fn, val_collate_fn
from .datasets import init_dataset, ImageDataset, ImageDatasetPart
from .samplers import RandomIdentitySampler, RandomIdentitySampler_Part
from .transforms import build_transforms
import random
import glob
import os.path as osp
import torch

def make_data_loader(cfg):
    train_transforms = build_transforms(cfg, is_train=True)
    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    if len(cfg.DATASETS.NAMES) == 1:
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)

    else:
        # TODO: add multi dataset to train
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)

    num_classes = dataset.num_train_pids

    train_set = ImageDataset(dataset.train, train_transforms)

    if cfg.DATALOADER.SAMPLER == 'softmax':
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            # sampler=RandomIdentitySampler_alignedreid(dataset.train, cfg.DATALOADER.NUM_INSTANCE),      # new add by gu
            num_workers=num_workers, collate_fn=train_collate_fn
        )

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, val_loader, len(dataset.query), num_classes


def make_data_loader_part(cfg):
    train_transforms = build_transforms(cfg, is_train=True)
    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)

    num_classes = dataset.num_train_pids
    train_set = ImageDatasetPart(dataset.train, transform=train_transforms)
    # train_set = ImageDatasetPart(dataset.train, cfg)

    if cfg.DATALOADER.SAMPLER == 'softmax':
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=part_train_collate_fn
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler_Part(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            # sampler=RandomIdentitySampler_alignedreid(dataset.train, cfg.DATALOADER.NUM_INSTANCE),      # new add by gu
            num_workers=num_workers, collate_fn=part_train_collate_fn
        )

    val_set = ImageDatasetPart(dataset.query + dataset.gallery, transform=val_transforms)

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=part_val_collate_fn
    )
    return train_loader, val_loader, len(dataset.query), num_classes


def make_data_loader_prcc(cfg, trial=0):
    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS

    random.seed(trial)
    ids = ['188', '005', '091', '309', '075', '162', '182', '223', '061', '006', '321', '324', '057',
          '279', '156', '328', '152', '282', '118', '004', '099', '319', '257', '008', '272', '214',
          '058', '146', '112', '230', '094', '186', '323', '120', '242', '071', '320', '264', '265',
          '001', '072', '097', '018', '056', '069', '030', '096', '263', '062', '002', '070', '216',
          '167', '117', '159', '212', '059', '007', '073', '064', '219', '326', '060', '202', '322',
          '183', '063', '260', '325', '028', '074']
    img_paths = []
    for id in ids:
        img = glob.glob(osp.join('/home/yhl/data/prcc/rgb/gallerycrop3', id + '*.jpg'))
        img.sort()
        img_paths.append(random.choice(img))
    pid_container = set()

    for img_path in img_paths:
        pid = int(img_path.split("/")[-1][:3])
        if pid == -1: continue  # junk images are just ignored
        pid_container.add(pid)

    gallery = []
    # kps = torch.load('/home/yhl/data/prcc/rgb/part4n/maskg.pt')
    for img_path in img_paths:
        pid = int(img_path.split("/")[-1][:3])
        camid = img_path.split("/")[-1][4]
        # mask = kps[img_path.split("/")[-1][:-4]]/
        # gallery.append((img_path, pid, camid, mask))
        gallery.append((img_path, pid, camid))

    img_paths = glob.glob(osp.join('/home/yhl/data/prcc/rgb/queryccrop3', '*.jpg'))
    pid_container = set()

    for img_path in img_paths:
        pid = int(img_path.split("/")[-1][:3])
        if pid == -1: continue  # junk images are just ignored
        pid_container.add(pid)

    query = []
    # kps = torch.load('/home/yhl/data/prcc/rgb/part4n/maskq.pt')
    for img_path in img_paths:
        pid = int(img_path.split("/")[-1][:3])
        camid = img_path.split("/")[-1][4]
        # mask = kps[img_path.split("/")[-1][:-4]]
        # query.append((img_path, pid, camid, mask))
        query.append((img_path, pid, camid))

    val_set = ImageDatasetPart(query + gallery, transform=val_transforms)

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return val_loader, len(query), val_set