# encoding: utf-8

import glob
import numpy as np
import torch
import os.path as osp

from .bases import BaseImageDataset


class PRCC(BaseImageDataset):
    """
    Dataset statistics:
    # identities: 221
    # images: 9449 (train) + 1020 (query) + 8591 (gallery)
    """
    dataset_dir = 'prcc/rgb'

    def __init__(self, root='/home/yhl/data', verbose=True, **kwargs):
        super(PRCC, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.val_dir = osp.join(self.dataset_dir, 'val')
        self.test_dir = osp.join(self.dataset_dir, 'test')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        val = self._process_dir(self.val_dir, relabel=False)
        test = self._process_dir(self.test_dir, relabel=False)

        if verbose:
            print("=> PRCC loaded")
            self.print_dataset_statistics(train, val, test)

        self.train = train
        self.val = val
        self.test = test

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        # if dir_path.find("train") != -1:
        #     kps = torch.load('/home/yhl/data/VC/partb/maskt.pt')
        # elif dir_path.find("gallery") != -1:
        #     kps = torch.load('/home/yhl/data/VC/partb/maskg.pt')
        # elif dir_path.find("query") != -1:
        #     kps = torch.load('/home/yhl/data/VC/partb/maskq.pt')
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))

        pid_container = set()
        for img_path in img_paths:
            pid = int(img_path[-24:-21])
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid = int(img_path[-24:-21])
            camid = int(img_path[-20])
            # mask = kps[img_path[-17:-4]]
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset
