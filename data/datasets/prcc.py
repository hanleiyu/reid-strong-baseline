# encoding: utf-8

import glob
import numpy as np
import torch
import random
import os.path as osp

from .bases import BaseImageDataset


class PRCC(BaseImageDataset):
    """
    Dataset statistics:
    # identities: 221
    # images: 17887 (train) + 10800 (test gallery(A):3383 query:7417(B 3873 C 3543)) + 5003 (val)
    """
    dataset_dir = 'prcc/rgb'

    def __init__(self, root='/home/yhl/data', verbose=True, **kwargs):
        super(PRCC, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'queryc')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> PRCC loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

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
        if dir_path.find("train") != -1:
            kps = torch.load('/home/yhl/data/prcc/rgb/part6n/maskt.pt')
        # elif dir_path.find("gallery") != -1:
        #     kps = torch.load('/home/yhl/data/VC/partb/maskg.pt')
        # elif dir_path.find("query") != -1:
        #     kps = torch.load('/home/yhl/data/VC/partb/maskq.pt')
        if dir_path.find("gallery") != -1:
            ids = ['188', '005', '091', '309', '075', '162', '182', '223', '061', '006', '321', '324', '057',
                   '279', '156', '328', '152', '282', '118', '004', '099', '319', '257', '008', '272', '214',
                   '058', '146', '112', '230', '094', '186', '323', '120', '242', '071', '320', '264', '265',
                   '001', '072', '097', '018', '056', '069', '030', '096', '263', '062', '002', '070', '216',
                   '167', '117', '159', '212', '059', '007', '073', '064', '219', '326', '060', '202', '322',
                   '183', '063', '260', '325', '028', '074']
            random.seed(0)
            img_paths = []
            for id in ids:
                img = glob.glob(osp.join(dir_path, id + '*.jpg'))
                img_paths.append(random.choice(img))
        else:
            img_paths = glob.glob(osp.join(dir_path, '*.jpg'))

        pid_container = set()
        for img_path in img_paths:
            pid = int(img_path.split("/")[-1][:3])
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []

        for img_path in img_paths:
            pid = int(img_path.split("/")[-1][:3])
            camid = img_path.split("/")[-1][4]
            if relabel: pid = pid2label[pid]
            if dir_path.find("train") != -1:
                mask = kps[img_path.split("/")[-1][:-4]]
            else:
                mask = ""
            dataset.append((img_path, pid, camid, mask))

        return dataset
