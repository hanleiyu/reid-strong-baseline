# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn

from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
from .backbones.resnet_ibn_a import resnet50_ibn_a
from .backbones.vit import vit_TransReID, vit_TransReID2
from .model_keypoints import ScoremapComputer, compute_local_features
from .gcn import generate_adj, GCN
from .pointnet import PointNetfeat


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class FeatureBlock(nn.Module):
    def __init__(self, in_planes):
        super(FeatureBlock, self).__init__()
        self.in_planes = in_planes
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.bottleneck(x)
        return x


class ClassBlock(nn.Module):
    def __init__(self, neck, num_classes, in_planes):
        super(ClassBlock, self).__init__()
        self.num_classes = num_classes
        self.in_planes = in_planes
        if neck == 'no':
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
        elif neck == 'bnneck':
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        score = self.classifier(x)
        return score


class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
        super(Baseline, self).__init__()
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])

        elif model_name == 'se_resnet50':
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 4, 6, 3],
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet101':
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 4, 23, 3],
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet152':
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 8, 36, 3],
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnext50':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3],
                              groups=32,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnext101':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 23, 3],
                              groups=32,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'senet154':
            self.base = SENet(block=SEBottleneck,
                              layers=[3, 8, 36, 3],
                              groups=64,
                              reduction=16,
                              dropout_p=0.2,
                              last_stride=last_stride)
        elif model_name == 'resnet50_ibn_a':
            self.base = resnet50_ibn_a(last_stride)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):

        global_feat = self.gap(self.base(x))  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softmax

        # if self.neck_feat == 'after':
        #     # print("Test with feature after BN")
        #     return feat
        # else:
        #     # print("Test with feature before BN")
        #     return global_feat

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path).state_dict()
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])


class Part(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
        super(Part, self).__init__()
        if model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        self.feature1 = FeatureBlock(self.in_planes)
        self.feature2 = FeatureBlock(self.in_planes)
        self.feature3 = FeatureBlock(self.in_planes)
        self.feature4 = FeatureBlock(self.in_planes)
        self.feature5 = FeatureBlock(self.in_planes)
        self.feature6 = FeatureBlock(self.in_planes)

        self.classifier1 = ClassBlock(neck, self.num_classes, self.in_planes)
        self.classifier2 = ClassBlock(neck, self.num_classes, self.in_planes)
        self.classifier3 = ClassBlock(neck, self.num_classes, self.in_planes)
        self.classifier4 = ClassBlock(neck, self.num_classes, self.in_planes)
        self.classifier5 = ClassBlock(neck, self.num_classes, self.in_planes)
        # self.classifier6 = ClassBlock(neck, self.num_classes, self.in_planes)
        # self.classifier6 = ClassBlock(neck, self.num_classes, 2304)
        self.classifier6 = ClassBlock(neck, self.num_classes, 2176)
        # self.classifier6 = ClassBlock(neck, self.num_classes, 2898)
        # self.classifier7 = ClassBlock(neck, self.num_classes, 1700)

        self.transformer = vit_TransReID()
        # self.transformer2 = vit_TransReID2()

        self.linked_edges = \
            [[0, 1], [0, 2], [1, 3], [3, 5], [2, 4], [4, 6],  # body
             [0, 7], [0, 8], [7, 9], [9, 11], [8, 10], [10, 12]  # libs
             ]
        # self.linked_edges = \
        #     [[0, 1], [0, 2], [1, 3], [2, 4],  # head
        #      [0, 5], [0, 6], [5, 7], [7, 9], [6, 8], [8, 10],  # body
        #     [0, 11], [0, 12], [11, 13], [13, 15], [12, 14], [14, 16]  # libs
        #      ]
        # print(next(self.base.parameters()).device)
        self.device = torch.device('cuda')
        self.adj = generate_adj(14, self.linked_edges, self_connect=0.0).to(self.device)
        # self.adj = generate_adj(17, self.linked_edges, self_connect=0.0)
        # self.gcn = GCN(100, 100, 100).to(self.device)
        # self.gcn = GCN(50, 50, 50)
        self.gcn = GCN(128, 128, 128)
        # self.gcn = GCN(256, 256, 256)

        # keypoints model
        # self.scoremap_computer = ScoremapComputer(10).to(self.device)
        self.scoremap_computer = ScoremapComputer(10)
        # self.scoremap_computer = nn.DataParallel(self.scoremap_computer).to(self.device)
        self.scoremap_computer = self.scoremap_computer.eval()

    def forward(self, x, mask=None):
        global_feat = self.base(x)

        with torch.no_grad():
            score_maps, keypoints_confidence, keypoints_location = self.scoremap_computer(x)
        feature_vector_list, keypoints_confidence = compute_local_features(
            global_feat, score_maps, keypoints_confidence)

        f_confidence = keypoints_confidence.unsqueeze(2).repeat([1, 1, 2048])
        f = f_confidence * torch.stack(feature_vector_list, 1)
        # vit_feat = self.transformer(f)

        # key = torch.zeros((keypoints_location.size()[0], 14, 126))
        # k = torch.cat((keypoints_location, key), 2).cuda()
        # self.adj = self.adj.to(k.device)
        # key_feat = self.gcn(k, self.adj)

        pointfeat = PointNetfeat(global_feat=False)
        k = pointfeat(keypoints_location.transpose(2, 1))
        k = k.transpose(2, 1).cuda()
        self.adj = self.adj.to(k.device)
        k_confidence = keypoints_confidence.unsqueeze(2).repeat([1, 1, 128])
        key_feat = k_confidence * self.gcn(k, self.adj)
        # key_feat = self.gcn(k, self.adj)

        f = torch.cat((f, key_feat), dim=2)
        vit_feat = self.transformer(f)

        # key = torch.zeros((keypoints_location.size()[0], 17, 48))
        # k = torch.cat((keypoints_location, key), 2).cuda()
        # self.adj = self.adj.to(k.device)
        # key_feat = self.gcn(k, self.adj)
        # key_feat = key_feat.reshape((key_feat.size()[0], -1))

        if self.training:
            # score = self.classifier1(feats[4])
            # score = [torch.zeros(256) for _ in range(num + 2)]
            # score[0] = self.classifier1(feats[0])
            # score[1] = self.classifier2(feats[1])
            # score[2] = self.classifier3(feats[2])
            # score[3] = self.classifier4(feats[3])
            # score[4] = self.classifier5(feats[4])
            # score[5] = self.classifier6(feats[5])
            # return score, feats

            score = [torch.zeros(128) for _ in range(2)]
            score[0] = self.classifier5(feature_vector_list[-1])
            score[1] = self.classifier6(vit_feat)
            # score[1] = self.classifier6(torch.cat((vit_feat, key_feat), 1))
            # score[2] = self.classifier7(key_feat)

            # return score, (feature_vector_list[-1], vit_feat, key_feat)
            # return score, (feature_vector_list[-1], torch.cat((vit_feat, key_feat), 1))
            return score, (feature_vector_list[-1], vit_feat)
            # return score, feats[4]
        else:
            if self.neck_feat == 'after':
                return feature_vector_list[-1]
            else:
                # return torch.cat((vit_feat, global_feat), 1)
                # return torch.cat((feats[4], global_feat), 1)
                return torch.cat((feature_vector_list[-1], vit_feat), 1)
                # return torch.cat((feature_vector_list[-1], vit_feat, key_feat), 1)

        # if self.training:
        #     global_feat = self.base(x)
        #     num = len(list(mask[0, :, 0, 0, 0]))
        #     feats = [torch.zeros(128, 2048) for _ in range(num + 1)]
        #     # score = [torch.zeros(256) for _ in range(num + 1)]
        #
        #     for i in range(num):
        #         feats[i] = torch.mul(global_feat, mask[:, i, :, :, :].cuda())
        #
        #     # global_feat = self.gap(global_feat)  # (b, 2048, 1, 1)
        #     # global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        #     # # feat = self.bottleneck5(global_feat)
        #     # feat = self.bottleneck(global_feat)
        #     # feats[num] = feat
        #     # for i in range(num):
        #     #     score[i] = self.classifier(feats[i])
        #     # score[num] = self.classifier(feat)
        #
        #     feats[0] = self.feature1(feats[0])
        #     feats[1] = self.feature2(feats[1])
        #     feats[2] = self.feature3(feats[2])
        #     feats[3] = self.feature4(feats[3])
        #     feats[4] = self.feature5(global_feat)
        #
        #     f = torch.stack((feats[0], feats[1], feats[2], feats[3], feats[4]), 1)
        #     feat = self.transformer(f)
        #     score = self.classifier1(feat)
        #
        #
        #     # score[0] = self.classifier1(feats[0])
        #     # score[1] = self.classifier2(feats[1])
        #     # score[2] = self.classifier3(feats[2])
        #     # score[3] = self.classifier4(feats[3])
        #     # score[4] = self.classifier5(feats[4])
        #     # return score, feats  # global feature for triplet loss
        #     return score, feat
        #
        # else:
        #     global_feat = self.base(x)
        #
        #     # global_feat = self.feature.gap(global_feat)
        #     global_feat = self.feature5.gap(global_feat)  # (b, 2048, 1, 1)
        #     global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        #
        #     if self.neck == 'no':
        #         feat = global_feat
        #     elif self.neck == 'bnneck':
        #         feat = self.feature5.bottleneck(global_feat)
        #         # feat = self.bottleneck5(global_feat)
        #         # feat = self.bottleneck(global_feat)
        #
        #     if self.neck_feat == 'after':
        #         # print("Test with feature after BN")
        #         return feat
        #     else:
        #         # print("Test with feature before BN")
        #         return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path).state_dict()
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
