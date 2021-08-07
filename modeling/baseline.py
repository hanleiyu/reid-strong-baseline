# encoding: utf-8
import torch
from torch import nn

from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.resnet_ibn_a import resnet50_ibn_a
from .backbones.vit import vit_TransReID
from .model_keypoints import ScoremapComputer, compute_local_features
from .gcn import generate_adj, GCN
from .pointnet import PointNetfeat
from .backbones.hmr import hmr



class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


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
        elif model_name == 'resnet50_ibn_a':
            self.base = resnet50_ibn_a(last_stride)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        if self.neck == 'no':
            # self.classifier = nn.Linear(self.in_planes, self.num_classes)
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)  # new add by luo
            self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

        self.l2norm = Normalize(2)

    def forward(self, x):
        global_feat = self.gap(self.base(x))  # (b, 2048, 1, 1)

        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softmax

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return self.l2norm(feat)
            else:
                # print("Test with feature before BN")
                return self.l2norm(global_feat)

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
        # self.feature6 = FeatureBlock(2176)
        # self.feature7 = FeatureBlock(1024)

        self.classifier1 = ClassBlock(neck, self.num_classes, self.in_planes)
        self.classifier2 = ClassBlock(neck, self.num_classes, self.in_planes)
        self.classifier3 = ClassBlock(neck, self.num_classes, self.in_planes)
        self.classifier4 = ClassBlock(neck, self.num_classes, self.in_planes)
        self.classifier5 = ClassBlock(neck, self.num_classes, self.in_planes)
        # self.classifier6 = ClassBlock(neck, self.num_classes, self.in_planes)
        self.classifier6 = ClassBlock(neck, self.num_classes, 2176)
        # self.classifier6 = ClassBlock(neck, self.num_classes, 2304)
        self.classifier7 = ClassBlock(neck, self.num_classes, 1024)
        # self.classifier8 = ClassBlock(neck, self.num_classes, 229)

        self.transformer = vit_TransReID()

        # self.linked_edges = \
        #     [[0, 1], [0, 2], [1, 3], [3, 5], [2, 4], [4, 6],  # body
        #      [0, 7], [0, 8], [7, 9], [9, 11], [8, 10], [10, 12]  # libs
        #      ]
        self.linked_edges = \
            [[0, 1], [0, 2], [1, 3], [3, 5], [2, 4], [4, 6],  # body
             [0, 7], [0, 8], [1, 7], [1, 8], [7, 9], [9, 11], [8, 10], [10, 12]  # libs
             ]
        self.linked_edges2 = \
            [[1, 5], [2, 6], [7, 11], [8, 12]
             ]
        # self.linked_edges3 = \
        #     [[0, 5], [0, 6], [0, 11], [0, 12],
        #      [1, 11], [2, 12], [5, 11], [5, 12]
        #      ]

        self.device = torch.device('cuda')
        # self.adj = generate_adj(14, self.linked_edges, self_connect=0.0).to(self.device)
        self.adj = generate_adj(14, self.linked_edges, self.linked_edges2, self_connect=0.0).to(self.device)
        # self.adj = generate_adj(14, self.linked_edges, self.linked_edges2, self.linked_edges3, self_connect=0.0).to(self.device)
        # self.gcn = GCN(100, 100, 100)
        # self.gcn = GCN(50, 50, 50)
        self.gcn = GCN(128, 128, 128)
        # self.gcn = GCN(256, 256, 256)

        # keypoints model
        self.scoremap_computer = ScoremapComputer(10)
        # self.scoremap_computer = nn.DataParallel(self.scoremap_computer).to(self.device)
        self.scoremap_computer = self.scoremap_computer.eval()

        # 3D model
        self.hmr = hmr('/home/yhl/.torch/models/smpl_mean_params.npz').to(self.device)
        checkpoint = torch.load('/home/yhl/.torch/models/hmr.pt')
        self.hmr.load_state_dict(checkpoint['model'], strict=False)
        self.hmr = self.hmr.eval()

        self.bottleneck1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck1.bias.requires_grad_(False)  # no shift
        self.bottleneck1.apply(weights_init_kaiming)

        # self.bottleneck2 = nn.BatchNorm1d(2048)
        self.bottleneck2 = nn.BatchNorm1d(2176)
        # self.bottleneck2 = nn.BatchNorm1d(2304)
        self.bottleneck2.bias.requires_grad_(False)  # no shift
        self.bottleneck2.apply(weights_init_kaiming)

        self.bottleneck3 = nn.BatchNorm1d(1024)
        self.bottleneck3.bias.requires_grad_(False)  # no shift
        self.bottleneck3.apply(weights_init_kaiming)

        # self.bottleneck4 = nn.BatchNorm1d(229)
        # self.bottleneck4.bias.requires_grad_(False)  # no shift
        # self.bottleneck4.apply(weights_init_kaiming)

        self.l2norm = Normalize(2)

    def forward(self, x, mask=None):
    # def forward(self, x, x2, mask=None):
        # resnet50
        global_feat = self.base(x)

        # keypoint estimation
        with torch.no_grad():
            score_maps, keypoints_confidence, keypoints_location = self.scoremap_computer(x)
        feature_vector_list, keypoints_confidence = compute_local_features(
            global_feat, score_maps, keypoints_confidence)

        # f_confidence = keypoints_confidence.unsqueeze(2).repeat([1, 1, 2048])
        # f = f_confidence * torch.stack(feature_vector_list, 1)
        f = torch.stack(feature_vector_list, 1)
        # vit_feat = self.transformer(f)

        pointfeat = PointNetfeat(global_feat=False)
        k = pointfeat(keypoints_location.transpose(2, 1))
        # # k = pointfeat(torch.cat((keypoints_location, keypoints_confidence.unsqueeze(2).cpu()), dim=2).transpose(2, 1))
        k = k.transpose(2, 1).cuda()
        # k = keypoints_location.cuda()
        self.adj = self.adj.to(k.device)
        # k_confidence = keypoints_confidence.unsqueeze(2).repeat([1, 1, 128])
        # key_feat = k_confidence * self.gcn(k, self.adj)
        key_feat = self.gcn(k, self.adj)

        bn3 = nn.BatchNorm1d(1024)
        conv3 = torch.nn.Conv1d(128, 1024, 1)
        # conv3 = torch.nn.Conv1d(128, 2048, 1)
        # conv3 = torch.nn.Conv1d(256, 1024, 1)
        key_global = bn3(conv3(key_feat.transpose(2, 1).cpu()))
        key_global = torch.max(key_global, 2, keepdim=True)[0]
        key_global = key_global.view(-1, 1024).cuda()

        f = torch.cat((f, key_feat), dim=2)
        vit_feat = self.transformer(f)


        # pred_rotmat, pred_betas, pred_camera = self.hmr(x2)
        # threeDF = torch.cat((pred_rotmat.view(pred_rotmat.size()[0], -1), pred_betas, pred_camera), 1)
        # vit_feat = torch.cat((vit_feat, pred_betas), 1)
        # bn4 = nn.BatchNorm1d(1024)
        # conv4 = torch.nn.Conv1d(10, 1024, 1)
        # threeDF = bn4(conv4(torch.unsqueeze(pred_betas, 2).cpu())).view(-1, 1024).cuda()

        if self.neck == 'bnneck':
            fb = self.bottleneck1(feature_vector_list[-1])
            vb = self.bottleneck2(vit_feat)
            kb = self.bottleneck3(key_global)
            # db = self.bottleneck4(threeDF)

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

            score = [torch.zeros(128) for _ in range(3)]
            # score[0] = self.classifier5(feature_vector_list[-1])
            # score[1] = self.classifier6(vit_feat)
            # score[2] = self.classifier7(key_global)
            score[0] = self.classifier5(fb)
            score[1] = self.classifier6(vb)
            score[2] = self.classifier7(kb)
            # score[3] = self.classifier8(db)

            return score, (feature_vector_list[-1], vit_feat, key_global)
            # return score, (feature_vector_list[-1], vit_feat, key_global, threeDF)
        else:
            if self.neck_feat == 'after':
                # return feature_vector_list[-1]
                # return self.l2norm(threeDF)
                return self.l2norm(fb)
                # return torch.cat((fb, threeDF), 1)
                # return self.l2norm(torch.cat((fb, vb), 1))
                # return self.l2norm(torch.cat((fb, db), 1))
                # return self.l2norm(torch.cat((fb, vb, db), 1))
            else:
                # return torch.cat((vit_feat, global_feat), 1)
                # return torch.cat((feats[4], global_feat), 1)
                # return self.l2norm(feature_vector_list[-1])
                return self.l2norm(torch.cat((feature_vector_list[-1], vit_feat), 1))
                # return torch.cat((feature_vector_list[-1], vit_feat, key_feat), 1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path).state_dict()
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
