import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import resnet50
from .utils import weights_init_kaiming
from .utils import weights_init_classifier
from .utils import normalize_fn, NormalizeByChannelMeanStd
from .utils import GeneralizedMeanPooling, GeneralizedMeanPoolingP
from .utils import shuffle_unit
from .vit import vit_base_patch16_224, vit_small_patch16_224

class BaselineResnet(nn.Module):
    pool_dim = 2048
    feat_dim = 2048

    def __init__(self, num_classes, pretrained=False, last_stride=1, dropout_rate=0.0):
        super(BaselineResnet, self).__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        self.backbone = resnet50(pretrained=pretrained, last_stride=last_stride)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.bnneck = nn.BatchNorm1d(self.feat_dim)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(self.feat_dim, self.num_classes, bias=False)

        self.bnneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x, x_tmp=None, mode=None):
        global_feat = self.avgpool(self.backbone(x))  # (bs, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        feat = self.bnneck(global_feat)  # (bs, 2048)

        if self.training:
            # global feature for triplet loss
            if self.dropout_rate > 0:
                feat = F.dropout(feat, p=self.dropout_rate)
            # return global_feat, self.classifier(feat)
            return global_feat, feat, self.classifier(feat)
        else:
            # test with feature before/after BN
            return global_feat, feat

class BaselineVit(nn.Module):
    feat_dim = 768
    def __init__(self, num_classes, pretrained=False, dropout_rate=0.0):
        super(BaselineVit, self).__init__()
        self.model_path = 'D:/Datasets/pretrain-models/imagenet/jx_vit_base_p16_224-80ecf9dd.pth'
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.in_planes = 768
        self.dropout_rate = dropout_rate
        self.normalize = NormalizeByChannelMeanStd(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.backbone = vit_base_patch16_224(img_size=(256, 128),
                                   stride_size=16,
                                   drop_rate=0.0,
                                   attn_drop_rate=0.0,
                                   drop_path_rate=0.1,
                                   camera=0,
                                   view=0,
                                   local_feature=False,
                                   sie_xishu=1.5,
                                   linear_block=False
                                   )
        if self.pretrained == True:
            self.backbone.load_param(self.model_path)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.bnneck = nn.BatchNorm1d(self.in_planes)
        self.bnneck.bias.requires_grad_(False)
        self.bnneck.apply(weights_init_kaiming)


    def forward(self, x, label=None):
        x = self.normalize(x)
        global_feat = self.backbone(x)
        feat = self.bnneck(global_feat)

        if self.training:
            # global feature for triplet loss
            if self.dropout_rate > 0:
                feat = F.dropout(feat, p=self.dropout_rate)
            return global_feat, feat, self.classifier(feat)

        else:
            return global_feat, feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(
            model_path))





