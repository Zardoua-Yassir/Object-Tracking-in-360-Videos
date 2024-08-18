"""
This script implements the SiamCAR tracker and its associated models.
"""
import torch
import torchvision.models as tormodels
from collections import OrderedDict
from toolbox.misc import _dw_xcorr
from config import cfg
import math


class ResnetBackbone50(torch.nn.Module):
    """
    Class to extract branch Resnet-50 feature map as described in SiamRPN++ and SiamCAR papers
    """

    def __init__(self):
        super(ResnetBackbone50, self).__init__()
        # Load pre-trained ResNet model
        self.pretrained_resnet50 = tormodels.resnet50(weights=tormodels.ResNet50_Weights.DEFAULT)
        # self.backbone = self.modify_pretrained_resnet50()
        self.modify_pretrained_resnet50()
        # extracting three interconnected backbones to produce the three feature maps F3, F4, and F5
        self.backbone_f3 = torch.nn.Sequential(OrderedDict([*(list(self.pretrained_resnet50.named_children())[0:6])]))
        self.backbone_f4 = torch.nn.Sequential(OrderedDict([*list((self.pretrained_resnet50.named_children()))[6:7]]))
        self.backbone_f5 = torch.nn.Sequential(OrderedDict([*list((self.pretrained_resnet50.named_children()))[7:-2]]))

        self.conv_pw_f3 = torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1))
        self.conv_pw_f4 = torch.nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=(1, 1))
        self.conv_pw_f5 = torch.nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=(1, 1))

    def forward(self, x):
        self.f3 = self.backbone_f3(x)
        self.f4 = self.backbone_f4(self.f3)
        self.f5 = self.backbone_f5(self.f4)

        self.f3 = self.conv_pw_f3(self.f3)
        self.f4 = self.conv_pw_f4(self.f4)
        self.f5 = self.conv_pw_f5(self.f5)

        self.f345_cat = torch.cat((self.f3, self.f4, self.f5), dim=1)
        return self.f345_cat

    def modify_pretrained_resnet50(self):
        """
        Returns the ResNet-50 modified according to paper: SiamRPN++: <Evolution of Siamese Visual Tracking With Very
        Deep Networks>
        """
        self.pretrained_resnet50.conv1.padding = (0, 0)
        # at (layer3):(0):Bottleneck(downsample):(0): change stride from (2,2) to (1,1)
        self.pretrained_resnet50.layer3[0].downsample[0].stride = (1, 1)
        # at (layer3):(0):Bottleneck(conv2): change stride from (2,2) to (1,1)
        self.pretrained_resnet50.layer3[0].conv2.stride = (1, 1)

        # at (layer3):[(1):Bottleneck(conv2) to (5):Bottleneck(conv2)]: change dilation and padding from (1,1) to (2,2)
        for idx in range(1, 6):
            self.pretrained_resnet50.layer3[idx].conv2.padding = (2, 2)
            self.pretrained_resnet50.layer3[idx].conv2.dilation = (2, 2)
        # at (layer4):(0):Bottleneck(conv2): change stride from (2,2) to (1,1), change dilation and padding from (1,1)
        #  to (2,2)
        self.pretrained_resnet50.layer4[0].conv2.stride = (1, 1)
        self.pretrained_resnet50.layer4[0].conv2.dilation = (2, 2)
        self.pretrained_resnet50.layer4[0].conv2.padding = (2, 2)
        # at (layer4):(0):downsample(0): change stride from (2,2) to (1,1), change dilation and padding from (1,1)
        #  to (2,2)
        self.pretrained_resnet50.layer4[0].downsample[0].stride = (1, 1)
        # at (layer4):(1):Bottleneck(conv2): change dilation and padding from (1,1) to (4,4)
        self.pretrained_resnet50.layer4[1].conv2.dilation = (4, 4)
        self.pretrained_resnet50.layer4[1].conv2.padding = (4, 4)
        # at (layer4):(2):Bottleneck(conv2): change dilation and padding from (1,1) to (4,4)
        self.pretrained_resnet50.layer4[2].conv2.dilation = (4, 4)
        self.pretrained_resnet50.layer4[2].conv2.padding = (4, 4)
        # Extract only the layers up to the last convolutional layer
        # self.backbone = torch.nn.Sequential(*list(resnet.children())[:-2])
        return torch.nn.Sequential(OrderedDict([*(list(self.pretrained_resnet50.named_children())[:-2])]))


class SiameseSubnet(torch.nn.Module):
    """
    The Siamese Subnetwork + correlation module, as described in SiamCAR. Returns compressed correlation map R*
    """

    def __init__(self, m=256, mode='train'):
        """
        :param m: number of channels in R* (See Figure 2)
        """
        if mode not in ['train', 'track_2d']:
            raise ValueError("'mode' must be either 'train' or 'track_2d'")
        self.mode = mode
        super(SiameseSubnet, self).__init__()
        self.r_star = None
        self.m = m
        self.n = 256 * 3  # Channels of response R = the 3 concatenated ResNet 256 channel feature map

        # Modules/layers
        self.resnet_siam_subnet = ResnetBackbone50()  # CNN (Figure 2)
        self.pw_conv_r_star = torch.nn.Conv2d(in_channels=self.n, out_channels=self.m, kernel_size=(1, 1))

    def forward(self, x, z=None):
        self.phi_x = self.resnet_siam_subnet(x)
        if self.mode == 'train':
            self.phi_z = self.resnet_siam_subnet(z)
            # todo: make sure to remove redundency in computing phi_z
            self.phi_z_h, self.phi_w = self.phi_z.shape[2], self.phi_z.shape[3]
            # cropping the center 7 × 7 regions as the temp_window feature to speed-up the correlation module
            self.phi_z = self.phi_z[:, :, ((self.phi_z_h - 7) // 2):((self.phi_z_h + 7) // 2),
                         ((self.phi_w - 7) // 2):((self.phi_w + 7) // 2)]
        self.r = _dw_xcorr(self.phi_x, self.phi_z)
        self.r_star = self.pw_conv_r_star(self.r)
        return self.r_star

    def initialize_temp_branch(self, z):
        if self.mode == 'track_2d':
            self.phi_z = self.resnet_siam_subnet(z)
            self.phi_z_h, self.phi_w = self.phi_z.shape[2], self.phi_z.shape[3]
            # cropping the center 7 × 7 regions as the temp_window feature to speed-up the correlation module
            self.phi_z = self.phi_z[:, :, ((self.phi_z_h - 7) // 2):((self.phi_z_h + 7) // 2),
                         ((self.phi_w - 7) // 2):((self.phi_w + 7) // 2)]
            print("Template branch output has been initialized")
            print("Shape = ", self.phi_z.shape)
            print("Value = ", self.phi_z)
            print("--")
        else:
            raise ValueError("'mode' attribute must be set to 'track_2d'")


class CarSubnet(torch.nn.Module):
    """
    The CAR Subnetwork
    """

    def __init__(self, in_channels=256):
        self.m = in_channels
        super(CarSubnet, self).__init__()
        self.cnn_branch1_layers = []
        self.cnn_branch2_layers = []
        self.cen_branch_layers = []
        self.conv_layer = torch.nn.Conv2d(self.m, self.m, (3, 3), (1, 1), (1, 1))
        self.group_norm = torch.nn.GroupNorm(32, self.m)
        self.relu = torch.nn.ReLU()

        self.layers_count = 4  # number of layers in the CNN of each branch (cls and reg branches)

        for _ in range(self.layers_count):
            self.cnn_branch1_layers.append(self.conv_layer)
            self.cnn_branch1_layers.append(self.group_norm)
            self.cnn_branch1_layers.append(self.relu)

            self.cnn_branch2_layers.append(self.conv_layer)
            self.cnn_branch2_layers.append(self.group_norm)
            self.cnn_branch2_layers.append(self.relu)

        self.add_module("car_cnn_branch1", torch.nn.Sequential(*self.cnn_branch1_layers))
        self.add_module("car_cnn_branch2", torch.nn.Sequential(*self.cnn_branch2_layers))

        self.out_layer_cls = torch.nn.Conv2d(self.m, 2, (3, 3), (1, 1), (1, 1))
        self.out_layer_cen = torch.nn.Conv2d(self.m, 1, (3, 3), (1, 1), (1, 1))
        self.out_layer_loc = torch.nn.Conv2d(self.m, 4, (3, 3), (1, 1), (1, 1))

        # self.initialize_car_layers()  # deprecated, read description

    def initialize_car_layers(self):
        """
        Initialize the CAR layers, as in the original paper.
        NOT USED IN THIS IMPLEMENTATION BECAUSE:
        Testing with default initialization and this customized initialization does not provide a clear advantage to
        early reduce the loss over the first 10 batches. In fact, the default initialization performed a little bit
        better.
        """
        self.car_blocks = [self.car_cnn_branch1, self.car_cnn_branch2, self.out_layer_cls, self.out_layer_cen,
                           self.out_layer_loc]
        for block in self.car_blocks:
            for layer in block.modules():
                if isinstance(layer, torch.nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        cls_logits_bias = -math.log((1 - cfg.TRAIN.PRIOR_PROB) / cfg.TRAIN.PRIOR_PROB)
        torch.nn.init.constant_(self.out_layer_cls.bias, cls_logits_bias)

    def forward(self, r_star):
        self.car_cnn_branch1_output = self.car_cnn_branch1(r_star)
        self.car_cnn_branch2_output = self.car_cnn_branch2(r_star)

        self.cls_pred = self.out_layer_cls(self.car_cnn_branch1_output)
        self.cen_pred = self.out_layer_cen(self.car_cnn_branch1_output)

        self.loc_pred = self.out_layer_loc(self.car_cnn_branch2_output)
        self.loc_pred = torch.exp(self.loc_pred)  # not mentioned in paper, but I believe the authors used exp to 1)
        # ensure positivity constraint of the output values LTRB, 2) Adds sensitivity to scale changes, 3) induce
        # further non-linear transformation.
        return self.cls_pred, self.cen_pred, self.loc_pred


class SiamCarV1(torch.nn.Module):
    def __init__(self, mode='train'):
        if mode not in ['train', 'track_2d']:
            raise ValueError("'mode' must be either 'train' or 'track_2d'")
        self.mode = mode
        super(SiamCarV1, self).__init__()
        self.siam_subnet = SiameseSubnet(mode=self.mode)
        self.car_subnet = CarSubnet()

    def initialize_temp_branch(self, z):
        self.siam_subnet.initialize_temp_branch(z)

    def forward(self, x, z=None):
        """
        :param x: srch_window
        :param z: temp_window
        """
        if self.mode == 'train':
            self.r_star = self.siam_subnet(x, z)
        elif self.mode == 'track_2d':
            self.r_star = self.siam_subnet(x)

        self.cls_pred, self.cen_pred, self.loc_pred = self.car_subnet(self.r_star)
        self.pred_dict = {"cls": self.cls_pred,
                          "cen": self.cen_pred,
                          "loc": self.loc_pred}
        return self.pred_dict
