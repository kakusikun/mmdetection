from abc import abstractmethod
import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init
from mmcv.runner import force_fp32

from mmdet.core import multi_apply
from ..builder import HEADS, build_loss
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin


@HEADS.register_module()
class CenterHandHead(BaseDenseHead, BBoxTestMixin):
    """Anchor-free head (FCOS, Fovea, RepPoints, etc.).

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels. Used in child classes.
        stacked_convs (int): Number of stacking convs of the head.
        strides (tuple): Downsample factor of each feature map.
        dcn_on_last_conv (bool): If true, use dcn in the last layer of
            towers. Default: False.
        conv_bias (bool | str): If specified as `auto`, it will be decided by
            the norm_cfg. Bias of conv will be set as True if `norm_cfg` is
            None, otherwise False. Default: "auto".
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
    """  # noqa: W605

    _version = 1

    def __init__(self,
                 in_channels,
                 feat_channels={'hm': 1, 'wh': 2, 'offset': 2, 'orie': 6},
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 dcn_on_last_conv=False,
                 conv_bias='auto',
                 loss_hm=dict(type='GaussianFocalLoss', loss_weight=1.0),
                 loss_wh=dict(type='CenterNetRegL1Loss', loss_weight=1.0),
                 loss_offset=dict(type='CenterNetRegL1Loss', loss_weight=1.0),
                 loss_orie=dict(type='CenterNetRegL1Loss', loss_weight=1.0),
                 conv_cfg=None,
                 norm_cfg=None,
                 train_cfg=None,
                 test_cfg=None):
        super(CenterHandHead, self).__init__()
        self.heads = list(feat_channels.keys())
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.dcn_on_last_conv = dcn_on_last_conv
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.loss_hm = build_loss(loss_hm)
        self.loss_wh = build_loss(loss_wh)
        self.loss_offset = build_loss(loss_offset)
        self.loss_orie = build_loss(loss_orie)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False

        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        self._init_head_convs()

    def _init_head_convs(self):
        """Initialize classification conv layers of the head."""
        self.head_convs = nn.ModuleDict()
        for head in self.heads:
            if head != 'hm':
                self.head_convs[head] = ConvModule(
                                            self.in_channels,
                                            self.feat_channels[head],
                                            1,
                                            stride=1,
                                            padding=0,
                                            conv_cfg=self.conv_cfg,
                                            norm_cfg=self.norm_cfg,
                                            bias=self.conv_bias
                                        )
            else:
                self.head_convs[head] = ConvModule(
                                            self.in_channels,
                                            self.feat_channels[head],
                                            1,
                                            stride=1,
                                            padding=0,
                                            conv_cfg=self.conv_cfg,
                                            norm_cfg=self.norm_cfg,
                                            bias=self.conv_bias,
                                            act_cfg=dict(type='Sigmoid')
                                        )

    def init_weights(self):
        """Initialize weights of the head."""
        for n, m in self.head_convs.items():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    if n == 'hm':
                        nn.init.constant_(m.bias, -2.19)
                    else:
                        nn.init.constant_(m.bias, 0)

    # def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
    #                           missing_keys, unexpected_keys, error_msgs):
    #     """Hack some keys of the model state dict so that can load checkpoints
    #     of previous version."""
    #     version = local_metadata.get('version', None)
    #     if version is None:
    #         # the key is different in early versions
    #         # for example, 'fcos_cls' become 'conv_cls' now
    #         bbox_head_keys = [
    #             k for k in state_dict.keys() if k.startswith(prefix)
    #         ]
    #         ori_predictor_keys = []
    #         new_predictor_keys = []
    #         # e.g. 'fcos_cls' or 'fcos_reg'
    #         for key in bbox_head_keys:
    #             ori_predictor_keys.append(key)
    #             key = key.split('.')
    #             conv_name = None
    #             if key[1].endswith('cls'):
    #                 conv_name = 'conv_cls'
    #             elif key[1].endswith('reg'):
    #                 conv_name = 'conv_reg'
    #             elif key[1].endswith('centerness'):
    #                 conv_name = 'conv_centerness'
    #             else:
    #                 assert NotImplementedError
    #             if conv_name is not None:
    #                 key[1] = conv_name
    #                 new_predictor_keys.append('.'.join(key))
    #             else:
    #                 ori_predictor_keys.pop(-1)
    #         for i in range(len(new_predictor_keys)):
    #             state_dict[new_predictor_keys[i]] = state_dict.pop(
    #                 ori_predictor_keys[i])
    #     super()._load_from_state_dict(state_dict, prefix, local_metadata,
    #                                   strict, missing_keys, unexpected_keys,
    #                                   error_msgs)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually contain classification scores and bbox predictions.
                cls_scores (list[Tensor]): Box scores for each scale level,
                    each is a 4D-tensor, the channel number is
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each scale
                    level, each is a 4D-tensor, the channel number is
                    num_points * 4.
        """
        return multi_apply(self.forward_single, feats)

    def forward_single(self, x):
        """Forward features of a single scale levle.

        Args:
            x (Tensor): FPN feature maps of the specified stride.

        Returns:
            tuple: Scores for each class, bbox predictions, features
                after classification and regression conv layers, some
                models needs these features like FCOS.
        """
        feats = []
        for head in self.heads:
            feats.append(self.head_convs[head](x))

        hm_feat, wh_feat, offset_feat, orie_feat = feats
        return hm_feat, wh_feat, offset_feat, orie_feat

    @force_fp32(apply_to=('hm_feat', 'wh_feat', 'offset_feat', 'orie_feat'))
    def loss(self,
             hm_feats,
             wh_feats,
             offset_feats,
             orie_feats,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_majors=None,
             gt_minors=None,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
        """

        assert gt_majors is not None
        assert gt_minors is not None

        hm_targets, wh_targets, offset_targets, orie_targets = self.get_targets(
            hm_feats,
            gt_bboxes,
            gt_majors,
            gt_minors,
            gt_labels
        )

        flatten_hm_feats = [
            hm_feat.permute(0, 2, 3, 1).reshape(-1, self.feat_channels['hm'])
            for hm_feat in hm_feats
        ]
        flatten_wh_feats = [
            wh_feat.permute(0, 2, 3, 1).reshape(-1, self.feat_channels['wh'])
            for wh_feat in wh_feats
        ]
        flatten_offset_feats = [
            offset_feat.permute(0, 2, 3, 1).reshape(-1, self.feat_channels['offset'])
            for offset_feat in offset_feats
        ]
        flatten_orie_feats = [
            orie_feat.permute(0, 2, 3, 1).reshape(-1, self.feat_channels['orie'])
            for orie_feat in orie_feats
        ]
        flatten_hm_targets = [
            hm_target.permute(1, 2, 0).reshape(-1, self.feat_channels['hm'])
            for hm_target in hm_targets
        ]

        flatten_hm_feats = torch.cat(flatten_hm_feats)
        flatten_wh_feats = torch.cat(flatten_wh_feats)
        flatten_offset_feats = torch.cat(flatten_offset_feats)
        flatten_orie_feats = torch.cat(flatten_orie_feats)

        device = flatten_hm_feats.device
        flatten_hm_targets = torch.cat(flatten_hm_targets).to(device)
        flatten_wh_targets = torch.cat(wh_targets).to(device)
        flatten_offset_targets = torch.cat(offset_targets).to(device)
        flatten_orie_targets = torch.cat(orie_targets).to(device)
        flatten_mask = flatten_hm_targets == 1.0
        flatten_mask = torch.nonzero(flatten_mask.squeeze(), as_tuple=False).squeeze()


        loss_hm = self.loss_hm(flatten_hm_feats, flatten_hm_targets)
        loss_wh = self.loss_wh(flatten_wh_feats[flatten_mask], flatten_wh_targets)
        loss_offset = self.loss_offset(flatten_offset_feats[flatten_mask], flatten_offset_targets)
        loss_orie = self.loss_orie(flatten_orie_feats[flatten_mask], flatten_orie_targets)

        return dict(
            loss_hm=loss_hm,
            loss_wh=loss_wh,
            loss_offset=loss_offset,
            loss_orie=loss_orie,
        )



    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   cfg=None,
                   rescale=None):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_points * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space
        """

        raise NotImplementedError

    def get_targets(self, 
            hm_feats,
            gt_bboxes_list,
            gt_majors_list,
            gt_minors_list,
            gt_labels_list,
        ):
        """Compute regression, classification and centerss targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).
        """
        num_batch = len(gt_bboxes_list)
        sizes = [(feat.size(2), feat.size(3)) for feat in hm_feats]
        hm_targets = []
        wh_targets = []
        offset_targets = []
        orie_targets = []
        for size, stride in zip(sizes, self.strides):
            hm_target, wh_target, offset_target, orie_target = multi_apply(
                self._get_targle_single,
                [size for _ in range(num_batch)],
                [stride for _ in range(num_batch)],
                gt_bboxes_list,
                gt_majors_list,
                gt_minors_list,
                gt_labels_list,
            )
            hm_targets.append(torch.cat(hm_target))
            wh_targets.append(torch.cat(wh_target))
            offset_targets.append(torch.cat(offset_target))
            orie_targets.append(torch.cat(orie_target))
        return hm_targets, wh_targets, offset_targets, orie_targets


    def _get_targle_single(self,
            size,
            stride,
            gt_bboxes,
            gt_majors,
            gt_minors,
            gt_labels
        ):
        feat_h, feat_w = size
        hm_target = torch.zeros(self.feat_channels['hm'], feat_h, feat_w)
        wh_target = defaultdict(list)
        offset_target = defaultdict(list)
        orie_target = defaultdict(list)
        for _gt_bbox, _gt_major, _gt_minor, gt_label in zip(gt_bboxes, gt_majors, gt_minors, gt_labels):
            gt_bbox = _gt_bbox.clone() / stride
            gt_major = _gt_major.clone() / stride
            gt_minor = _gt_minor.clone() / stride
            bh, bw = gt_bbox[3] - gt_bbox[1], gt_bbox[2] - gt_bbox[0]
            radius = self._gaussian_radius((math.ceil(bh), math.ceil(bw)))
            radius = max(0, int(radius))
            center = torch.Tensor([gt_bbox[0] + bw / 2, gt_bbox[1] + bh / 2])
            center_int = center.int()
            self._draw_umich_gaussian(hm_target[gt_label], center_int, radius)
            wh_target[center_int].append(torch.Tensor([bw, bh]))
            offset_target[center_int].append(center-center_int)
            orie_target[center_int].append(
                torch.stack([
                    (gt_major[0]-center[0])/bw,
                    (gt_major[1]-center[1])/bh,
                    torch.log(torch.abs(gt_minor[2]-gt_minor[0])/bw+0.1),
                    torch.log(torch.abs(gt_minor[3]-gt_minor[1])/bh+0.1),
                    torch.log(torch.abs(gt_major[2]-gt_major[0])/bw+0.1),
                    torch.log(torch.abs(gt_major[3]-gt_major[1])/bh+0.1),
                ])
            )

        wh_target = torch.stack([torch.mean(torch.stack(wh), dim=0) for wh in wh_target.values()])
        offset_target = torch.stack([torch.mean(torch.stack(offset), dim=0) for offset in offset_target.values()]).cuda()
        orie_target = torch.stack([torch.mean(torch.stack(orie), dim=0) for orie in orie_target.values()])

        return hm_target, wh_target, offset_target, orie_target

    def aug_test(self, feats, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            feats (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains features for all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[ndarray]: bbox results of each class
        """
        return self.aug_test_bboxes(feats, img_metas, rescale=rescale)

    def _gaussian_radius(self, det_size, min_overlap=0.7):
        height, width = det_size

        a1    = 1
        b1    = (height + width)
        c1    = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1    = (b1 + sq1) / 2

        a2    = 4
        b2    = 2 * (height + width)
        c2    = (1 - min_overlap) * width * height
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2    = (b2 + sq2) / 2

        a3    = 4 * min_overlap
        b3    = -2 * min_overlap * (height + width)
        c3    = (min_overlap - 1) * width * height
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3    = (b3 + sq3) / 2
        return min(r1, r2, r3)

    def _draw_umich_gaussian(self, heatmap, center, radius, k=1):
        x, y = int(center[0]), int(center[1])
        if radius == 0:
            heatmap[y, x] = 1.
        else:
            diameter = 2 * radius
            gaussian = self._gaussian2D((diameter, diameter), sigma=diameter / 6)

            height, width = heatmap.shape[0:2]
                    
            left, right = min(x, radius), min(width - x, radius)
            top, bottom = min(y, radius), min(height - y, radius)

            masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
            masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
            if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
                heatmap[y - top:y + bottom, x - left:x + right] = torch.max(torch.stack((masked_heatmap, masked_gaussian * k)), dim=0)[0]
        return heatmap       

    def _gaussian2D(self, shape, sigma=1):
        m, n = [np.ceil((ss - 1.) / 2.) for ss in shape]
        y, x = np.ogrid[-m:m+1,-n:n+1]

        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return torch.from_numpy(h)                      