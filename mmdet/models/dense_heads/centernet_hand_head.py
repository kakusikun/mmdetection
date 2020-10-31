from abc import abstractmethod
import math

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
                 feat_channels={'hm': 1, 'wh': 2, 'offset': 2, 'orie': 8},
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 dcn_on_last_conv=False,
                 conv_bias='auto',
                 loss_hm=dict(type='CenterNetFocalLoss', loss_weight=1.0),
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

    @abstractmethod
    @force_fp32(apply_to=('hm_feat', 'wh_feat', 'offset_feat', 'orie_feat'))
    def loss(self,
             hm_feat,
             wh_feat,
             offset_feat,
             orie_feat,
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



        raise NotImplementedError

    @abstractmethod
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

    @abstractmethod
    def get_targets(self, 
            hm_feat_sizes,
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
        hm_targets = torch.zeros_like(hm_feat)
        for bidx, gt_bbox in enumerate(gt_bboxes):
            h, w = gt_bbox[3] - gt_bbox[1], gt_bbox[2] - gt_bbox[0]
            radius = self._gaussian_radius((math.ceil(h), math.ceil(w)))
            radius = max(0, int(radius))
            center = torch.Tensor([(gt_bbox[0] + gt_bbox[2]) / 2, (gt_bbox[1] + gt_bbox[3]) / 2])
            center_int = center.int()
            self._draw_umich_gaussian(hm_targets[0][bidx], ct_int, radius)
        wh[k] = 1. * w, 1. * h
        ind[k] = ct_int[1] * output_w + ct_int[0]
        reg[k] = center - ct_int
        reg_mask[k] = 1
        cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
        cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
        if self.opt.dense_wh:
          draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
        gt_det.append([center[0] - w / 2, center[1] - h / 2, 
                       center[0] + w / 2, center[1] + h / 2, 1, cls_id])


        

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points of a single scale level."""
        h, w = featmap_size
        x_range = torch.arange(w, dtype=dtype, device=device)
        y_range = torch.arange(h, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        if flatten:
            y = y.flatten()
            x = x.flatten()
        return y, x

    def get_points(self, featmap_sizes, dtype, device, flatten=False):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self._get_points_single(featmap_sizes[i], self.strides[i],
                                        dtype, device, flatten))
        return mlvl_points

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
        diameter = 2 * radius + 1
        gaussian = self._gaussian2D((diameter, diameter), sigma=diameter / 6)
        
        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]
                
        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
                np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap       

    def _gaussian2D(self, shape, sigma=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m+1,-n:n+1]

        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return torch.from_numpy(h)                      