import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (ConvModule, DepthwiseSeparableConvModule,
                      bias_init_with_prob)
from mmcv.ops.nms import batched_nms
from mmcv.runner import force_fp32

from mmdet.core import (MlvlPointGenerator, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from ..builder import HEADS, build_loss
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin
import torchvision
import copy



@HEADS.register_module()
class OTAHead(BaseDenseHead, BBoxTestMixin):
    def __init__(
            self,
            num_classes,
            width=1.0,
            strides=[8, 16, 32],
            in_channels=[576, 576, 576],
            loss_ref_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                reduction='sum',
                loss_weight=1.0),
            clip_len = 1,
            heads=4,
            drop=0.0,
            use_score=True,
            defualt_p=30,
            sim_thresh=0.75,
            pre_nms=0.75,
            ave=True,
            defulat_pre=750,
            use_mask=False,
            train_cfg=None,
            test_cfg=None,
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super(OTAHead, self).__init__()

        self.Afternum = defualt_p
        self.Prenum = defulat_pre
        self.simN = defualt_p
        self.nms_thresh = pre_nms
        self.n_anchors = 1
        self.use_score = use_score
        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False
        self.clip_len = clip_len

        self.width = int(in_channels[0] * width)
        self.trans = MSA_yolov(dim=self.width, out_dim=4 * self.width, num_heads=heads, attn_drop=drop)
        self.linear_pred = nn.Linear(int(self.width),
                                     num_classes)  # Mlp(in_features=512,hidden_features=self.num_classes+1)
        self.linear_output = nn.Linear(int(self.Afternum*clip_len*num_classes), num_classes)
        self.sim_thresh = sim_thresh
        self.ave = ave
        self.use_mask = use_mask
        self.loss_ref_cls = build_loss(loss_ref_cls)

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)
        self.prior_generator = MlvlPointGenerator(strides, offset=0)
        self.cls_out_channels = num_classes

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg

        # 设置assigner
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)



    def forward_train(self, cls_feat, reg_feat, cls_scores, bbox_preds, objectnesses,
                      img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore, nms_thresh=0.5):
        outputs = []
        outputs_decode = []
        origin_preds = []
        before_nms_features = []
        before_nms_regf = []

        for k, (c_f, r_f, cls_output, reg_output, obj_output, stride_this_level) in enumerate(
                zip(cls_feat, reg_feat, cls_scores, bbox_preds, objectnesses, self.strides)
        ):
            output = torch.cat([reg_output, obj_output, cls_output], 1)
            output_decode = torch.cat(
                [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
            )

            if self.use_l1:
                batch_size = reg_output.shape[0]
                hsize, wsize = reg_output.shape[-2:]
                reg_output = reg_output.view(
                    batch_size, self.n_anchors, 4, hsize, wsize
                )
                reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                    batch_size, -1, 4
                )
                origin_preds.append(reg_output.clone())
            outputs.append(output)
            before_nms_features.append(c_f)
            before_nms_regf.append(r_f)

            outputs_decode.append(output_decode)

        self.hw = [x.shape[-2:] for x in outputs_decode]

        outputs_decode = torch.cat([x.flatten(start_dim=2) for x in outputs_decode], dim=2
                                   ).permute(0, 2, 1)
        decode_res = self.decode_outputs(outputs_decode)

        pred_result, topk_idx = self.postpro_woclass(decode_res, num_classes=self.num_classes, nms_thre=self.nms_thresh,
                                                     topK=self.Afternum)  # postprocess(decode_res,num_classes=30)

        cls_feat_flatten = torch.cat(
            [x.flatten(start_dim=2) for x in before_nms_features], dim=2
        ).permute(0, 2, 1)  # [b,features,channels]
        reg_feat_flatten = torch.cat(
            [x.flatten(start_dim=2) for x in before_nms_regf], dim=2
        ).permute(0, 2, 1)

        features_cls, features_reg, topk_cls_scores, fg_scores = self.find_feature_score(cls_feat_flatten, topk_idx,
                                                                                    reg_feat_flatten, None,
                                                                                    pred_result)

        features_reg = features_reg.unsqueeze(0)
        features_cls = features_cls.unsqueeze(0)
        if not self.training:
            topk_cls_scores = topk_cls_scores.to(cls_feat_flatten.device)
            fg_scores = fg_scores.to(cls_feat_flatten.device)
        # MSA在这里
        if self.use_score:
            trans_cls = self.trans(features_cls, features_reg, topk_cls_scores, fg_scores, sim_thresh=self.sim_thresh,
                                   ave=self.ave, use_mask=self.use_mask)
        else:
            trans_cls = self.trans(features_cls, features_reg, None, None, sim_thresh=self.sim_thresh, ave=self.ave)
        fc_pred = self.linear_pred(trans_cls)
        fc_output = self.linear_output(torch.flatten(fc_pred))# 全连接层出预测结果

        target = self.get_one_hot(torch.cat(gt_labels), self.num_classes)

        loss = self.loss(fc_output.unsqueeze(0), target)
        return loss

    def forward_test(self, cls_feat, reg_feat, cls_scores, bbox_preds, objectnesses, nms_thresh=0.5):
        outputs_decode = []

        before_nms_features = []
        before_nms_regf = []

        for k, (c_f, r_f, cls_output, reg_output, obj_output, stride_this_level) in enumerate(
                zip(cls_feat, reg_feat, cls_scores, bbox_preds, objectnesses, self.strides)
        ):
            output_decode = torch.cat(
                [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
            )

            # which features to choose
            before_nms_features.append(c_f)
            before_nms_regf.append(r_f)
            outputs_decode.append(output_decode)

        self.hw = [x.shape[-2:] for x in outputs_decode]

        outputs_decode = torch.cat([x.flatten(start_dim=2) for x in outputs_decode], dim=2
                                   ).permute(0, 2, 1)
        decode_res = self.decode_outputs(outputs_decode)


        pred_result, pred_idx = self.postpro_woclass(decode_res, num_classes=self.num_classes, nms_thre=self.nms_thresh,
                                                     topK=self.Afternum)  # postprocess(decode_res,num_classes=30)
        # return pred_result
        # if not self.training and cls_output.shape[0] == 1:
        #     return self.postprocess_single_img(pred_result, self.num_classes)

        cls_feat_flatten = torch.cat(
            [x.flatten(start_dim=2) for x in before_nms_features], dim=2
        ).permute(0, 2, 1)  # [b,features,channels]
        reg_feat_flatten = torch.cat(
            [x.flatten(start_dim=2) for x in before_nms_regf], dim=2
        ).permute(0, 2, 1)

        features_cls, features_reg, cls_scores, fg_scores = self.find_feature_score(cls_feat_flatten, pred_idx,
                                                                                    reg_feat_flatten, None,
                                                                                    pred_result)
        features_reg = features_reg.unsqueeze(0)
        features_cls = features_cls.unsqueeze(0)
        if not self.training:
            cls_scores = cls_scores.to(cls_feat_flatten.dtype)
            fg_scores = fg_scores.to(cls_feat_flatten.dtype)
        # MSA在这里
        if self.use_score:
            trans_cls = self.trans(features_cls, features_reg, cls_scores, fg_scores, sim_thresh=self.sim_thresh,
                                   ave=self.ave, use_mask=self.use_mask)
        else:
            trans_cls = self.trans(features_cls, features_reg, None, None, sim_thresh=self.sim_thresh, ave=self.ave)
        fc_pred = self.linear_pred(trans_cls)
        fc_output = self.linear_output(torch.flatten(fc_pred))# 全连接层出预测结果

        return fc_output # result

    def get_output_and_grid(self, output, k, stride):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2)
            self.grids[k] = grid

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.n_anchors * hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2).to(output.device)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid

    def get_one_hot(self, label, num_classes):
        batch_size = label.shape[0]
        onehot_label = torch.zeros((batch_size, num_classes))
        onehot_label = onehot_label.scatter_(1, label.unsqueeze(1).detach().cpu(), 1)
        onehot_label = (onehot_label.type(torch.FloatTensor)).to(label.device)
        return onehot_label

    def decode_outputs(self, outputs):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).to(outputs.device)
        strides = torch.cat(strides, dim=1).to(outputs.device)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs

    def find_feature_score(self, features, idxs, reg_features, imgs=None, predictions=None, roi_features=None):
        features_cls = []
        features_reg = []
        cls_scores = []
        fg_scores = []
        for i, feature in enumerate(features):
            features_cls.append(feature[idxs[i][:self.simN]])
            features_reg.append(reg_features[i, idxs[i][:self.simN]])
            cls_scores.append(predictions[i][:self.simN, 5])
            fg_scores.append(predictions[i][:self.simN, 4])
        features_cls = torch.cat(features_cls)
        features_reg = torch.cat(features_reg)
        cls_scores = torch.cat(cls_scores)
        fg_scores = torch.cat(fg_scores)
        return features_cls, features_reg, cls_scores, fg_scores

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'objectnesses'))
    def loss(self,
             cls_preds,
             cls_targets):

        loss_ref_cls = self.loss_ref_cls(
            cls_preds,
            cls_targets)

        loss_dict = dict(
            loss_ref_cls=loss_ref_cls)

        return loss_dict

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignment_last_frame(self,
             cls_scores,
             bbox_preds,
             objectnesses,
             gt_bboxes,
             gt_labels,
             gt_bboxes_ignore=None):

        num_imgs = cls_scores[0].shape[0]
        frame_num = cls_scores[0].shape[0]
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)

        #
        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.cls_out_channels)[-1].unsqueeze(0)
            for cls_pred in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)[-1].unsqueeze(0)
            for bbox_pred in bbox_preds
        ]
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)[-1].unsqueeze(0)
            for objectness in objectnesses
        ]

        
        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1)
        flatten_priors = torch.cat(mlvl_priors)
        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)

        (pos_masks, cls_targets, obj_targets, bbox_targets, l1_targets,
         num_fg_imgs) = multi_apply(
            self._get_target_single, flatten_cls_preds.detach(),
            flatten_objectness.detach(),
            flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1),
            flatten_bboxes.detach(), gt_bboxes, gt_labels)

        return pos_masks, cls_targets, obj_targets, bbox_targets, l1_targets, num_fg_imgs

    def _bbox_decode(self, priors, bbox_preds):
        xys = (bbox_preds[..., :2] * priors[:, 2:]) + priors[:, :2]
        whs = bbox_preds[..., 2:].exp() * priors[:, 2:]

        tl_x = (xys[..., 0] - whs[..., 0] / 2)
        tl_y = (xys[..., 1] - whs[..., 1] / 2)
        br_x = (xys[..., 0] + whs[..., 0] / 2)
        br_y = (xys[..., 1] + whs[..., 1] / 2)

        decoded_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], -1)
        return decoded_bboxes

    @torch.no_grad()
    def _get_target_single(self, cls_preds, objectness, priors, decoded_bboxes,
                           gt_bboxes, gt_labels):
        """Compute classification, regression, and objectness targets for
        priors in a single image.
        Args:
            cls_preds (Tensor): Classification predictions of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            objectness (Tensor): Objectness predictions of one image,
                a 1D-Tensor with shape [num_priors]
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Decoded bboxes predictions of one image,
                a 2D-Tensor with shape [num_priors, 4] in [tl_x, tl_y,
                br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
        """

        num_priors = priors.size(0)
        num_gts = gt_labels.size(0)
        gt_bboxes = gt_bboxes.to(decoded_bboxes.dtype)
        # No target
        if num_gts == 0:
            cls_target = cls_preds.new_zeros((0, self.num_classes))
            bbox_target = cls_preds.new_zeros((0, 4))
            l1_target = cls_preds.new_zeros((0, 4))
            obj_target = cls_preds.new_zeros((num_priors, 1))
            foreground_mask = cls_preds.new_zeros(num_priors).bool()
            return (foreground_mask, cls_target, obj_target, bbox_target,
                    l1_target, 0)

        # YOLOX uses center priors with 0.5 offset to assign targets,
        # but use center priors without offset to regress bboxes.
        offset_priors = torch.cat(
            [priors[:, :2] + priors[:, 2:] * 0.5, priors[:, 2:]], dim=-1)

        assign_result = self.assigner.assign(
            cls_preds.sigmoid() * objectness.unsqueeze(1).sigmoid(),
            offset_priors, decoded_bboxes, gt_bboxes, gt_labels)

        sampling_result = self.sampler.sample(assign_result, priors, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        num_pos_per_img = pos_inds.size(0)

        pos_ious = assign_result.max_overlaps[pos_inds]
        # IOU aware classification score
        cls_target = F.one_hot(sampling_result.pos_gt_labels,
                               self.num_classes) * pos_ious.unsqueeze(-1)
        obj_target = torch.zeros_like(objectness).unsqueeze(-1)
        obj_target[pos_inds] = 1
        bbox_target = sampling_result.pos_gt_bboxes
        l1_target = cls_preds.new_zeros((num_pos_per_img, 4))
        if self.use_l1:
            l1_target = self._get_l1_target(l1_target, bbox_target,
                                            priors[pos_inds])
        foreground_mask = torch.zeros_like(objectness).to(torch.bool)
        foreground_mask[pos_inds] = 1
        return (foreground_mask, cls_target, obj_target, bbox_target,
                l1_target, num_pos_per_img)

    def _get_l1_target(self, l1_target, gt_bboxes, priors, eps=1e-8):
        """Convert gt bboxes to center offset and log width height."""
        gt_cxcywh = bbox_xyxy_to_cxcywh(gt_bboxes)
        l1_target[:, :2] = (gt_cxcywh[:, :2] - priors[:, :2]) / priors[:, 2:]
        l1_target[:, 2:] = torch.log(gt_cxcywh[:, 2:] / priors[:, 2:] + eps)
        return l1_target

    def get_in_boxes_info(
            self,
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
    ):
        expanded_strides_per_image = expanded_strides[0]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
                .unsqueeze(0)
                .repeat(num_gt, 1)
        )  # [n_anchor] -> [n_gt, n_anchor]
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
                .unsqueeze(0)
                .repeat(num_gt, 1)
        )

        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
                .unsqueeze(1)
                .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
                .unsqueeze(1)
                .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
                .unsqueeze(1)
                .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
                .unsqueeze(1)
                .repeat(1, total_num_anchors)
        )

        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        # in fixed center

        center_radius = 4.5

        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (
                is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds

    def postpro_woclass(self, prediction, num_classes, nms_thre=0.75, topK=75):
        # find topK predictions, play the same role as RPN
        '''

        Args:
            prediction: [batch,feature_num,5+clsnum]
            num_classes:
            conf_thre:
            conf_thre_high:
            nms_thre:

        Returns:
            [batch,topK,5+clsnum]
        '''
        self.topK = topK
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]
        output = [None for _ in range(len(prediction))]
        output_index = [None for _ in range(len(prediction))]
        features_list = []
        for i, image_pred in enumerate(prediction):

            if not image_pred.size(0):
                continue
            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

            detections = torch.cat(
                (image_pred[:, :5], class_conf, class_pred.float(), image_pred[:, 5: 5 + num_classes]), 1)

            conf_score = image_pred[:, 4]
            top_pre = torch.topk(conf_score, k=self.Prenum)
            sort_idx = top_pre.indices[:self.Prenum]
            detections_temp = detections[sort_idx, :]
            nms_out_index = torchvision.ops.batched_nms(
                detections_temp[:, :4],
                detections_temp[:, 4] * detections_temp[:, 5],
                detections_temp[:, 6],
                nms_thre,
            )

            topk_idx = sort_idx[nms_out_index[:self.topK]]
            output[i] = detections[topk_idx, :]
            output_index[i] = topk_idx

        return output, output_index

    def postprocess_single_img(self, prediction, num_classes, conf_thre=0.001, nms_thre=0.5):

        output_ori = [None for _ in range(len(prediction))]
        prediction_ori = copy.deepcopy(prediction)
        for i, detections in enumerate(prediction):

            if not detections.size(0):
                continue

            detections_ori = prediction_ori[i]

            conf_mask = (detections_ori[:, 4] * detections_ori[:, 5] >= conf_thre).squeeze()
            detections_ori = detections_ori[conf_mask]
            nms_out_index = torchvision.ops.batched_nms(
                detections_ori[:, :4],
                detections_ori[:, 4] * detections_ori[:, 5],
                detections_ori[:, 6],
                nms_thre,
            )
            detections_ori = detections_ori[nms_out_index]
            output_ori[i] = detections_ori
        # print(output)
        return output_ori, output_ori

class MSA_yolov(nn.Module):
    def __init__(self, dim, out_dim, num_heads=4, qkv_bias=False, attn_drop=0., scale=25):
        super(MSA_yolov, self).__init__()
        self.msa = Attention_msa(dim, num_heads, qkv_bias, attn_drop, scale=scale)
        self.linear1 = nn.Linear(2 * dim, 2 * dim)
        self.linear2 = nn.Linear(4 * dim, out_dim)

    def find_similar_round2(self, features, sort_results):
        key_feature = features[0]
        support_feature = features[0]
        if not self.training:
            sort_results = sort_results.to(features.dtype)
        soft_sim_feature = (
                    sort_results @ support_feature)  # .transpose(1, 2)#torch.sum(softmax_value * most_sim_feature, dim=1)
        cls_feature = torch.cat([soft_sim_feature, key_feature], dim=-1)
        return cls_feature

    def forward(self, x_cls, x_reg, cls_score=None, fg_score=None, sim_thresh=0.75, ave=True, use_mask=False):
        trans_cls, trans_reg, sim_round2 = self.msa(x_cls, x_reg, cls_score, fg_score, sim_thresh=sim_thresh, ave=ave,
                                                    use_mask=use_mask)

        return trans_cls

class Attention_msa(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., scale=25):
        # dim :input[batchsize,sequence length, input dimension]-->output[batchsize, sequence lenght, dim]
        # qkv_bias : Is it matter?
        # qk_scale, attn_drop,proj_drop will not be used
        # object = Attention(dim,num head)
        super(Attention_msa, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = scale  # qk_scale or head_dim ** -0.5

        self.qkv_cls = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_reg = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x_cls, x_reg, cls_score=None, fg_score=None, return_attention=False, ave=True, sim_thresh=0.75,
                use_mask=False):
        B, N, C = x_cls.shape

        qkv_cls = self.qkv_cls(x_cls).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                                    4)  # 3, B, num_head, N, c
        qkv_reg = self.qkv_reg(x_reg).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_cls, k_cls, v_cls = qkv_cls[0], qkv_cls[1], qkv_cls[2]  # make torchscript happy (cannot use tensor as tuple)
        q_reg, k_reg, v_reg = qkv_reg[0], qkv_reg[1], qkv_reg[2]

        q_cls = q_cls / torch.norm(q_cls, dim=-1, keepdim=True)
        k_cls = k_cls / torch.norm(k_cls, dim=-1, keepdim=True)
        q_reg = q_reg / torch.norm(q_reg, dim=-1, keepdim=True)
        k_reg = k_reg / torch.norm(k_reg, dim=-1, keepdim=True)
        v_cls_normed = v_cls / torch.norm(v_cls, dim=-1, keepdim=True)

        if cls_score == None:
            cls_score = 1
        else:
            cls_score = torch.reshape(cls_score, [1, 1, 1, -1]).repeat(1, self.num_heads, N, 1)

        if fg_score == None:
            fg_score = 1
        else:
            fg_score = torch.reshape(fg_score, [1, 1, 1, -1]).repeat(1, self.num_heads, N, 1)

        attn_cls_raw = v_cls_normed @ v_cls_normed.transpose(-2, -1)

        attn_cls = (q_cls @ k_cls.transpose(-2, -1))
        attn_cls = attn_cls.softmax(dim=-1)
        attn_cls = self.attn_drop(attn_cls)

        attn_reg = (q_reg @ k_reg.transpose(-2, -1))
        attn_reg = attn_reg.softmax(dim=-1)
        attn_reg = self.attn_drop(attn_reg)

        attn = (attn_reg + attn_cls) / 2
        x = (attn @ v_cls).transpose(1, 2).reshape(B, N, C)

        x_ori = v_cls.permute(0, 2, 1, 3).reshape(B, N, C)

        x_output = x + x_cls

        if ave:
            ones_matrix = torch.ones(attn.shape[2:]).to(x_cls.device)
            zero_matrix = torch.zeros(attn.shape[2:]).to(x_cls.device)

            attn_cls_raw = torch.sum(attn_cls_raw, dim=1, keepdim=False)[0] / self.num_heads
            sim_mask = torch.where(attn_cls_raw > sim_thresh, ones_matrix, zero_matrix)
            sim_attn = torch.sum(attn, dim=1, keepdim=False)[0] / self.num_heads

            sim_round2 = torch.softmax(sim_attn, dim=-1)
            sim_round2 = sim_mask * sim_round2 / (torch.sum(sim_mask * sim_round2, dim=-1, keepdim=True))
            return x_output, None, sim_round2
        else:
            return x_output, None, None

def postprocess(prediction, num_classes, fc_outputs, conf_thre=0.001, nms_thre=0.5):
    output = [None for _ in range(len(prediction))]
    output_ori = [None for _ in range(len(prediction))]
    prediction_ori = copy.deepcopy(prediction)
    cls_conf, cls_pred = torch.max(fc_outputs, -1, keepdim=False)

    for i, detections in enumerate(prediction):

        if not detections.size(0):
            continue

        detections[:, 5] = cls_conf[i].sigmoid()
        detections[:, 6] = cls_pred[i]
        tmp_cls_score = fc_outputs[i].sigmoid()
        cls_mask = tmp_cls_score >= conf_thre
        cls_loc = torch.where(cls_mask)
        scores = torch.gather(tmp_cls_score[cls_loc[0]],dim=-1,index=cls_loc[1].unsqueeze(1))#[:,cls_loc[1]]#tmp_cls_score[torch.stack(cls_loc).T]#torch.gather(tmp_cls_score, dim=1, index=torch.stack(cls_loc).T)

        detections[:, -num_classes:] = tmp_cls_score
        detections_raw = detections[:, :7]
        new_detetions = detections_raw[cls_loc[0]]
        new_detetions[:, -1] = cls_loc[1]
        new_detetions[:,5] = scores.squeeze()
        detections_high = new_detetions  # new_detetions
        detections_ori = prediction_ori[i]
        #print(len(detections_high.shape))

        conf_mask = (detections_high[:, 4] * detections_high[:, 5] >= conf_thre).squeeze()
        detections_high = detections_high[conf_mask]

        if not detections_high.shape[0]:
            continue
        if len(detections_high.shape)==3:
            detections_high = detections_high[0]
        nms_out_index = torchvision.ops.batched_nms(
            detections_high[:, :4],
            detections_high[:, 4] * detections_high[:, 5],
            detections_high[:, 6],
            nms_thre,
        )

        detections_high = detections_high[nms_out_index]
        output[i] = detections_high
        detections_ori = detections_ori[:, :7]
        conf_mask = (detections_ori[:, 4] * detections_ori[:, 5] >= conf_thre).squeeze()
        detections_ori = detections_ori[conf_mask]
        nms_out_index = torchvision.ops.batched_nms(
            detections_ori[:, :4],
            detections_ori[:, 4] * detections_ori[:, 5],
            detections_ori[:, 6],
            nms_thre,
        )

        detections_ori = detections_ori[nms_out_index]
        output_ori[i] = detections_ori

    return output, output_ori