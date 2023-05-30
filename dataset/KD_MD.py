import contextlib
import copy
import io
import itertools
import logging
import os
import cv2
import os.path as osp
import tempfile
import warnings
import pandas as pd
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, \
    precision_score, recall_score, f1_score, classification_report, confusion_matrix
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .builder import DATASETS
from .custom import CustomDataset
from mmdet.core import eval_map, eval_recalls, get_best_results



@DATASETS.register_module()
class KDMDDataset(CustomDataset):

    # CLASSES = ("Atypical benign", "Typical benign",
    #            "RCC-1", "RCC-2", "RCC-3", "RCC-4")

    CLASSES = ("benign", "RCC")

    # CLASSES = ("benign_RCC", )

    DOMAIN=("B", "CEUS")

    def load_annotations(self, ann_file):
        """Load annotation from .csv annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info.
        """
        ID_CLASSES = {i: c for i, c in enumerate(self.CLASSES)}
        CLASSES_ID = {c: i for i, c in enumerate(self.CLASSES)}


        total_file = os.listdir(ann_file)
        data_infos = []
        ids_data = dict()

        for id, file_name in enumerate(total_file):
            img_id = id
            file_name = file_name.replace(".txt", ".jpg")
            img_ = cv2.imdecode(np.fromfile(os.path.join(self.img_prefix, self.DOMAIN[0], file_name),
                                            dtype=np.uint8), 1)

            width = img_.shape[1]
            height = img_.shape[0]

            data_infos.append(dict(id=img_id, filename=file_name, width=width, height=height))


        return data_infos

    def get_ann_info(self, idx):
        """Get annotation from XML file by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        file_name = self.data_infos[idx]["filename"].replace(".jpg", ".txt")
        width = self.data_infos[idx]["width"]
        height = self.data_infos[idx]["height"]

        txtFile = open(os.path.join(self.ann_file, file_name))
        txtList = txtFile.readlines()

        # 读取txt文件
        bboxes = []
        labels = []
        for line in txtList:
            oneline = line.strip().split(" ")
            # 读取label
            label = oneline[0]

            # 调整label，将非典型良性和典型良性合并成良性，将RCC1234合并成RCC
            if int(label) in [0,1] and len(self.CLASSES) == 2:
                label = '0'
            elif int(label) in [2,3,4,5,6,7,8,9] and len(self.CLASSES) == 2:
                label = '1'
            elif len(self.CLASSES) == 1:
                # 调整label，所有都合成一类
                label = '0'
            # if int(label) is 0 and len(self.CLASSES) == 2:
            #     label = '0'
            # elif int(label) is 1 and len(self.CLASSES) == 2:
            #     label = '1'
            # elif len(self.CLASSES) == 1:
            #     # 调整label，所有都合成一类
            #     label = '0'


            labels.append(label)

            # 读取bbox
            xmin = int(((float(oneline[1])) * width + 1) - (float(oneline[3])) * 0.5 * width)
            ymin = int(((float(oneline[2])) * height + 1) - (float(oneline[4])) * 0.5 * height)
            xmax = int(((float(oneline[1])) * width + 1) + (float(oneline[3])) * 0.5 * width)
            ymax = int(((float(oneline[2])) * height + 1) + (float(oneline[4])) * 0.5 * height)

            bbox = [xmin, ymin, xmax, ymax]
            bboxes.append(bbox)

        bboxes_ignore = np.zeros((0, 4))
        labels_ignore = np.zeros((0,))

        ann = dict(
            bboxes=np.array(bboxes).astype(np.float32),
            labels=np.array(labels).astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """

        if self.test_mode:
            data_s =  self.prepare_test_img(idx)
            return data_s
        while True:
            data_s = self.prepare_train_img(idx)
            if data_s is None:
                idx = self._rand_another(idx)
                continue
            return data_s

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        img = self.pipeline(results)

        return img

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by \
                pipeline.
        """

        img_info = self.data_infos[idx]
        results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        img = self.pipeline(results)

        return img

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall', 'best_mAP', 'classify']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()

        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, _ = eval_map(
                    results,
                    annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=iou_thr,
                    dataset=self.CLASSES,
                    logger=logger)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thrs):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        elif metric == 'best_mAP':
            assert isinstance(iou_thrs, list)
            best_results = get_best_results(results, len(self.CLASSES))
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, _ = eval_map(
                    best_results,
                    annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=iou_thr,
                    dataset=self.CLASSES,
                    logger=logger)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['best_mAP'] = sum(mean_aps) / len(mean_aps)
        elif metric == 'classify':
            best_results = get_best_results(results, len(self.CLASSES))
            label_all = []
            pred_all = []
            prob_all = []
            for anno in annotations:
                label = torch.tensor(anno['labels']).squeeze()
                label_all.append(label)

            for result in best_results:
                if result[0].shape == (0,5):
                    pred_all.append(1)
                else:
                    pred_all.append(0)

            acc = accuracy_score(label_all, pred_all)
            precision = precision_score(label_all, pred_all, average="macro")
            recall = recall_score(label_all, pred_all, average="macro")
            f1_s = f1_score(label_all, pred_all, average="macro")

            print_log('\n' + "acc = " + str(acc), logger=logger)
            print_log('\n' + "precision = " + str(precision), logger=logger)
            print_log('\n' + "recall = " + str(recall), logger=logger)
            print_log('\n' + "f1_s = " + str(f1_s), logger=logger)
            eval_results['acc'] = acc
            eval_results['precision'] = precision
            eval_results['recall'] = recall
            eval_results['f1_s'] = f1_s

        return eval_results