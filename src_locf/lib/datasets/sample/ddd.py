from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import pycocotools.coco as coco
import numpy as np
import torch
import json
import cv2
import os
import math
import copy
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from datasets.sample.preprocessing3d import compute_3d_center, \
    dimension_encode, orientation_encode, depth_encode, get_new_alpha, \
    orientation_quaternion_encode
import pycocotools.coco as coco


class DddDataset(data.Dataset):
    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def _convert_alpha(self, alpha):
        return math.radians(alpha + 45) if self.alpha_in_degree else alpha

    def __getitem__(self, index):
        # index = 4
        # img_id = 631
        img_id = self.images[index]
        img_info = self.coco.loadImgs(ids=[img_id])[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = cv2.imread(img_path)
        if 'calib' in img_info:
            calib = np.array(img_info['calib'], dtype=np.float32)
        else:
            calib = self.calib

        height, width = img.shape[0], img.shape[1]
        # image center
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.])
        if self.opt.keep_res:
            s = np.array([self.opt.input_w, self.opt.input_h], dtype=np.int32)
        else:
            s = np.array([width, height], dtype=np.int32)

        aug = False
        flipped = False

        K = calib
        
        if (self.split == 'train' or 'trainval') and (np.random.random() < self.opt.flip):
            flipped = True
            # print(flipped)
            img = img[:, ::-1, :]
            c[0] = width - c[0] - 1

            calib_new = calib
            calib_new[0, 2] = width - calib[0, 2] - 1
            K = calib_new

        if (self.split == 'train' or 'trainval') and (np.random.random() < 0.3):
            aug = True
            sf = self.opt.scale
            cf = self.opt.shift
            # s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            s = s * np.random.choice(np.arange(1 - sf, 1 + sf + 0.1, 0.1))
            c[0] += img.shape[1] * np.random.choice(np.arange(-2 * cf, 2 * cf + 0.1, 0.1))
            c[1] += img.shape[0] * np.random.choice(np.arange(-2 * cf, 2 * cf + 0.1, 0.1))
        
        trans_input = get_affine_transform(
            c, s, 0, [self.opt.input_w, self.opt.input_h])
        inp = cv2.warpAffine(img, trans_input,
                             (self.opt.input_w, self.opt.input_h),
                             flags=cv2.INTER_LINEAR)
        inp = (inp.astype(np.float32) / 255.)
        # if self.split == 'train' and not self.opt.no_color_aug:
        #   color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        # Todo: check if transformation of intrinsic correct here

        # cxcy = np.array([calib[0, 2], calib[1, 2]])
        # cxcy_transform = affine_transform(cxcy, trans_input)
        # calib[0, 2],  calib[1, 2] = cxcy_transform[0], cxcy_transform[1]

        num_classes = self.opt.num_classes
        trans_output = get_affine_transform(
            c, s, 0, [self.opt.output_w, self.opt.output_h])

        hm = np.zeros(
            (num_classes, self.opt.output_h, self.opt.output_w), dtype=np.float32)
        cls_ids = np.zeros((self.max_objs), dtype=np.int32)
        roty = np.zeros((self.max_objs), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        ind_2d = np.zeros((self.max_objs, 2), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        flip_mask = np.zeros((self.max_objs), dtype=np.uint8)
        reg_3d = np.zeros((self.max_objs, 8), dtype=np.float32)

        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)
        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
            draw_umich_gaussian
        gt_det = []
        for k in range(num_objs):
            ann = anns[k]
            new_ann = copy.deepcopy(ann)
            cls_id = int(self.cat_ids[ann['category_id']])
            truncation = ann['truncated']
            # if cls_id <= -99:
            if cls_id < 0:
                continue
            # in [x1, y1, x2, y2] form
            # bbox = self._coco_box_to_bbox(ann['bbox'])

            new_ann['new_alpha'] = get_new_alpha(new_ann['alpha'])
            new_ann, ct_3d, bboxfrom3d = compute_3d_center(new_ann, K, flipped)

            # if flipped:
            #     # bbox[[0, 2]] = width - bbox[[2, 0]] - 1
            #     ann['alpha'] = - ann['alpha']
            #     ann['location'][0] = - ann['location'][0]

            ct_3d = affine_transform(ct_3d, trans_output)
            bboxfrom3d[:2] = affine_transform(bboxfrom3d[:2], trans_output)
            bboxfrom3d[2:] = affine_transform(bboxfrom3d[2:], trans_output)
            # make sure bbox lies in image
            bboxfrom3d[[0, 2]] = np.clip(bboxfrom3d[[0, 2]], 0, self.opt.output_w - 1)
            bboxfrom3d[[1, 3]] = np.clip(bboxfrom3d[[1, 3]], 0, self.opt.output_h - 1)
            h, w = bboxfrom3d[3] - bboxfrom3d[1], bboxfrom3d[2] - bboxfrom3d[0]

            # Todo: check if we need h, w here
            # if (ct_3d[0] < 0 or ct_3d[0] > self.opt.output_w) or (ct_3d[1] < 0 or ct_3d[1] > self.opt.output_h):
            #     ct_3dint = ct_3d.astype(np.int32)
            #     ct_3dint[0] = int(np.clip(ct_3dint[0], 0, self.opt.output_w))
            #     delta_ct3d = ct_3d - ct_3dint

            if (0 < ct_3d[0] < self.opt.output_w) and (0 < ct_3d[1] < self.opt.output_h):
                ct_3dint = ct_3d.astype(np.int32)
                delta_ct3d = ct_3d - ct_3dint

            # else:
            #     ct_3dint = ct_3d.astype(np.int32)
            #     ct_3dint[0] = int(np.clip(ct_3dint[0], 0, self.opt.output_w - 1))
            #     ct_3dint[1] = int(np.clip(ct_3dint[1], 0, self.opt.output_h - 1))
            #     delta_ct3d = ct_3d - ct_3dint

                radius = gaussian_radius((h, w))
                radius = max(0, int(radius))
                # ct = np.array(
                #     [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)

                draw_gaussian(hm[cls_id], ct_3d, radius)

                # TODO: how to parametrize depth here?

                dep = np.asarray([new_ann['depth']])
                # delta_ct3d = ct_3d - ct_3dint
                delta_depth = depth_encode(dep, self.depth_pre)
                delta_dim = dimension_encode(cls_id, new_ann['dim'], self.dim_avg)
                orientation = orientation_encode(new_ann['new_alpha'])
                # orientation = orientation_quaternion_encode(new_ann['new_alpha'])

                cls_ids[k] = cls_id
                roty[k] = new_ann['rotation_y']
                reg_3d[k] = np.concatenate([delta_depth, delta_ct3d, delta_dim, orientation])
                ind[k] = ct_3dint[1] * self.opt.output_w + ct_3dint[0]
                ind_2d[k] = np.asarray([ct_3dint[0], ct_3dint[1]])
                reg_mask[k] = 1 if not aug else 0
                flip_mask[k] = 1 if not aug and flipped else 0
            # wh[k] = 1. * w, 1. * h
            # gt_det.append([ct_3d[0], ct_3d[1], 1] + \
            #               self._alpha_to_8(self._convert_alpha(ann['alpha'])) + \
            #               [ann['depth']] + (np.array(ann['dim']) / 1).tolist() + [cls_id])
            # if self.opt.reg_bbox:
            #     gt_det[-1] = gt_det[-1][:-1] + [w, h] + [gt_det[-1][-1]]
            # # if (not self.opt.car_only) or cls_id == 1: # Only estimate ADD for cars !!!
            # if 1:
            #     alpha = self._convert_alpha(ann['alpha'])
            #     # print('img_id cls_id alpha rot_y', img_path, cls_id, alpha, ann['rotation_y'])
            #     if alpha < np.pi / 6. or alpha > 5 * np.pi / 6.:
            #         rotbin[k, 0] = 1
            #         rotres[k, 0] = alpha - (-0.5 * np.pi)
            #     if alpha > -np.pi / 6. or alpha < -5 * np.pi / 6.:
            #         rotbin[k, 1] = 1
            #         rotres[k, 1] = alpha - (0.5 * np.pi)
            #     dep[k] = ann['depth']
            #     dim[k] = ann['dim']
            #     # print('        cat dim', cls_id, dim[k])
            #     ind[k] = ct_int[1] * self.opt.output_w + ct_int[0]
            #     reg[k] = ct_3d - ct_int
            #     reg_mask[k] = 1 if not aug else 0
            #     rot_mask[k] = 1
        # print('gt_det', gt_det)
        # print('')
        ret = {'img_id': img_id,
               'cls_id': cls_ids,
               'input': inp,
               'hm': hm,
               'roty': roty,
               'reg_3d': reg_3d,
               'reg_mask': reg_mask,
               'flip_mask': flip_mask,
               'ind': ind,
               '3dct': ind_2d,
               'K': K,
               'trans': np.float32(trans_output), }
        # if self.opt.reg_bbox:
        #     ret.update({'wh': wh})
        # if self.opt.reg_offset:
        #     ret.update({'reg': reg})
        # if self.opt.debug > 0 or not ('train' in self.split):
        #     gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
        #         np.zeros((1, 18), dtype=np.float32)
        #     meta = {'c': c, 's': s, 'gt_det': gt_det, 'calib': calib,
        #             'image_path': img_path, 'img_id': img_id}
        #     ret['meta'] = meta
        return ret
