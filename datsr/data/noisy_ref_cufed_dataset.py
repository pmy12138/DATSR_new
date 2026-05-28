import os

import cv2
import mmcv
import numpy as np
import torch.utils.data as data
from PIL import Image

from datsr.data.transforms import augment, mod_crop, totensor
from datsr.data.util import paired_paths_from_ann_file, paired_paths_from_folder
from datsr.utils import FileClient


class NoisyRefCUFEDDataset(data.Dataset):
    """CUFED-style dataset with noisy input, clean GT and clean reference.

    Expected folders:
        dataroot_in: noisy HR input images, filename-aligned with GT.
        dataroot_gt: clean HR target images.
        dataroot_ref: clean reference images.

    Returned tensors:
        img_in: clean HR GT.
        img_in_lq: noisy LR input for the restoration network.
        img_in_lq_clean: clean LR target for the denoising branch.
        img_in_up: noisy LR bicubic-upsampled to HR, kept as fallback.
        img_in_up_clean: clean LR bicubic-upsampled to HR.
        img_ref: clean HR reference image.
    """

    def __init__(self, opt):
        super(NoisyRefCUFEDDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend'].copy()

        self.in_folder = opt['dataroot_in']
        self.ref_folder = opt['dataroot_ref']
        self.gt_folder = opt['dataroot_gt']
        self.filename_tmpl = opt.get('filename_tmpl', '{}')

        if 'ann_file' in self.opt:
            self.paths = paired_paths_from_ann_file(
                [self.in_folder, self.ref_folder], ['in', 'ref'],
                self.opt['ann_file'])
        else:
            self.paths = paired_paths_from_folder(
                [self.in_folder, self.ref_folder], ['in', 'ref'],
                self.filename_tmpl)

    def _read_img(self, path, key):
        img_bytes = self.file_client.get(path, key)
        return mmcv.imfrombytes(img_bytes).astype(np.float32) / 255.

    def _gt_path_from_input(self, in_path):
        rel_path = os.path.relpath(in_path, self.in_folder)
        gt_path = os.path.join(self.gt_folder, rel_path)
        if os.path.exists(gt_path):
            return gt_path
        return os.path.join(self.gt_folder, os.path.basename(in_path))

    @staticmethod
    def _resize_to(img, size):
        h, w = img.shape[:2]
        target_h, target_w = size
        if h == target_h and w == target_w:
            return img
        pil_img = Image.fromarray(
            cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
        pil_img = pil_img.resize((target_w, target_h), Image.BICUBIC)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR).astype(
            np.float32) / 255.

    @staticmethod
    def _make_lr_and_up(img_hr, lq_size, hr_size):
        lq_h, lq_w = lq_size
        hr_h, hr_w = hr_size
        img_pil = Image.fromarray(
            cv2.cvtColor((img_hr * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
        img_lq = img_pil.resize((lq_w, lq_h), Image.BICUBIC)
        img_up = img_lq.resize((hr_w, hr_h), Image.BICUBIC)
        img_lq = cv2.cvtColor(np.array(img_lq), cv2.COLOR_RGB2BGR).astype(
            np.float32) / 255.
        img_up = cv2.cvtColor(np.array(img_up), cv2.COLOR_RGB2BGR).astype(
            np.float32) / 255.
        return img_lq, img_up

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        in_path = self.paths[index]['in_path']
        ref_path = self.paths[index]['ref_path']
        gt_path = self._gt_path_from_input(in_path)

        img_in_noisy = self._read_img(in_path, 'in')
        img_ref = self._read_img(ref_path, 'ref')
        img_gt = self._read_img(gt_path, 'gt')

        if self.opt['phase'] == 'train':
            gt_h = gt_w = self.opt['gt_size']
            img_in_noisy = self._resize_to(img_in_noisy, (gt_h, gt_w))
            img_gt = self._resize_to(img_gt, (gt_h, gt_w))
            img_ref = self._resize_to(img_ref, (gt_h, gt_w))

            img_gt, img_in_noisy, img_ref = augment(
                [img_gt, img_in_noisy, img_ref],
                self.opt.get('use_flip', False),
                self.opt.get('use_rot', False))
            padding = False
            original_size = (gt_h, gt_w)
        else:
            img_in_noisy = mod_crop(img_in_noisy, scale)
            img_gt = mod_crop(img_gt, scale)
            img_ref = mod_crop(img_ref, scale)

            img_in_h, img_in_w = img_in_noisy.shape[:2]
            img_ref_h, img_ref_w = img_ref.shape[:2]
            gt_h, gt_w = img_gt.shape[:2]
            padding = False

            target_h = max(img_in_h, img_ref_h, gt_h)
            target_w = max(img_in_w, img_ref_w, gt_w)
            if (img_in_h, img_in_w) != (target_h, target_w):
                padding = True
                img_in_noisy = mmcv.impad(
                    img_in_noisy, shape=(target_h, target_w), pad_val=0)
            if (img_ref_h, img_ref_w) != (target_h, target_w):
                padding = True
                img_ref = mmcv.impad(
                    img_ref, shape=(target_h, target_w), pad_val=0)
            if (gt_h, gt_w) != (target_h, target_w):
                padding = True
                img_gt = mmcv.impad(
                    img_gt, shape=(target_h, target_w), pad_val=0)

            gt_h, gt_w = target_h, target_w
            original_size = (img_in_h, img_in_w)

        lq_h, lq_w = gt_h // scale, gt_w // scale
        img_in_lq, img_in_up = self._make_lr_and_up(
            img_in_noisy, (lq_h, lq_w), (gt_h, gt_w))
        img_in_lq_clean, img_in_up_clean = self._make_lr_and_up(
            img_gt, (lq_h, lq_w), (gt_h, gt_w))

        img_gt, img_in_lq, img_in_lq_clean, img_in_up, img_in_up_clean, img_ref = totensor(
            [img_gt, img_in_lq, img_in_lq_clean, img_in_up, img_in_up_clean,
             img_ref],
            bgr2rgb=True,
            float32=True)

        return_dict = {
            'img_in': img_gt,
            'img_in_lq': img_in_lq,
            'img_in_lq_clean': img_in_lq_clean,
            'img_in_up': img_in_up,
            'img_in_up_clean': img_in_up_clean,
            'img_ref': img_ref,
        }

        if self.opt['phase'] != 'train':
            return_dict['lq_path'] = in_path
            return_dict['padding'] = padding
            return_dict['original_size'] = original_size

        return return_dict

    def __len__(self):
        return len(self.paths)
