# datsr/data/mc_denoising_dataset.py

import cv2
import mmcv
import numpy as np
import os
import random

import torch
import torch.utils.data as data
from PIL import Image
from datsr.data.transforms import augment, mod_crop, totensor
from datsr.utils import FileClient


def generate_ref_affine(img, angle_range=5.0, scale_range=(0.95, 1.05),
                        max_translate=10):
    """对图像施加轻微仿射变换，模拟不同视角的参考图。

    在 GT crop (160×160) 上操作，确保参考图与 GT 高度重叠。

    Args:
        img: numpy array, HWC, float32, [0, 1]
        angle_range: 最大旋转角度（度），实际范围 [-angle_range, angle_range]
        scale_range: 缩放范围 (min_scale, max_scale)
        max_translate: 最大平移像素数
    Returns:
        ref_img: 变换后的图像，与输入同尺寸
    """
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)

    # 随机参数
    angle = random.uniform(-angle_range, angle_range)
    scale = random.uniform(scale_range[0], scale_range[1])
    tx = random.uniform(-max_translate, max_translate)
    ty = random.uniform(-max_translate, max_translate)

    # 构建仿射矩阵：旋转 + 缩放
    M = cv2.getRotationMatrix2D(center, angle, scale)
    # 叠加平移
    M[0, 2] += tx
    M[1, 2] += ty

    # 使用 BORDER_REFLECT 避免黑边
    ref_img = cv2.warpAffine(img, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REFLECT_101)
    return ref_img


class MCDenoisingDataset(data.Dataset):
    """MC渲染降噪+超分数据集。

    文件夹结构:
        dataroot_in/    — 有噪 LR (320×180)
        dataroot_ref/   — default HR (1280×720)，同时作为 GT 和参考图来源
        dataroot_albedo/ — albedo HR (1280×720)，辅助输入通道
    """

    def __init__(self, opt):
        super(MCDenoisingDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend'].copy()

        self.in_folder = opt['dataroot_in']
        self.ref_folder = opt['dataroot_ref']
        self.albedo_folder = opt.get('dataroot_albedo', None)

        # ---------- 扫描文件并建立路径映射 ----------
        self.paths = []
        in_names = sorted([
            f for f in os.listdir(self.in_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))
        ])

        for name in in_names:
            in_path = os.path.join(self.in_folder, name)
            ref_path = os.path.join(self.ref_folder, name)

            albedo_path = None
            if self.albedo_folder is not None:
                albedo_name = name.replace('_spp_', '_albedo_spp_', 1)
                candidate = os.path.join(self.albedo_folder, albedo_name)
                if os.path.exists(candidate):
                    albedo_path = candidate

            if os.path.exists(ref_path):
                entry = {
                    'in_path': in_path,
                    'ref_path': ref_path,
                }
                if albedo_path is not None:
                    entry['albedo_path'] = albedo_path
                self.paths.append(entry)

        print(f"[MCDenoisingDataset] Found {len(self.paths)} image pairs")

    # datsr/data/mc_denoising_dataset.py — __getitem__ 方法

    def __getitem__(self, index):
        scale = self.opt.get('scale', 4)
        gt_size = self.opt.get('gt_size', 160)
        lq_size = gt_size // scale  # 40

        # ========== 1. 加载图像 ==========
        # (a) 有噪 LR 输入 (320×180 或 256×256)
        in_path = self.paths[index]['in_path']
        img_in_noisy = cv2.imread(in_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.

        # (b) default HR (1280×720) — 同时作为 GT 来源和参考图来源
        ref_path = self.paths[index]['ref_path']
        img_ref_full = cv2.imread(ref_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        h_hr, w_hr = img_ref_full.shape[:2]  # 1280×720 → h=720, w=1280

        # (c) albedo HR (1280×720)
        img_albedo_full = None
        if 'albedo_path' in self.paths[index] and self.paths[index]['albedo_path'] is not None:
            albedo_path = self.paths[index]['albedo_path']
            img_albedo_full = cv2.imread(albedo_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.

            # ========== 2. 将有噪输入 resize 到标准 LQ 尺寸 ==========
        # LQ 尺寸 = HR 尺寸 / scale，确保 4x 空间对应关系
        target_lq_h, target_lq_w = h_hr // scale, w_hr // scale  # 180, 320
        h_in, w_in = img_in_noisy.shape[:2]
        if h_in != target_lq_h or w_in != target_lq_w:
            img_in_noisy = cv2.resize(img_in_noisy, (target_lq_w, target_lq_h),
                                      interpolation=cv2.INTER_CUBIC)

            # ========== 3. 将 albedo HR 下采样到 LQ 尺寸 ==========
        if img_albedo_full is not None:
            img_albedo_lq = cv2.resize(img_albedo_full, (target_lq_w, target_lq_h),
                                       interpolation=cv2.INTER_CUBIC)
        else:
            # 如果没有 albedo，用零填充
            img_albedo_lq = np.zeros_like(img_in_noisy)

            # ========== 4. 裁剪 ==========
        if self.opt['phase'] == 'train':
            # --- 随机裁剪 ---
            # 在 LQ 空间随机选取裁剪位置
            top_lq = random.randint(0, max(0, target_lq_h - lq_size))
            left_lq = random.randint(0, max(0, target_lq_w - lq_size))

            # LQ 裁剪 (40×40)
            img_in_noisy_patch = img_in_noisy[top_lq:top_lq + lq_size,
                                 left_lq:left_lq + lq_size, :]
            img_albedo_patch = img_albedo_lq[top_lq:top_lq + lq_size,
                               left_lq:left_lq + lq_size, :]

            # 对应 HR 裁剪位置 (160×160)
            top_gt = top_lq * scale
            left_gt = left_lq * scale
            img_gt = img_ref_full[top_gt:top_gt + gt_size,
                     left_gt:left_gt + gt_size, :]

            # --- 从 GT patch 生成仿射变换参考图 (方案二) ---
            img_ref = self._generate_ref_affine(img_gt)

            # --- 数据增强（对 GT、Ref、LQ noisy、LQ albedo 施加相同的翻转/旋转）---
            if self.opt.get('use_flip', False) or self.opt.get('use_rot', False):
                img_gt, img_ref, img_in_noisy_patch, img_albedo_patch = augment(
                    [img_gt, img_ref, img_in_noisy_patch, img_albedo_patch],
                    self.opt.get('use_flip', False),
                    self.opt.get('use_rot', False))

        else:
            # --- 验证/测试阶段：中心裁剪 ---
            center_lq_h = (target_lq_h - lq_size) // 2
            center_lq_w = (target_lq_w - lq_size) // 2

            img_in_noisy_patch = img_in_noisy[center_lq_h:center_lq_h + lq_size,
                                 center_lq_w:center_lq_w + lq_size, :]
            img_albedo_patch = img_albedo_lq[center_lq_h:center_lq_h + lq_size,
                               center_lq_w:center_lq_w + lq_size, :]

            top_gt = center_lq_h * scale
            left_gt = center_lq_w * scale
            img_gt = img_ref_full[top_gt:top_gt + gt_size,
                     left_gt:left_gt + gt_size, :]

            # 验证时参考图不做变换，直接用 GT（或施加非常轻微的变换）
            img_ref = img_gt.copy()

            # ========== 5. 生成 bicubic 上采样图（用于 VGG 特征匹配）==========
        # 将 LQ noisy patch (40×40) 上采样到 GT 尺寸 (160×160)
        img_in_noisy_pil = Image.fromarray(
            cv2.cvtColor((img_in_noisy_patch * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
        img_in_up = img_in_noisy_pil.resize((gt_size, gt_size), Image.BICUBIC)
        img_in_up = cv2.cvtColor(np.array(img_in_up), cv2.COLOR_RGB2BGR).astype(np.float32) / 255.

        # ========== 6. 拼接 6ch 输入 (noisy RGB + albedo RGB) ==========
        img_in_6ch = np.concatenate([img_in_noisy_patch, img_albedo_patch], axis=2)  # (40, 40, 6) HWC BGR

        # ========== 7. 转 tensor ==========
        # 3ch 图像用 totensor（自动 BGR→RGB）
        img_gt_t, img_in_up_t, img_ref_t = totensor(
            [img_gt, img_in_up, img_ref], bgr2rgb=True, float32=True)

        # 6ch 需要手动处理 BGR→RGB（前3ch 和后3ch 分别翻转通道）
        img_in_6ch_rgb = np.concatenate([
            img_in_6ch[:, :, 2::-1],  # noisy BGR → RGB (ch 0,1,2)
            img_in_6ch[:, :, 5:2:-1],  # albedo BGR → RGB (ch 3,4,5)
        ], axis=2)  # (40, 40, 6)
        img_in_6ch_t = torch.from_numpy(
            np.ascontiguousarray(img_in_6ch_rgb.transpose(2, 0, 1))).float()

        # ========== 8. 组装返回字典 ==========
        return_dict = {
            'img_in_lq': img_in_6ch_t,  # (6, 40, 40) — 降噪模块输入
            'img_in_up': img_in_up_t,  # (3, 160, 160) — VGG 特征匹配用
            'img_ref': img_ref_t,  # (3, 160, 160) — 参考图
            'img_in': img_gt_t,  # (3, 160, 160) — GT 监督信号
            'lq_path': in_path,  # 用于日志中显示图片名
        }

        if self.opt['phase'] != 'train':
            return_dict['padding'] = False
            return_dict['original_size'] = (gt_size, gt_size)

        return return_dict

    def _generate_ref_affine(self, img_patch):
        """对 GT patch 施加随机仿射变换，生成"伪不同视角"参考图。

        Args:
            img_patch: (H, W, 3) float32, [0, 1], BGR

        Returns:
            ref_patch: 同尺寸的仿射变换后图像
        """
        h, w = img_patch.shape[:2]
        center = (w / 2.0, h / 2.0)

        # 从 yml 读取变换参数，提供默认值
        angle = random.uniform(
            -self.opt.get('ref_angle_range', 5.0),
            self.opt.get('ref_angle_range', 5.0))
        scale = random.uniform(
            self.opt.get('ref_scale_min', 0.95),
            self.opt.get('ref_scale_max', 1.05))
        tx = random.uniform(
            -self.opt.get('ref_max_translate', 10),
            self.opt.get('ref_max_translate', 10))
        ty = random.uniform(
            -self.opt.get('ref_max_translate', 10),
            self.opt.get('ref_max_translate', 10))

        # 构建仿射矩阵：旋转 + 缩放 + 平移
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty

        ref_patch = cv2.warpAffine(
            img_patch, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT_101)

        return ref_patch

    def __len__(self):
        return len(self.paths)