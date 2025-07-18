"""
本脚本定义了用于加载RoadScene数据集的 `RoadScene` 类，它继承自
`jittor.dataset.Dataset`。

该类负责：
1. 从 `train.txt` 或 `test.txt` 文件中读取图像列表。
2. 定义训练和测试时所需的不同图像转换（transform）流水线。
3. 实现 `__getitem__` 方法，用于加载单张红外图像、可见光图像及其
   对应的显著性掩码（仅在训练时），并进行预处理。
4. 实现 `collate_batch` 方法，这是一个Jittor `Dataset` 的关键方法，
   用于将多个样本高效地整合成一个批次（batch）以供模型训练。
"""
import jittor as jt
from jittor.dataset import Dataset
from jittor.transform import Compose, Resize, ToTensor

import logging
from pathlib import Path
from typing import Literal
import os

from configs import ConfigDict
from utils.img_read import img_read

class RoadScene(Dataset):
    """
    用于加载RoadScene数据集的自定义Dataset类。
    """
    def __init__(self, cfg: ConfigDict, mode: Literal['train', 'val', 'test'], batch_size: int = 1, shuffle: bool = False):
        """
        初始化RoadScene数据集。

        Args:
            cfg (ConfigDict): 包含数据集路径、图像尺寸等信息的配置对象。
            mode (Literal['train', 'val', 'test']): 数据集模式。
            batch_size (int, optional): 批次大小。默认为1。
            shuffle (bool, optional): 是否打乱数据。默认为False。
        """
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        # 从 .txt 文件加载图像文件名列表
        self.img_list = Path(Path(self.cfg.dataset_root) / f'{mode}.txt').read_text().splitlines()
        logging.info(f'load {len(self.img_list)} images for {mode} mode')

        # 定义训练时使用的图像转换流水线
        # Jittor特性：`Compose` 用于串联多个transform。`ToTensor` 会自动将
        # [0, 255] 的图像归一化到 [0.0, 1.0]，并转换维度 (H,W,C -> C,H,W)。
        self.train_transforms = Compose([
            Resize((cfg.img_size, cfg.img_size)), # 训练时将图像缩放到指定尺寸
            ToTensor() 
        ])

        # 定义测试时使用的图像转换流水线
        self.test_transforms = Compose([
            # 测试时通常不应改变图像原始尺寸
            ToTensor()
        ])
        
        # 根据不同模式设置图像文件的具体路径
        if self.mode == 'train':
            self.ir_path = Path(Path(self.cfg.dataset_root) / 'ir')
            self.vi_path = Path(Path(self.cfg.dataset_root) / 'vi')
            self.mask_path = Path(Path(self.cfg.dataset_root) / ('labels' if cfg.have_seg_label else 'mask'))
        elif self.mode == 'test':
            self.ir_path = Path(Path(self.cfg.dataset_root) / 'test' / 'ir')
            self.vi_path = Path(Path(self.cfg.dataset_root) / 'test' / 'vi')
        
        # Jittor特性：set_attrs() 是Jittor Dataset的必要方法，用于设置数据集的
        # 核心属性，如总长度、批大小、是否打乱等。
        self.set_attrs(batch_size=batch_size, total_len=len(self.img_list), shuffle=shuffle)

    def __getitem__(self, index: int) -> tuple:
        """
        Jittor Dataset的核心方法之一。
        根据给定的索引 `index`，加载并返回一个样本的数据。

        Args:
            index (int): 数据样本的索引。

        Returns:
            tuple: 包含红外图像、可见光图像、掩码和图像名的元组。
        """
        img_name = self.img_list[index]
        # 1. 加载原始图像
        ir_pil  = img_read(os.path.join(self.ir_path, img_name), mode='L')
        # 将可见光图像转换为YCbCr空间，以分离亮度(Y)和色度(CbCr)
        vi_pil_ycbcr  = img_read(os.path.join(self.vi_path, img_name), mode='YCbCr')

        mask = None
        # 2. 根据模式进行不同的预处理
        if self.mode == 'train':
            # a. 训练模式：加载掩码，并对所有图像应用训练转换
            mask_pil = img_read(os.path.join(self.mask_path, img_name), mode='L')
            ir_img = self.train_transforms(ir_pil)
            vi_img_ycbcr = self.train_transforms(vi_pil_ycbcr)
            # 提取亮度通道Y作为模型的输入
            vi_img = vi_img_ycbcr[0:1, :, :]
            mask = self.train_transforms(mask_pil)
        else:
            # b. 测试模式：应用测试转换
            ir_img = self.test_transforms(ir_pil)
            vi_img_ycbcr = self.test_transforms(vi_pil_ycbcr)
            # 提取亮度通道Y
            vi_img = vi_img_ycbcr[0:1, :, :]

            # 为确保模型可以正确处理，将图像尺寸裁剪为偶数
            c, h, w = ir_img.shape
            if h % 2 != 0 or w % 2 != 0:
                 ir_img = ir_img[:, :h // 2 * 2, :w // 2 * 2]
                 vi_img = vi_img[:, :h // 2 * 2, :w // 2 * 2]

        return ir_img, vi_img, mask, img_name
    
    def collate_batch(self, batch: list) -> tuple:
        """
        Jittor Dataset的核心方法之一。
        将一个由 `__getitem__` 返回的样本列表 `batch`，整合成一个单一的、
        可供模型直接使用的大批次。

        Args:
            batch (list): 一个包含多个样本元组的列表。

        Returns:
            tuple: 包含批次化后的红外图像、可见光图像、掩码和图像名的元组。
        """
        # 1. 解压样本列表
        ir_img_batch, vi_img_batch, mask_batch, img_name_batch = zip(*batch)
        # 2. 将图像列表堆叠成一个批次张量
        ir_img_batch = jt.stack(ir_img_batch, dim=0)
        vi_img_batch = jt.stack(vi_img_batch, dim=0)
        
        # 3. 特殊处理掩码批次（因为在测试模式下可能为None）
        if self.mode == 'train':
            # 过滤掉可能存在的None值
            valid_masks = [m for m in mask_batch if m is not None]
            if valid_masks:
                mask_batch = jt.stack(valid_masks, dim=0)
            else:
                mask_batch = None
        
        return ir_img_batch, vi_img_batch, mask_batch, img_name_batch