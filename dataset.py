import jittor as jt
from jittor.dataset import Dataset
from jittor.transform import Compose, Resize, ToTensor, Gray

import logging
from pathlib import Path
from typing import Literal
import os

from configs import ConfigDict
from utils.img_read import img_read

class RoadScene(Dataset):
    def __init__(self, cfg: ConfigDict, mode: Literal['train', 'val', 'test'], batch_size=1, shuffle=False):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.img_list = Path(Path(self.cfg.dataset_root) / f'{mode}.txt').read_text().splitlines()
        logging.info(f'load {len(self.img_list)} images for {mode} mode')

        # 关键修改 1: 定义标准的 transform 流水线
        # ToTensor() 会自动处理归一化 ([0, 255] -> [0.0, 1.0]) 和维度转换 (H, W, C -> C, H, W)
        self.train_transforms = Compose([
            Resize((cfg.img_size, cfg.img_size)),
            ToTensor() 
        ])
        
        if self.mode == 'train':
            self.ir_path = Path(Path(self.cfg.dataset_root) / 'ir')
            self.vi_path = Path(Path(self.cfg.dataset_root) / 'vi')
            self.mask_path = Path(Path(self.cfg.dataset_root) / ('labels' if cfg.have_seg_label else 'mask'))
        elif self.mode == 'test':
            self.ir_path = Path(Path(self.cfg.dataset_root) / 'test' / 'ir')
            self.vi_path = Path(Path(self.cfg.dataset_root) / 'test' / 'vi')
        
        self.set_attrs(batch_size=batch_size, total_len=len(self.img_list), shuffle=shuffle)

    def __getitem__(self, index):
        img_name = self.img_list[index]
        ir_pil  = img_read(os.path.join(self.ir_path, img_name), mode='L')
        # print(f"Jittor img_read 'ir_img' shape: {ir_pil.shape}") 
        vi_pil_ycbcr  = img_read(os.path.join(self.vi_path, img_name), mode='YCbCr')
        # print(f"Jittor img_read 'vi_img' shape: {vi_pil_ycbcr}")

        mask = None
        if self.mode == 'train':
            mask_pil = img_read(os.path.join(self.mask_path, img_name), mode='L')
            ir_img = self.train_transforms(ir_pil)
            vi_img_ycbcr = self.train_transforms(vi_pil_ycbcr)
            vi_img = vi_img_ycbcr[0:1, :, :] # 取出 Y 通道
            mask = self.train_transforms(mask_pil)
        else:            
            # 测试模式
            ir_img = self.test_transforms(ir_pil)
            vi_img_ycbcr = self.test_transforms(vi_pil_ycbcr)
            vi_img = vi_img_ycbcr[0:1, :, :]

            # 裁剪逻辑现在可以正常工作
            c, h, w = ir_img.shape
            if h % 2 != 0 or w % 2 != 0:
                 ir_img = ir_img[:, :h // 2 * 2, :w // 2 * 2]
                 vi_img = vi_img[:, :h // 2 * 2, :w // 2 * 2]

        return ir_img, vi_img, mask, img_name
    
    def collate_batch(self, batch):
        ir_img_batch, vi_img_batch, mask_batch, img_name_batch = zip(*batch)
        ir_img_batch = jt.stack(ir_img_batch, dim=0)
        vi_img_batch = jt.stack(vi_img_batch, dim=0)
        
        if self.mode == 'train':
            valid_masks = [m for m in mask_batch if m is not None]
            if valid_masks:
                mask_batch = jt.stack(valid_masks, dim=0)
            else:
                mask_batch = None
        
        return ir_img_batch, vi_img_batch, mask_batch, img_name_batch