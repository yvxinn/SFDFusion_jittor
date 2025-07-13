import jittor as jt
from jittor.dataset import Dataset
from jittor.transform import Resize, ToTensor, to_pil_image

import logging
from pathlib import Path
from typing import Literal
import os

from SFDFusion_jittor.configs import ConfigDict
from SFDFusion_jittor.utils.img_read import img_read

class RoadScene(Dataset):
    def __init__(self, cfg: ConfigDict, mode: Literal['train', 'val', 'test'], batch_size=1, shuffle=False):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.img_list = Path(Path(self.cfg.dataset_root) / f'{mode}.txt').read_text().splitlines()
        logging.info(f'load {len(self.img_list)} images for {mode} mode')

        self.resize_transform = Resize((cfg.img_size, cfg.img_size))
        self.to_tensor_transform = ToTensor()

        if self.mode == 'train':
            self.ir_path = Path(Path(self.cfg.dataset_root) / 'ir')
            self.vi_path = Path(Path(self.cfg.dataset_root) / 'vi')
            self.mask_path = Path(Path(self.cfg.dataset_root) / ('labels' if cfg.have_seg_label else 'mask'))
        elif self.mode == 'test':
            self.ir_path = Path(Path(self.cfg.dataset_root) / 'test' / 'ir')
            self.vi_path = Path(Path(self.cfg.dataset_root) / 'test' / 'vi')
        
        self.set_attrs(batch_size=batch_size, total_len=len(self.img_list), shuffle=shuffle)

   # 在 SFDFusion_jittor/dataset.py 文件中

    def __getitem__(self, index):
        img_name = self.img_list[index]
        
        # 1. 调用 img_read，得到 Jittor 张量 (jt.Var)
        ir_img_var = img_read(os.path.join(self.ir_path, img_name), mode='L')
        vi_img_var, _ = img_read(os.path.join(self.vi_path, img_name), mode='YCbCr')

        mask_var = None
        if self.mode == 'train':
            mask_var = img_read(os.path.join(self.mask_path, img_name), mode='L')

        if self.mode == 'train':
            # "张量 -> PIL -> 缩放 -> 张量" 的往返转换
            
            # 关键修改 1: to_pil_image 后手动指定模式为 'L'
            # 这确保了后续的 ToTensor() 知道这是一个单通道灰度图
            ir_pil = to_pil_image(ir_img_var).convert('L')
            vi_pil = to_pil_image(vi_img_var).convert('L')
            
            ir_pil_resized = self.resize_transform(ir_pil)
            vi_pil_resized = self.resize_transform(vi_pil)

            ir_img = self.to_tensor_transform(ir_pil_resized)
            vi_img = self.to_tensor_transform(vi_pil_resized)

            mask = None
            if mask_var is not None:
                mask_pil = to_pil_image(mask_var).convert('L')
                mask_pil_resized = self.resize_transform(mask_pil)
                mask = self.to_tensor_transform(mask_pil_resized)

        else: # 测试模式
            ir_img = ir_img_var
            vi_img = vi_img_var
            
            _, h, w = ir_img.shape
            if h % 2 != 0 or w % 2 != 0:
                 ir_img = ir_img[:, : h // 2 * 2, : w // 2 * 2]
                 vi_img = vi_img[:, : h // 2 * 2, : w // 2 * 2]
            mask = None

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