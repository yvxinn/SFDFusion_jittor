"""
SFDFusion Jittor版本 训练脚本
=================================

本脚本负责整个SFDFusion模型的训练流程，包括：
1. 环境和随机种子设置。
2. 加载和解析配置文件。
3. 初始化Wandb进行实验跟踪。
4. 构建Fuse模型和Adam优化器。
5. 加载预训练权重或从断点恢复。
6. 定义损失函数。
7. 设置Jittor DataLoader。
8. 执行训练循环，包括前向传播、损失计算、反向传播和参数更新。
9. 手动实现学习率衰减。
10. 定期保存模型检查点。
"""

import os
# 设置环境变量，以在某些系统上避免Intel MKL库的错误。
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# 指定使用的GPU设备
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import jittor as jt
import jittor.nn as nn
from jittor.dataset import DataLoader
from modules import *
from utils.loss import *
# from utils.get_params_group import get_param_groups # 此项目未使用参数分组
from configs import *
import dataset
import logging
import yaml
from tqdm import tqdm
import argparse
import numpy as np
import wandb

class AverageMeter:
    """
    一个用于计算和存储数值的平均值和当前值的辅助类。
    常用于跟踪损失或评估指标。
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """重置所有统计数据。"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """用新值更新统计数据。"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def set_seed(seed: int):
    """
    为所有相关的随机数生成器设置随机种子，以确保实验的可复现性。
    
    Args:
        seed (int): 随机种子。
    """
    random.seed(seed)
    np.random.seed(seed)
    # 设置Jittor的全局随机种子
    jt.misc.set_global_seed(seed)
    
    # Jittor特性：如果检测到CUDA，则强制启用CUDA后端。
    # Jittor的JIT编译器会在运行时动态选择后端。
    if jt.compiler.has_cuda:
        jt.flags.use_cuda = 1


def train(cfg_path: str, wb_key: str, load_initial_weights: str = None):
    """
    主训练函数。
    
    Args:
        cfg_path (str): YAML配置文件的路径。
        wb_key (str): Weights & Biases 的API密钥，用于登录。
        load_initial_weights (str, optional): 初始权重的路径。默认为None。
    """
    # --- 1. 配置加载和初始化 ---
    config = yaml.safe_load(open(cfg_path))
    cfg = from_dict(config)
    set_seed(cfg.seed)
    log_f = '%(asctime)s | %(filename)s[line:%(lineno)d] | %(levelname)s | %(message)s'
    logging.basicConfig(level='INFO', format=log_f)
    # 登录并初始化Wandb
    wandb.login(key=wb_key)
    runs = wandb.init(project=cfg.project_name, name=cfg.dataset_name + '_' + cfg.exp_name, config=cfg, mode=cfg.wandb_mode)

    # --- 2. 模型、优化器和权重加载 ---
    if jt.compiler.has_cuda:
        jt.flags.use_cuda = 1
        logging.info("Jittor is using CUDA")
    else:
        logging.info("Jittor is using CPU")
        
    fuse_net = Fuse()

    # Jittor Adam优化器。为确保与PyTorch版本结果对齐，
    # 显式设置所有参数（如betas, eps, weight_decay）以匹配PyTorch的默认值。
    optimizer = jt.optim.Adam(
        fuse_net.parameters(),
        lr=cfg.lr_i,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0
    )

    # 根据提供的参数加载权重
    if load_initial_weights:
        if cfg.resume:
            logging.warning(f"同时设置了 --load_initial_weights 和 resume。将优先使用 {load_initial_weights} 的初始权重。")
        logging.info(f"正在从 {load_initial_weights} 加载初始权重...")
        try:
            # Jittor的权重通常保存为.pkl或.bin文件
            import pickle
            with open(load_initial_weights, 'rb') as f:
                initial_weights = pickle.load(f)
            fuse_net.load_state_dict(initial_weights)
            logging.info("✅ 成功加载初始权重。")
        except Exception as e:
            logging.error(f"从 {load_initial_weights} 加载初始权重失败: {e}")
            raise
    elif cfg.resume is not None:
        logging.info(f'从 {cfg.resume} 恢复训练')
        checkpoint = jt.load(cfg.resume)
        if isinstance(checkpoint, dict) and 'fuse_net' in checkpoint:
            fuse_net.load_state_dict(checkpoint['fuse_net'])
        else:
            fuse_net.load_state_dict(checkpoint)

    # --- 3. 损失函数和数据加载器 ---
    loss_ssim = SSIMLoss(window_size=11)
    loss_grad_pixel = PixelGradLoss()

    train_d = getattr(dataset, cfg.dataset_name)
    train_dataset = train_d(cfg, 'train')

    trainloader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    # --- 4. 训练循环 ---
    logging.info('开始训练...')
    for epoch in range(cfg.start_epoch, cfg.num_epochs):
        # 为每个epoch重置损失计量器
        total_loss_meter = AverageMeter()
        content_loss_meter = AverageMeter()
        ssim_loss_meter = AverageMeter()
        saliency_loss_meter = AverageMeter()
        fre_loss_meter = AverageMeter()

        log_dict = {}
        loss_dict = {}
        # 使用tqdm来显示训练进度
        pbar = tqdm(trainloader, total=len(trainloader), ncols=80)
        for data_ir, data_vi, mask, _ in pbar:
            fuse_net.train()
            
            # 前向传播
            fus_data, amp, pha = fuse_net(data_ir, data_vi)

            # 计算各项损失
            content_loss = loss_grad_pixel(data_vi, data_ir, fus_data)
            ssim_loss_v = loss_ssim(data_vi, fus_data)
            ssim_loss_i = loss_ssim(data_ir, fus_data)
            ssim_loss = ssim_loss_i + ssim_loss_v
            saliency_loss = cal_saliency_loss(fus_data, data_ir, data_vi, mask)
            fre_loss = cal_fre_loss(amp, pha, data_ir, data_vi, mask)

            # 加权求和得到总损失
            total_loss = cfg.coeff_content * content_loss + cfg.coeff_ssim * ssim_loss + cfg.coeff_saliency * saliency_loss + cfg.coeff_fre * fre_loss

            # Jittor特性：optimizer.step(loss) 会自动执行 backward() 和 zero_grad()
            optimizer.step(total_loss)

            # 更新损失字典用于tqdm显示
            loss_dict['total_loss'] = total_loss.item()
            total_loss_meter.update(total_loss.item())
            content_loss_meter.update(content_loss.item())
            ssim_loss_meter.update(ssim_loss.item())
            saliency_loss_meter.update(saliency_loss.item())
            fre_loss_meter.update(fre_loss.item())
            pbar.set_description(f'Epoch {epoch + 1}/{cfg.num_epochs}')
            pbar.set_postfix(loss_dict)

        # Jittor特性：手动实现学习率衰减。
        # 此处使用线性衰减策略，而不是依赖Jittor的lr_scheduler。
        lr_decay_factor = (1 - epoch / cfg.num_epochs) * (1 - cfg.lr_f) + cfg.lr_f
        optimizer.lr = cfg.lr_i * lr_decay_factor

        # --- 5. 日志记录和模型保存 ---
        print('*' * 60 + '\tepoch finished!')
        logging.info(
            f'Epoch {epoch + 1}/{cfg.num_epochs}, lr:{optimizer.lr}, total_loss: {total_loss_meter.avg}, content_loss: {content_loss_meter.avg}, ssim_loss: {ssim_loss_meter.avg}, saliency_loss: {saliency_loss_meter.avg}, fre_loss: {fre_loss_meter.avg}'
        )

        log_dict.update({
            'total_loss': total_loss_meter.avg,
            'content_loss': content_loss_meter.avg,
            'ssim_loss': ssim_loss_meter.avg,
            'saliency_loss': saliency_loss_meter.avg,
            'fre_loss': fre_loss_meter.avg,
            'lr': optimizer.lr,
        })
        # 将epoch的统计数据上传到Wandb
        runs.log(log_dict)

        # 定期保存模型检查点
        if (epoch + 1) % cfg.epoch_gap == 0:
            checkpoint = {'fuse_net': fuse_net.state_dict()}
            logging.info(f'保存检查点到 models/{cfg.exp_name}.pkl')
            save_path = os.path.join("models", f'{cfg.exp_name}.pkl')
            if not os.path.exists('models'):
                os.makedirs('models')
            # Jittor特性：使用jt.save保存模型
            jt.save(checkpoint, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='configs/cfg.yaml', help='config file path')
    parser.add_argument('--auth', default='', help='wandb auth api key')
    parser.add_argument('--load_initial_weights', type=str, default=None, help='Path to load initial weights from a Jittor-compatible .bin file.')
    args = parser.parse_args()
    train(args.cfg, args.auth, args.load_initial_weights)
    # 训练结束后，在后台启动验证脚本
    os.system(f'nohup python3 val.py &')