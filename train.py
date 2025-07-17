import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import jittor as jt
import jittor.nn as nn
from jittor.dataset import DataLoader
from modules import *
from utils.loss import *
from utils.get_params_group import get_param_groups
from configs import *
import dataset
import logging
import yaml
from tqdm import tqdm
import argparse
import numpy as np
import wandb

# 为 Jittor 实现的 AverageMeter 辅助类
class AverageMeter:
    """计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    jt.misc.set_global_seed(seed)
    
    if jt.compiler.has_cuda:
        jt.flags.use_cuda = 1


def train(cfg_path, wb_key, load_initial_weights=None):
    config = yaml.safe_load(open(cfg_path))
    cfg = from_dict(config)
    set_seed(cfg.seed)
    log_f = '%(asctime)s | %(filename)s[line:%(lineno)d] | %(levelname)s | %(message)s'
    logging.basicConfig(level='INFO', format=log_f)
    wandb.login(key=wb_key)
    runs = wandb.init(project=cfg.project_name, name=cfg.dataset_name + '_' + cfg.exp_name, config=cfg, mode=cfg.wandb_mode)

    if jt.compiler.has_cuda:
        jt.flags.use_cuda = 1
        logging.info("Jittor is using CUDA")
    else:
        logging.info("Jittor is using CPU")
        
    fuse_net = Fuse()

    optimizer = jt.optim.Adam(
        fuse_net.parameters(),
        lr=cfg.lr_i,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0
    )

    if load_initial_weights:
        if cfg.resume:
            logging.warning(f"同时设置了 --load_initial_weights 和 resume。将优先使用 {load_initial_weights} 的初始权重。")
        logging.info(f"正在从 {load_initial_weights} 加载初始权重...")
        try:
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

    loss_ssim = SSIMLoss(window_size=11)
    loss_grad_pixel = PixelGradLoss()

    train_d = getattr(dataset, cfg.dataset_name)
    train_dataset = train_d(cfg, 'train')

    trainloader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    logging.info('开始训练...')
    for epoch in range(cfg.start_epoch, cfg.num_epochs):
        total_loss_meter = AverageMeter()
        content_loss_meter = AverageMeter()
        ssim_loss_meter = AverageMeter()
        saliency_loss_meter = AverageMeter()
        fre_loss_meter = AverageMeter()

        log_dict = {}
        loss_dict = {}
        pbar = tqdm(trainloader, total=len(trainloader), ncols=80)
        for data_ir, data_vi, mask, _ in pbar:
            fuse_net.train()
            
            fus_data, amp, pha = fuse_net(data_ir, data_vi)
            content_loss = loss_grad_pixel(data_vi, data_ir, fus_data)
            ssim_loss_v = loss_ssim(data_vi, fus_data)
            ssim_loss_i = loss_ssim(data_ir, fus_data)
            ssim_loss = ssim_loss_i + ssim_loss_v
            saliency_loss = cal_saliency_loss(fus_data, data_ir, data_vi, mask)
            fre_loss = cal_fre_loss(amp, pha, data_ir, data_vi, mask)
            total_loss = cfg.coeff_content * content_loss + cfg.coeff_ssim * ssim_loss + cfg.coeff_saliency * saliency_loss + cfg.coeff_fre * fre_loss

            # 【【【最终正确版本 - 经过验证】】】
            # 步骤 1: 优化器负责计算梯度
            optimizer.backward(total_loss)
            
            # 步骤 2: 手动对计算出的梯度进行裁剪
            optimizer.clip_grad_norm(max_norm=1.0, norm_type=2)
            
            # 步骤 3: 优化器使用被裁剪后的梯度来更新权重
            optimizer.step()

            # loss dict
            loss_dict['total_loss'] = total_loss.item()
            total_loss_meter.update(total_loss.item())
            content_loss_meter.update(content_loss.item())
            ssim_loss_meter.update(ssim_loss.item())
            saliency_loss_meter.update(saliency_loss.item())
            fre_loss_meter.update(fre_loss.item())
            pbar.set_description(f'Epoch {epoch + 1}/{cfg.num_epochs}')
            pbar.set_postfix(loss_dict)

        lr_decay_factor = (1 - epoch / cfg.num_epochs) * (1 - cfg.lr_f) + cfg.lr_f
        optimizer.lr = cfg.lr_i * lr_decay_factor

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
        runs.log(log_dict)

        if (epoch + 1) % cfg.epoch_gap == 0:
            checkpoint = {'fuse_net': fuse_net.state_dict()}
            logging.info(f'保存检查点到 models/{cfg.exp_name}.pkl')
            save_path = os.path.join("models", f'{cfg.exp_name}.pkl')
            if not os.path.exists('models'):
                os.makedirs('models')
            jt.save(checkpoint, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='configs/cfg.yaml', help='config file path')
    parser.add_argument('--auth', default='9f7cff4767e982880d4259c5134e17aa9e91b530', help='wandb auth api key')
    parser.add_argument('--load_initial_weights', type=str, default=None, help='Path to load initial weights from a Jittor-compatible .bin file.')
    args = parser.parse_args()
    train(args.cfg, args.auth, args.load_initial_weights)
    os.system(f'nohup python3 val.py &')