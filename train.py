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

# Helper classes implemented for Jittor
class AverageMeter:
    """Computes and stores the average and current value"""
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

def _gaussian(window_size, sigma):
    gauss = jt.exp(-(jt.arange(window_size, dtype='float32') - window_size // 2) ** 2 / float(2 * sigma ** 2))
    return gauss / gauss.sum()

def _create_window(window_size, channel, sigma):
    _1D_window = _gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = jt.matmul(_1D_window, _1D_window.transpose(1, 0)).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

class SSIMLoss(jt.nn.Module):
    def __init__(self, window_size=11, sigma=1.5):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.channel = 1
        self.window = _create_window(window_size, self.channel, sigma)

    def execute(self, img1, img2):
        (_, channel, _, _) = img1.shape
        if channel == self.channel and jt.flags.use_cuda == 1:
            window = self.window
        else:
            window = _create_window(self.window_size, channel, self.sigma)
            self.window = window
            self.channel = channel

        mu1 = jt.nn.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = jt.nn.conv2d(img2, window, padding=self.window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = jt.nn.conv2d(img1 * img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = jt.nn.conv2d(img2 * img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = jt.nn.conv2d(img1 * img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean()


# This function is not used, the logic is now applied directly when creating the optimizer.
# def init_params_group(mlist):
#     pg0, pg1, pg2 = [], [], []
#     for m in mlist:
#         pg = get_param_groups(m)
#         pg0.extend(pg[0])
#         pg1.extend(pg[1])
#         pg2.extend(pg[2])
#     return pg0, pg1, pg2


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    jt.seed(seed)


def train(cfg_path, wb_key, load_initial_weights=None):
    config = yaml.safe_load(open(cfg_path))
    cfg = from_dict(config)
    set_seed(cfg.seed)
    log_f = '%(asctime)s | %(filename)s[line:%(lineno)d] | %(levelname)s | %(message)s'
    logging.basicConfig(level='INFO', format=log_f)
    # wandb
    wandb.login(key=wb_key)  # wandb api key
    runs = wandb.init(project=cfg.project_name, name=cfg.dataset_name + '_' + cfg.exp_name, config=cfg, mode=cfg.wandb_mode)

    # Model
    if jt.compiler.has_cuda:
        jt.flags.use_cuda = 1
        logging.info("Jittor is using CUDA")
    else:
        logging.info("Jittor is using CPU")
        
    fuse_net = Fuse()

    # --- 关键修改: 与 PyTorch 版本对齐 ---
    # 移除参数分组，直接将所有参数传入优化器
    optimizer = jt.optim.Adam(
        fuse_net.parameters(),
        lr=cfg.lr_i
    )
    # --- 修改结束 ---

    # Jittor does not have LambdaLR, so we will update lr manually
    # lr_func = lambda x: (1 - x / cfg.num_epochs) * (1 - cfg.lr_f) + cfg.lr_f
    # scheduler = jt.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    if load_initial_weights:
        if cfg.resume:
            logging.warning(f"Both --load_initial_weights and resume are set. Prioritizing initial weights from {load_initial_weights}.")
        logging.info(f"Loading initial weights from {load_initial_weights}...")
        try:
            import pickle
            with open(load_initial_weights, 'rb') as f:
                initial_weights = pickle.load(f)
            fuse_net.load_state_dict(initial_weights)
            logging.info("✅ Successfully loaded initial weights.")
        except Exception as e:
            logging.error(f"Failed to load initial weights from {load_initial_weights}: {e}")
            raise
    elif cfg.resume is not None:
        logging.info(f'Resume from {cfg.resume}')
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

    '''
    ------------------------------------------------------------------------------
    Train
    ------------------------------------------------------------------------------
    '''
    logging.info('Start training...')
    for epoch in range(cfg.start_epoch, cfg.num_epochs):
        '''train'''
        total_loss_meter = AverageMeter()
        content_loss_meter = AverageMeter()
        ssim_loss_meter = AverageMeter()
        saliency_loss_meter = AverageMeter()
        fre_loss_meter = AverageMeter()

        log_dict = {}
        loss_dict = {}
        iter = tqdm(trainloader, total=len(trainloader), ncols=80)
        for data_ir, data_vi, mask, _ in iter:
            fuse_net.train()
            
            fus_data, amp, pha = fuse_net(data_ir, data_vi)
            # conten_loss
            content_loss = loss_grad_pixel(data_vi, data_ir, fus_data)

            # SSIM-loss
            ssim_loss_v = loss_ssim(data_vi, fus_data)
            ssim_loss_i = loss_ssim(data_ir, fus_data)
            ssim_loss = ssim_loss_i + ssim_loss_v

            # saliency_loss
            saliency_loss = cal_saliency_loss(fus_data, data_ir, data_vi, mask)

            # fre_loss
            fre_loss = cal_fre_loss(amp, pha, data_ir, data_vi, mask)

            total_loss = cfg.coeff_content * content_loss + cfg.coeff_ssim * ssim_loss + cfg.coeff_saliency * saliency_loss + cfg.coeff_fre * fre_loss

            optimizer.step(total_loss)

            # loss dict
            loss_dict['total_loss'] = total_loss.item()
            total_loss_meter.update(total_loss.item())
            content_loss_meter.update(content_loss.item())
            ssim_loss_meter.update(ssim_loss.item())
            saliency_loss_meter.update(saliency_loss.item())
            fre_loss_meter.update(fre_loss.item())
            # 设置进度条
            iter.set_description(f'Epoch {epoch + 1}/{cfg.num_epochs}')
            iter.set_postfix(loss_dict)

        # Manually update learning rate
        lr_decay_factor = (1 - epoch / cfg.num_epochs) * (1 - cfg.lr_f) + cfg.lr_f
        optimizer.lr = cfg.lr_i * lr_decay_factor

        # 打印信息
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

        # update wandb
        runs.log(log_dict)

        # 每隔几个epoch保存一次模型
        if (epoch + 1) % cfg.epoch_gap == 0:
            checkpoint = {'fuse_net': fuse_net.state_dict()}

            logging.info(f'Save checkpoint to models/{cfg.exp_name}.pkl')
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
    # 运行命令行代码
    os.system(f'nohup python3 val.py &')
