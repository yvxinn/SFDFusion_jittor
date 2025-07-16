from modules import *
import os
import numpy as np
from utils.evaluator import Evaluator
import jittor as jt
from utils.img_read import *
import argparse
import logging
from tqdm import tqdm
import warnings
import yaml
from configs import from_dict
from jittor.dataset import DataLoader
import time
import cv2

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

log_f = '%(asctime)s | %(filename)s[line:%(lineno)d] | %(levelname)s | %(message)s'
logging.basicConfig(level='INFO', format=log_f)

# Helper class implemented for Jittor
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


def test(args, cfg):
    # Dynamically import the dataset based on cfg
    import dataset as dataset_module
    test_d = getattr(dataset_module, cfg.dataset_name)
    test_dataset = test_d(cfg, 'test')

    testloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=cfg.num_workers
    )
    fuse_out_folder = args.out_dir
    if not os.path.exists(fuse_out_folder):
        os.makedirs(fuse_out_folder)

    fuse_net = Fuse()
    ckpt = jt.load(args.ckpt_path)
    fuse_net.load_state_dict(ckpt['fuse_net'])
    fuse_net.eval()

    time_list = []
    logging.info(f'fusing images ...')
    iter = tqdm(testloader, total=len(testloader), ncols=80)
    for data_ir, data_vi, _, img_name in iter:
        
        ts = time.time()
        fus_data, _, _ = fuse_net(data_ir, data_vi)
        jt.sync_all(True) # Ensure completion for timing
        te = time.time()
        time_list.append(te - ts)
        
        if args.mode == 'gray':
            fi = np.squeeze((fus_data.numpy() * 255)).astype(np.uint8)
            img_save(fi, img_name[0], fuse_out_folder)
        elif args.mode == 'RGB':
            # This part is complex and depends on how YCbCr data is handled.
            # It seems vi_cbcr was missing in the original dataloader for this mode.
            # We will replicate the gray logic for now.
            logging.warning("RGB mode in val.py is not fully supported due to missing 'vi_cbcr' data. Defaulting to gray.")
            fi = np.squeeze((fus_data.numpy() * 255)).astype(np.uint8)
            img_save(fi, img_name[0], fuse_out_folder)

    logging.info(f'fusing images done!')
    # Skip the first image for timing calculation as it may include compilation time
    if len(time_list) > 1:
        logging.info(f'time: {np.round(np.mean(time_list[1:]), 6)}s')
    else:
        logging.info(f'time: {np.round(np.mean(time_list), 6)}s')
    evaluate(fuse_out_folder, cfg)


def evaluate(fuse_out_folder, cfg):
    # Dynamically import the dataset based on cfg
    import dataset as dataset_module
    test_d = getattr(dataset_module, cfg.dataset_name)
    test_dataset = test_d(cfg, 'test')
    
    testloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=cfg.num_workers
    )
    metric_result = [AverageMeter() for _ in range(6)]

    logging.info(f'evaluating images ...')
    iter = tqdm(testloader, total=len(testloader), ncols=80)

    for data_ir, data_vi, _, img_name in iter:
        ir = data_ir.numpy().squeeze() * 255
        vi = data_vi.numpy().squeeze() * 255
        
        # 修正: 先将 img_read 返回的 PIL Image 转为 numpy array
        fi_pil = img_read(os.path.join(fuse_out_folder, img_name[0]), 'L')
        fi = np.array(fi_pil).squeeze() # fi is now a numpy array, no need for .numpy()
        h, w = fi.shape
        if h % 2 != 0 or w % 2 != 0:
            fi = fi[: h // 2 * 2, : w // 2 * 2]
        if fi.shape != ir.shape or fi.shape != vi.shape:
            fi = cv2.resize(fi, (ir.shape[1], ir.shape[0]))

        metric_result[0].update(Evaluator.EN(fi))
        metric_result[1].update(Evaluator.SD(fi))
        metric_result[2].update(Evaluator.SF(fi))
        metric_result[3].update(Evaluator.MI(fi, ir, vi))
        metric_result[4].update(Evaluator.VIFF(fi, ir, vi))
        metric_result[5].update(Evaluator.Qabf(fi, ir, vi))

    # 结果写入文件
    with open(f'{fuse_out_folder}_result.txt', 'w') as f:
        f.write('EN: ' + str(np.round(metric_result[0].avg, 3)) + '\n')
        f.write('SD: ' + str(np.round(metric_result[1].avg, 3)) + '\n')
        f.write('SF: ' + str(np.round(metric_result[2].avg, 3)) + '\n')
        f.write('MI: ' + str(np.round(metric_result[3].avg, 3)) + '\n')
        f.write('VIF: ' + str(np.round(metric_result[4].avg, 3)) + '\n')
        f.write('Qabf: ' + str(np.round(metric_result[5].avg, 3)) + '\n')

    logging.info(f'writing results done!')
    print("\n" * 2 + "=" * 80)
    print("The test result :")
    print("\t\t EN\t SD\t SF\t MI\tVIF\tQabf")
    print(
        'result:\t'
        + '\t'
        + str(np.round(metric_result[0].avg, 3))
        + '\t'
        + str(np.round(metric_result[1].avg, 3))
        + '\t'
        + str(np.round(metric_result[2].avg, 3))
        + '\t'
        + str(np.round(metric_result[3].avg, 3))
        + '\t'
        + str(np.round(metric_result[4].avg, 3))
        + '\t'
        + str(np.round(metric_result[5].avg, 3))
    )
    print("=" * 80)


if __name__ == "__main__":
    if jt.compiler.has_cuda:
        jt.flags.use_cuda = 1
        logging.info("Jittor is using CUDA for validation")

    config = yaml.safe_load(open('configs/cfg.yaml'))
    cfg = from_dict(config)
    parse = argparse.ArgumentParser()
    # 修正: 将权重文件的默认扩展名从 .pth 改为 .pkl
    parse.add_argument('--ckpt_path', type=str, default=f'models/{cfg.exp_name}.pkl')
    parse.add_argument('--dataset_name', type=str, default=cfg.dataset_name)
    parse.add_argument('--out_dir', type=str, default=f'test_result/{cfg.dataset_name}/{cfg.exp_name}')
    parse.add_argument('--mode', type=str, default='gray')
    args = parse.parse_args()

    test(args, cfg)
    # evaluate("./test_result/res.txt")
