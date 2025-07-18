"""
SFDFusion Jittor版本 模型验证与评估脚本
==========================================

本脚本负责对训练好的SFDFusion模型进行验证和定量评估。

主要流程分为两部分：
1. **`test` 函数**:
   - 加载指定的模型检查点（checkpoint）。
   - 遍历测试数据集，对每一对图像执行融合操作。
   - 将生成的融合图像保存到指定的输出目录。
   - 记录并计算模型的平均推理时间。
   - 调用 `evaluate` 函数进行后续的定量评估。

2. **`evaluate` 函数**:
   - 读取 `test` 函数生成的融合图像以及原始的测试图像。
   - 计算多个标准的图像融合评估指标，包括：
     - EN (Entropy)
     - SD (Standard Deviation)
     - SF (Spatial Frequency)
     - MI (Mutual Information)
     - VIFF (Visual Information Fidelity for Fusion)
     - Qabf (A novel quality metric for image fusion)
   - 将所有指标的平均值打印到控制台，并保存到一个结果文件中。
"""
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

# --- 1. 环境设置 ---
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

log_f = '%(asctime)s | %(filename)s[line:%(lineno)d] | %(levelname)s | %(message)s'
logging.basicConfig(level='INFO', format=log_f)

class AverageMeter:
    """计算并存储数值的平均值和当前值。"""
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


def test(args: argparse.Namespace, cfg: dict):
    """
    执行模型推理，生成融合图像并保存。

    Args:
        args (argparse.Namespace): 包含模型路径、输出目录等信息的命令行参数。
        cfg (dict): 从YAML加载的配置字典。
    """
    # --- 2. 数据加载器准备 ---
    import dataset as dataset_module
    test_d = getattr(dataset_module, cfg.dataset_name)
    test_dataset = test_d(cfg, 'test')

    testloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=cfg.num_workers
    )
    fuse_out_folder = args.out_dir
    if not os.path.exists(fuse_out_folder):
        os.makedirs(fuse_out_folder)

    # --- 3. 模型加载 ---
    fuse_net = Fuse()
    ckpt = jt.load(args.ckpt_path)
    fuse_net.load_state_dict(ckpt['fuse_net'])
    fuse_net.eval() # 设置为评估模式

    # --- 4. 图像融合循环 ---
    time_list = []
    logging.info(f'正在融合图像...')
    pbar = tqdm(testloader, total=len(testloader), ncols=80)
    for data_ir, data_vi, _, img_name in pbar:
        
        # Jittor特性：为确保计时准确，在操作前后进行同步。
        start_time = time.time()
        fus_data, _, _ = fuse_net(data_ir, data_vi)
        jt.sync_all(True) # 保证所有Jittor操作已在设备上完成
        end_time = time.time()
        time_list.append(end_time - start_time)
        
        # --- 5. 结果保存 ---
        if args.mode == 'gray':
            fi = np.squeeze((fus_data.numpy() * 255)).astype(np.uint8)
            img_save(fi, img_name[0], fuse_out_folder)
        elif args.mode == 'RGB':
            # 注意：当前验证脚本中的RGB模式支持不完整，因为它依赖于
            # 数据加载器提供CbCr通道，而当前实现中未提供。
            logging.warning("val.py中的RGB模式支持不完整，将默认保存为灰度图。")
            fi = np.squeeze((fus_data.numpy() * 255)).astype(np.uint8)
            img_save(fi, img_name[0], fuse_out_folder)

    logging.info(f'图像融合完成！')
    # 忽略第一次的编译和加载时间，以获得更准确的平均推理时间
    if len(time_list) > 1:
        logging.info(f'平均处理时间: {np.round(np.mean(time_list[1:]), 6)}s/张')
    else:
        logging.info(f'处理时间: {np.round(np.mean(time_list), 6)}s/张')
    
    # --- 6. 调用评估函数 ---
    evaluate(fuse_out_folder, cfg)


def evaluate(fuse_out_folder: str, cfg: dict):
    """
    对生成的融合图像进行定量评估。

    Args:
        fuse_out_folder (str): 存放融合图像的目录。
        cfg (dict): 从YAML加载的配置字典。
    """
    # --- 7. 数据加载器准备 ---
    import dataset as dataset_module
    test_d = getattr(dataset_module, cfg.dataset_name)
    test_dataset = test_d(cfg, 'test')
    
    testloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=cfg.num_workers
    )
    # 为每个评估指标创建一个AverageMeter
    metric_result = [AverageMeter() for _ in range(6)]

    logging.info(f'正在评估图像...')
    pbar = tqdm(testloader, total=len(testloader), ncols=80)

    # --- 8. 评估循环 ---
    for data_ir, data_vi, _, img_name in pbar:
        # 将Jittor张量转为NumPy数组以进行评估
        ir = data_ir.numpy().squeeze() * 255
        vi = data_vi.numpy().squeeze() * 255
        
        # 从文件中读取已生成的融合图像
        fi_pil = img_read(os.path.join(fuse_out_folder, img_name[0]), 'L')
        fi = np.array(fi_pil).squeeze()
        
        # 确保所有图像尺寸一致以进行评估
        h, w = fi.shape
        if h % 2 != 0 or w % 2 != 0:
            fi = fi[: h // 2 * 2, : w // 2 * 2]
        if fi.shape != ir.shape or fi.shape != vi.shape:
            fi = cv2.resize(fi, (ir.shape[1], ir.shape[0]))

        # --- 9. 计算各项指标 ---
        metric_result[0].update(Evaluator.EN(fi))
        metric_result[1].update(Evaluator.SD(fi))
        metric_result[2].update(Evaluator.SF(fi))
        metric_result[3].update(Evaluator.MI(fi, ir, vi))
        metric_result[4].update(Evaluator.VIFF(fi, ir, vi))
        metric_result[5].update(Evaluator.Qabf(fi, ir, vi))

    # --- 10. 结果保存与打印 ---
    with open(f'{fuse_out_folder}_result.txt', 'w') as f:
        f.write('EN: ' + str(np.round(metric_result[0].avg, 3)) + '\n')
        f.write('SD: ' + str(np.round(metric_result[1].avg, 3)) + '\n')
        f.write('SF: ' + str(np.round(metric_result[2].avg, 3)) + '\n')
        f.write('MI: ' + str(np.round(metric_result[3].avg, 3)) + '\n')
        f.write('VIF: ' + str(np.round(metric_result[4].avg, 3)) + '\n')
        f.write('Qabf: ' + str(np.round(metric_result[5].avg, 3)) + '\n')

    logging.info(f'评估结果已写入文件！')
    print("\n" * 2 + "=" * 80)
    print("测试结果:")
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
    # --- 11. 主程序入口 ---
    if jt.compiler.has_cuda:
        jt.flags.use_cuda = 1
        logging.info("Jittor is using CUDA for validation")

    config = yaml.safe_load(open('configs/cfg.yaml'))
    cfg = from_dict(config)
    parse = argparse.ArgumentParser()
    
    parse.add_argument('--ckpt_path', type=str, default=f'models/{cfg.exp_name}.pkl')
    parse.add_argument('--dataset_name', type=str, default=cfg.dataset_name)
    parse.add_argument('--out_dir', type=str, default=f'test_result/{cfg.dataset_name}/{cfg.exp_name}')
    parse.add_argument('--mode', type=str, default='gray')
    args = parse.parse_args()

    test(args, cfg)
