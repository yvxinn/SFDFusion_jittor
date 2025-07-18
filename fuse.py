"""
本脚本用于加载已训练好的SFDFusion模型，并对指定的红外（IR）和可见光（VI）
图像对进行批量融合。

主要流程包括：
1. 设置Jittor环境和日志记录。
2. 定义并解析命令行参数，以指定模型路径、输入图像目录和输出目录。
3. 加载模型权重并将其设置为评估模式（`eval()`）。
4. 遍历输入目录中的所有图像。
5. 对每对图像进行预处理，包括读取、转换为YCbCr（如果需要）、
   调整形状和尺寸以符合模型输入。
6. 在 `jt.no_grad()` 上下文中执行模型的前向传播以进行融合，并记录处理时间。
7. 对融合结果进行后处理，并将其保存为灰度图和（可选的）RGB彩色图。
8. 计算并报告平均处理时间。
"""
import jittor as jt
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import argparse
import yaml
import logging
import time
import warnings

# 导入 Jittor 项目的组件
from modules import Fuse
from utils.img_read import img_read, img_save, ycbcr_to_rgb, tensor_to_image
from configs import from_dict

# --- 1. 环境设置 ---
warnings.filterwarnings("ignore")
# Jittor特性：检查CUDA是否可用，如果可用，则强制Jittor使用CUDA后端。
if jt.compiler.has_cuda:
    jt.flags.use_cuda = 1

log_f = '%(asctime)s | %(filename)s[line:%(lineno)d] | %(levelname)s | %(message)s'
logging.basicConfig(level='INFO', format=log_f)


def fuse_all(args: argparse.Namespace):
    """
    加载模型并对指定目录中的所有图像对执行融合。

    Args:
        args (argparse.Namespace): 包含所有命令行参数的命名空间对象。
    """
    # --- 2. 路径准备和模型加载 ---
    ir_dir = Path(args.ir_path)
    vi_dir = Path(args.vi_path)
    
    out_dir_gray = Path(args.out_dir) / 'gray'
    out_dir_rgb = Path(args.out_dir) / 'rgb'
    
    out_dir_gray.mkdir(parents=True, exist_ok=True)
    if args.mode == 'RGB':
        out_dir_rgb.mkdir(parents=True, exist_ok=True)

    # 初始化模型
    fuse_net = Fuse()
    # Jittor的 `jt.load` 可以加载 `.pkl` 或 `.bin` 文件
    weights = jt.load(args.ckpt_path)

    # 权重文件可能将整个 state_dict 存在一个特定的键下（如此处的 'fuse_net'）
    state_dict_to_load = weights['fuse_net']

    fuse_net.load_state_dict(state_dict_to_load)
    # Jittor特性：将模型设置为评估模式。这会禁用Dropout和BatchNorm等层。
    fuse_net.eval()
    logging.info(f"模型权重已从 {args.ckpt_path} 加载。")

    # --- 3. 图像遍历与融合 ---
    img_names = sorted([f for f in os.listdir(ir_dir) if f.endswith(('.png', '.jpg', '.bmp'))])
    
    time_list = []
    
    logging.info(f"开始融合来自 {ir_dir} 和 {vi_dir} 的 {len(img_names)} 张图像...")
    
    # Jittor特性：jt.no_grad() 上下文管理器。
    # 在此上下文中的所有操作都不会被记录梯度，从而节省内存并加速推理。
    with jt.no_grad():
        for img_name in tqdm(img_names, ncols=80):
            ir_path = ir_dir / img_name
            vi_path = vi_dir / img_name

            if not vi_path.exists():
                logging.warning(f"跳过 {img_name}，因为在可见光图像目录中找不到对应的文件。")
                continue

            # a. 预处理红外图像
            # Jittor的维度顺序是 (B, C, H, W)
            ir_img = jt.array(np.array(img_read(str(ir_path), mode='L'))).unsqueeze(0).unsqueeze(0)
            ir_img = ir_img.float() / 255.0
            
            # b. 预处理可见光图像
            if args.mode == 'RGB':
                # 先将 (H, W, C) 的NumPy数组转置为 (C, H, W)
                vi_img_ycbcr = jt.array(np.array(img_read(str(vi_path), mode='YCbCr')).transpose(2, 0, 1))
                vi_img_ycbcr = vi_img_ycbcr.float() / 255.0
                # 分离Y通道（用于融合）和CbCr通道（用于着色）
                vi_y_img = vi_img_ycbcr[0:1].unsqueeze(0)
                vi_cbcr_img = vi_img_ycbcr[1:3].unsqueeze(0)
            else: # 灰度模式
                vi_y_img = jt.array(np.array(img_read(str(vi_path), mode='L'))).unsqueeze(0).unsqueeze(0)
                vi_y_img = vi_y_img.float() / 255.0

            # c. 确保输入尺寸为偶数，以适应模型的下采样操作
            _, _, h, w = ir_img.shape
            if h % 2 != 0 or w % 2 != 0:
                h_new, w_new = h // 2 * 2, w // 2 * 2
                ir_img = ir_img[:, :, :h_new, :w_new]
                vi_y_img = vi_y_img[:, :, :h_new, :w_new]
                if args.mode == 'RGB':
                    vi_cbcr_img = vi_cbcr_img[:, :, :h_new, :w_new]
            
            # d. 执行融合并计时
            start_time = time.time()
            fus_data, _, _ = fuse_net(ir_img, vi_y_img)
            time_list.append(time.time() - start_time)

            # --- 4. 结果后处理与保存 ---
            # a. 保存灰度融合图像
            # Jittor -> NumPy: 使用 .numpy() 方法
            fi_gray = np.squeeze(fus_data.numpy() * 255)
            fi_gray = np.round(fi_gray).astype(np.uint8)
            img_save(fi_gray, img_name, str(out_dir_gray))
            
            # b. 如果是RGB模式，合并CbCr通道并保存彩色图
            if args.mode == 'RGB':
                fi_rgb = jt.concat((fus_data, vi_cbcr_img), dim=1)
                fi_rgb = ycbcr_to_rgb(fi_rgb)
                fi_rgb = tensor_to_image(fi_rgb) * 255
                fi_rgb = np.round(fi_rgb).astype(np.uint8)
                img_save(fi_rgb, img_name, str(out_dir_rgb), mode='RGB')

    # --- 5. 报告总结 ---
    # 忽略第一次的编译时间，以获得更准确的平均推理时间
    avg_time = np.mean(time_list[1:]) if len(time_list) > 1 else np.mean(time_list)
    logging.info(f"所有图像融合完成！平均处理时间: {avg_time:.4f} 秒/张。")
    logging.info(f"灰度结果保存在: {out_dir_gray}")
    if args.mode == 'RGB':
        logging.info(f"RGB 结果保存在: {out_dir_rgb}")

if __name__ == "__main__":
    # --- 6. 加载配置和解析参数 ---
    config = yaml.safe_load(open('configs/cfg.yaml'))
    cfg = from_dict(config)
    
    parse = argparse.ArgumentParser()
    # 使用训练配置中定义的名字来定位模型文件
    parse.add_argument('--ckpt_path', type=str, default=f'models/{cfg.exp_name}.pkl')
    parse.add_argument('--ir_path', type=str, default='./ir_imgs/')
    parse.add_argument('--vi_path', type=str, default='./vi_imgs/')
    parse.add_argument('--out_dir', type=str, default=f'test_result/fuse_result/')
    parse.add_argument('--mode', type=str, default='RGB', choices=['gray', 'RGB'], help='输出模式 (灰度或彩色)')
    
    args = parse.parse_args()

    fuse_all(args)