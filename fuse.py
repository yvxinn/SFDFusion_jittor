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

# --- 配置 ---
warnings.filterwarnings("ignore")
if jt.has_cuda:
    jt.flags.use_cuda = 1

log_f = '%(asctime)s | %(filename)s[line:%(lineno)d] | %(levelname)s | %(message)s'
logging.basicConfig(level='INFO', format=log_f)


def fuse_all(args):
    """
    批量处理整个文件夹的图像融合。
    """
    # --- 1. 准备路径和模型 ---
    ir_dir = Path(args.ir_path)
    vi_dir = Path(args.vi_path)
    
    out_dir_gray = Path(args.out_dir) / 'gray'
    out_dir_rgb = Path(args.out_dir) / 'rgb'
    
    out_dir_gray.mkdir(parents=True, exist_ok=True)
    if args.mode == 'RGB':
        out_dir_rgb.mkdir(parents=True, exist_ok=True)

    fuse_net = Fuse()
    weights = np.load(args.ckpt_path, allow_pickle=True)

    # 权重文件将整个 state_dict 存在一个 'fuse_net' 键下
    # 我们需要提取这个 state_dict 来加载
    state_dict_to_load = weights['fuse_net']

    fuse_net.load_state_dict(state_dict_to_load)
    fuse_net.eval()
    logging.info(f"模型权重已从 {args.ckpt_path} 加载。")

    # --- 2. 遍历并处理图像 ---
    img_names = sorted([f for f in os.listdir(ir_dir) if f.endswith(('.png', '.jpg', '.bmp'))])
    
    time_list = []
    
    logging.info(f"开始融合来自 {ir_dir} 和 {vi_dir} 的 {len(img_names)} 张图像...")
    
    with jt.no_grad():
        for img_name in tqdm(img_names, ncols=80):
            ir_path = ir_dir / img_name
            vi_path = vi_dir / img_name

            if not vi_path.exists():
                logging.warning(f"跳过 {img_name}，因为在可见光图像目录中找不到对应的文件。")
                continue

            # 读取图像并增加 batch 维度, 转换为 float32 并归一化
            ir_img = jt.array(np.array(img_read(str(ir_path), mode='L'))).unsqueeze(0).unsqueeze(0)
            ir_img = ir_img.float() / 255.0
            
            if args.mode == 'RGB':
                # YCbCr图 (H, W, C) -> (B, C, H, W)
                vi_img_ycbcr = jt.array(np.array(img_read(str(vi_path), mode='YCbCr')).transpose(2, 0, 1))
                vi_img_ycbcr = vi_img_ycbcr.float() / 255.0
                # 分离 Y 和 CbCr 通道, 并增加 batch 维度
                vi_y_img = vi_img_ycbcr[0:1].unsqueeze(0)
                vi_cbcr_img = vi_img_ycbcr[1:3].unsqueeze(0)
            else: # 灰度模式
                vi_y_img = jt.array(np.array(img_read(str(vi_path), mode='L'))).unsqueeze(0).unsqueeze(0)
                vi_y_img = vi_y_img.float() / 255.0

            # 确保尺寸为偶数
            _, _, h, w = ir_img.shape
            if h % 2 != 0 or w % 2 != 0:
                h_new, w_new = h // 2 * 2, w // 2 * 2
                ir_img = ir_img[:, :, :h_new, :w_new]
                vi_y_img = vi_y_img[:, :, :h_new, :w_new]
                if args.mode == 'RGB':
                    vi_cbcr_img = vi_cbcr_img[:, :, :h_new, :w_new]
            
            # 执行融合
            start_time = time.time()
            fus_data, _, _ = fuse_net(ir_img, vi_y_img)
            time_list.append(time.time() - start_time)

            # --- 保存结果 ---
            # 总是保存灰度图
            fi_gray = np.squeeze(fus_data.numpy() * 255)
            fi_gray = np.round(fi_gray).astype(np.uint8)
            img_save(fi_gray, img_name, str(out_dir_gray))
            
            # 如果是 RGB 模式，额外保存彩色图
            if args.mode == 'RGB':
                fi_rgb = jt.concat((fus_data, vi_cbcr_img), dim=1)
                fi_rgb = ycbcr_to_rgb(fi_rgb)
                fi_rgb = tensor_to_image(fi_rgb) * 255
                fi_rgb = np.round(fi_rgb).astype(np.uint8)
                img_save(fi_rgb, img_name, str(out_dir_rgb), mode='RGB')

    avg_time = np.mean(time_list[1:]) if len(time_list) > 1 else np.mean(time_list)
    logging.info(f"所有图像融合完成！平均处理时间: {avg_time:.4f} 秒/张。")
    logging.info(f"灰度结果保存在: {out_dir_gray}")
    if args.mode == 'RGB':
        logging.info(f"RGB 结果保存在: {out_dir_rgb}")

if __name__ == "__main__":
    # 加载配置
    config = yaml.safe_load(open('configs/cfg.yaml'))
    cfg = from_dict(config)
    
    # 设置命令行参数
    parse = argparse.ArgumentParser()
    # 注意：我们这里使用之前测试生成的 .npz 权重
    parse.add_argument('--ckpt_path', type=str, default=f'models/{cfg.exp_name}.pkl')
    parse.add_argument('--ir_path', type=str, default='./ir_imgs/')
    parse.add_argument('--vi_path', type=str, default='./vi_imgs/')
    parse.add_argument('--out_dir', type=str, default=f'test_result/fuse_result/')
    parse.add_argument('--mode', type=str, default='RGB', choices=['gray', 'RGB'], help='输出模式 (灰度或彩色)')
    
    args = parse.parse_args()

    fuse_all(args)