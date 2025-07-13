from modules import * # 导入自定义的网络模块，包含融合网络的核心架构
import os # 导入操作系统接口，用于文件和目录操作
import numpy as np # 导入NumPy数值计算库，用于数组和矩阵运算
from utils.evaluator import Evaluator # 导入自定义的评估指标类，用于计算图像质量指标
import torch # 导入PyTorch深度学习框架
from utils.img_read import * # 导入自定义的图像读取工具函数
import logging # 导入日志记录模块，用于输出运行信息
from kornia.metrics import AverageMeter # 导入Kornia库的平均值计算器
from tqdm import tqdm # 导入tqdm进度条显示库
import warnings # 导入警告处理模块
import yaml # 导入YAML配置文件解析库
from configs import from_dict # 导入配置字典转换函数
import dataset # 导入数据集处理模块
from torch.utils.data import DataLoader # 导入PyTorch数据加载器
# from thop import profile, clever_format # 导入模型复杂度分析工具 (移除，因为依赖 PyTorch)
import time # 导入时间计算模块
import cv2 # 导入OpenCV图像处理库
import argparse # 导入命令行参数解析模块


# 忽略所有警告信息，避免输出干扰
warnings.filterwarnings("ignore")
# 设置CUDA可见设备为第0块GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 自动检测并选择计算设备（GPU优先，否则使用CPU）
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 定义日志输出格式：时间 | 文件名[行号] | 日志级别 | 消息内容
log_f = '%(asctime)s | %(filename)s[line:%(lineno)d] | %(levelname)s | %(message)s'
# 配置日志系统，设置为INFO级别
logging.basicConfig(level='INFO', format=log_f)


def fuse(args):
    """
    主要的图像融合函数
    
    Args:
        args: 命令行参数对象，包含模型路径、图像路径、输出路径等配置
    """
    
    # 获取融合结果输出目录路径
    fuse_out_folder = args.out_dir
    # 检查输出目录是否存在，不存在则创建
    if not os.path.exists(fuse_out_folder):
        os.makedirs(fuse_out_folder)

    # 创建SFDFusion融合网络实例
    fuse_net = Fuse()
    # 从指定路径加载预训练模型权重，并映射到当前设备
    ckpt = torch.load(args.ckpt_path, map_location=device)
    # 加载融合网络的状态字典（模型参数）
    fuse_net.load_state_dict(ckpt['fuse_net'])
    # 将模型移动到计算设备（GPU或CPU）
    fuse_net.to(device)
    # 设置模型为评估模式，关闭dropout和batch normalization的训练行为
    fuse_net.eval()

    # 初始化时间记录列表，用于统计每张图像的处理时间
    time_list = []

    # 获取红外图像目录下所有文件名
    img_names = [i for i in os.listdir(args.ir_path)]
    # 批量读取红外图像：以灰度模式('L')读取，并添加batch维度
    ir_imgs = [img_read(os.path.join(args.ir_path, i), mode='L').unsqueeze(0) for i in img_names]
    
    # 批量读取可见光图像：分别保存Y通道和CbCr通道
    vi_y_imgs = []
    vi_cbcr_imgs = []
    for img_name in img_names:
        y, cbcr = img_read(os.path.join(args.vi_path, img_name), mode='YCbCr')
        vi_y_imgs.append(y.unsqueeze(0))  # Y通道用于融合
        vi_cbcr_imgs.append(cbcr.unsqueeze(0))  # CbCr通道用于RGB重建
    
    # 遍历所有图像，确保尺寸为偶数（FFT变换的要求）
    for idx,img in enumerate(ir_imgs):
        # 获取图像的批次、通道、高度、宽度
        _,_, h, w = img.shape
        # 检查高度或宽度是否为奇数
        if h // 2 != 0 or w // 2 != 0:
            # 将红外图像裁剪为偶数尺寸（向下取整到最近的偶数）
            ir_imgs[idx] = ir_imgs[idx][:, : h // 2 * 2, : w // 2 * 2]
            # 同步裁剪对应的可见光图像，保持尺寸一致
            vi_y_imgs[idx] = vi_y_imgs[idx][:, : h // 2 * 2, : w // 2 * 2]
            vi_cbcr_imgs[idx] = vi_cbcr_imgs[idx][:, : h // 2 * 2, : w // 2 * 2]
    
    # 将红外图像、可见光图像和文件名打包为数据列表
    data_list=zip(ir_imgs, vi_y_imgs, vi_cbcr_imgs, img_names)
    
    # 使用torch.no_grad()上下文管理器，禁用梯度计算以节省内存和加速推理
    with torch.no_grad():
        # 记录开始融合的日志信息
        logging.info(f'fusing images ...')
        # 创建进度条，显示处理进度
        iter = tqdm(data_list, total=len(img_names), ncols=80)
        
        # 遍历每对红外和可见光图像
        for data_ir, data_vi_y, data_vi_cbcr, img_name in iter:
            # 将图像数据移动到计算设备（GPU或CPU）
            data_vi_y, data_ir = data_vi_y.to(device), data_ir.to(device)

            # 记录推理开始时间
            ts = time.time()
            # 执行图像融合：输入红外和可见光图像，返回融合结果、幅度和相位信息
            fus_data, _, _ = fuse_net(data_ir, data_vi_y)
            # print(fus_data.shape)  # 调试用：可以打印融合结果的张量形状
            # 记录推理结束时间
            te = time.time()
            # 将单张图像的处理时间添加到时间列表
            time_list.append(te - ts)
            
            # 根据输出模式保存融合结果
            if args.mode == 'gray':
                # 灰度模式：将融合结果转换为0-255范围的8位灰度图像
                # np只能处理cpu上的数据，所以需要将数据移动到cpu上
                fi = np.squeeze((fus_data * 255).cpu().numpy()).astype(np.uint8)
                # 保存灰度图像到输出目录
                img_save(fi, img_name, fuse_out_folder)
            elif args.mode == 'RGB':
                # RGB模式：需要重建彩色图像
                # 将CbCr通道移动到计算设备
                data_vi_cbcr = data_vi_cbcr.to(device)
                # 将融合的Y通道与原始的CbCr通道拼接，重建YCbCr图像
                fi = torch.cat((fus_data, data_vi_cbcr), dim=1)
                # 将YCbCr格式转换为RGB格式
                fi = ycbcr_to_rgb(fi)
                # 将张量转换为图像格式并缩放到0-255范围
                fi = tensor_to_image(fi) * 255
                # 转换为8位整数类型
                fi = fi.astype(np.uint8)
                # 保存RGB图像到输出目录
                img_save(fi, img_name, fuse_out_folder, mode='RGB')

    # 以下代码已注释，可用于输出处理完成信息和平均处理时间
    # logging.info(f'fusing images done!')
    # logging.info(f'time: {np.round(np.mean(time_list[1:]), 6)}s')


# 主程序入口点
if __name__ == "__main__":
    # 加载YAML配置文件
    config = yaml.safe_load(open('configs/cfg.yaml'))
    # 将配置字典转换为配置对象
    cfg = from_dict(config)
    
    # 创建命令行参数解析器
    parse = argparse.ArgumentParser()
    # 添加模型检查点路径参数，默认使用配置文件中的实验名称
    parse.add_argument('--ckpt_path', type=str, default=f'models/{cfg.exp_name}.pth')
    # 添加红外图像目录路径参数
    parse.add_argument('--ir_path', type=str, default='./ir_imgs/')
    # 添加可见光图像目录路径参数
    parse.add_argument('--vi_path', type=str, default='./vi_imgs/')
    # 添加输出目录路径参数
    parse.add_argument('--out_dir', type=str, default=f'test_result/fuse_result/')
    # 添加输出模式参数（'gray'为灰度输出，'RGB'为彩色输出）
    parse.add_argument('--mode', type=str, default='RGB')
    # 解析命令行参数
    args = parse.parse_args()

    # 调用融合函数开始处理
    fuse(args)
    
