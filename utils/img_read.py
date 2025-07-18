"""
SFDFusion Jittor版本 图像读写工具
====================================

本脚本提供了一系列用于图像处理的辅助函数，主要围绕使用PIL库进行
图像的读取、保存以及在Jittor张量和NumPy数组/PIL图像之间的转换。

主要功能包括：
- `img_read`: 从指定路径读取图像，并根据需要转换为特定模式（灰度、RGB、YCbCr）。
- `ycbcr_to_rgb`: 将YCbCr颜色空间的Jittor张量转换为RGB颜色空间。
- `tensor_to_image`: 将Jittor张量（通常为CHW格式）转换回适用于图像处理的
                   NumPy数组（HWC格式）。
- `img_save`: 将Jittor张量或NumPy数组保存为图像文件。
"""
from PIL import Image
import os
import jittor as jt
import numpy as np

def img_read(path: str, mode: str = 'L') -> Image.Image:
    """
    从文件路径读取图像并转换为指定模式。

    Args:
        path (str): 图像文件的路径。
        mode (str, optional): 目标图像模式。支持 'L' (灰度), 'RGB', 'YCbCr'。
                              默认为 'L'。

    Returns:
        Image.Image: 一个PIL图像对象。
    """
    img = Image.open(path)
    
    # 根据指定模式转换图像
    if mode == 'L':
        return img.convert('L')
    elif mode == 'RGB':
        return img.convert('RGB')
    elif mode == 'YCbCr':
        return img.convert('YCbCr')
    else:
        raise ValueError(f"不支持的图像模式: {mode}")

def ycbcr_to_rgb(image: jt.Var) -> jt.Var:
    """
    将一个批次的YCbCr Jittor张量转换为RGB Jittor张量。
    输入张量的值域应为 [0, 1]。

    Args:
        image (jt.Var): 形状为 (N, 3, H, W) 的YCbCr张量。

    Returns:
        jt.Var: 形状为 (N, 3, H, W) 的RGB张量，值域被裁剪到 [0, 1]。
    """
    if not isinstance(image, jt.Var) or image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("输入必须是形状为(N, 3, H, W)的Jittor Var")

    # 1. 分离Y, Cb, Cr通道，并将Cb, Cr中心化到[-0.5, 0.5]
    y = image[:, 0:1, :, :]
    cb = image[:, 1:2, :, :] - 0.5
    cr = image[:, 2:3, :, :] - 0.5

    # 2. 应用标准的YCbCr到RGB转换矩阵
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    
    # 3. 合并RGB通道并裁剪值域
    rgb = jt.concat([r, g, b], dim=1)
    return jt.clamp(rgb, 0, 1)

def tensor_to_image(tensor: jt.Var) -> np.ndarray:
    """
    将一个Jittor张量（通常是 B,C,H,W 或 C,H,W 格式）转换为
    一个适用于图像显示的NumPy数组（H,W,C 格式）。

    Args:
        tensor (jt.Var): 输入的Jittor张量。

    Returns:
        np.ndarray: HWC格式的NumPy数组。
    """
    # 如果有批次维度，移除它
    if tensor.ndim == 4:
        tensor = jt.squeeze(tensor, 0)
    # 将 C,H,W 格式转换为 H,W,C 格式
    return tensor.numpy().transpose(1, 2, 0)


def img_save(image, imagename: str, savedir: str, mode: str = 'L'):
    """
    将Jittor张量或NumPy数组保存为图像文件。
    该函数会自动处理数据类型转换和维度重排。

    Args:
        image: 要保存的图像，可以是Jittor张量或NumPy数组。
        imagename (str): 输出图像的文件名。
        savedir (str): 保存图像的目录。
        mode (str, optional): 保存PIL图像时使用的模式。默认为 'L'。
    """
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    # 1. 如果是Jittor张量，先转换为NumPy数组
    if isinstance(image, jt.Var):
        image = image.numpy()

    # 2. 如果是CHW格式，转换为HWC格式
    if image.ndim == 3 and image.shape[0] in [1, 3]:
         image = image.transpose(1, 2, 0)
    
    # 3. 如果是单通道的HWC（即H,W,1），则压缩掉最后的维度
    if image.ndim == 3 and image.shape[2] == 1:
        image = image.squeeze(axis=2)

    # 4. 如果是浮点数（通常在[0,1]范围），转换为uint8（[0,255]范围）
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)

    # 5. 从NumPy数组创建PIL图像并保存
    img_pil = Image.fromarray(image, mode=mode)
    path = os.path.join(savedir, imagename)
    img_pil.save(path)