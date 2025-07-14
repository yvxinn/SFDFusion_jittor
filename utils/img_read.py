from PIL import Image
import os
import jittor as jt
import numpy as np
from jittor import transform

def img_read(path: str, mode: str = 'L'):
    """
    Jittor 版图像读取函数。
    - 读取图像
    - 根据模式（L, RGB, YCbCr）进行转换
    - 明确将 numpy array 转换为 Jittor.Var
    """
    img = Image.open(path)
    
    # 统一使用 ToTensor，它在当前环境下返回 numpy array
    to_tensor_transform = transform.ToTensor()

    if mode == 'L':
        img = img.convert('L')
        img_np = to_tensor_transform(img)
        return jt.array(img_np)

    elif mode == 'RGB':
        img = img.convert('RGB')
        img_np = to_tensor_transform(img)
        return jt.array(img_np)
        
    elif mode == 'YCbCr':
        img = img.convert('YCbCr')
        img_np = to_tensor_transform(img)
        img_tensor = jt.array(img_np)
        # 分离 Y 和 CbCr 通道
        y = img_tensor[0:1, :, :]
        cbcr = img_tensor[1:3, :, :]
        return y, cbcr
        
    else:
        raise ValueError(f"Unsupported mode: {mode}")

def ycbcr_to_rgb(image: jt.Var):
    """
    将 YCbCr Jittor Var 转换为 RGB Jittor Var.
    image: (N, 3, H, W) 的 Jittor Var
    """
    if not isinstance(image, jt.Var) or image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("Input must be a 4D Jittor Var with 3 channels (YCbCr)")

    y = image[:, 0:1, :, :]
    cb = image[:, 1:2, :, :] - 0.5
    cr = image[:, 2:3, :, :] - 0.5

    # YCbCr to RGB conversion matrix (standard for JPEG)
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    
    rgb = jt.concat([r, g, b], dim=1)
    return jt.clamp(rgb, 0, 1)

def tensor_to_image(tensor: jt.Var):
    """
    将 Jittor Var (C, H, W) 转换为 NumPy 数组 (H, W, C)
    """
    if not isinstance(tensor, jt.Var):
        raise ValueError("Input must be a Jittor Var")
        
    # 如果有 batch 维度，移除它
    if tensor.ndim == 4:
        tensor = jt.squeeze(tensor, 0)
        
    # C, H, W -> H, W, C
    return tensor.numpy().transpose(1, 2, 0)


def img_save(image, imagename, savedir, mode='L'):
    """
    Jittor 图像保存函数
    """
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    # .numpy() is needed if image is a Jittor Var
    if isinstance(image, jt.Var):
        image = image.numpy()

    # Ensure image is in HWC format for to_pil_image if it's a numpy array
    if image.ndim == 3 and image.shape[0] in [1, 3]: # CHW -> HWC
         image = image.transpose(1, 2, 0)
    
    # Squeeze single-channel dimension if present, e.g. (H, W, 1) -> (H, W)
    if image.ndim == 3 and image.shape[2] == 1:
        image = image.squeeze(axis=2)

    # Convert from float [0,1] to uint8 [0,255] for saving
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)

    img_pil = Image.fromarray(image, mode=mode)
    path = os.path.join(savedir, imagename)
    img_pil.save(path)