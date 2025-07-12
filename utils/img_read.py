import jittor as jt
import numpy as np
from PIL import Image
import os
from jittor import transform

# 手动实现RGB到YCbCr的转换
def rgb_to_ycbcr_jt(image: jt.Var) -> jt.Var:
    if image.shape[0] != 3:
        raise ValueError("输入图像必须是3通道的RGB图像")
    r, g, b = image[0], image[1], image[2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    ycbcr = jt.stack([y, cb, cr], dim=0)
    return ycbcr

def img_read(path, mode):
    '''
    使用 Jittor 读取图像
    input: path, mode
    output: jittor.Var, [c, h, w]
    '''
    assert mode in ['RGB', 'L', 'YCbCr'], "mode should be 'RGB', 'L', or 'YCbCr'"
    img_pil = Image.open(path).convert(mode if mode != 'YCbCr' else 'RGB')

    # ======================================================================
    # Jittor 的 to_tensor 函数行为与 PyTorch 的不等价，
    # 应该返回一个可以通过 .numpy() 转换的对象（即 jittor.Var）
    # 但Jittor 的 to_tensor 直接返回了 numpy.ndarray，
    # 所以这里需要手动转换
    # img_tensor = image_to_tensor(img, keepdim=True) / 255.0
    # ======================================================================
    img_np = np.array(img_pil, dtype=np.float32)

    if img_np.ndim == 2:  # 处理灰度图
        # 增加通道维度: [H, W] -> [1, H, W]
        img_np = np.expand_dims(img_np, axis=0)
    else:  # 处理 RGB 图
        # 转换维度: [H, W, C] -> [C, H, W]
        img_np = img_np.transpose((2, 0, 1))
    
    # 归一化: [0, 255] -> [0.0, 1.0]
    img_np /= 255.0

    # 从 numpy 数组创建 Jittor Var
    img_tensor = jt.array(img_np)
    # ======================================================================
    
    if mode == 'RGB' or mode == 'L':
        return img_tensor
    elif mode == 'YCbCr':
        img_ycbcr = rgb_to_ycbcr_jt(img_tensor)
        y, cb, cr = jt.split(img_ycbcr, 1, dim=0)
        cbcr = jt.concat([cb, cr], dim=0)
        return y, cbcr

def img_save(image, imagename, savedir, mode='L'):
    """
    健壮的 Jittor 图像保存函数。
    它能正确处理 jittor.Var 和 numpy.ndarray 两种输入。
    """
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    if isinstance(image, jt.Var):
        # to_pil_image 需要 HWC 格式的输入.
        # 我们的 image Var 是 CHW 格式, 所以需要转置.
        # CHW (0, 1, 2) -> HWC (1, 2, 0)
        image_hwc = image.transpose(1, 2, 0)
        img = transform.to_pil_image(image_hwc)
    else:
        # 如果输入已经是 NumPy 数组，直接使用
        img = Image.fromarray(image, mode=mode)

    path = os.path.join(savedir, imagename)
    img.save(path)
