from PIL import Image
import os
import jittor as jt
from jittor import transform

def img_read(path: str, mode: str = 'L'):
    """
    Jittor 版图像读取函数
    职责单一：只负责用 PIL 打开图像并按需转换模式
    返回原始的 PIL.Image 对象
    """
    img = Image.open(path)
    # 对于 YCbCr，直接让 PIL 处理转换，返回的也是 PIL Image 对象
    return img.convert(mode)

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