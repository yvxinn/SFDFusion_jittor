"""
本脚本定义了SFDFusion模型训练所使用的全部损失函数。
这些损失函数从不同维度（像素、梯度、结构相似性、显著性、频域）
来约束模型的输出，引导其生成高质量的融合图像。

主要内容包括：
- `Sobelxy`: 使用卷积实现的Sobel梯度算子。
- `PixelGradLoss`: 结合了L1像素损失和梯度损失。
- `cal_saliency_loss`: 显著性损失，用于保留红外目标和可见光背景。
- `cc`: 计算两个张量之间的相关系数，是频域损失的一部分。
- `cal_fre_loss`: 频域损失，通过重建的频谱来约束模型。
- `SSIMLoss`: 结构相似性损失（SSIM），衡量图像结构的相似度。
"""
import jittor as jt
from jittor import nn
from .fft_utils import irfftn

class Sobelxy(jt.Module):
    """
    使用固定的卷积核来高效计算图像x和y方向上的Sobel梯度。
    """
    def __init__(self):
        super().__init__()
        # 定义x和y方向的Sobel算子卷积核
        kernelx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        kernely = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        
        kernelx = jt.array(kernelx, dtype=jt.float32).unsqueeze(0).unsqueeze(0)
        kernely = jt.array(kernely, dtype=jt.float32).unsqueeze(0).unsqueeze(0)
        
        # Jittor特性：将卷积核定义为不可训练的参数（通过 stop_grad()）。
        # 这比直接创建新的jt.array更高效，因为它避免了每次执行都重新创建变量。
        self.weightx = kernelx.stop_grad()
        self.weighty = kernely.stop_grad()

    def execute(self, x):
        """对输入x应用Sobel滤波，并返回x和y方向梯度绝对值之和。"""
        sobelx = nn.conv2d(x, self.weightx, padding=1)
        sobely = nn.conv2d(x, self.weighty, padding=1)
        return jt.abs(sobelx) + jt.abs(sobely)

class PixelGradLoss(jt.Module):
    """
    一个复合损失，结合了L1像素损失和梯度损失。
    """
    def __init__(self):
        super().__init__()
        self.sobel=Sobelxy()
    def execute(self, image_vis, image_ir, fus_img):
        """
        计算像素损失和梯度损失的加权和。
        - 像素损失：鼓励融合图像接近红外和可见光中的较亮部分。
        - 梯度损失：鼓励融合图像的梯度接近红外和可见光中的较强梯度。
        """
        # 1. 计算像素损失
        y = image_vis[:, :1]
        x_in = jt.maximum(y, image_ir)
        loss_in =nn.l1_loss(x_in, fus_img)
        
        # 2. 计算梯度损失
        gy = self.sobel(y)
        gir = self.sobel(image_ir)
        gf = self.sobel(fus_img)
        gtarget = jt.maximum(gy, gir)
        loss_grad = nn.l1_loss(gtarget, gf)

        # 3. 返回加权和
        return 5*loss_in + 10*loss_grad

def cal_saliency_loss(fus: jt.Var, ir: jt.Var, vi: jt.Var, mask: jt.Var) -> jt.Var:
    """
    计算显著性损失。
    使用mask来区分前景（通常来自红外图像）和背景（通常来自可见光图像），
    并分别计算损失。

    Args:
        fus (jt.Var): 融合图像。
        ir (jt.Var): 红外图像。
        vi (jt.Var): 可见光图像。
        mask (jt.Var): 显著性掩码，前景为1，背景为0。

    Returns:
        jt.Var: 加权后的显著性损失。
    """
    # 前景损失：鼓励融合图像的前景区域接近红外图像
    loss_tar = nn.l1_loss(fus * mask, ir * mask)
    # 背景损失：鼓励融合图像的背景区域接近可见光图像
    loss_back = nn.l1_loss(fus * (1 - mask), vi * (1 - mask))
    return 5 * loss_tar + loss_back

def cc(img1: jt.Var, img2: jt.Var) -> jt.Var:
    """
    计算两个图像批次之间的相关系数（Correlation Coefficient）。
    这是频域损失`cal_fre_loss`的辅助函数。
    """
    # 添加一个小的epsilon以防止除以零
    eps = 1e-7 

    N, C, H, W = img1.shape
    # 将图像展平为向量
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    # 中心化
    img1 = img1 - img1.mean(dim=-1, keepdims=True)
    img2 = img2 - img2.mean(dim=-1, keepdims=True)

    # 计算相关系数
    num = jt.sum(img1 * img2, dim=-1)
    den = jt.multiply(jt.sqrt(jt.sum(img1**2, dim=-1)), jt.sqrt(jt.sum(img2**2, dim=-1)))
    
    # 将结果限制在[-1, 1]范围内，并取平均值
    return jt.clamp(jt.divide(num, (eps + den)), -1.0, 1.0).mean()

def cal_fre_loss(amp: jt.Var, pha: jt.Var, ir: jt.Var, vi: jt.Var, mask: jt.Var) -> jt.Var:
    """
    计算频域损失。
    通过将融合后的幅度和相位谱重建回空间域，并计算其与原始图像
    在显著性区域内的相关性来实现。
    """
    # 1. 从融合后的幅度和相位重建半复数频谱
    real = amp * jt.cos(pha) + 1e-8
    imag = amp * jt.sin(pha) + 1e-8
    half_spec = jt.stack([real, imag], dim=-1)
    
    # 2. 调用我们自定义的irfftn算子执行IFFT
    x_ifft = irfftn(half_spec)

    # 3. 取绝对值得到重建的空间域表示
    x_ifft_abs = jt.abs(x_ifft)

    # 4. 分别计算重建结果与红外前景、可见光背景的相关性
    loss_ir = cc(x_ifft_abs * mask, ir * mask)
    loss_vi = cc(x_ifft_abs * (1 - mask), vi * (1 - mask))
    return loss_ir + loss_vi

def _create_window(window_size: int, channel: int, sigma: float) -> jt.Var:
    """
    创建一个用于SSIM计算的高斯窗口。
    该实现与Kornia/PyTorch的行为对齐。
    """
    # 生成一维高斯分布
    gauss = jt.exp(-(jt.arange(window_size, dtype='float32') - window_size // 2) ** 2 / float(2 * sigma ** 2))
    _1D_window = jt.divide(gauss, gauss.sum())
    # 通过外积将一维窗口扩展为二维
    _2D_window = jt.matmul(jt.unsqueeze(_1D_window, 1), jt.unsqueeze(_1D_window, 0))
    # 扩展为适合卷积的形状 (C, 1, H, W)
    window = _2D_window.unsqueeze(0).unsqueeze(0).expand(channel, 1, window_size, window_size).contiguous()
    return window

class SSIMLoss(jt.nn.Module):
    """
    结构相似性（SSIM）损失函数。
    该实现接近Kornia库，返回一个表示损失的map。
    """
    def __init__(self, window_size: int = 11, reduction: str = 'mean', max_val: float = 1.0):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.reduction = reduction
        self.max_val = max_val
        # 创建一个可复用的高斯窗口
        self.window = _create_window(window_size, 1, 1.5)
        
        self.C1 = (0.01 * self.max_val) ** 2
        self.C2 = (0.03 * self.max_val) ** 2

    def execute(self, img1, img2):
        """
        计算img1和img2之间的SSIM损失。
        """
        (_, channel, _, _) = img1.shape
        
        # Jittor特性：使用分组卷积（groups=channel）来对每个通道独立应用高斯滤波。
        # 这是计算SSIM中局部均值、方差和协方差的标准方法。
        window = self.window.expand(channel, 1, self.window_size, self.window_size)
        
        padding = self.window_size // 2
        
        mu1 = jt.nn.conv2d(img1, window, padding=padding, groups=channel)
        mu2 = jt.nn.conv2d(img2, window, padding=padding, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = jt.nn.conv2d(img1 * img1, window, padding=padding, groups=channel) - mu1_sq
        sigma2_sq = jt.nn.conv2d(img2 * img2, window, padding=padding, groups=channel) - mu2_sq
        sigma12 = jt.nn.conv2d(img1 * img2, window, padding=padding, groups=channel) - mu1_mu2
        
        # 计算SSIM map
        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))
        
        # 将SSIM map转换为loss map
        loss_map = jt.clamp(1.0 - ssim_map, min_v=0, max_v=1) / 2.0
        
        # 根据指定的reduction策略返回最终损失
        if self.reduction == 'mean':
            return loss_map.mean()
        elif self.reduction == 'sum':
            return loss_map.sum()
        else: # 'none'
            return loss_map