"""
本脚本定义了构成SFDFusion模型的所有核心神经网络模块。
这些模块共同协作，以实现对红外和可见光图像的特征提取与融合。

主要模块包括：
- `fft`: 一个手动实现的函数，用以模拟PyTorch的`rfft`，计算输入张量的
          幅度和相位谱。
- `Att_Block`: 一个简单的注意力模块。
- `Sobelxy`: 使用Sobel算子计算图像梯度的模块。
- `DMRM`: (Detail-preserving Mutual-attention-based Registration Module)
          一个用于提取细节和互补信息的关键模块。
- `Fuse_block`: 将从不同分支提取的特征进行最终融合的模块。
- `IFFT`: 一个自定义模块，用于从幅度和相位谱重建空间域特征，
          并在内部调用我们自定义的`irfftn`算子。
- `AmpFuse` & `PhaFuse`: 用于融合幅度和相位谱的简单卷积模块。
- `Fuse`: 最终的顶层模型，整合了上述所有模块，定义了完整的前向传播路径。
"""
import jittor as jt
import jittor.nn as nn
import numpy as np
from utils.fft_utils import irfftn


def fft(input_real: jt.Var) -> (jt.Var, jt.Var):
    """
    手动实现的`rfft`，用于计算2D实数输入的幅度和相位。
    Jittor特性：由于旧版本Jittor缺少`fft`模块，此函数通过调用底层的`_fft2`
    并手动进行频谱的截取和重塑，来精确模拟PyTorch `rfft2`的行为。

    Args:
        input_real (jt.Var): 形状为 (N, C, H, W) 的实数输入张量。

    Returns:
        tuple[jt.Var, jt.Var]: 幅度谱和相位谱，形状均为 (N, C, H, W//2 + 1)。
    """
    # 1. 准备 `_fft2` 的输入
    # 输入是实数，所以虚部为0
    batch_size, channels, H, W = input_real.shape
    input_imag = jt.zeros_like(input_real)

    # 将实部和虚部堆叠，形成 (N, C, H, W, 2) 的复数表示
    x_complex = jt.stack([input_real, input_imag], dim=-1)

    # `_fft2` 要求输入是4维的 (B, H, W, 2)，因此合并N和C维度
    x_complex_reshaped = x_complex.reshape((batch_size * channels, H, W, 2))

    # 2. 调用Jittor底层的FFT实现
    fft_full = nn._fft2(x_complex_reshaped, inverse=False)

    # 3. 手动截取，模拟 `rfft` 对实数输入的处理
    # 实数FFT的结果是厄米共轭的，因此只需保留一半的频谱
    output_width = W // 2 + 1
    fft_r_reshaped = fft_full[:, :, :output_width, :]

    # 4. 将形状恢复为 (N, C, H, W_out, 2)
    fft_r = fft_r_reshaped.reshape(batch_size, channels, H, output_width, 2)

    # 5. 从截取后的复数张量计算幅度和相位
    real_part = fft_r[..., 0]
    imag_part = fft_r[..., 1]
    
    amp = jt.sqrt(real_part**2 + imag_part**2)
    pha = jt.atan2(imag_part, real_part)
    
    return amp, pha


class Att_Block(nn.Module):
    """一个简单的自注意力模块。"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.att = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1), 
            nn.Sigmoid()
        )

    def execute(self, x):
        """通过计算出的注意力权重对输入进行加权。"""
        att = self.att(x)
        x = x * att
        return x


class Sobelxy(nn.Module):
    """
    使用不可训练的卷积层来实现Sobel梯度算子。
    Jittor特性：通过将会长和宽固定的卷积核(`sobel_filter`)赋值给
    `conv.weight`，并设置`groups=channels`，可以对每个通道独立地
    应用相同的Sobel滤波，效率很高。
    """
    def __init__(self, channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        # x方向的梯度卷积
        self.convx = nn.Conv2d(
            channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels, bias=False
        )
        self.convx.weight.data = jt.array(sobel_filter).float().view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        # y方向的梯度卷积
        self.convy = nn.Conv2d(
            channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels, bias=False
        )
        self.convy.weight.data = jt.array(sobel_filter.T).float().view(1, 1, 3, 3).repeat(channels, 1, 1, 1)

    def execute(self, x):
        """分别计算x和y方向的梯度，并返回其绝对值之和。"""
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x = jt.abs(sobelx) + jt.abs(sobely)
        return x


class DMRM(nn.Module):
    """
    Detail-preserving Mutual-attention-based Registration Module (DMRM)
    该模块是模型的核心之一，用于提取和增强红外与可见光图像的特征。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.ir_embed = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.vi_embed = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.ir_att1 = Att_Block(out_channels, out_channels)
        self.ir_att2 = Att_Block(out_channels, out_channels)
        self.vi_att1 = Att_Block(out_channels, out_channels)
        self.vi_att2 = Att_Block(out_channels, out_channels)
        self.grad_ir = Sobelxy(out_channels)
        self.grad_vi = Sobelxy(out_channels)

    def execute(self, x, y):
        """执行DMRM模块的前向传播。"""
        x = self.ir_embed(x)
        y = self.vi_embed(y)
        t = x + y
        x1 = self.ir_att1(x)
        y1 = self.vi_att1(y)
        x2 = self.ir_att2(t)
        y2 = self.vi_att2(t)
        ir_grad = self.grad_ir(x)
        vi_grad = self.grad_vi(y)
        return x1 + x2 + ir_grad, y1 + y2 + vi_grad


class Fuse_block(nn.Module):
    """将来自空间域和频域的特征进行最终融合的模块。"""
    def __init__(self, dim, channels=32):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(dim, channels, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.down_conv = nn.Sequential(
            nn.Sequential(nn.Conv2d(channels, channels * 4, kernel_size=3, stride=1, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(channels * 4, channels * 2, kernel_size=3, stride=1, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1), nn.ReLU()),
            # 使用Tanh激活函数将输出范围限制在[-1, 1]
            nn.Sequential(nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1), nn.Tanh()),
        )

    def execute(self, ir, vi, frefus):
        """将三个输入特征沿通道维度拼接后进行卷积处理。"""
        x = jt.concat([ir, vi, frefus], dim=1)
        x = self.encoder(x)
        x = self.down_conv(x)
        return x

class IFFT(nn.Module):
    """
    一个用于从幅度和相位重建空间域特征的自定义模块。
    """
    def __init__(self, out_channels=8):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, out_channels // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def execute(self, amp, pha):
        """
        执行逆傅里叶变换，并从结果中提取统计特征。
        """
        # 1. 从幅度和相位重建半复数频谱
        real_part = amp * jt.cos(pha) + 1e-8
        imag_part = amp * jt.sin(pha) + 1e-8
        half_spec = jt.stack([real_part, imag_part], dim=-1)

        # 2. 调用我们自定义的、梯度计算正确的irfftn算子
        x_ifft = irfftn(half_spec)

        # 3. 取绝对值得到空间域特征
        x = jt.abs(x_ifft)
        
        # 4. 沿通道维度计算最大值和平均值作为统计特征
        x_max = jt.max(x, dim=1, keepdims=True)
        x_mean = jt.mean(x, dim=1, keepdims=True)
        x_cat = jt.concat([x_max, x_mean], dim=1)
        
        # 5. 对统计特征进行卷积
        x_out = self.conv1(x_cat)
        return x_out

class AmpFuse(nn.Module):
    """一个用于融合幅度谱的简单卷积模块。"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
        )

    def execute(self, f1, f2):
        x = jt.concat([f1, f2], dim=1)
        x = self.conv1(x)
        return x


class PhaFuse(nn.Module):
    """一个用于融合相位谱的简单卷积模块。"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
        )

    def execute(self, f1, f2):
        x = jt.concat([f1, f2], dim=1)
        x = self.conv1(x)
        return x

class Fuse(nn.Module):
    """
    SFDFusion的顶层模型，整合所有子模块。
    """
    def __init__(self):
        super().__init__()
        self.channel = 8
        self.dmrm = DMRM(1, self.channel)
        self.ff1 = AmpFuse()
        self.ff2 = PhaFuse()
        self.ifft = IFFT(self.channel)
        self.fus_block = Fuse_block(self.channel * 3)

    def execute(self, ir, vi):
        """
        定义完整的前向传播路径。
        """
        # 1. 频域分支：分别计算和融合幅度和相位
        ir_amp, ir_pha = fft(ir)
        vi_amp, vi_pha = fft(vi)
        
        amp = self.ff1(ir_amp, vi_amp)
        pha = self.ff2(ir_pha, vi_pha)
        
        # 从融合后的频谱重建空间特征
        frefus = self.ifft(amp, pha)
        
        # 2. 空间域分支：使用DMRM提取细节特征
        ir, vi = self.dmrm(ir, vi)
        
        # 3. 最终融合：将空间域特征和频域特征送入融合块
        fus = self.fus_block(ir, vi, frefus)
        
        # 4. 归一化操作，将输出值缩放到[0, 1]范围
        min_val = fus.min()
        max_val = fus.max()
        fus = (fus - min_val) / (max_val - min_val)
        
        # 返回最终的融合图像，以及中间的幅度和相位（用于计算损失）
        return fus, amp, pha
