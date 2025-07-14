import jittor as jt
import jittor.nn as nn
import numpy as np


def fft(input_real):
    '''
    input_real: jittor var of shape (N, C, H, W)
    使用底层的 _fft2 精确模拟 PyTorch 的 rfft
    '''
    # 1. 准备 _fft2 的输入
    # 输入是实数，所以虚部为0
    batch_size, channels, H, W = input_real.shape
    input_imag = jt.zeros_like(input_real)

    # 堆叠实部和虚部，形成 (N, C, H, W, 2) 的5维张量
    x_complex = jt.stack([input_real, input_imag], dim=-1)

    # _fft2 要求输入是4维的 (N', H, W, 2)，所以合并 N 和 C 维度
    x_complex_reshaped = x_complex.view(batch_size * channels, H, W, 2)

    # 2. 调用底层的 _fft2
    # 输出 fft_full 的形状是 (N*C, H, W, 2)
    fft_full = nn._fft2(x_complex_reshaped, inverse=False)

    # 3. 手动截取，模拟 'rfft' (实数FFT) 的行为
    # PyTorch rfft 的输出宽度是 W // 2 + 1
    output_width = W // 2 + 1
    fft_r_reshaped = fft_full[:, :, :output_width, :]

    # 4. 将形状恢复为 (N, C, H, W_out, 2)
    fft_r = fft_r_reshaped.view(batch_size, channels, H, output_width, 2)

    # 5. 从截取后的复数张量计算幅度和相位
    real_part = fft_r[..., 0]
    imag_part = fft_r[..., 1]
    
    amp = jt.sqrt(real_part**2 + imag_part**2)
    pha = jt.atan2(imag_part, real_part)
    
    # amp 和 pha 的形状将是 (N, C, H, W//2 + 1)，与 PyTorch 完全一致
    return amp, pha


class Att_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.att = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1), 
            nn.Sigmoid()
        )

    def execute(self, x):
        att = self.att(x)
        x = x * att
        return x


class Sobelxy(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).astype(np.float32)
        sobel_filter_y = sobel_filter_x.T

        weight_x = np.zeros((channels, 1, kernel_size, kernel_size), dtype=np.float32)
        weight_y = np.zeros((channels, 1, kernel_size, kernel_size), dtype=np.float32)

        for i in range(channels):
            weight_x[i, 0] = sobel_filter_x
            weight_y[i, 0] = sobel_filter_y

        self.convx = nn.Conv2d(
            channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels, bias=False
        )
        self.convx.weight = jt.array(weight_x)
        
        self.convy = nn.Conv2d(
            channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels, bias=False
        )
        self.convy.weight = jt.array(weight_y)

        # ======================= 最终修正 =======================
        # 明确设置Sobel算子的权重不需要计算梯度，统一框架行为
        self.convx.weight.stop_grad()
        self.convy.weight.stop_grad()
        # ========================================================

    def execute(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x = jt.add(jt.abs(sobelx), jt.abs(sobely))
        return x


class DMRM(nn.Module):
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
    def __init__(self, dim, channels=32):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(dim, channels, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.down_conv = nn.Sequential(
            nn.Sequential(nn.Conv2d(channels, channels * 4, kernel_size=3, stride=1, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(channels * 4, channels * 2, kernel_size=3, stride=1, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1), nn.Tanh()),
        )

    def execute(self, ir, vi, frefus):
        x = jt.concat([ir, vi, frefus], dim=1)
        x = self.encoder(x)
        x = self.down_conv(x)
        return x

class IFFT(nn.Module):
    def __init__(self, out_channels=8):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, out_channels // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def execute(self, amp, pha):
        # amp 和 pha 的形状是 (N, C, H, W//2 + 1)
        # 1. 重建半复数频谱，与 PyTorch 版本对齐增加 1e-8
        real_part = amp * jt.cos(pha) + 1e-8
        imag_part = amp * jt.sin(pha) + 1e-8
        half_spectrum_complex = jt.stack([real_part, imag_part], dim=-1)

        # 2. 从 rfft 的输出重建完整的复数频谱
        mirrored_part = half_spectrum_complex[:, :, :, 1:-1, :]
        mirrored_part_conj = jt.stack([mirrored_part[..., 0], -mirrored_part[..., 1]], dim=-1)
        
        # 将共轭部分沿宽度维度(维度3)翻转
        mirrored_part_conj_flipped = jt.flip(mirrored_part_conj, dim=3)

        # 拼接成完整频谱
        full_spectrum_complex = jt.concat([half_spectrum_complex, mirrored_part_conj_flipped], dim=3)
        
        batch_size, channels, H, W_full = full_spectrum_complex.shape[:4]
        full_spectrum_reshaped = full_spectrum_complex.view(batch_size * channels, H, W_full, 2)

        # 3. 执行逆 FFT
        x_ifft_complex = nn._fft2(full_spectrum_reshaped, inverse=True)

        # 取实部并恢复形状
        x_real_reshaped = x_ifft_complex[..., 0]

        # 与 PyTorch 版本对齐，对逆变换后的实数结果取绝对值
        x_abs = jt.abs(x_real_reshaped)
        x = x_abs.view(batch_size, channels, H, W_full)
        
        # 4. 沿通道维度计算统计特征，模拟PyTorch版本
        # x shape: (N, C, H, W)
        x_max = jt.max(x, dim=1, keepdims=True)
        x_mean = jt.mean(x, dim=1, keepdims=True)
        
        # 5. 拼接特征并通过卷积
        x_cat = jt.concat([x_max, x_mean], dim=1)
        
        x_out = self.conv1(x_cat)
        return x_out

class AmpFuse(nn.Module):
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
    def __init__(self):
        super().__init__()
        self.channel = 8
        self.dmrm = DMRM(1, self.channel)
        # --- 修复：与 PyTorch 版本对齐，不传入参数 ---
        self.ff1 = AmpFuse()
        self.ff2 = PhaFuse()
        # --- 修复结束 ---
        self.ifft = IFFT(self.channel)
        # Fuse_block 的输入维度是 3 * self.channel
        self.fus_block = Fuse_block(self.channel * 3)

    def execute(self, ir, vi):
        ir_amp, ir_pha = fft(ir)
        vi_amp, vi_pha = fft(vi)
        
        amp = self.ff1(ir_amp, vi_amp)
        pha = self.ff2(ir_pha, vi_pha)
        
        frefus = self.ifft(amp, pha)
        
        # 与 PyTorch 版本对齐，使用 dmrm 的输出覆盖 ir 和 vi
        ir, vi = self.dmrm(ir, vi)
        
        # 确保所有张量都是4维的
        # ir_f: (N, 8, H, W)
        # vi_f: (N, 8, H, W)
        # frefus: (N, 8, H, W)
        fus = self.fus_block(ir, vi, frefus)
        
        # 归一化操作
        min_val = fus.min()
        max_val = fus.max()
        # 修正：与 PyTorch 版本对齐，移除 epsilon 以确保数值一致
        fus = (fus - min_val) / (max_val - min_val)
        
        return fus, amp, pha
