import jittor as jt
from jittor import nn, Function
import numpy as np

class IRFFTN(Function):
    def execute(self, half_spec):
        """
        前向传播：执行您手动实现的 IFFT 逻辑
        """
        N, C, H, W_half, _ = half_spec.shape
        W = (W_half - 1) * 2
        
        full_spec_np = np.zeros((N, C, H, W, 2), dtype='float32')
        half_spec_np = half_spec.numpy()
        
        full_spec_np[:, :, :, :W_half, :] = half_spec_np

        # --- 最终优化: 鲁棒的混合向量化 ---
        # h=0 的情况 (无垂直翻转)
        h0_slice = half_spec_np[:, :, 0, 1:W_half-1, :]
        h0_slice_conj = h0_slice.copy()
        h0_slice_conj[..., 1] *= -1
        full_spec_np[:, :, 0, W_half:, :] = np.flip(h0_slice_conj, axis=3)

        # h>0 的情况 (需要垂直和水平翻转)
        if H > 1:
            h_rest_slice = half_spec_np[:, :, 1:, 1:W_half-1, :]
            h_rest_slice_conj = h_rest_slice.copy()
            h_rest_slice_conj[..., 1] *= -1
            # 同时翻转 h (axis=2) 和 w (axis=3)
            full_spec_np[:, :, 1:, W_half:, :] = np.flip(h_rest_slice_conj, axis=(2, 3))
        # --- 优化结束 ---

        full_spec = jt.array(full_spec_np)

        # IFFT
        x_complex = nn._fft2(full_spec.reshape(-1, H, W, 2), inverse=True)
        # 返回实部
        return x_complex.reshape(N, C, H, W, 2)[..., 0]

    def grad(self, grad_output):
        """
        反向传播：IFFT的梯度是FFT
        """
        # 1. 准备FFT的输入
        N, C, H, W = grad_output.shape
        grad_output_imag = jt.zeros_like(grad_output)
        grad_complex = jt.stack([grad_output, grad_output_imag], dim=-1)

        # 2. 调用FFT
        fft_full = nn._fft2(grad_complex.reshape(-1, H, W, 2), inverse=False)

        # 3. 截取半谱，以匹配前向输入的形状
        W_half = W // 2 + 1
        grad_half_spec = fft_full.reshape(N, C, H, W, 2)[:, :, :, :W_half, :]

        # 返回的梯度需要与前向输入的形状完全一致
        return grad_half_spec

def irfftn(half_spec: jt.Var) -> jt.Var:
    return IRFFTN.apply(half_spec)