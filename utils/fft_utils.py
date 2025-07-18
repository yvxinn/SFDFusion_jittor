"""
本脚本定义了一个自定义的Jittor算子 `IRFFTN`，用以实现
逆快速傅里叶变换（IFFT），特别是针对厄米共轭对称的半谱进行操作，
这在处理实数信号的FFT时非常常见。

背景：
由于Jittor的旧版本缺少官方的`fft`模块，我们必须手动实现该功能。
一个简单的Python函数实现虽然在前向传播时数值正确，但在反向传播（梯度计算）时
会因为计算图过于复杂而导致显存溢出和性能低下。

解决方案：
通过继承 `jt.Function`，我们创建了一个自定义的底层算子。这允许我们：
1. **定义前向传播 (`execute`)**: 使用我们自己的逻辑（这里是经过优化的NumPy实现）
   来确保计算结果的正确性。
2. **定义反向传播 (`grad`)**: 手动指定一个高效且正确的梯度计算方法。根据傅里叶
   变换的性质，IFFT的梯度就是FFT。

这种方式将复杂的前向逻辑和其梯度计算解耦，完美地解决了我们遇到的所有问题。
"""
import jittor as jt
from jittor import nn, Function
import numpy as np

class IRFFTN(Function):
    """
    一个自定义的Jittor算子，用于执行逆实数快速傅里叶变换（IRFFT）。
    """
    def execute(self, half_spec: jt.Var):
        """
        前向传播方法。
        从一个半谱重建一个完整的厄米共轭对称的全谱，然后执行IFFT。
        
        Jittor特性：
        为了性能，我们将计算密集且难以在Jittor中高效向量化的频谱构建部分，
        转移到NumPy中完成。这利用了NumPy高度优化的C语言后端，并避免了在
        Jittor计算图中创建大量中间变量。
        """
        N, C, H, W_half, _ = half_spec.shape
        W = (W_half - 1) * 2
        
        # 1. 在NumPy中构建完整的厄米共轭对称频谱
        full_spec_np = np.zeros((N, C, H, W, 2), dtype='float32')
        # Jittor -> NumPy: .numpy() 会触发一次GPU到CPU的数据同步
        half_spec_np = half_spec.numpy()
        
        full_spec_np[:, :, :, :W_half, :] = half_spec_np

        # 2. 使用混合向量化技术高效地填充共轭部分
        # a. h=0 的情况 (只进行水平翻转)
        h0_slice = half_spec_np[:, :, 0, 1:W_half-1, :]
        h0_slice_conj = h0_slice.copy()
        h0_slice_conj[..., 1] *= -1 # 取虚部的相反数
        full_spec_np[:, :, 0, W_half:, :] = np.flip(h0_slice_conj, axis=3)

        # b. h>0 的情况 (需要同时进行垂直和水平翻转)
        if H > 1:
            h_rest_slice = half_spec_np[:, :, 1:, 1:W_half-1, :]
            h_rest_slice_conj = h_rest_slice.copy()
            h_rest_slice_conj[..., 1] *= -1 # 取虚部的相反数
            # 同时翻转 h (axis=2) 和 w (axis=3) 轴
            full_spec_np[:, :, 1:, W_half:, :] = np.flip(h_rest_slice_conj, axis=(2, 3))

        # 3. 将构建好的全谱传回Jittor，并执行IFFT
        # NumPy -> Jittor: jt.array() 会触发一次CPU到GPU的数据同步
        full_spec = jt.array(full_spec_np)
        x_complex = nn._fft2(full_spec.reshape(-1, H, W, 2), inverse=True)
        
        # 4. 返回IFFT结果的实部
        return x_complex.reshape(N, C, H, W, 2)[..., 0]

    def grad(self, grad_output: jt.Var):
        """
        反向传播（梯度计算）方法。
        根据数学原理，IRFFT算子的梯度就是FFT算子。
        因此，我们直接对输出的梯度执行一次前向FFT即可。
        """
        # 1. 准备FFT的输入（一个实数张量）
        N, C, H, W = grad_output.shape
        # FFT需要复数输入，因此创建一个全零的虚部
        grad_output_imag = jt.zeros_like(grad_output)
        grad_complex = jt.stack([grad_output, grad_output_imag], dim=-1)

        # 2. 调用Jittor底层的FFT实现
        fft_full = nn._fft2(grad_complex.reshape(-1, H, W, 2), inverse=False)

        # 3. 截取半谱，以确保返回的梯度形状与前向输入的形状完全一致
        W_half = W // 2 + 1
        grad_half_spec = fft_full.reshape(N, C, H, W, 2)[:, :, :, :W_half, :]

        return grad_half_spec

def irfftn(half_spec: jt.Var) -> jt.Var:
    """
    一个用户友好的封装函数，用于调用我们自定义的IRFFTN算子。

    Args:
        half_spec (jt.Var): 形状为(N,C,H,W_half,2)的半谱张量。

    Returns:
        jt.Var: 形状为(N,C,H,W)的重建后的实数张量。
    """
    return IRFFTN.apply(half_spec)