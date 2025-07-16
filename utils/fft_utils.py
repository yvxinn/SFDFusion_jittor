import jittor as jt
from jittor import nn

def jittor_irfftn_backward(half_spec: jt.Var) -> jt.Var:
    """
    Jittor 逆 FFT 实现
    
    参数:
    half_spec: 半谱，形状为 (N, C, H, W_half, 2)
    
    返回:
    重建的实数图像，形状为 (N, C, H, W)
    """
    N, C, H, W_half, _ = half_spec.shape
    W = (W_half - 1) * 2
    full_spec = jt.zeros((N, C, H, W, 2), dtype='float32')
    full_spec[:, :, :, :W_half, :] = half_spec

    for h in range(H):
        for w in range(1, W_half - 1):  # 不包括 DC 和 Nyquist
            h_conj = (-h) % H
            w_conj = (-w) % W
            re = half_spec[:, :, h, w, 0]
            im = -half_spec[:, :, h, w, 1]
            full_spec[:, :, h_conj, w_conj, 0] = re
            full_spec[:, :, h_conj, w_conj, 1] = im

    # IFFT
    x = nn._fft2(full_spec.reshape(-1, H, W, 2), inverse=True)
    return x.reshape(N, C, H, W, 2)[..., 0]