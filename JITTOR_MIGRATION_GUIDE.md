# SFDFusion Jittor 迁移指南

## 项目结构

```
SFDFusion_jittor/          # Jittor 版本项目
├── utils/                 # 工具模块 (优先迁移)
│   ├── img_read.py       # ⭐ 第1步：图像读取
│   ├── evaluator.py      # ⭐ 第2步：评估指标
│   ├── loss.py           # ⭐ 第3步：损失函数
│   └── saliency.py       # 第4步：显著性检测
├── configs.py            # ⭐ 第5步：配置管理
├── dataset.py            # ⭐ 第6步：数据集
├── modules.py            # ⭐ 第7步：网络模块
├── train.py              # 第8步：训练脚本
├── val.py                # 第9步：验证脚本
└── fuse.py               # 第10步：推理脚本
```

## 迁移步骤

### 阶段 1：基础工具模块

- [ ] `utils/img_read.py` - 图像读取和转换
- [ ] `utils/evaluator.py` - 评估指标计算
- [ ] `utils/loss.py` - 损失函数

### 阶段 2：数据处理

- [ ] `configs.py` - 配置管理
- [ ] `dataset.py` - 数据集加载

### 阶段 3：网络模块

- [ ] `modules.py` - 核心网络结构

### 阶段 4：训练和推理

- [ ] `train.py` - 训练脚本
- [ ] `val.py` - 验证脚本
- [ ] `fuse.py` - 推理脚本

## 关键替换映射

### 导入替换

```python
# PyTorch → Jittor
import torch → import jittor as jt
import torch.nn as nn → import jittor.nn as nn
import torch.nn.functional as F → import jittor.nn as F
from torch.utils.data import Dataset, DataLoader → from jittor.dataset import Dataset
```

### 方法替换

```python
# PyTorch → Jittor
.forward() → .execute()
torch.cat() → jt.concat()
torch.stack() → jt.stack()
torch.fft.rfftn() → jt.fft.rfftn()
.cuda() → # 自动处理
```

### Kornia 替换

```python
# 需要自实现或替换
kornia.image_to_tensor() → jt.array()
kornia.tensor_to_image() → tensor.numpy()
kornia.losses.SSIMLoss() → 自实现
kornia.metrics.AverageMeter() → 自实现
```

## 验证策略

每个模块迁移后，在 Jupyter 中验证：

```python
# 1. 导入原始和 Jittor 版本
from utils.img_read import img_read as img_read_torch
from utils_jt.img_read import img_read as img_read_jt

# 2. 对比测试
result_torch = img_read_torch("test.jpg")
result_jt = img_read_jt("test.jpg")
assert np.allclose(result_torch.numpy(), result_jt.numpy())
```

## 注意事项

1. **不要混合使用**：每个文件要么全部 PyTorch，要么全部 Jittor
2. **保持接口一致**：函数名和参数保持不变
3. **逐步验证**：每个模块迁移后立即测试
4. **备份原始**：保留原始 PyTorch 版本作为参考

## 开始迁移

建议从 `utils/img_read.py` 开始，这是最基础且相对独立的模块。
