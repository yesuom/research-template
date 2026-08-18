# Research Workflow

本模板使用 **PyTorch + Lightning + Hydra** 组织科研代码。推荐遵循下面的工作流开发新的实验。

核心思路：

```text
DataModule
    ↓
data config

PyTorch Model
    ↓
LightningModule
    ↓
model config

data + model + trainer + hyperparameters
    ↓
experiment config

    ↓
train.py
```

其中：

* `src/data/`：数据读取与 DataLoader
* `src/models/components/`：纯 PyTorch 模型
* `src/models/`：LightningModule，负责训练逻辑
* `configs/data/`：数据配置
* `configs/model/`：模型、优化器等配置
* `configs/experiment/`：完整实验配置
* `src/train.py`：统一训练入口，通常不需要修改

---

## 1. 创建 DataModule

首先实现数据模块：

```text
src/data/<dataset>_datamodule.py
```

例如：

```text
src/data/cifar10_datamodule.py
```

DataModule 一般负责：

```python
class MyDataModule(LightningDataModule):

    def __init__(self, ...):
        ...

    def prepare_data(self):
        # 下载或准备数据
        ...

    def setup(self, stage=None):
        # 创建 train / val / test dataset
        ...

    def train_dataloader(self):
        ...

    def val_dataloader(self):
        ...

    def test_dataloader(self):
        ...
```

建议在继续之前先单独测试 DataModule：

```bash
python -c "
from src.data.cifar10_datamodule import CIFAR10DataModule

dm = CIFAR10DataModule()
dm.prepare_data()
dm.setup()

x, y = next(iter(dm.train_dataloader()))

print(x.shape)
print(y.shape)
"
```

---

## 2. 创建 Data Config

创建：

```text
configs/data/<dataset>.yaml
```

例如：

```text
configs/data/cifar10.yaml
```

```yaml
_target_: src.data.cifar10_datamodule.CIFAR10DataModule

data_dir: ${paths.data_dir}

batch_size: 128
num_workers: 4
pin_memory: true
```

Hydra 会根据 `_target_` 自动实例化 DataModule：

```python
datamodule = hydra.utils.instantiate(cfg.data)
```

以后可以通过命令行覆盖配置：

```bash
python src/train.py data=cifar10 data.batch_size=256
```

---

## 3. 创建 PyTorch Model

模型本体放在：

```text
src/models/components/
```

例如：

```text
src/models/components/cifar_resnet.py
```

这一层只负责定义神经网络：

```python
class MyModel(torch.nn.Module):

    def __init__(self, ...):
        ...

    def forward(self, x):
        ...
```

这里尽量不要包含：

```text
optimizer
scheduler
training loop
validation
checkpoint
logger
```

原则：

> `components/` 中的模型只回答“网络结构是什么”。

---

## 4. 创建 LightningModule

创建：

```text
src/models/<task>_module.py
```

例如：

```text
src/models/cifar10_module.py
```

LightningModule 负责：

```text
forward
loss
training_step
validation_step
test_step
metrics
optimizer
scheduler
```

典型结构：

```python
class MyLitModule(LightningModule):

    def __init__(
        self,
        net,
        optimizer,
        scheduler=None,
        compile=False,
    ):
        super().__init__()

        self.save_hyperparameters(
            logger=False,
            ignore=["net"],
        )

        self.net = net

    def forward(self, x):
        return self.net(x)

    def model_step(self, batch):
        ...

    def training_step(self, batch, batch_idx):
        ...

    def validation_step(self, batch, batch_idx):
        ...

    def test_step(self, batch, batch_idx):
        ...

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(
            params=self.parameters()
        )

        ...
```

建议把训练、验证、测试公用的前向和 loss 计算放到：

```python
model_step()
```

中复用。

---

## 5. 创建 Model Config

创建：

```text
configs/model/<model>.yaml
```

例如：

```text
configs/model/cifar10_resnet18.yaml
```

```yaml
_target_: src.models.cifar10_module.CIFAR10LitModule

optimizer:
  _target_: torch.optim.AdamW
  _partial_: true
  lr: 0.001
  weight_decay: 0.0005

scheduler: null

net:
  _target_: src.models.components.cifar_resnet.CIFARResNet18
  num_classes: 10

compile: false
```

对应关系：

```text
model config
     │
     ▼
LightningModule
     │
     ├── PyTorch Model
     │
     └── Optimizer
```

`_partial_: true` 表示 Hydra 暂时创建 optimizer factory，真正的 optimizer 在 `configure_optimizers()` 中通过模型参数实例化。

---

## 6. 先跑一个 Smoke Test

在创建正式 experiment 之前，先直接组合 data、model 和 trainer：

```bash
python src/train.py \
  data=cifar10 \
  model=cifar10_resnet18 \
  trainer=gpu \
  trainer.max_epochs=1
```

这一步的目标不是获得好的实验结果，而是确认完整流程可以运行：

```text
DataModule
    ↓
Model
    ↓
LightningModule
    ↓
Optimizer
    ↓
Training
    ↓
Validation
    ↓
Checkpoint
    ↓
Test
```

如果只想快速检查几个 batch：

```bash
python src/train.py \
  data=cifar10 \
  model=cifar10_resnet18 \
  trainer=gpu \
  trainer.max_epochs=1 \
  +trainer.limit_train_batches=20 \
  +trainer.limit_val_batches=10 \
  test=false
```

Hydra 中，如果参数原本不存在，需要使用 `+` 添加。

---

## 7. 创建 Experiment Config

当基本训练流程跑通以后，再把一次正式实验固化下来。

创建：

```text
configs/experiment/<experiment>.yaml
```

例如：

```text
configs/experiment/cifar10_resnet18.yaml
```

```yaml
# @package _global_

defaults:
  - override /data: cifar10
  - override /model: cifar10_resnet18
  - override /trainer: gpu

tags:
  - cifar10
  - resnet18
  - baseline

seed: 42

trainer:
  min_epochs: 1
  max_epochs: 10

data:
  batch_size: 128
  num_workers: 4

model:
  optimizer:
    lr: 0.001
    weight_decay: 0.0005
```

然后运行：

```bash
python src/train.py experiment=cifar10_resnet18
```

相比：

```bash
python src/train.py \
  data=cifar10 \
  model=cifar10_resnet18 \
  trainer=gpu \
  trainer.max_epochs=10 \
  data.batch_size=128 \
  model.optimizer.lr=0.001
```

experiment config 可以完整记录一次科研实验使用的配置。

---

## 8. 临时覆盖 Experiment 参数

Experiment config 是实验的默认配置，但仍然可以通过命令行临时覆盖。

修改 epochs：

```bash
python src/train.py \
  experiment=cifar10_resnet18 \
  trainer.max_epochs=20
```

修改学习率：

```bash
python src/train.py \
  experiment=cifar10_resnet18 \
  model.optimizer.lr=0.0003
```

修改 batch size：

```bash
python src/train.py \
  experiment=cifar10_resnet18 \
  data.batch_size=256
```

关闭 test：

```bash
python src/train.py \
  experiment=cifar10_resnet18 \
  test=false
```

原则：

> 固定、需要复现的参数写进 experiment config；临时调试参数可以通过 CLI override。

---

## 9. 管理正式科研实验

推荐让论文中的实验与配置文件一一对应：

```text
configs/experiment/
├── baseline.yaml
├── ours.yaml
├── ours_large.yaml
├── ablation_no_a.yaml
├── ablation_no_b.yaml
└── ablation_no_c.yaml
```

运行：

```bash
python src/train.py experiment=baseline

python src/train.py experiment=ours

python src/train.py experiment=ablation_no_a
```

这样论文中的结果可以追溯到明确的配置文件：

```text
Table 2 / Baseline
        ↓
configs/experiment/baseline.yaml
```

不要依赖 shell history 或记忆保存实验超参数。

---

## 10. Hydra Multirun

如果只想比较某一个参数，可以使用 Hydra multirun。

例如比较三个学习率：

```bash
python src/train.py -m \
  experiment=cifar10_resnet18 \
  model.optimizer.lr=0.0001,0.0003,0.001
```

可以把它理解为：

```text
固定实验配置
      +
待比较变量
      ↓
多个独立实验
```

正式科研中，建议优先先写好 baseline experiment，再进行参数搜索或 ablation。

---

## Recommended Workflow

每次开始一个新的任务，可以按照下面的顺序：

```text
[ ] 1. 实现 DataModule
[ ] 2. 创建 data config
[ ] 3. 实现 PyTorch Model
[ ] 4. 实现 LightningModule
[ ] 5. 创建 model config
[ ] 6. 跑 1 epoch / smoke test
[ ] 7. 创建 experiment config
[ ] 8. 正式训练
[ ] 9. 保存结果并提交 experiment config
```

最重要的两个原则：

```text
data/*.yaml 和 model/*.yaml
描述“组件是什么”

experiment/*.yaml
描述“这次实验是什么”
```

以及：

> 正常情况下，开发一个新的数据集、模型或实验时，不应该修改 `src/train.py`。
