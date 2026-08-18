# Research Workflow

本模板使用 **PyTorch + Lightning + Hydra** 组织科研代码，并推荐使用 **SwanLab** 记录实验指标与训练曲线。

核心工作流：

```text
DataModule
    ↓
data config

PyTorch Model
    ↓
LightningModule
    ↓
model config

data + model + trainer + logger + hyperparameters
    ↓
experiment config
    ↓
train.py
```

目录职责：

- `src/data/`：数据读取与 DataLoader
- `src/models/components/`：纯 PyTorch 模型
- `src/models/`：LightningModule，负责训练逻辑
- `configs/data/`：数据配置
- `configs/model/`：模型、优化器等配置
- `configs/logger/`：实验记录器配置
- `configs/experiment/`：完整实验配置
- `src/train.py`：统一训练入口，通常不需要修改

---

## 1. 创建 DataModule

创建：

```text
src/data/<dataset>_datamodule.py
```

DataModule 一般负责：

```python
class MyDataModule(LightningDataModule):
    def __init__(self, ...):
        ...

    def prepare_data(self):
        ...

    def setup(self, stage=None):
        ...

    def train_dataloader(self):
        ...

    def val_dataloader(self):
        ...

    def test_dataloader(self):
        ...
```

先单独确认 Dataset、shape 和 DataLoader 正常，再继续后面的训练流程。

---

## 2. 创建 Data Config

创建：

```text
configs/data/<dataset>.yaml
```

例如：

```yaml
_target_: src.data.cifar10_datamodule.CIFAR10DataModule

data_dir: ${paths.data_dir}
batch_size: 128
num_workers: 4
pin_memory: true
```

Hydra 会通过：

```python
datamodule = hydra.utils.instantiate(cfg.data)
```

自动创建对应的 DataModule。

命令行可以临时覆盖参数：

```bash
python src/train.py data=cifar10 data.batch_size=256
```

---

## 3. 创建 PyTorch Model

纯模型放在：

```text
src/models/components/
```

例如：

```python
class MyModel(torch.nn.Module):
    def __init__(self, ...):
        ...

    def forward(self, x):
        ...
```

这一层只负责网络结构，不负责 optimizer、scheduler、training loop、checkpoint 或 logger。

---

## 4. 创建 LightningModule

创建：

```text
src/models/<task>_module.py
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
    def __init__(self, net, optimizer, scheduler=None, compile=False):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
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
        optimizer = self.hparams.optimizer(params=self.parameters())
        ...
```

训练、验证和测试共享的 forward/loss 逻辑建议放在 `model_step()` 中。

---

## 5. 创建 Model Config

创建：

```text
configs/model/<model>.yaml
```

例如：

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

其中 `_partial_: true` 会先创建 optimizer factory，之后由 `configure_optimizers()` 传入模型参数完成实例化。

---

## 6. 先跑 Smoke Test

在写正式 experiment 之前，先组合 data、model 和 trainer 跑 1 个 epoch：

```bash
python src/train.py \
  data=cifar10 \
  model=cifar10_resnet18 \
  trainer=gpu \
  trainer.max_epochs=1
```

目标不是取得好的结果，而是确认以下链路完整工作：

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

只想快速检查少量 batch 时：

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

Hydra 中新增原配置不存在的字段时需要使用 `+`。

---

## 7. 使用 SwanLab 记录实验

模板提供：

```text
configs/logger/swanlab.yaml
```

首次使用先登录：

```bash
swanlab login
```

临时启用 SwanLab：

```bash
python src/train.py \
  data=cifar10 \
  model=cifar10_resnet18 \
  trainer=gpu \
  logger=swanlab
```

LightningModule 中通过 `self.log(...)` 记录的指标会由 `SwanLabLogger` 自动记录，例如：

```text
train/loss
train/acc
val/loss
val/acc
test/loss
test/acc
```

不需要在 `training_step()`、`validation_step()` 中额外调用 `swanlab.log()`。

默认配置：

```yaml
swanlab:
  _target_: swanlab.integration.pytorch_lightning.SwanLabLogger
  project: "research-project"
  experiment_name: null
```

建议在具体科研项目的 experiment config 中覆盖 `project` 和 `experiment_name`，而不是修改模板里的默认值。

---

## 8. 创建 Experiment Config

当 Smoke Test 跑通后，把一次正式实验固化为：

```text
configs/experiment/<experiment>.yaml
```

例如：

```yaml
# @package _global_

defaults:
  - override /data: cifar10
  - override /model: cifar10_resnet18
  - override /trainer: gpu
  - override /logger: swanlab

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

logger:
  swanlab:
    project: "cifar10-research"
    experiment_name: "cifar10_resnet18"
```

运行：

```bash
python src/train.py experiment=cifar10_resnet18
```

这样一个 experiment config 就同时描述了：

```text
data
model
trainer
logger
seed
hyperparameters
experiment metadata
```

并且 SwanLab 会自动记录训练/验证/测试曲线。

---

## 9. 临时覆盖 Experiment 参数

修改 epochs：

```bash
python src/train.py experiment=cifar10_resnet18 trainer.max_epochs=20
```

修改学习率：

```bash
python src/train.py experiment=cifar10_resnet18 model.optimizer.lr=0.0003
```

修改 batch size：

```bash
python src/train.py experiment=cifar10_resnet18 data.batch_size=256
```

关闭 test：

```bash
python src/train.py experiment=cifar10_resnet18 test=false
```

原则：

> 固定、需要复现的参数写进 experiment config；临时调试参数通过 CLI override。

---

## 10. 管理正式科研实验

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

例如：

```bash
python src/train.py experiment=baseline
python src/train.py experiment=ours
python src/train.py experiment=ablation_no_a
```

对应关系尽量保持：

```text
论文实验 / 表格结果
        ↓
experiment config
        ↓
SwanLab run
        ↓
本地 checkpoint + Hydra config
```

这样每个结果都可以追溯到明确的代码、配置和训练记录。

---

## 11. Hydra Multirun

比较三个学习率：

```bash
python src/train.py -m \
  experiment=cifar10_resnet18 \
  model.optimizer.lr=0.0001,0.0003,0.001
```

可以理解为：

```text
固定实验配置
      +
待比较变量
      ↓
多个独立实验
```

正式科研中建议先固定 baseline，再进行参数搜索或 ablation。

---

## Recommended Workflow

```text
[ ] 1. 实现 DataModule
[ ] 2. 创建 data config
[ ] 3. 实现 PyTorch Model
[ ] 4. 实现 LightningModule
[ ] 5. 创建 model config
[ ] 6. 跑 1 epoch / smoke test
[ ] 7. 用 SwanLab 检查训练曲线
[ ] 8. 创建 experiment config
[ ] 9. 正式训练
[ ] 10. 保存结果并提交 experiment config
```

最重要的原则：

```text
data/*.yaml 和 model/*.yaml
描述“组件是什么”

logger/*.yaml
描述“实验如何记录”

experiment/*.yaml
描述“这次完整实验是什么”
```

正常情况下，开发一个新的数据集、模型、logger 或实验时，不应该修改 `src/train.py`。
