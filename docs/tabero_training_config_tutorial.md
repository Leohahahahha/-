# Tabero-VTLA 训练配置入门：从 Python 配置到 LoRA 与全量微调

> 适用仓库版本：`5c765e6`（2026-09-19）  
> 面向读者：刚开始学习 Python、深度学习和 VLA，希望能读懂并安全修改本项目训练配置。  
> 本章只解释和检查配置，不会启动训练。

## 1. 学完本章应该能回答什么

读完后，你应该能回答：

1. 一条训练命令是怎样找到模型、数据、优化器和保存目录的？
2. LoRA 和全量微调的根本区别在哪里？
3. 为什么全量微调配置的模型名字里仍可能出现 `lora`？
4. `rank=32` 改变了什么，没有改变什么？
5. BF16、FP32、AdamW、FSDP 分别控制什么，它们为什么不是同一个概念？
6. 配置文件写了 12,000 步，而命令行传入 20,000 步时，到底以谁为准？
7. 如何在不启动训练的情况下检查最终生效的配置？

## 2. 先建立整体心智模型

这个项目的一次训练可以拆成四层：

```text
模型层：模型有哪些模块和参数？是否创建 LoRA 参数？
   ↓
数据层：读哪个数据集？输入/标签是什么？如何归一化和变换？
   ↓
优化层：哪些参数可训练？用什么精度、优化器和学习率？
   ↓
运行层：用几张卡？跑多少步？何时评估、保存和上传 W&B？
```

对应到代码：

| 层 | 主要代码 | 负责内容 |
|---|---|---|
| 模型 | `src/openpi/models/pi0_config.py` | Pi0 结构、backbone LoRA、触觉配置 |
| 触觉模型 | `src/openpi/models/tactile_encoder.py` | 触觉 TCN 和触觉 LoRA |
| 数据与实验配置 | `src/openpi/training/config.py` | 数据集、模型配置、冻结规则、精度、训练超参 |
| 优化器 | `src/openpi/training/optimizer.py` | 学习率、AdamW/SGD、梯度裁剪 |
| 训练执行 | `scripts/train.py` | 创建模型、求损失、反向传播、更新、评估、保存 |
| 启动编排 | `scripts/run_tabero_whiteboard_pair_full_ft_2gpu_12k.sh` | GPU、环境变量、命令行覆盖、连续训练两个模型 |

最关键的一句话是：**配置描述训练，`train.py` 执行训练。**

运行入口是：

```python
if __name__ == "__main__":
    main(_config.cli())
```

`_config.cli()` 先根据命令行中的配置名取出一个 `TrainConfig`，再用后续 `--参数=值` 覆盖它，最后把完整配置传给 `main()`。

## 3. 阅读配置所需的 Python 语法

### 3.1 `@dataclasses.dataclass(frozen=True)`

例子：

```python
@dataclasses.dataclass(frozen=True)
class ParameterDtypeRule:
    path_regex: str
    dtype: Literal["bfloat16", "float32"]
```

它相当于自动生成一个保存配置数据的类：

```python
rule = ParameterDtypeRule(".*norm.*", "float32")
print(rule.path_regex)
```

- `path_regex: str` 是类型提示，说明它应该是字符串。
- `Literal[...]` 表示只接受列出的字符串值。
- `frozen=True` 表示对象创建后不能直接赋值修改。

因此以下写法会失败：

```python
rule.dtype = "bfloat16"
```

项目使用不可变配置，是为了避免某处悄悄改掉配置，导致实验难以复现。

### 3.2 `dataclasses.replace`

既然 frozen 对象不能直接修改，就复制一份并替换指定字段：

```python
new_config = dataclasses.replace(old_config, batch_size=2)
```

`old_config` 不变，`new_config.batch_size` 是 2。

本项目大量使用这种“基于一个可靠配置，只改少数字段”的方式。嵌套对象也要嵌套替换：

```python
new_config = dataclasses.replace(
    old_config,
    model=dataclasses.replace(old_config.model, tactile_prefix_lora_rank=32),
)
```

外层替换 `TrainConfig.model`，内层替换 `Pi0Config.tactile_prefix_lora_rank`。

### 3.3 函数参数中的单独 `*`

```python
def make_config(*, name: str, batch_size: int = 4):
    ...
```

`*` 后面的参数必须写名字：

```python
make_config(name="demo", batch_size=2)  # 正确
make_config("demo", 2)                  # 错误
```

训练参数很多，强制写名字能减少把参数顺序传错的风险。

### 3.4 `tuple(...)`、推导式和 `not in`

```python
episodes=tuple(i for i in range(39) if i not in (1, 7, 17))
```

逐步理解：

- `range(39)` 产生 0 到 38；
- `if i not in (1, 7, 17)` 排除三条验证轨迹；
- `tuple(...)` 把结果保存成不可变元组。

所以训练集是 36 条 episode，验证集是 1、7、17。

### 3.5 `**字典` 展开

```python
policy_metadata={
    **base.policy_metadata,
    "optimization_method": "full_parameter_finetuning",
}
```

含义是先复制 `base.policy_metadata` 的所有键值，再增加或覆盖新键。这是配置继承常用写法。

### 3.6 `A | None` 和 `tuple[str, ...]`

```python
ema_decay: float | None = 0.99
tactile_streams: tuple[str, ...] = ()
```

- `float | None`：可以是浮点数，也可以是空值 `None`。
- `tuple[str, ...]`：包含任意数量字符串的元组。
- `()`：空元组。

### 3.7 `@property`

```python
@property
def trainable_filter(self):
    return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))
```

调用时写 `config.trainable_filter`，看起来像字段，实际会执行函数。这里返回“所有参数中，不属于冻结集合的部分”。

## 4. `TrainConfig`：训练实验的总目录

中心类位于 `src/openpi/training/config.py` 的 `TrainConfig`。可以把它理解为一张实验登记表。

### 4.1 四组关键字段

模型和初始化：

```python
model: BaseModelConfig
weight_loader: WeightLoader
freeze_filter: Filter
parameter_dtype_policy: ParameterDtypePolicy | None
```

- `model`：要创建什么结构。
- `weight_loader`：从哪里载入初始权重。
- `freeze_filter`：哪些参数不更新。
- `parameter_dtype_policy`：参数、梯度和优化器状态用什么存储精度。

数据：

```python
data: DataConfigFactory
assets_base_dir: str
```

优化：

```python
lr_schedule: LRScheduleConfig
optimizer: OptimizerConfig
ema_decay: float | None
```

运行：

```python
batch_size: int
num_train_steps: int
eval_interval: int
save_interval: int
wandb_enabled: bool
fsdp_devices: int
```

### 4.2 `freeze_filter` 才是“训练谁”的最终依据

核心代码是：

```python
return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))
```

把它翻译成中文：

```text
可训练参数 = 所有参数 ∩ 非冻结参数
```

因此不要只凭配置名里有没有 `lora` 判断训练范围。可靠判断方法是查看：

1. `freeze_filter`；
2. 启动日志里的 `Trainable parameters: 可训练数 / 总数`；
3. 启动日志里的 `Trainable paths`。

## 5. LoRA 到底做了什么

### 5.1 原理

普通线性层近似写作：

```text
y = xW
```

LoRA 不直接修改大矩阵 `W`，而是加入两个小矩阵：

```text
y = xW + (alpha / rank) · xAB
```

若 `W` 是 `d_in × d_out`：

- 原矩阵参数量：`d_in × d_out`；
- LoRA 参数量：`d_in × rank + rank × d_out`。

当 `rank` 远小于输入和输出维度时，训练参数和优化器状态显著减少。

`rank` 越大，适配容量通常越强，但参数量、显存和计算量也随之增加。它不是“越大效果一定越好”，小数据下也可能更容易过拟合。

### 5.2 本项目有两类 LoRA

#### Backbone LoRA

`Pi0Config` 中：

```python
paligemma_variant="gemma_2b_lora"
action_expert_variant="gemma_300m_lora"
```

包含 `lora` 的 variant 会在 PaliGemma VLM 和 Action Expert 中创建 LoRA 参数。

`Pi0Config.get_freeze_filter()` 的逻辑是：冻结原始 LLM 大权重，但从冻结集合中排除名字含 `lora` 的叶子，于是 LoRA 叶子能训练。

#### 触觉 TCN LoRA

这是为传感器域差异添加的单独适配层：

```python
tactile_prefix_lora_rank=32
tactile_prefix_lora_alpha=32.0
```

`TactileLoRALinear` 保存原来的 `kernel/bias` 路径，并增加 `lora_a/lora_b`。初始化时：

```python
lora_a = small_random
lora_b = zeros
```

因此初始 `xAB=0`，刚载入时不会立即改变预训练模型输出；训练后由 LoRA 学习你自己的触觉传感器分布。

### 5.3 r32 白板 LoRA 配置做了什么

配置注册位置：

```python
_make_tabero_whiteboard_config(
    name="pi0_lora_tabero_whiteboard_next_state_force_tactile_r32_12k",
    ...,
    tactile_lora_rank=32,
    tactile_lora_alpha=32.0,
    batch_size=2,
    num_train_steps=12_000,
)
```

它完成了四件事：

1. 继承 Tabero 已有 backbone LoRA 架构和权重；
2. 给触觉 TCN 添加 rank 32 的 LoRA；
3. 冻结触觉 TCN 原始大部分参数，只训练允许的 LoRA 叶子；
4. 同时预测 7D 动作和 6D wrist wrench。

项目测试记录的 rank-16 触觉 LoRA 是 779,008 个参数、16 个叶子。rank 近似翻倍时，这部分参数量也近似翻倍，但模型其他部分不随触觉 rank 改变。最终数字应以启动日志为准。

## 6. 全量微调做了什么

全量微调工厂是 `_make_tabero_whiteboard_full_finetune_config()`。

### 6.1 先构造共同的数据与模型合同

```python
base = _make_tabero_whiteboard_config(
    ...,
    tactile_lora_rank=0,
    batch_size=2,
    num_train_steps=12_000,
    wait_for_checkpoint_on_save=True,
)
```

- `tactile_lora_rank=0`：不再新增触觉 LoRA，直接训练原 TCN。
- `batch_size=2`：这是全局 batch，不是每卡 batch。
- `wait_for_checkpoint_on_save=True`：每次保存时等待 checkpoint 完整落盘，更稳但会短暂停顿。

### 6.2 严格加载 Tabero 49999

```python
weight_loader=_tabero_local_smoke.weight_loader
```

这个 loader 指向发布的 Tabero 49999 参数，并使用严格结构检查，避免某些权重缺失后被随机初始化而没有被发现。

### 6.3 解冻所有参数

```python
freeze_filter=nnx.Nothing
```

`nnx.Nothing` 表示冻结集合为空，因此：

```text
可训练参数 = 所有参数 - 空集 = 所有参数
```

这才是“全量微调”成立的核心代码。

### 6.4 为什么模型里仍然有 backbone LoRA

Tabero 49999 本身就是在含 backbone LoRA 叶子的结构上保存的。若把模型改成完全不含这些叶子的普通 Gemma，checkpoint 的参数树会与模型结构不一致，严格加载会失败。

所以当前全量微调采用：

```text
保留 Tabero 49999 的原始结构（包含已有 backbone LoRA 叶子）
+ 不新增触觉 LoRA
+ 解冻全部叶子
= 全量微调
```

“结构里存在 LoRA”不等于“只训练 LoRA”。训练范围由 `freeze_filter` 决定。

### 6.5 关闭 EMA

```python
ema_decay=None
```

EMA 会额外保存一份指数滑动平均参数。全量微调时这份副本很大，因此这里关闭，以减少显存/内存和 checkpoint 开销。

## 7. LoRA 与全量微调对比

| 项目 | LoRA 微调 | 当前全量微调 |
|---|---|---|
| 主要更新对象 | 低秩适配参数 | 所有参数叶子 |
| 触觉部分 | 新增 TCN LoRA | 不新增 LoRA，直接更新 TCN |
| 训练参数量 | 少 | 很大 |
| AdamW 状态显存 | 少 | 很大 |
| 数据少时风险 | 容量受限，但较稳 | 更易过拟合或破坏预训练能力 |
| 学习率敏感度 | 相对低 | 更敏感 |
| checkpoint 结构 | 含适配器 | 仍兼容 Tabero 原结构，但所有叶子都更新 |

LoRA、全量微调只是“更新哪些参数”的选择。它们与 BF16/FP32、AdamW/SGD、单卡/FSDP 是不同维度，可以组合。

## 8. 可配置混合精度做了什么

### 8.1 精度策略的数据结构

```python
class ParameterDtypePolicy:
    name: str
    default_trainable_dtype: Literal["bfloat16", "float32"]
    overrides: tuple[ParameterDtypeRule, ...] = ()
    gradient_dtype: Literal["match_parameter", "float32"] = "match_parameter"
    optimizer_state_dtype: Literal["match_parameter", "float32"] = "match_parameter"
```

这使精度成为配置，而不是把 `.astype(...)` 写死在模型代码里。以后卡够用时，可以切换全 FP32 策略，不需要重写模型。

### 8.2 当前 mixed BF16 参数策略

默认将大部分可训练参数存为 BF16，但以下路径规则覆盖为 FP32：

- LayerNorm/RMSNorm 的 `bias/scale`；
- 视觉输入 embedding；
- state/action 投影；
- time MLP；
- action 输出层；
- 触觉 encoder。

然后派生策略：

```python
gradient_dtype="float32"
optimizer_state_dtype="float32"
```

因此当前 `mixed_bf16_adamw` 不是“所有东西 BF16”，而是：

```text
大 Transformer 参数：主要 BF16
敏感/任务专用参数：FP32
标量 loss 归约：FP32
梯度存储与裁剪：FP32
AdamW 一阶、二阶状态：FP32
```

### 8.3 训练循环中真正应用的位置

初始化参数后：

```python
params = apply_parameter_dtype_policy(params, config)
```

反向传播后、进入优化器前：

```python
grads = apply_gradient_dtype_policy(grads, config)
updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
```

loss 和全局范数也用 FP32 累加，以减少 BF16 归约误差。

### 8.4 全 FP32 如何切换

全 FP32 策略是：

```python
ParameterDtypePolicy(
    name="full_float32",
    default_trainable_dtype="float32",
)
```

当前 launcher 的 profile 对应关系：

| `TABERO_FULL_FT_PROFILE` | 参数 | 梯度/Adam 状态 | 优化器 |
|---|---|---|---|
| `mixed_bf16_adamw` | 混合 BF16/FP32 | FP32 | AdamW |
| `mixed_bf16_lowmem_adamw` | 混合 BF16/FP32 | 跟随参数 | AdamW |
| `fp32_adamw` | FP32 | FP32 | AdamW |
| `fp32_sgd` | FP32 | 无动量状态 | Stateless SGD |

## 9. AdamW、SGD、梯度和优化器状态

### 9.1 梯度是什么

梯度表示“参数向哪个方向改变，会让当前 loss 增大得最快”。优化器沿相反方向更新参数。

训练代码：

```python
(loss, aux), grads = nnx.value_and_grad(...)(...)
```

这一步同时得到 loss 和可训练参数的梯度。

### 9.2 AdamW 为什么更占显存

AdamW 对每个可训练参数通常还保存：

- 一阶动量 `m`：梯度的滑动平均；
- 二阶动量 `v`：梯度平方的滑动平均。

如果全量参数、梯度、`m`、`v` 都是 FP32，粗略看就是多份大模型大小的存储，再加激活、临时张量和通信缓冲，因此显存压力很大。

当前 AdamW 默认值：

```python
b1 = 0.9
b2 = 0.95
eps = 1e-8
clip_gradient_norm = 1.0
```

### 9.3 梯度裁剪

```python
clip_gradient_norm=1.0
```

当所有梯度的全局范数超过 1.0 时，整体按比例缩小，避免一次异常 batch 产生过大的更新。它不是让每个梯度元素都截断到 `[-1, 1]`。

### 9.4 Stateless SGD

Stateless SGD 没有一阶/二阶动量张量，显存更省，但更新只依赖当前梯度。它可能比 AdamW 更难优化大型预训练模型。此前加入它的目的首先是解决全量 AdamW 的显存约束，不代表它在当前任务上一定更好。

## 10. 数据配置：两个白板模型哪里不同

共同点：

- 39 条轨迹；
- episode 1、7、17 做验证集；
- 输入包含 RGB、state、触觉 marker motion；
- 标签是 7D action + 6D K 坐标系 wrist wrench；
- 使用正确的 SO(3) 相对旋转，而不是 rotvec 分量直接相减；
- 从数据集 task 字段读取文本提示。

主要差别：

| 配置 | 数据集 | action 标签 |
|---|---|---|
| next-state | `test2_tabero_next_state_compact` | 机械臂下一时刻实际到达状态 |
| sent-command | `test2_tabero_sent_command_compact` | 同步的遥操作绝对发送指令 |

两个数据集必须各自计算训练集 normalization statistics，不能共用 action 统计量，否则比较不公平，也可能造成错误的反归一化。

## 11. 学习率、step、batch 与 epoch

### 11.1 一个 step 是什么

这里一个训练 step 是一次优化器更新：

```python
for step in range(start_step, config.num_train_steps):
    train_state, info = ptrain_step(...)
```

不是“一条数据”，也不是“一个 epoch”。

### 11.2 全局 batch

`TrainConfig.batch_size` 在本项目中是全局 batch。两张卡、`batch_size=2` 时，大致是每张卡各处理 1 个样本，而不是总 batch 4。

代码要求：

```text
batch_size % 可见设备数 == 0
```

所以三张卡配全局 batch 2 会直接报错。

### 11.3 用 step 估算 epoch

若训练集有 `N` 帧，全局 batch 为 `B`，训练 `S` 步：

```text
近似 epoch = S × B / N
```

白板训练集记录为 11,125 帧，若 batch=2、20,000 steps：

```text
20000 × 2 / 11125 ≈ 3.60 个等效 epoch
```

若想达到约 20 个等效 epoch：

```text
20 × 11125 / 2 ≈ 111250 steps
```

但不能因为“20 epoch 常见”就机械增加到这个数。机器人连续帧高度相似，有效独立样本远少于帧数；应先看 validation loss、离线轨迹指标和真机 shadow test。训练更久只保证看数据更多，不保证性能更好。

## 12. 配置覆盖优先级

本项目实际生效顺序是：

```text
dataclass 默认值
  < 共享配置工厂传入的值
  < 具体命名配置传入的值
  < scripts/train.py 后面的命令行参数
```

例如配置里是：

```python
num_train_steps=12_000
```

启动命令中又有：

```bash
--num-train-steps=20000
```

最终是 20,000。

而 `TABERO_TARGET_STEPS` 本身不是 `TrainConfig` 字段。它先被 shell 脚本读取，再变成 `--num-train-steps="$target_steps"`，最后才覆盖 Python 配置。

## 13. FSDP 控制什么

```python
fsdp_devices=2
```

FSDP 负责把模型/训练状态按规则分片到设备上，主要目的是降低单卡存储压力。它不决定：

- 是 LoRA 还是全量微调；
- 是 BF16 还是 FP32；
- 是 AdamW 还是 SGD；
- 总共训练多少步。

这些是互相独立的配置轴。当前两卡全量脚本使用 `--fsdp-devices=2`，并且只允许恰好两张可见 GPU。

## 14. W&B、日志和 checkpoint

### 14.1 W&B

```python
wandb_enabled=True
wandb_log_images=False
```

表示上传标量、配置等信息，但不上传首个 batch 的相机拼图。网络异常是否终止训练还取决于 W&B 客户端行为和运行环境；本项目同时把指标写到本地 `metrics.jsonl`，不能只依赖网页状态判断训练结果。

### 14.2 保存间隔

```python
save_interval=4_000
keep_period=4_000
```

训练 12,000 次更新时，应保留 4,000、8,000、12,000 checkpoint。最终一步即使不整除保存间隔，训练循环也会保存。

### 14.3 本地证据优先级

判断训练是否完成，建议依次检查：

1. 训练进程是否仍在；
2. 本地日志最后的异常或完成信息；
3. checkpoint 的 `params` 是否完整；
4. `metrics.jsonl` 最后 step；
5. W&B 状态。

W&B 的 `running/crashed` 不是本地进程真相的唯一来源。

## 15. 启动脚本做了什么

`scripts/run_tabero_whiteboard_pair_full_ft_2gpu_12k.sh`：

1. 检查恰好暴露两张不同 GPU；
2. 根据 `TABERO_FULL_FT_PROFILE` 选择命名配置；
3. 分别准备两组数据的 normalization assets；
4. 先训练 next-state；
5. 只有前一个命令退出码为 0，才训练 sent-command；
6. 每个模型使用独立 checkpoint、日志和 W&B run；
7. `--resume` 时检查 checkpoint 和 `wandb_id.txt` 后续训。

注意：脚本文件名和部分提示中仍含 `12k`，但 `TABERO_TARGET_STEPS` 可以覆盖步数。真正完成判断已使用 `$target_steps`；某一行提示固定打印 `{4000,8000,12000}`，在目标不是 12,000 时只是提示不准确，不会改变训练逻辑。

## 16. 我们对训练代码做过的关键修改

### 16.1 Tabero 白板 LoRA/全量配置

文件：`src/openpi/training/config.py`

- 增加两组白板数据的共同配置工厂；
- 固定按 episode 的训练/验证划分；
- 加入 7D action + 6D wrench 数据合同；
- 增加触觉 LoRA rank 16/rank 32 配置；
- 增加全量微调配置，明确解冻全部参数；
- 增加全 FP32、mixed BF16、低显存 mixed BF16、stateless SGD profile；
- 把精度策略记录进 metadata，便于复现实验。

### 16.2 触觉 LoRA

文件：`src/openpi/models/tactile_encoder.py` 和 `src/openpi/models/pi0_config.py`

- 为 TCN 线性层增加可选 `TactileLoRALinear`；
- rank 为 0 时保持旧结构；
- 校验 rank/alpha 和触觉 TCN 是否真的启用；
- 让新触觉 LoRA 可以在不破坏发布权重严格加载的前提下初始化。

### 16.3 混合精度和数值安全

文件：`scripts/train.py`、`src/openpi/training/optimizer.py`、`src/openpi/training/config.py`

- 精度由 config 路径规则控制；
- loss、梯度范数和参数范数使用 FP32 归约；
- 可选择把梯度和 Adam 状态保持 FP32；
- 日志打印参数和优化器状态的 dtype 分布；
- 每步检查输入、loss、梯度、更新、候选参数和优化器状态是否有限；
- 非有限更新会被拒绝并保存诊断，而不是继续污染模型。

### 16.4 运行与恢复

文件：`scripts/run_tabero_whiteboard_pair_full_ft_2gpu_12k.sh`

- 串行训练两个数据定义；
- 独立日志、assets、checkpoint、W&B run；
- 支持 profile、目标步数和恢复模式；
- 保存时可等待 checkpoint 原子完成；
- 完成判断使用实际 `target_steps`。

## 17. 不启动训练的安全检查方法

先进入环境：

```bash
cd /home/yanghaojun/Tabero-VTLA
source /data/yanghaojun/envs/tabero-smoke/bin/activate
```

### 17.1 查看命名配置的关键字段

以下是一条完整单行命令，避免该主机的反斜杠续行问题：

```bash
JAX_PLATFORMS=cpu python -c 'from openpi.training import config as c; x=c.get_config("pi0_tabero_whiteboard_next_state_force_full_ft_mixed_bf16_adamw_12k"); print("name=",x.name); print("batch=",x.batch_size,"steps=",x.num_train_steps,"fsdp=",x.fsdp_devices); print("optimizer=",x.optimizer); print("dtype_policy=",x.parameter_dtype_policy); print("freeze_filter=",x.freeze_filter); print("dataset=",x.data.repo_id); print("train_eps=",len(x.data.base_config.episodes),"val_eps=",x.data.base_config.validation_episodes)'
```

它只创建 Python 配置对象，不创建模型、不读取数据、不占 GPU。

### 17.2 查看脚本最终将执行的命令

```bash
CUDA_VISIBLE_DEVICES=0,1 TABERO_FULL_FT_PROFILE=mixed_bf16_adamw TABERO_TARGET_STEPS=20000 bash scripts/run_tabero_whiteboard_pair_full_ft_2gpu_12k.sh --dry-run learning_check
```

`--dry-run` 只打印准备和训练命令，不启动训练。重点检查：

- config 名称；
- `--num-train-steps`；
- `--batch-size`；
- `--fsdp-devices`；
- 学习率；
- W&B 是否开启；
- checkpoint 和 assets 根目录。

### 17.3 查看某个字段来自哪里

```bash
rg -n 'pi0_tabero_whiteboard_next_state_force_full_ft_mixed_bf16_adamw_12k|_make_tabero_whiteboard_full_finetune_config|ParameterDtypePolicy' src/openpi/training/config.py
```

阅读配置时，不要从头通读两千行文件。先搜具体配置名，再追到它调用的工厂。

## 18. 初学者修改配置的安全步骤

每次只改一个实验变量：

1. 写清楚问题，例如“rank 16 与 rank 32 哪个验证更好”；
2. 复制/调用共同配置，避免整段重复；
3. 只改目标字段；
4. 使用全新的 config name 和 exp name；
5. 运行 config 单元测试；
6. 用 CPU 打印配置；
7. 用 launcher `--dry-run` 检查命令行覆盖；
8. 保存 Git diff 和最终 `run_config.txt`；
9. 最后才由你启动训练。

不要同时改 rank、学习率、batch、数据划分和训练步数，否则结果变化时无法知道原因。

## 19. 第一组练习

建议先自己回答，再看下方答案。

1. `paligemma_variant` 含 `lora`，是否必然是 LoRA-only 微调？
2. `freeze_filter=nnx.Nothing` 表示冻结所有参数，还是不冻结参数？
3. 两卡 FSDP、`batch_size=2` 时全局 batch 是多少？
4. 配置写 12,000 steps，CLI 写 20,000，以哪个为准？
5. 把触觉 rank 从 16 改成 32，会不会自动把 backbone LoRA rank 也改成 32？
6. mixed BF16 是否意味着梯度和 Adam 状态一定也是 BF16？

答案：

1. 不必然；还要看 `freeze_filter`，全量微调也可保留 LoRA 结构。
2. 不冻结任何参数。
3. 2。
4. 20,000，CLI 优先级更高。
5. 不会，它只改变触觉 TCN LoRA。
6. 不一定；当前 `mixed_bf16_adamw` 明确将梯度和 Adam 状态设为 FP32。

## 20. 面试时如何简洁说明

可以这样说：

> 我把训练配置分成模型结构、数据合同、可训练参数范围、优化精度和运行编排五部分。LoRA 实验保留预训练主体，通过路径过滤只更新 backbone 和触觉适配器；全量实验为兼容 Tabero 49999 仍保留 checkpoint 中已有的 backbone LoRA 叶子，但把 freeze filter 设为空，从而更新所有参数，并直接微调触觉 TCN。为了让两张 A6000 上的全量 AdamW 更可行，我实现了配置化混合精度：大 Transformer 参数主要用 BF16，归一化、输入/输出投影和触觉编码器保留 FP32，同时 loss、梯度归约和 Adam 状态使用 FP32。实验参数可由命名配置和 CLI 分层覆盖，最终通过启动日志、run_config 和 dtype/可训练参数统计审计。

## 21. 复习卡片

- 模型结构决定“有哪些参数”，freeze filter 决定“训练哪些参数”。
- LoRA 的核心是冻结大矩阵，学习低秩增量 `AB`。
- rank 控制适配容量和参数量，不直接控制学习率或训练步数。
- 全量微调的核心证据是所有参数叶子可训练，而不是配置名称。
- BF16/FP32 是数值存储策略；AdamW/SGD 是更新算法；FSDP 是多卡分片策略。
- `batch_size` 是全局 batch，必须能被可见设备数整除。
- step 是优化器更新次数；epoch 只能由数据量和 batch 近似换算。
- CLI 覆盖命名配置；判断实际实验必须查看最终 `run_config.txt` 和日志。
- 数据划分必须按 episode，normalization statistics 只能用训练集计算。
- W&B 是观测渠道，不是判断本地训练是否仍运行的唯一真相。

