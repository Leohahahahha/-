# A/F/T 慢专家源码学习指南

更新：2026-09-23。当前为新增架构实现；CPU单元/集成验证不代表训练收敛或真机部署验证。原Pi0路径继续保留。

## 从哪里开始读

按 `aft_types.py → aft_data.py → aft_policy.py → aft_config.py → aft.py → train.py` 阅读。这里的文件分别位于 `src/openpi/models`、`src/openpi/training`、`src/openpi/policies` 和 `scripts`。模型入口是 `AFTConfig.create()`，核心是 `AFTModel.flow()`。

## 一次训练的数据流

```text
单episode Parquet + 当前两路RGB + task文本
  ├─ 当前state、历史marker[9,198,2]、历史wrench[8,6] → Observation
  └─ 未来actions[50,7]、shear[50,396]、wrench[50,6]及mask → AFTTargets
       ↓ SO(3)相对动作、训练集统计归一化；actions/state padding到32
同一tau + 各模态独立高斯噪声 → 三路带噪未来序列
       ↓
VLM视觉/文本上下文 + A条件/未来 + T条件/未来 + F条件/未来
       ↓ 每层独立QKV/FFN、联合attention
三路flow → 分别按有效元素均值 → LA + 0.1LT + 0.1LF
```

`Observation`没有未来shear/wrench字段。`AFTTargets`是独立Flax pytree，由loader作为batch第二项传递。原模型仍收到动作数组。

## 模块与张量形状

B为batch，H生产默认50，Nv为视觉/语言token总数；下表为正式配置，单元测试使用小宽度。

| 文件/符号 | 输入 | 输出 | 职责 |
|---|---|---|---|
| models/aft_types.py `AFTTargets` | 三路标签及时间mask | 独立pytree | 与推理条件隔离 |
| training/aft_data.py `build_episode_windows` | 单episode数组、anchor、审计邻接边 | force[8,6]、marker[9,198,2]、targets | 历史补齐、未来offset、跨缺失间隔屏蔽 |
| training/aft_data.py `AFTDataset` | LeRobot当前样本+Parquet | 原始AFT样本 | 两类动作语义校验，RGB仍由LeRobot解码 |
| policies/aft_policy.py `AFTInputs` | 两路RGB、state[7]、marker、force | 图像字典、tactile_prefix[9,396]、force[8,6]及独立标签键 | 检查参考网格、夹爪与有限值 |
| policies/aft_policy.py `NormalizeSensors` | force历史/未来、shear未来 | 同shape归一化数组 | wrench共用6分量统计；shear396分量统计 |
| models/model.py `Observation` | 图像、state、prompt、历史感知 | 模型条件 | 新增默认None的force_history及mask，旧配置兼容 |
| models/aft.py `PaliGemma.img/llm` | RGB、文本；π0.5离散state在prompt中 | [B,Nv,2048] | 保留视觉语言初始化路径 |
| models/tactile_encoder.py 既有TCN | [B,9,396] | [B,2048] | 第0槽参考+8槽历史；完整9槽输入，不先减参考 |
| models/aft.py `tactile_history_proj` | [B,2048] | [B,1,512] | 历史触觉条件适配 |
| models/aft.py `force_history_in/out` | [B,8,6]→[B,48] | [B,1,256] | 48→512→256，Swish；不足历史重复最早有效帧 |
| models/aft.py 旧动作入口/时间MLP | [B,H,32]、tau | [B,H,1024] | 复用Pi0动作参数路径；仅7维参与loss |
| models/aft.py `tactile_expert` | [B,H,396]、tau | token[B,H,512]、flow[B,H,396] | 直接shear空间生成，不是VAE latent |
| models/aft.py `force_expert` | [B,H,6]、tau | token[B,H,256]、flow[B,H,6] | 直接K系wrench生成 |
| models/gemma.py 既有多专家 `Module/Block/Attention` | 四路不同宽度token | 各路保持自身宽度 | 18层，Q8/KV1/head256；三专家FFN4096/2048/1024 |
| models/aft.py `sample_predictions` | Observation、rng、num_steps | actions[B,50,32]、shear[B,50,396]、wrench[B,50,6] | 共享去噪时间，每轮同时更新三路 |
| policies/aft_policy.py `AFTOutputs` | 已还原绝对动作、归一化感知预测 | actions[50,7]、tactile_shear[50,198,2]、wrist_wrench[50,6] | 感知反归一化，不发送机器人指令 |
| policies/aft_policy.py `AFTPolicy/create_aft_policy` | 单次当前观测字典 | 三路物理预测字典 | 同步推理、共享归一化与SO(3)逆变换 |
| training/aft_assets.py `validate_assets` | manifest、统计、adjacency、model/data config | 已验证manifest或异常 | 校验模型、数据来源、split及文件哈希 |

## 四路如何交互

不是三个输出头共用一个Transformer。Gemma的第0路是VLM，第1路动作，第2路触觉，第3路力；各有自己的FFN、normalization和投影。触觉/力历史分别进入自己的专家，不拼进VLM。

注意力组：VLM=0，干净条件=1，未来带噪=2。query可读取组号不大于自己的有效key，因此VLM看不到感知/未来；历史条件看不到未来；三个未来模态互相可见。共同head维度使不同隐藏宽度可做attention。

三个未来序列的RoPE位置使用相同slot索引，不用串联后的不同绝对序号冒充物理时间。π0保留显式state token和动作时间MLP；π0.5保留离散state路由与AdaRMS，T/F同样使用各自时间条件。

## 历史、标签和单位

力历史为t-7…t共8个真实观测槽，没有“参考力帧”。第一个时刻不足历史就重复首帧；跨已知缺口只用最新连续段。mask保留哪些历史槽是真实存在的，MLP首版消费补齐的48维数值。

next_state的actions[t]是state[t+1]；sent_command的actions[t]是发送目标，不保证下一时刻到达。两者未来感知从t+1开始。末尾缺少未来观测时分别屏蔽action/sensor目标；compact缺口后的目标槽不当成严格等间隔监督。

触觉标签是marker最后槽减第0参考槽，再flatten到396。它是**二维位移场**，不是物理时间导数；flow matching预测的“速度场”是去噪时间的导数，不能当作机械运动速度。

力/力矩保持K坐标系，前三维N、后三维N·m，逐分量z-score；不做逐窗口去均值，不自动宣称已去重力/零偏。动作复用`RelativePoseActions/AbsolutePoseActions`：Rrel=Rstate⁻¹Raction，恢复时Raction=RstateRrel，不做rotvec分量相加。

## 精度、训练范围与权重

`training/aft_configs.py`集中生成8个配置。混合策略：VLM/A大矩阵BF16；TCN、T/F完整专家、感知投影/MLP、norm与敏感投影FP32；梯度存储与Adam动量FP32；EMA关闭。共享attention计算dtype由model.dtype控制，FP32参数不等于所有算子全程FP32。

π0从Tabero49999加载，保留已有主干/动作LoRA叶子。π0.5主干从pi05_base加载，TCN单独来自Tabero。`AFTWeightLoader`按来源和白名单加载，形状错误、共享叶子缺失、部分LoRA缺失直接报错。新T/F专家与适配层初始化；LoRA模式下新增主体仍完整训练，不冻结随机主体只训adapter。

π0.5沿用分位数方式归一化state/actions；两种backbone的触觉历史都单独做z-score，保留迁移TCN的输入规则。未来396D shear和历史/未来6D wrench分别使用训练集统计，力历史与目标共享同一组统计。

## 推理与边界

通用`policy_config.create_trained_policy()`遇到AFTConfig会转到专用AFTPolicy，保留checkpoint各叶子的存储精度，不再把所有参数强制BF16。`infer()`返回三个物理数组；调用时提供两路RGB、state、prompt、tactile_marker_motion、force_history和force_history_mask。LoRA模式冻结预训练视觉编码器，LLM/A只更新LoRA及未冻结投影，新T/F模块和TCN完整更新。

`sample_predictions`从三路独立噪声出发，在同一tau计算全部flow，然后同时Euler更新。输出是归一化三路预测；`eval_aft_offline.py`负责物理还原和误差统计。

首版没有快层、门控、控制器或安全限幅，也不保证快层修改动作后预测感知仍对应真实执行。调用方必须按anchor/执行步对齐预测与观测。此代码不是直接真机部署批准。

首版为便于核对，去噪循环重新计算上下文，没有实现VLM KV缓存优化；真实推理延迟尚未测量。TA-VLA/N0-TWAM是已讨论的编码/独立专家联合注意力设计参考，本实现复用本仓库JAX/Gemma，不是复制它们的PyTorch模型。
