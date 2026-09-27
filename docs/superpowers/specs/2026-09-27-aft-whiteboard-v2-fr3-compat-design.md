# A/F/T 慢专家对白板 v2 数据与 FR3 部署的兼容设计

状态：设计待用户书面审阅；本文不代表代码已实现、模型已训练或真机已验证。日期：2026-09-27。

## 1. 目的、范围与成功标准

从 `做触觉/力联合输入预测的慢专家` 建立独立分支 `部署代码兼容版本`，让现有 π0 A/F/T 慢专家能读取 `/data/yanghaojun/datasets/whiteboard_20260922_v2_tabero_next_state_filtered`，按 `fix/fr3-tactile-shadow-deploy` 白板 v2 30k 全量微调实验的超参数训练，并通过现有 FR3 同步部署客户端完成三路推理接口的 shadow 验证准备。训练命令由用户执行；本任务不启动训练、不向机器人发送命令。

真机 RGB、腕部 RGB、7D 状态、任务文本与 DM-Tac marker 输入继续使用现有部署处理。唯一新增的真机输入是 `/getstate` 同一次采样的 K 系 `force[3] + torque[3]`，组成包含当前帧的 8 帧力历史。`external_wrench_base` 表达于 O 系，禁止替代。慢专家继续输出 `actions[50,7]`、`tactile_shear[50,198,2]`、`wrist_wrench[50,6]`；本次不实现快专家、误差门控或基于预测反馈的控制修正。

成功标准：39 条轨迹通过严格时间/标签预检；新训练配置与既有白板 v2 全量配置的相关超参数逐项一致；服务端和客户端对新模型完成无真机命令的契约测试；旧 Pi0 部署与 AFT 八套旧配置的既有测试保持通过。真实权重加载、GPU 显存、收敛、推理时延、shadow 质量和真机任务成功率在实际运行前均标为未验证。

## 2. 现状与必须修复的阻塞

1. 现有 `aft_data.audit_edges` 假定转换报告的帧数只比保留行数多一个省略的末帧。新 v2 数据先构造 next-state 标签、后筛除共 28 个源帧，还包含 14 个压缩的候选时间步；episode 0 有 349 行，报告 351 帧，当前预检必然拒绝。把 compact 后的行号/时间戳当作原始连续时间会制造错误监督。
2. AFT 现有八套配置指向旧 `test2` 数据，20k 步、4k 保存，不存在新 v2 的 30k 配置。
3. `scripts/serve_tabero.py` 只识别旧 7D 或 13D Pi0 契约，会拒绝 `AFTConfig`。它调用 AFT 的 `data.create()` 时还会检查训练数据源；推理机不应被要求挂载原数据集。
4. FR3 客户端已能发送现有图像、状态与 marker，尚不发送 `force_history[8,6]` 和 `force_history_mask[8]`；元数据校验只认识旧输出布局，并丢弃未来触觉输出。
5. 默认部署文本仍是装配任务。白板模型必须使用训练任务文本 `Pick up the yellow whiteboard eraser and erase the X-shaped mark on the whiteboard.` 和新数据集 conversion 文件，不能沿用旧默认文件。

以上是静态代码与数据契约阻塞，不是训练质量结论。旧 13D checkpoint 的 7:13 物理槽与随机初始化的 T/F 专家可能影响训练稳定性，但尚无这次新模型的实测证据；不把它们伪装成上述确定性部署错误，也不在兼容改动中暗改模型架构。

## 3. 训练数据与配置

### 3.1 严格 episode-local 邻接审计

从每条 `meta/source_mapping/episode_XXXXXX.json` 的保留行建立 `contiguous_edges[N-1]`。仅当下一保留行恰为当前行的 `action_source_row_index` / `action_source_frame_index` 指向的观测、原始帧索引只前进一个 10 Hz 槽、原始时间差在采样容差内、episode 与输出行号一致时，边才为真。任何缺字段、重复/倒退、长度不符或转换报告不一致都应 fail closed；不得以紧凑化后的输出时间戳或全 true 数组补齐未知边。显式外部 adjacency 文件仍可作为审计输入，但必须与 source mapping 和 conversion 的哈希及形状一致。

next-state 的 `actions[t]` 已在筛帧之前由后继观测构造。跨断点的 `action_mask` 和未来触觉/力 `sensor_mask` 均不得监督；后续预测槽不能穿越断点。8 帧力历史也只能来自锚点所在连续片段，片段开头重复首个力值并用布尔 mask 标记无效历史。保留原有 SO(3) 相对姿态验证，只在有效 next-state 边上比较 `actions[t]` 与 `state[t+1]`，不修改动作维度或绝对目标语义。

预检须遍历 39 条轨迹，并汇总原始缺步、筛帧、被屏蔽边和有效 action/sensor 目标数；与 conversion/source mapping 不符则停止。归一化统计只从训练 episode 1–38 计算，episode 0 只作验证。

### 3.2 新配置与资产

新增独立的 π0 A/F/T next-state 全量微调配置，使用 Tabero 49999 作为兼容模块初始化，新 T/F 模块按现有 AFT 加载规则初始化。不要将旧 13D 单头 Pi0 模型配置当成 AFT 配置。资产和 checkpoint 目录与旧 20k、旧 30k 实验分开，固定数据集名、分割、conversion 哈希、邻接哈希、任务文本和模型契约进入 manifest/元数据。

与当前 `fix/fr3-tactile-shadow-deploy` 的白板 v2 30k 配置逐项对齐：train episodes 1–38，val episode 0；global batch 2，FSDP 2 卡，workers 4；全量参数训练，AdamW `b1=0.9,b2=0.95,eps=1e-8,weight_decay=1e-10,clip=1.0`，FP32 梯度与优化器状态、混合 BF16/FP32 参数、无 EMA；warmup 500，余弦峰值 `2e-5`、衰减至 `2e-6`，总更新 30,000；每 1,000 步验证，174 个验证 batch；每 6,000 步保存并保留，启用 W&B，不上传图像，seed 42，等待 checkpoint 保存完成。AFT 专用模型宽度、H=50、`L_A + 0.1L_T + 0.1L_F` 和独立 sensor stats 保持原 AFT 契约，不被旧 13D 配置覆盖。

既有白板 v2 30k Pi0 配置是当前训练工作区中的未提交改动；新分支应以实读到的配置值重建 AFT 专用配置，而不是把用户的未提交文件搬入或混合提交。旧 30k 运行曾在第三次更新出现非有限值，根因未定；相同超参数不保证新 AFT 训练稳定。交付命令前必须清楚标注这项风险及 checkpoint/数值监测方法。

## 4. FR3 观测：只增加力历史

沿用 `RobotHttp.read_state`、`LiveObservations.poll_state` 和 10 Hz `sample_loop`：同一次 `/getstate` 响应解析现有 7D 状态、`stamp.to_sec` 以及 `force[3]`、`torque[3]`。两个三维字段按 `[Fx,Fy,Fz,Tx,Ty,Tz]` 合并为 float32 `[6]`，分别为 N、N·m，必须有限且形状准确。状态与力作为一次原子采样共享机器人采集时间戳；不增加第二个 HTTP 请求，不读取 `external_wrench_base`，不修改相机、裁剪、marker 参考网格或旧模型的输入。

AFT 模式每 100 ms 从已有最新机器人采样追加一个力值，构造 `force_history[8,6]` 和 `force_history_mask[8]`。训练样本的规则是片段开头重复首值，重复位为 false，当前位为 true；部署照此处理。状态过期、时间戳倒退、采样停顿超过现有 150 ms 连续性门槛、缺字段或非有限力值时重置/拒绝样本，绝不复用过期力作为有效历史。力历史与该帧图像/marker 使用现有传感器 age/skew 校验；旧策略无需力字段，行为保持不变。

当前 AFT 编码器虽传递 `force_history_mask`，力 MLP 实际输入为重复填充后的 `[8,6]` 展平；本次保持训练和部署一致，不假称 mask 已作为网络注意力权重使用。

## 5. 服务、输出与部署保护

服务端对 AFT 开一个显式分支：识别 `AFTConfig`，验证 AFT checkpoint 的 `assets/aft` 里 manifest、norm stats、adjacency 与模型/划分/模式契约；比较传入 conversion 文件哈希与 manifest。不要让推理机因没有训练数据根目录而失败，也不要放宽旧 Pi0 的校验。分别验证 7D action、396D shear、6D wrench stats，生成与实际三路独立输出相符的元数据；不得谎称 AFT 仍是旧 13D 拼接头。

客户端保留旧协议路径，新增显式 AFT 元数据分支。AFT 模式要求力历史输入、精确的白板任务文本、H=50、绝对 7D 末端动作、K 系 wrench 与 `[50,198,2]` shear。`RemotePolicy` 对三路输出做 shape/finite 检查，才形成 chunk；动作继续由现有同步执行与 `TargetGuard` 管理，不将预测力/触觉直接变成机器人命令。保存预测 wrench/shear 与观测时间/当前实测 wrench 用于后续离线对比；本次不计算误差门控，也不改变 `--execute` 所需的原使能心跳和硬件停止条件。旧装配 config/conversion 保留，白板运行使用明确传入的新文本配置与新 conversion 文件。

现有请求超时 1 s、结果最大年龄 0.35 s；AFT 采样每步都会重新计算多模态前向，目前没有真卡时延实测。即便接口测试通过，也不得声称可实时真机执行。首次只进行离线和 shadow 时延/契约测试；若不满足现有时效门槛，客户端必须拒绝过期 chunk，不能靠提高门槛掩盖。真实执行需要后续单独确认。

## 6. 测试、记录与 Git 边界

先写失败测试，再做最小实现：合成筛帧/压缩/末帧的邻接与三路 mask 测试；真实 39 条数据只读预检；力字段顺序、单位、时间戳、首帧填充、断流重置和无力/NaN/stale 拒绝测试；AFT/旧 Pi0 元数据、资产哈希、conversion、prompt、三路输出及不发命令的 shadow mock 测试。复跑现有 AFT 与 FR3 测试。基线已在新分支跑通定向 CPU 测试 54 项；那不等于兼容实现已经通过。

更新 `docs/tabero_interview_incident_log.md` 概览及新条目，明确已实现、仅 CPU 验证、训练未运行、shadow/真机未验证。设计、数据/配置、服务/客户端、测试/文档分别提交；只提交本分支相关文件，不采集当前训练工作区的未提交文件。交付提交哈希、差异摘要和回退方式。训练命令单行给出，由用户决定 GPU 和运行时启动；本任务不得启动训练或向机器人发送控制请求。
