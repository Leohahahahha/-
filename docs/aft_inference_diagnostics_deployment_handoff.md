# AFT 推理诊断：交给部署端 AI 助手的接入说明

更新：2026-09-28。代码分支：`部署代码兼容版本`。当前隔离工作区：`/tmp/tabero-vtla-deploy-compatible`。

诊断代码提交：`2e4a9c8`。数据预检加强提交：`17cd525`；三路部署日志提交：`d734786`。原训练工作区保持原分支及用户改动，未合并/推送。仅需撤销新增诊断时应针对`2e4a9c8`做Git revert（先保存当前改动、核对所在分支），不要reset或覆盖正在训练的工作区。

## 1. 范围与必须保留的行为

新增的是推理解释接口，不是训练目标或机器人控制功能。无需重新训练已有的同架构 AFT checkpoint；不增加参数叶子，不改变 checkpoint 参数名。默认不开启诊断，旧 Pi0/13D serving 路径保留。原始未适配 AFT 的部署代码不能直接加载这个新架构：仍需接入 K 系力历史和三路输出契约。

负责部署的 AI 助手应先同步本分支代码，再核对 checkpoint 的精确 config、`assets/aft`、转换文件 SHA256 和任务文本。不要只复制热力图脚本，也不要把 AFT 三路输出重新塞回旧 13D 头。

保留原 RGB、state、marker 历史处理和 `TargetGuard`、限速、超时、watchdog、停止路径。诊断数组不许进入动作执行器，也不能自动变成快层门控阈值。首次只做离线或 shadow；增加超时绕过安全检查不是接入方案。

## 2. 模块与数据流

| 文件/入口 | 用途 | 输入 → 输出 |
|---|---|---|
| `src/openpi/models/gemma.py::Attention/Block/Module` | 可选导出共享 attention 实际使用的概率，按注意力头平均，沿层 scan 堆叠 | 选定 query 索引 → `[L,B,Q,K]` FP32；不开启仍返回原特征/cache |
| `src/openpi/models/aft.py::sample_attention` | 用同一 RNG、同一 Euler 步程回放到指定去噪步 | 归一化 Observation + RNG → attention、key/query 布局及该步 noisy/velocity |
| `src/openpi/policies/aft_diagnostics.py` | 选项、模态汇总、物理差异和安全文件格式 | attention → mass/per-token/counts；两次物理预测 → mm/deg/N/Nm 差异 |
| `src/openpi/policies/aft_policy.py::configure_diagnostics/infer` | 默认关闭的诊断入口；同噪声历史扰动 | 原观测 → 原三路预测 + 可选 `diagnostics` 字典 |
| `scripts/serve_tabero.py` | 仅对 AFT 开启诊断、提供 WebSocket 元数据 | checkpoint + CLI 选项 → 策略服务 |
| `scripts/plot_aft_diagnostics.py` | 无模型/无 GPU/无机器人连接的绘图 | JSON+NPZ bundle → 2 张 attention 图、可选第 3 张扰动图 |

流程：原观测 → 原 encode/归一化 → 原三路采样 → 原物理解码；只有被采样的诊断请求才另外进行 attention 回放和可选历史扰动采样 → 可选记录/响应 → 部署端独立保存/离线绘图。基础动作始终来自原始观测的正常采样，不来自扰动实验。

H=50、10 Hz。默认 `query_stride=1`，三个预测流各选 50 个 future query，因此 Q=150。K 是拼接后的全部 key token 数，包括图像 patch、文本 padding、状态及 A/T/F 历史/未来；无效 key 的 attention 为零。

触觉 TCN 将现有 `[9,198,2]` 历史压成一个条件 token；力 MLP 将 `[8,6]` 历史压成一个条件 token。联合 attention 不能直接解释每个原始历史帧/marker 点。π0.5 没有独立 state token，离散状态在语言序列里，不能把 language 汇总解释成纯任务文本。

## 3. 请求观测契约（与兼容分支一致）

- `image`、`wrist_image`、`state[7]`、`prompt`、`tactile_marker_motion[9,198,2]`：沿用原部署处理。
- `force_history[8,6]`：float32，包含当前帧和此前7帧，10 Hz，顺序 `[Fx,Fy,Fz,Tx,Ty,Tz]`，N/N·m，K坐标系。
- `force_history_mask[8]`：bool，当前帧必须有效，开头重复最早帧补齐并将补齐处标 false。
- 力与状态来自同次 `/getstate` 的 `force`+`torque`，使用机器人 `stamp`。不得拿 O 系 `external_wrench_base` 顶替 K 系力。时间回退、过期或缺失时停止/重置历史。

这些已在兼容分支的 `examples/fr3_deploy/{core,transport,observations,run}.py` 接入；远端部署副本若仍是旧分支，需要移植相应改动。真机来源由用户给出的只读响应/源码说明确认，本任务没有调用机器人。

## 4. 开关与默认值

训练无需添加诊断 flags。仅在推理服务端使用：

| 选项 | 默认 | 解释 |
|---|---|---|
| `--aft-diagnostics` | off | 开启额外 attention 回放与诊断响应 |
| `--aft-diagnostics-every` | 10 | 从第1次请求开始，每 N 次采集一次；每个推理时刻都看设为1 |
| `--aft-diagnostics-denoise-step` | -1 | -1表示最后一轮速度场计算；10步时等价于索引9、flow time=0.1，而非去噪完成后t=0 |
| `--aft-diagnostics-query-stride` | 1 | 按 horizon 索引0,stride,...选 query，不改变实际预测 H |
| `--aft-diagnostics-layers` | 空 | 空选最后一层；`0,8,17`选这些0-based层。π0当前18层 |
| `--aft-diagnostics-ablations` | off | 额外3次三路推理，分别替换触觉历史、力历史、两者 |
| `--aft-diagnostics-dir` | 无 | 可选服务端 JSON+NPZ 落盘；不指定仍返回诊断并打印动作 query 的汇总 |

注意：层选择发生在导出侧，内部仍计算/收集各层选定 query 的 head mean；这不是峰值显存保证。`query_stride` 可降低诊断矩阵大小。首次 JIT 编译和额外回放/3次扰动采样及同步写盘都会增加延迟；默认控制路径不要开启。

工作站存在反斜杠续行引发 Bash 崩溃的问题，命令每行完整执行。以下仅为用户运行的示例，GPU编号须先确认，不要占用训练中的卡：

```bash
cd /tmp/tabero-vtla-deploy-compatible
TABERO_CHECKPOINT=/data/yanghaojun/outputs/checkpoints/aft_pi0_whiteboard_20260922_v2_next_state_full_30k/你的实际实验名/30000
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda PYTHONPATH=src:. /data/yanghaojun/envs/tabero-smoke/bin/python -u scripts/serve_tabero.py --config=aft_pi0_whiteboard_20260922_v2_next_state_full_30k --checkpoint="$TABERO_CHECKPOINT" --conversion=/data/yanghaojun/datasets/whiteboard_20260922_v2_tabero_next_state_filtered/meta/tabero_conversion.json --host=127.0.0.1 --port=8000 --num-denoise-steps=10 --aft-diagnostics --aft-diagnostics-every=1 --aft-diagnostics-layers=0,8,17 --aft-diagnostics-ablations --aft-diagnostics-dir=/data/yanghaojun/outputs/aft-diagnostics/你的新诊断运行名
```

这是本机绑定服务；远端访问按原安全连接方式走SSH隧道，不要求开放公网端口。不要把示例中的 checkpoint 当作已训练出来的文件。

## 5. 响应 schema 和部署端修改位置

正常响应仍包含：`actions[50,7]`、`tactile_shear[50,198,2]`、`wrist_wrench[50,6]`。选中诊断请求多出 `diagnostics`，未选中时没有该键；客户端不能要求每次都有。

`diagnostics.schema == "aft_diagnostics_v1"`。主要字段：

| 字段 | shape/类型 | 含义 |
|---|---|---|
| `attention` | `[Ls,Q,K]` float32 | 选中层、head平均后的概率；不能称为因果融合权重 |
| `layer_indices` | list[int] | 上述层的原始0-based编号 |
| `query_stream_id` | `[Q]` int | 0动作、1触觉、2力 |
| `query_horizon_index` | `[Q]` int | 预测chunk里的0-based槽位 |
| `key_group_id` | `[K]` int | 对应 `key_group_names` |
| `key_group_names` | 10个字符串 | front_rgb/wrist_rgb/other_rgb/language/state/action_future/tactile_history/tactile_future/force_history/force_future |
| `key_valid` | `[K]` bool | 不把遮罩图像或文本padding算入有效token数 |
| `key_camera_id/key_patch_index` | `[K]` int | 图像key的相机编号/patch索引，其余为-1；相机名见 `image_camera_names` |
| `modality_mass` | `[Ls,Q,10]` | 对每组有效key求和；受各组token数影响 |
| `valid_key_counts` | `[10]` | 有效key数量 |
| `modality_per_token` | `[Ls,Q,10]` | mass/count；空组为0。这不是重新softmax后的概率分布 |
| `denoise_step/flow_time/num_denoise_steps` | 数值 | 注意力对应哪轮速度场计算 |
| `rng_key` | `[2]` uint32 | 基础采样、回放和扰动复用的JAX随机key数据 |
| `record_id/inference_index` | 字符串/int | 服务端记录标识及configure后请求编号 |
| `capture_status/capture_error` | 字符串 | 未请求写盘为not_requested，成功为saved，失败为error并附错误；写盘失败保留baseline预测响应 |
| `diagnostics_latency_ms` | float | 额外回放/扰动计算耗时，不包含正常推理、最终压缩写盘及网络 |
| `ablations` | dict | 未开扰动时空字典 |

部署端 AI 具体要改：

1. **`transport.RemotePolicy.infer` 的接收解析**：现有 `Chunk` 只保存三路物理预测，会丢弃额外diagnostics。可以在Chunk末尾增加默认None的可选字段，或把诊断分离到独立事件通道；不改旧构造参数顺序。先验证schema、shape、finite及布局长度，不接受pickle。
2. **`run.control_loop` 的记录位置**：收到响应时关联本地 `plan_id`、观测采集时间、当前力采集时间、请求/响应时间、实际执行槽位、checkpoint/config/hash/seed。这里的 `inference_index` 不是机器人采样序号，也不是plan_id；未来50槽不是已经发生的传感器时刻。
3. **独立记录/绘图**：可调用 `save_bundle(response, directory, safe_record_id, context=...)`。函数会创建目录但拒绝覆盖同名JSON/NPZ。大矩阵压缩/写盘/绘图应移到有界后台队列，队列满时丢诊断并计数，不阻塞控制。模型回放和扰动计算本身仍在服务端同步发生，后台写盘不能消除这部分延迟。
4. **先测试shadow**：开启诊断可能触发现有超时/结果年龄保护，应记录并停止，而不是放宽安全门限。所有快速修正、预测误差门控、接触阈值仍是后续任务。

服务端文件已包含三路预测及诊断。部署端若只想看图，可直接读取服务端bundle，无需立刻改客户端；若要与真实传感器/执行时刻精确对齐，仍需要上述本地context。

Bundle先在同文件系统临时目录完成JSON/NPZ，再以不覆盖的方式发布NPZ和最后的JSON；普通写盘/发布失败会清理本次已发布文件。读取方以JSON存在作为完成信号。进程被强制杀死/主机掉电可能留下孤立NPZ，不能把它当作完整记录；没有承诺文件系统级抗掉电事务。

## 6. 三类图如何画

拿一条完成的bundle共同前缀（去掉.json/.npz），每个bundle用新的图输出目录：

```bash
PYTHONPATH=src:. JAX_PLATFORMS=cpu /data/yanghaojun/envs/tabero-smoke/bin/python scripts/plot_aft_diagnostics.py --bundle-prefix=/data/yanghaojun/outputs/aft-diagnostics/你的诊断运行名/inference_实际时间戳_000000 --output-dir=/data/yanghaojun/outputs/aft-diagnostics/你的诊断运行名/figures_000000
```

- `attention_heatmap.png`：三行面板分别是动作/触觉/力future query；纵轴预测槽，横轴各类key，按选定层和heads平均。图像块是patch序列，不是自动叠加到RGB图上的空间解释图。
- `modality_attention.png`：三预测流分别画总mass与每token平均曲线，并标有效token数。T/F future是带噪预测状态，T/F history才是实际传感器历史条件；两者不能混称“触觉输入权重”。
- `history_sensitivity.png`：只有开启ablations才有。对基础动作的位置mm、SO(3)姿态deg、单指夹爪mm画差异，而不是对真实标签的预测误差。数据中还存有shear RMS、力N范数、力矩N·m范数差异供后续绘图。

历史扰动是在**归一化后的**Observation里把指定历史设为0：代表对应训练统计的均值，不代表原始传感器全零、不代表关闭TCN/专家，也不等于无触觉重新训练模型。保留同一状态、图像、文本、mask、三路future生成结构与随机key。对异常/远分布均值历史的敏感性只能作诊断，不可直接称为因果贡献或触觉性能增益。

## 7. 两卡训练不变

训练config仍是 `aft_pi0_whiteboard_20260922_v2_next_state_full_30k`：2卡FSDP2、global batch2、30k更新、每6k保存；warmup500，AdamW 2e-5→2e-6，混合权重/FP32梯度动量，W&B开启。历史/未来契约与loss不因诊断改变。不要把上面的推理诊断flags传给 `train.py`。

操作顺序：检查空闲GPU → W&B登录 → CPU准备正式assets（已有目录不能直接覆盖）→ 在tmux里启动训练。以下每行独立执行，GPU 0/1仅为示例：

```bash
cd /tmp/tabero-vtla-deploy-compatible
export HF_HOME=/data/yanghaojun/cache/huggingface
export OPENPI_DATA_HOME=/data/yanghaojun/cache/openpi
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv
CUDA_VISIBLE_DEVICES=0,1 JAX_PLATFORMS=cuda /data/yanghaojun/envs/tabero-smoke/bin/python -c 'import jax; d=jax.devices(); print(d); assert len(d)==2 and all(x.platform=="gpu" for x in d)'
/data/yanghaojun/envs/tabero-smoke/bin/wandb login
PYTHONPATH=src:. JAX_PLATFORMS=cpu /data/yanghaojun/envs/tabero-smoke/bin/python -u scripts/prepare_aft.py --config=aft_pi0_whiteboard_20260922_v2_next_state_full_30k --output-dir=/data/yanghaojun/outputs/assets/aft_pi0_whiteboard_20260922_v2_next_state_full_30k/aft
mkdir -p /data/yanghaojun/outputs/logs
tmux new -s aft-v2-30k
```

进入tmux后重新 `cd /tmp/tabero-vtla-deploy-compatible`、设置上述cache环境变量，然后运行：

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0,1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_MEM_FRACTION=0.70 PYTHONPATH=src:. /data/yanghaojun/envs/tabero-smoke/bin/python -u scripts/train.py aft_pi0_whiteboard_20260922_v2_next_state_full_30k --exp-name=whiteboard_v2_aft_full_30k_20260928 --fsdp-devices=2 --batch-size=2 2>&1 | tee /data/yanghaojun/outputs/logs/whiteboard_v2_aft_full_30k_20260928.log
```

Ctrl+B再D脱离tmux；重新连接用 `tmux attach -t aft-v2-30k`。另一终端看日志：

```bash
tail -F /data/yanghaojun/outputs/logs/whiteboard_v2_aft_full_30k_20260928.log
```

本任务不启动训练，不承诺两卡实测峰值/收敛，也不把临时CPU测试的assets替代正式训练资产。代码在隔离worktree，原训练分支未改；切换/合并到部署副本前先保存其本地改动，不要强制覆盖。

## 8. 部署端必须补的验收测试

- 诊断关闭/抽样跳过时三路输出与原接口完全兼容；固定seed的基础动作不受开关影响。
- 诊断schema/维度不匹配、NaN、过大矩阵安全拒绝；错误不许绕过动作保护。
- 本地capture时间、力stamp、plan_id和执行槽位能关联；不能把flow_time当作机器人时间。
- 正常执行只能使用baseline `actions`，不使用ablations里的替代动作。
- 后台诊断队列满/磁盘慢/磁盘满不阻塞控制；观测过期与响应超时仍触发原停止路径。
- 测过真实模型延迟及显存后，再决定在线抽样频率。当前单元测试和绘图验证不是ROS、shadow或真机安全验证。

## 9. 本次验证与已知遗留失败

97项AFT/部署定向CPU测试通过，独立审阅另运行22项通过。Gemma的选定query采集、两backbone回放、参数树/基础预测保留、固定噪声历史扰动、SO(3)差异、bundle拒绝覆盖/故障清理、capture失败保留响应、三图及旧部署接口均有覆盖。E/F lint忽略原有jaxtyping F722误报后通过。

扩大项目CPU回归最初233通过、9失败，不能称全仓绿色。已复现：

- `models/tokenizer_test.py::{test_tokenize,test_fast_tokenizer}`、`transforms_test.py::{test_tokenize_prompt,test_tokenize_no_prompt}`默认缓存路径只读。指向已有OPENPI缓存后除FAST外三项通过；FAST Hugging Face资产未缓存，未下载。
- `policies/tabero_offline_test.py::test_factory_uses_checkpoint_stats_strict_restore_and_absolute_outputs`的两个触觉case及`training/tabero_baseline_test.py::test_no_tactile_data_transform_matches_shared_inputs_and_inverse`旧fixture不满足固定参考网格契约。
- `training/data_loader_test.py::test_with_fake_dataset`和`scripts/train_test.py::test_train[debug]`依赖未注册的debug配置。

后五项在变更前提交d734786的干净归档中也同样失败，非本次诊断回归；未将它们静默改为通过。巨大模型、联网下载、真实数据下载和Orbax roundtrip未纳入本次受限CPU测试。上游JAX/Flax弃用警告仍在。不要据此执行未经验证的真机控制。
