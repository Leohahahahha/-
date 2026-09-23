# AFT慢专家：准备、训练和离线评估

更新：2026-09-23。命令供用户执行，实施过程中不启动正式训练。

## 当前先决条件

白板数据conversion报告98个缺失候选间隔。compact导出时间戳不能定位原始断点。必须从原始采集时间/导出映射恢复每条轨迹的邻接布尔数组，保存为独立JSON：`{"0":[true,true,false,...], "1":[...]}`。每个数组长度为该episode帧数减1，false表示这一对原始观测不连续。禁止为了通过检查填写全true。

在恢复断点之前，预检将有意报错；不能把代码通过CPU测试理解成当前数据已可正式训练。保持数据集原目录不变。

## 八套配置

名字为 `aft_{pi0|pi05}_{next_state|sent_command}_{full|lora}`，例如 `aft_pi0_next_state_full`。每套配置独立assets/checkpoints，验证episode固定1、7、17，其余36条训练。H50，力历史8帧；默认20k步，每4k保存，W&B启用。

π0初始化Tabero49999；π0.5初始化pi05_base并单独加载Tabero TCN。当前未下载/实载验证π0.5大权重。不要把π0 checkpoint路径替换成π0.5来源，或反过来。

## 准备命令（每行单独执行，不用反斜杠续行）

```bash
cd /home/yanghaojun/Tabero-VTLA
source /data/yanghaojun/envs/tabero-smoke/bin/activate
export HF_HOME=/data/yanghaojun/cache/huggingface
export OPENPI_DATA_HOME=/data/yanghaojun/cache/openpi
TABERO_CONFIG=aft_pi0_next_state_full
JAX_PLATFORMS=cpu python scripts/prepare_aft.py --config "$TABERO_CONFIG" --output-dir "/data/yanghaojun/outputs/assets/$TABERO_CONFIG/aft" --edges-json /绝对路径/经过审计的邻接数组.json
```

最后一项是需要你替换的路径，不是已存在文件。脚本生成norm_stats.json、adjacency.json、manifest.json，已有输出目录会拒绝覆盖。训练自动从本配置assets目录读取这些文件。统计只读取训练episode；预检检查训练和验证两部分。

## 训练（预检成功及显存验证后才执行）

先进入tmux会话，避免SSH断开导致训练退出。以下两卡global batch2/FSDP2只是候选，尚未做真实A6000完整优化步显存验证。`CUDA_VISIBLE_DEVICES`与`fsdp-devices`需要同时正确；前者不会自动分片。

```bash
tmux new -s aft_training
cd /home/yanghaojun/Tabero-VTLA
source /data/yanghaojun/envs/tabero-smoke/bin/activate
TABERO_CONFIG=aft_pi0_next_state_full
TABERO_RUN=aft_pi0_next_state_full_20k_20260922
mkdir -p /data/yanghaojun/outputs/logs
set -o pipefail
CUDA_VISIBLE_DEVICES=0,1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 python -u scripts/train.py "$TABERO_CONFIG" --exp-name="$TABERO_RUN" --fsdp-devices=2 --batch-size=2 2>&1 | tee "/data/yanghaojun/outputs/logs/$TABERO_RUN.log"
```

请将0,1换成当时空闲且获准使用的GPU编号。W&B账号登录由你完成；本次实现没有上传任何日志。训练日志新增action_loss/tactile_loss/wrench_loss，loss=LA+0.1LT+0.1LF。

退出终端前按Ctrl-b再按d分离tmux。查看：

```bash
tmux attach -t aft_training
tail -F /data/yanghaojun/outputs/logs/aft_pi0_next_state_full_20k_20260922.log
```

完整FP32需要同时将model.dtype改为float32，并在`training/aft_configs.py`把parameter_dtype_policy替换为`ParameterDtypePolicy(name='aft_float32', default_trainable_dtype='float32', gradient_dtype='float32', optimizer_state_dtype='float32', include_frozen=True)`；不能只改计算dtype就称为全FP32。该策略是配置项，不需要改模型forward。

## 离线评估

```bash
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda python scripts/eval_aft_offline.py --config aft_pi0_next_state_full --checkpoint /data/yanghaojun/outputs/checkpoints/aft_pi0_next_state_full/aft_pi0_next_state_full_20k_20260922/20000 --output-dir /data/yanghaojun/outputs/aft_eval_next_state_20k --max-anchors 100 --stride 10
```

输出errors.npz和metrics.json，按horizon报告位置mm、SO(3)角误差度、单指夹爪mm、shear RMSE以及6个wrench分量的绝对误差和有效计数。只读验证集，checkpoint目录必须包含匹配的三专家参数和归一化资产，不能使用旧13D输出模型冒充新架构。脚本不会连接机器人。
