# A/F/T Slow Expert Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现可训练、可离线推理的A/F/T慢专家，以及八套配置和逐模块源码学习文档。

**Architecture:** 保留旧Pi0路径；新增独立AFT模型/配置，复用Gemma四路联合注意力与TCN。多目标在独立target容器中传递，不把未来观测伪装成推理输入。π0/π0.5保留各自状态和时间条件契约。

**Tech Stack:** Python、JAX、Flax NNX/Linen、Optax、LeRobot v2.1、pytest；沿用现有环境，不升级依赖。

**Spec:** `docs/superpowers/specs/2026-09-21-aft-slow-expert-design.md`

## Global Constraints

- 分支：`做触觉/力联合输入预测的慢专家`；原分支不再修改，不切换worktree。
- 三路H=50；力历史包含当前观测共8帧；触觉保持现有9槽TCN契约。
- π0/π0.5 × next_state/sent_command × full/lora共8套。
- 混合策略仅VLM/A大矩阵及其LoRA BF16；完整T/F分支参数FP32；梯度/Adam状态FP32；EMA关闭。
- 不启动正式训练、不操纵机器人、不上传W&B、不下载大权重；CPU小模型测试允许。
- 用户指定的研究流程图只作为慢层需求；快层门控及控制不实现。
- 每个可独立验证单元提交前，先运行测试及git diff --check；只提交该任务文件。

## Review Focus

- episode首尾与缺失未来观测：不能跨轨迹、不能将padding参与损失（任务1）。
- π0.5离散state与新增未来目标：不能把未来标签送入条件（任务2、3）。
- 两种dtype专家连接：共享attention显式转换，不偷偷把T/F参数存成BF16（任务3、6）。
- 源checkpoint部分存在的LoRA/模块：不得静默随机补齐已有模块（任务5）。
- 旧模型接口与新三路输出：旧训练/推理结果结构不得改变（任务4、7）。

## 执行环境与基线

先确认`/data/yanghaojun/envs/tabero-smoke/bin/python`存在。下文`PY`仅在命令说明中代表该解释器，执行命令使用绝对路径。所有pytest命令加`JAX_PLATFORMS=cpu`，避免占用用户GPU。若现有基线失败，按systematic-debugging检查并报告，不将旧失败误称新回归。

## Task 1: 独立目标契约与episode安全取样

**Files:** 新增`src/openpi/training/aft_data.py`、`src/openpi/training/aft_data_test.py`、`src/openpi/models/aft_types.py`；修改`src/openpi/training/data_loader.py`的新模型分支。

**Interfaces:** `AFTTargets(actions, shear, wrench, action_mask, sensor_mask)`为JAX pytree；`build_episode_windows(rows, anchor, horizon=50, force_history=8, sensor_offset=1, contiguous_edges=None)`返回dict，包含`force_history`、`marker_history`、`targets`。rows为单episode字段数组，边界裁剪只用于安全读取，mask保存真实有效性。

- [ ] 建立失败测试：以10帧单episode、wrench[:,0]=arange(10)构造rows；anchor=2的历史首通道必须为[0,0,0,0,0,0,1,2]，首个未来wrench为3，末尾mask必须False。

```python
assert window['force_history'].shape == (8, 6)
np.testing.assert_array_equal(window['force_history'][:, 0], [0,0,0,0,0,0,1,2])
assert window['targets'].wrench[0, 0] == 3
assert not window['targets'].sensor_mask[-1]
```

- [ ] 运行`JAX_PLATFORMS=cpu /data/yanghaojun/envs/tabero-smoke/bin/python -m pytest src/openpi/training/aft_data_test.py -q`，确认因缺少新接口失败。
- [ ] 实现索引：`past=anchor+np.arange(-7,1)`、`future=anchor+1+np.arange(horizon)`；先算有效mask，再clip；shear使用未来marker最后槽减参考槽。episode身份不一致、非有限有效观测、形状错误直接ValueError。
- [ ] 读取两套meta/conversion与现有转换代码，验证真实标签offset。compact断点若无可恢复原时间信息，严格模式拒绝并报告数据准备阻塞；不能凭compact时间轴构造“全部连续”。
- [ ] 扩展测试至末帧、短episode、缺失间隔、next_state动作验证、sent_command不要求等于下一state；重复运行上述测试通过后提交`feat: add episode-safe AFT target windows`。

## Task 2: 独立变换与统计契约

**Files:** 新增`src/openpi/policies/aft_policy.py`、`aft_policy_test.py`；修改`src/openpi/models/model.py`仅加入默认None的历史观测字段；复用`src/openpi/transforms.py`已有SO(3)变换。

**Interfaces:** `AFTInputs`从任务1字段生成旧图像/state/token字段及`force_history`；`AFTOutputs`接受模型三路预测还原物理数组；`AFTTargets`始终独立于Observation。

- [ ] 先写测试：输入当前marker保持TCN契约；未来shear不出现在Observation；六维force历史与目标使用同一组统计；训练split以外样本不进入统计。

```python
assert 'targets' not in observation.to_dict()
assert output['actions'].shape == (50, 7)
assert output['tactile_shear'].shape == (50, 198, 2)
assert output['wrist_wrench'].shape == (50, 6)
```

- [ ] 运行`JAX_PLATFORMS=cpu /data/yanghaojun/envs/tabero-smoke/bin/python -m pytest src/openpi/policies/aft_policy_test.py -q`确认失败。
- [ ] 按`(x-mean)/max(std,eps)`实现独立shear/wrench统计与逆变换；复用旧action的SO(3)正逆流程，不分量相加；数据集根目录不写入。
- [ ] 验证归一化roundtrip、π0.5state token、全mask模态、非法gripper和非有限输出；测试通过后提交`feat: add AFT transforms and physical output contract`。

## Task 3: 模型与联合注意力

**Files:** 新增`src/openpi/models/aft.py`、`aft_config.py`、`aft_test.py`；必要时修改`gemma.py`与`gemma_test.py`；复用`tactile_encoder.py`。

**Interfaces:** `AFTConfig`继承`Pi0Config`并实现create/inputs_spec/freeze_filter；`AFTModel.compute_loss(..., targets:AFTTargets, return_components=False)`；`sample_predictions(rng, observation, num_steps=10)`返回dict。暴露`make_aft_attention_mask(valid, groups)`与`masked_mean(error, mask)`供直接测试。

- [ ] 写失败测试验证可见性，groups明确为VLM=0、干净条件=1、未来噪声=2；未来跨模态可见、VLM不可读条件/未来、条件不可读未来。

```python
mask = make_aft_attention_mask(valid, groups)
assert not mask[0, 0, -1]
assert mask[0, -1, 0]
assert mask[0, -1, -2]
assert not mask[0, first_condition, first_future]
```

- [ ] 运行`JAX_PLATFORMS=cpu /data/yanghaojun/envs/tabero-smoke/bin/python -m pytest src/openpi/models/aft_test.py -q`确认缺失接口失败。
- [ ] 新建模型四路configs，保留主干参数路径便于加载；A宽1024/T512/F256，TCN2048→512，force48→512→256。每路使用独立输入投影、时间条件和输出头；未来位置由显式slice获取，不从混合总序列任取末H。
- [ ] 新attention入口将Q/K/V统一到配置compute dtype，保留现有稳定softmax路径；参数dtype不在forward中重建。保留旧Gemma默认行为。
- [ ] 小模型测试π0/π0.5两模式：目标H=3用于单元测试、生产默认50；损失有限，全部无效目标返回零贡献，T/F分支至少一个参数梯度非零。若旧TCN固定大宽不适合CPU，使用配置允许的小TCN测试，不初始化3B模型。
- [ ] 运行模型与旧Gemma相关测试通过后提交`feat: add joint action tactile wrench experts`。

## Task 4: 采样与训练器集成

**Files:** 修改`scripts/train.py`、评估入口及`src/openpi/training/data_loader.py`；新增`scripts/aft_training_test.py`。

**Interfaces:** batch第二元素允许旧Actions或AFTTargets；仅AFTModel走三路loss分支。旧模型返回值不变。AFT组件日志键固定`action_loss/tactile_loss/wrench_loss`。

- [ ] 先测试`masked_mean`不稀释有效项；固定flow函数常量时三路都以同一个dt同步更新。

```python
assert masked_mean(jnp.array([1., 9.]), jnp.array([True, False])) == 1.
# 三路不同常量速度，步进结果分别等于初值 + dt*各自速度。
```

- [ ] 运行`JAX_PLATFORMS=cpu /data/yanghaojun/envs/tabero-smoke/bin/python -m pytest scripts/aft_training_test.py -q`确认失败。
- [ ] 损失使用各模态有效元素均值再加权，采样先获取全部v再更新全部x；不要在预测F之前先修改A。eval沿用确定性seed与独立valid masks。
- [ ] 用小模型执行CPU单次梯度、优化状态与保存恢复测试（不是数据集训练），比较恢复前固定seed三路结果；旧训练路径回归通过后提交`feat: integrate AFT loss and synchronous sampling`。

## Task 5: 分模块权重加载

**Files:** 修改`src/openpi/training/weight_loaders.py`；新增`src/openpi/training/aft_weights_test.py`。

**Interfaces:** `AFTWeightLoader(backbone_params_path, tactile_params_path, allowed_new_patterns)`实现WeightLoader；源路径与模块归属写入manifest。

- [ ] 用NumPy微型树测试主干与TCN分别来自指定源，新增参数保留初始化；已有模块shape错误/部分LoRA缺失必须失败。

```python
np.testing.assert_array_equal(loaded['PaliGemma']['test'], backbone['PaliGemma']['test'])
np.testing.assert_array_equal(loaded['tactile_prefix_encoder']['test'], tactile['tactile_prefix_encoder']['test'])
with pytest.raises(ValueError):
    loader.load(reference_with_wrong_shared_shape)
```

- [ ] 运行`JAX_PLATFORMS=cpu /data/yanghaojun/envs/tabero-smoke/bin/python -m pytest src/openpi/training/aft_weights_test.py -q`确认失败；实现按白名单merge，禁止全局missing_regex='.*'掩盖既有模块缺失。
- [ ] π0载入Tabero、π0.5载入pi05_base；TCN可选择Tabero或随机；严格校验新旧动作维度映射和LoRA保留。测试通过后提交`feat: add audited multi-source AFT initialization`。

## Task 6: 八套配置与精度冻结规则

**Files:** 新增`src/openpi/training/aft_configs.py`、`aft_configs_test.py`，修改`config.py`注册工厂（避免模块顶层循环导入）。

**Interfaces:** `make_aft_configs()`生成8个TrainConfig，名称`aft_{pi0|pi05}_{next_state|sent_command}_{full|lora}`；参数存储策略先命中T/F模块FP32，再处理VLM/A规则。

- [ ] 先写矩阵测试与filter测试：所有配置H50、force_history8、验证episode(1,7,17)，两数据集独立路径，混合策略T/F每个参数FP32。

```python
assert len(configs) == 8
assert len({c.name for c in configs}) == 8
assert all(c.model.action_horizon == 50 for c in configs)
# 用微型参数树验证full全可训练，lora保留新T/F主体可训练而非冻结随机权重。
```

- [ ] 运行`JAX_PLATFORMS=cpu /data/yanghaojun/envs/tabero-smoke/bin/python -m pytest src/openpi/training/aft_configs_test.py -q`确认失败；实现所有配置及CLI可覆盖字段。
- [ ] FSDP、dtype、EMA、optimizer显式配置；检查新LayerNorm、时间MLP、LoRA路径。不得把Pi05离散state设置为False来回避预处理。
- [ ] 测试解析与参数filter通过后提交`feat: register eight AFT training configurations`。

## Task 7: 预检、统计与离线推理入口

**Files:** 新增`scripts/prepare_aft.py`、`scripts/eval_aft_offline.py`及对应`*_test.py`；新增或扩展`src/openpi/policies/aft_policy.py`策略包装。

**Interfaces:** prepare接收`--config --output-dir`，只审计数据/生成训练统计与manifest；eval接收`--config --checkpoint --output-dir`，默认只做离线预测，不连接机器人。三路评估按物理单位与horizon位置统计。

- [ ] 用临时目录模拟meta/episode测试：缺少时间契约失败、不覆盖已有输出、训练与验证分离，测试不得使用真实dataset写入。
- [ ] `JAX_PLATFORMS=cpu /data/yanghaojun/envs/tabero-smoke/bin/python -m pytest scripts/prepare_aft_test.py scripts/eval_aft_offline_test.py -q`先失败后实现。
- [ ] prepare保存输入契约/统计哈希；eval保存位置/SO(3)/夹爪、shear与wrench分量和分horizon误差及有效计数。输出不把flow当作物理marker速度。
- [ ] 以小模型fixture校验输出维度、反归一化、checkpoint配置不匹配拒绝；通过后提交`feat: add AFT preparation and offline evaluation tools`。

## Task 8: 学习文档、回归与交付

**Files:** 新增`docs/aft_slow_expert_code_guide.md`、`docs/aft_slow_expert_runbook.md`；更新incident log、本计划和设计的实现状态。

- [ ] 每个真实模块写明文件/类/函数、输入输出shape、参数精度、调用者；分别梳理训练forward、反向和推理去噪循环，解释历史条件与未来target的区别。
- [ ] runbook给出8套配置选择、初始化来源、单行准备/训练/评估命令和日志查看方法；训练命令只展示不执行。单卡/双卡说明batch是全局值，FSDP不等于只设置CUDA_VISIBLE_DEVICES。
- [ ] 执行新增测试、相关旧TCN/数据/权重/SO(3)/配置回归、`git diff --check`；报告测试数量和失败详情。验证前不宣称可训练或显存可用。
- [ ] 使用requesting-code-review进行最终审阅；发现问题先补测试修复；提交`docs: explain AFT modules shapes and training workflow`。
- [ ] 交付实际完成项、未验证项、准确命令及用户可点击源码文档；不合并原分支、不push、不启动正式训练。

## 自检结果与实施交接

### 2026-09-23实施记录

八个任务对应的模块已落地；本计划上面的逐项清单保留原始验收范围，不把未进行的真实数据/大模型验证标为完成。实现清单和张量学习说明见`docs/aft_slow_expert_code_guide.md`，运行及数据阻塞说明见`docs/aft_slow_expert_runbook.md`。

独立审阅发现并修复五项：无效未来标签进入attention、来源权重dtype不匹配、冻结敏感叶子被降BF16、资产来源未校验、同步三路policy缺失。三项新增反例先失败后修复，资产/policy接口测试也验证过RED→GREEN。另按真实conversion修正省略末帧后的长度审计，并明确LoRA冻结预训练视觉模块。

执行调整：复用Gemma原有多路attention入口，不修改其实现；其统一embed_dtype已提供共享QKV精度。无KV缓存优化，首版逐步重算上下文。保存恢复测试使用仓库实际CheckpointManager；沙箱内保存等待，在沙箱外CPU测试通过。不下载大权重、不运行正式训练。未恢复的98个采样间隔定位仍阻塞真实数据预检；严格检查没有被放宽。

规格覆盖：数据/归一化1–2，四路模型3，训练采样4，权重5，8配置/精度6，预检与评估7，学习文档与回归8。新接口集中在AFTTargets、AFTConfig、AFTModel，不改变旧调用默认行为。

用户已选择在本会话直接实施。真实数据契约冲突需要记录并保留严格检查；具体阻塞与已验证范围见上方2026-09-23实施记录及runbook。
