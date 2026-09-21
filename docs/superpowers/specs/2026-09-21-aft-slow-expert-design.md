# A/F/T 慢专家设计与源码学习说明

状态：用户已确认架构并要求实施；尚未实现。日期：2026-09-21。

## 1. 目标与边界

在当前OpenPI/Tabero上增加独立的动作(A)、触觉(T)、腕部力(F)生成专家。慢层同步生成基础动作chunk及其对应的未来感知预测，供后续执行、预测与观测比较使用。快层、门控、安全控制和实际机器人执行不在本次实现范围。

固定三路预测H=50；力历史包含当前帧共8帧；触觉沿用现有TCN历史契约。保留π0和π0.5，以及next_state和sent_command两类数据训练配置；分别支持全量和LoRA，总计8套。所有新功能显式配置启用，原配置行为不变。不得启动正式训练。

## 2. 方案选择

选择在当前Gemma多专家实现上增加T/F分支：复用JAX/Flax、权重资产和训练器，参考N0-TWAM的独立QKV/FFN与联合注意力思想，不移植其PyTorch/Wan和视频级联mask。

共享主干三头可作为后续消融，不属于首版8套配置的交付；整体移植N0-TWAM会破坏当前权重和训练栈，不采用。

力历史参考TA-VLA的展平、两层MLP和单token编码，但信号是本项目K系6D wrench，不是原论文关节力矩。

## 3. 模块、形状与数据流

B=batch，H=50，dA=1024，dT=512，dF=256。三专家18层；FFN分别4096/2048/1024。沿用共同注意力接口：8个query heads、1个KV head、head_dim=256。以下是设计形状，不是当前代码已实现形状。

| 模块 | 输入 | 输出 | 用途 |
|---|---|---|---|
| VLM | 两路RGB、任务文本；π0.5另含离散state | [B,Nv,2048] | 语义视觉条件 |
| state处理 | [B,7]并沿用模型padding/归一化 | π0动作条件token；π0.5文本前缀state | 保持各backbone预训练路由 |
| 现有触觉TCN | [B,9,198,2]按旧契约转为[B,9,396] | [B,2048] | 已观测触觉历史摘要 |
| 触觉条件投影 | [B,2048] | [B,1,512] | 适配新触觉专家 |
| 力历史MLP | [B,8,6]归一化并展平[B,48] | [B,1,256] | Linear48→512、Swish、Linear512→256 |
| A入口 | [B,H,Da]，保留backbone模型action_dim | [B,H,1024] | 未来带噪动作token，物理有效维度7 |
| T入口 | [B,H,396] | [B,H,512] | 未来带噪shear，不使用VAE |
| F入口 | [B,H,6] | [B,H,256] | 未来带噪wrench |
| 三专家联合层 | A/T/F各自条件与未来tokens，VLM上下文 | 各分支原宽度 | 独立QKV投影、联合注意力、独立输出投影/FFN |
| A输出头 | A未来位置[B,H,1024] | [B,H,Da] | 动作flow，推理还原7D动作 |
| T输出头 | T未来位置[B,H,512] | [B,H,396] | shear flow |
| F输出头 | F未来位置[B,H,256] | [B,H,6] | wrench flow |
| 联合采样器 | 三路噪声、相同去噪时间 | 三路干净预测 | 每轮先计算全部flow，再同时更新 |
| 输出还原 | 模型归一化输出 | actions[B,H,7]、shear[B,H,198,2]、wrench[B,H,6] | 供同步调用方消费 |

TCN第0槽是否为参考网格以及历史槽语义，必须与数据转换器和原权重实际契约核对，不将9槽武断解释为9个未来/历史时刻。没有校验通过的输入契约不得进入训练。

## 4. 注意力与时间条件

VLM不读取A/T/F；三路未来tokens可读取VLM、干净历史条件和三路未来带噪tokens。干净条件不读取未来带噪tokens，便于保持条件稳定。全部mask还叠加样本、padding有效性，不允许跨样本访问。

未来三个模态使用相同物理slot索引，但注意力内部序列位置和物理slot编码分开，不因串联序号不同误认为不同预测时刻。首版chunk内未来带噪tokens可双向交互；禁止暴露未来干净标签。

π0沿用动作原时间MLP，T/F采用各自时间条件投影；π0.5沿用AdaRMS时间条件并扩展到T/F。不得把π0的时间权重当作π0.5权重加载。

## 5. 数据、标签与归一化

历史force取t-7…t，开头不足重复episode首帧，并维护历史有效信息；禁止跨episode。历史触觉只使用截至t的真实数据。

每个样本明确记录动作数据行offset和未来传感器offset。预期next_state的action[t+k]对应state[t+k+1]，因此T/F目标取t+k+1。sent_command保留发送指令作为动作标签，T/F取对应指令执行后下一观测槽；它不保证机器人已经到达命令目标。

上述offset必须由两套数据meta、转换说明及相邻state/action核验，不能仅凭目录名推断。若数据契约不支持该解释，数据预检失败并报告，不静默改标签。compact跨缺失时间间隔的目标槽在首版屏蔽；若无法识别间隔，则严格时间对齐配置拒绝通过预检。末尾无未来观测的目标独立mask。

未来shear为相对真实参考网格的二维位移，双侧各99点。历史TCN保持原marker输入语义和对应归一化；未来shear另设统计字段，避免破坏TCN权重契约。所有统计只用训练episode计算。

force按6分量固定z-score，历史和未来共用训练统计；保留K系、N/N·m，不按窗口减均值。不声称模型自动完成零偏或重力补偿。

动作沿用已验证SO(3)相对旋转与夹爪米制契约。两数据集使用相同episode划分，默认沿用已有固定验证episode，先核对各自总数和episode对应关系。

## 6. Flow训练目标

沿用本地时间方向：x_tau=tau*noise+(1-tau)*target，监督noise-target；推理从tau=1走向0。A/T/F共用tau，各自独立噪声。

L=L_A+0.1L_T+0.1L_F。各模态分别按有效时间和有效维度平均；A的模型padding维度不计入物理loss。无有效目标的模态样本不制造零分母。单独记录三路loss、有效计数及物理评估指标。

## 7. 初始化与训练配置

π0：主干及动作兼容模块来自Tabero49999，TCN来自同checkpoint。原动作/力联合头的有效维度需按现有映射审计；新F专家不能冒充继承旧力输出头。原有LoRA若保留必须完整加载。

π0.5：主干、动作专家和时间条件来自pi05_base；默认尝试显式选择Tabero TCN来源，只有完整结构与输入契约验证通过才使用。TCN随机初始化提供独立选项。不得将π0的主干LoRA套在π0.5基底上。

T/F专家、force MLP和新adapter显式初始化。新权重载入规则按模块白名单配置，缺失/多余/形状冲突必须报告；不使用全局忽略错误策略。

LoRA配置：已有主干/动作专家采用LoRA；随机新增模块全部训练；TCN训练策略显式配置。全量配置：所有参数可训练。两者都打印分模块总参数、可训练参数、dtype与权重来源。

精度可选全FP32或混合策略：VLM/动作专家大Transformer矩阵及其LoRA为BF16；触觉和力的全部分支参数（含完整专家、TCN、MLP及投影）为FP32；敏感层/输入embedding/动作投影/时间MLP为FP32；梯度和Adam状态FP32。参数存储与共同QKV计算dtype分离配置，联合注意力默认BF16接口、softmax按现有FP32稳定路径，不能宣传为整个感知分支全程FP32计算。新模块必须纳入dtype规则。EMA默认关闭。保留FSDP选项，双卡候选FSDP=2/global batch=2，但不宣称已验证可运行。

八套配置命名按backbone、dataset、tuning组合生成，分别使用独立assets/checkpoint/run目录；W&B按用户既有要求启用。训练步数、学习率和保存间隔均显式可覆盖，不在本次设计中擅自启动训练或覆盖旧run。

## 8. 同步推理接口

策略调用返回三路反归一化预测。调用端负责步号/时间关联，plan_id可作为可选日志字段，不作为模型预测。固定H有效时不强制输出运行mask，但保留训练mask。快层修正后的感知偏差不保证代表异常；本次不实现任何基于预测误差的机器人控制。

## 9. 预计源码落点与学习文档

| 现有位置 | 职责/预计扩展 |
|---|---|
| src/openpi/models/gemma.py | 增加可配置专家结构与四路参数、mask调用支持 |
| src/openpi/models/pi0.py、pi0_config.py | 保留旧路径，新慢层使用独立模块/配置入口避免继续膨胀 |
| src/openpi/models/model.py | 新观测与多目标数据契约，不破坏旧Actions接口 |
| src/openpi/models/tactile_encoder.py | 复用TCN，新增独立力历史编码模块 |
| src/openpi/policies/libero_policy.py中的Tabero输入类、transforms.py | 现有物理字段映射与SO(3)变换；新增独立慢层policy模块处理shear和输出还原 |
| src/openpi/training/data_loader.py | 历史/未来offset与mask，多目标取样 |
| src/openpi/training/weight_loaders.py | 分模块双来源严格加载 |
| src/openpi/training/config.py | 八套配置和可配置精度/初始化/冻结策略 |
| scripts/train.py、评估入口 | 多路loss日志、三路预测评估，旧训练兼容 |

实现时新增docs/aft_slow_expert_code_guide.md，逐模块列出实际文件/符号、输入输出形状、配置、前向/训练/采样路径及设计原因。记录实际修改而不是将本设计误写为已实现。保持docs/tabero_interview_incident_log.md同步。

## 10. 验证与交付边界

先测试再实现：合成episode验证历史边界与未来offset、next_state/sent_command语义、compact与末帧mask；归一化与SO(3)roundtrip；小尺寸模型A/T/F形状、注意力可见性、各新分支梯度、联合采样更新时间；π0/π0.5初始化和LoRA冻结集合；八套配置矩阵；checkpoint保存恢复及策略物理输出；旧模型相关测试回归。

模型/参数测试优先CPU小模型或抽象shape，不下载整套权重或启动GPU训练。真实checkpoint审计和完整GPU优化步在另行授权下执行。测试通过不等于新架构收敛、双卡容量已验证或真机可部署。

## 11. 审阅与执行

本文件由已讨论方案整理，新增明确了mask、严格时间审计、双来源加载和验证边界。按用户指定Superpowers架构流程：先审阅本文件；通过后编写具体实施计划，并选择执行方式，再实施模型代码。
