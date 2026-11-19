# vLLM 并行机制详解

> 整理日期：2026-11-19  
> 内容：TP、PP、DP、EP、PCP、DCP、SP、DBO 的基本原理、区别、配置示例与组合关系。  
> 范围说明：本文以 vLLM 官方文档为依据。具体功能是否可用，取决于安装版本、模型结构、硬件平台和执行后端；vLLM 主线与 vllm-ascend 的支持范围需要分别确认。命令主要用于解释参数含义，并非适用于所有模型的通用部署方案。

## 1. 总览：每种机制到底在“分”什么？

理解并行机制时，可以先问三个问题：

1. 分的是模型权重、输入请求，还是同一请求的上下文？
2. 不同设备之间需要交换什么数据？
3. 目标是降低单卡显存、改善延迟，还是提高吞吐量？

| 机制 | 英文全称 | 核心思想 | 主要解决的问题 |
| --- | --- | --- | --- |
| TP：张量并行 | Tensor Parallelism | 切分同一层内部的权重和计算 | 单层计算分摊、模型权重显存 |
| PP：流水线并行 | Pipeline Parallelism | 将不同层放到不同流水线阶段 | 模型容量、跨设备或跨节点部署 |
| DP：数据并行 | Data Parallelism | 不同请求由不同数据并行组处理 | 服务吞吐量、并发请求承载能力 |
| EP：专家并行 | Expert Parallelism | 将 MoE 专家分布到不同设备 | 专家权重显存、专家计算分摊 |
| PCP：预填充上下文并行 | Prefill Context Parallelism | 分担同一长输入的 Prefill 计算 | 长输入的首 Token 延迟 |
| DCP：解码上下文并行 | Decode Context Parallelism | 沿上下文维度分片 KV Cache 并协作完成 Attention | KV Cache 重复、长上下文容量 |
| SP：序列并行 | Sequence Parallelism | 部分算子的激活值沿 Token/序列维度分片 | 激活值重复、部分计算和通信开销 |
| DBO：双批次重叠 | Dual Batch Overlap | 让两个 Micro-batch 的计算与通信重叠 | 隐藏通信等待时间 |

CP（Context Parallelism，上下文并行）是 PCP、DCP 所属的一类方法。DBO 则属于执行优化，不是另一种独立的模型权重切分方式。

**术语纠正：DCP 的全称是 Decode Context Parallelism，不能写成“数据上下文并行”。**

参考：[并行策略概览][1]、[上下文并行][5]、[并行配置][6]。

## 2. TP：张量并行

### 2.1 基本原理

TP 把一个 Transformer Layer 内部的矩阵计算分配到多张设备上。每张设备拥有部分权重，并参与同一层的计算。

以线性层为例，忽略偏置：

$$
Y = XW
$$

其中：

$$
X \in \mathbb{R}^{N \times H_{\mathrm{in}}},
\qquad
W \in \mathbb{R}^{H_{\mathrm{in}} \times H_{\mathrm{out}}}
$$

这里的 $N$ 表示参与当前计算的 Token 数；为了说明方便，把 Batch 与序列维度合并。

**沿输出维度切分权重：**

$$
W = [W_0,\ W_1,\ W_2,\ W_3]
$$

当 TP=4 时，第 $i$ 张设备计算：

$$
Y_i = XW_i
$$

从数学上看，完整结果为：

$$
Y = \operatorname{Concat}(Y_0,Y_1,Y_2,Y_3)
$$

实际实现不一定立即执行 AllGather。如果后续算子能够继续处理分片结果，就可以保留分片布局。

**沿输入维度切分权重：**

把 $X$ 和 $W$ 按匹配的输入维度分别切分，可得到：

$$
Y = \sum_{i=0}^{3} X_iW_i
$$

每张设备先计算局部结果，再通过 AllReduce 或适合下一步布局的归约方式完成合并。列并行与行并行通常配合使用。

### 2.2 设备分工

假设模型有 32 层，TP=4：

| 设备 | 工作内容 |
| --- | --- |
| GPU 0 | 各层中属于 Rank 0 的权重与计算分片 |
| GPU 1 | 各层中属于 Rank 1 的权重与计算分片 |
| GPU 2 | 各层中属于 Rank 2 的权重与计算分片 |
| GPU 3 | 各层中属于 Rank 3 的权重与计算分片 |

四张设备协同计算同一层，然后继续后面的层。

### 2.3 特点与配置

TP 可以分摊权重和矩阵计算，但通常伴随较频繁的层内通信。增加 TP 大小后的收益，要结合计算规模和设备互联速度判断。

```bash
vllm serve /path/to/model \
    --tensor-parallel-size 4
```

这表示使用 4 路张量并行。具体权重如何分片，由模型实现决定，并非所有张量都按完全相同的方式切分。

参考：[Parallelism and Scaling][2]、[Optimization and Tuning][1]。

## 3. PP：流水线并行

### 3.1 基本原理

PP 将模型的不同层分配到不同 Stage。一个 Stage 可以只有一张设备，也可以由一个 TP 组共同执行。

假设模型有 32 层，PP=4、TP=1：

| Stage | 设备 | 示意层分配 |
| --- | --- | --- |
| Stage 0 | GPU 0 | Layer 0–7 |
| Stage 1 | GPU 1 | Layer 8–15 |
| Stage 2 | GPU 2 | Layer 16–23 |
| Stage 3 | GPU 3 | Layer 24–31 |

这是用于理解的均匀分层示例；Embedding、Final Norm、LM Head 等模块还会影响实际负载。

一次前向传播依次经过这些 Stage。相邻 Stage 传递中间激活值，例如 Hidden States；某些模型还需要传递 Residual 等中间张量。

### 3.2 为什么叫“流水线”？

不同批次可以在不同阶段重叠执行。例如，假设每个 Stage 的计算时间相同，并且有四个能够独立进入流水线的微批次 A、B、C、D：

| 时隙 | Stage 0 | Stage 1 | Stage 2 | Stage 3 |
| --- | --- | --- | --- | --- |
| 1 | A | 空闲 | 空闲 | 空闲 |
| 2 | B | A | 空闲 | 空闲 |
| 3 | C | B | A | 空闲 |
| 4 | D | C | B | A |
| 5 | 空闲 | D | C | B |
| 6 | 空闲 | 空闲 | D | C |
| 7 | 空闲 | 空闲 | 空闲 | D |

表中的空闲时间称为流水线气泡（Pipeline Bubble）。

这只是解释原理的理想调度图。实际 vLLM 是否以及如何形成这样的重叠，要看版本、执行器、调度方式和工作负载。

**同一请求的自回归 Decode 仍存在依赖：普通解码中，下一个 Token 的输入依赖前一个 Token 的生成结果。** 因而，不能因为 PP=4 就推断同一请求的四个连续 Token 可以无依赖地同时执行。

### 3.3 配置与收益

```bash
vllm serve /path/to/model \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 2
```

含义是：

- 模型分为 2 个 Stage。
- 每个 Stage 使用 4 路 TP。
- 在不启用额外 DP、PCP 的情况下，共使用 8 张设备。

PP 能扩大模型可部署规模；有足够可重叠工作时，也可能提高吞吐量。它不保证降低单个请求的延迟，Stage 负载不均衡和通信都会影响效果。

参考：[Parallelism and Scaling][2]。

## 4. DP：数据并行

### 4.1 Dense 模型：不同副本处理不同请求

对于通常的 Dense 模型副本部署，DP 可以理解成：准备多份模型，每份处理自己的请求批次。一个副本内部还可以使用 TP 或 PP。

例如，一个模型副本需要 4 张设备：

| DP 大小 | 副本数量 | 每个副本设备数 | 总设备数 |
| --- | --- | --- | --- |
| DP=1 | 1 | 4 | 4 |
| DP=2 | 2 | 4 | 8 |
| DP=4 | 4 | 4 | 16 |

DP=2、TP=4 时，可以把请求 A、B 分配给第一个副本，把请求 C、D 分配给第二个副本。

```bash
vllm serve /path/to/dense-model \
    --tensor-parallel-size 4 \
    --data-parallel-size 2
```

DP 的主要目标是增加请求处理能力。复制模型本身不会降低单个模型副本的权重显存需求。

### 4.2 MoE 模型：不能直接套用“完整独立副本”

vLLM 的 MoE 部署可以让 Attention 按 DP 分开处理不同请求，而专家层在多个 DP Rank 之间共同分布和执行。

因此，对于 MoE + EP：

- 不同 DP 组可以处理不同请求。
- 专家权重可能跨这些 DP 组分布。
- 专家层执行涉及跨组通信与同步。

这里的 DP 组并不等于互相完全独立的完整模型副本。这一点在阅读 DeepSeek 等 MoE 模型的并行实现时尤其重要。

参考：[Data Parallel Deployment][3]、[Expert Parallel Deployment][4]。

## 5. EP：专家并行

### 5.1 基本原理

MoE 层包含 Router 和多个专家网络。对一个 Token，Router 通常选择 Top-K 个专家，汇总这些专家的计算结果。

一种简化表达为：

$$
y_t = \sum_{e \in \mathcal{E}_t} g_{t,e}\, f_e(x_t)
$$

其中：

- $x_t$：第 $t$ 个 Token 的输入。
- $\mathcal{E}_t$：该 Token 被选中的专家集合。
- $g_{t,e}$：对应的路由权重。
- $f_e$：专家 $e$ 的计算。
- $y_t$：加权组合后的输出。

这个式子用于说明被路由专家的聚合，不包含某些模型额外设置的共享专家分支。

EP 把不同专家放到不同设备。例如，有 8 个专家、EP=4：

| 设备 | 示意专家分配 |
| --- | --- |
| GPU 0 | Expert 0、1 |
| GPU 1 | Expert 2、3 |
| GPU 2 | Expert 4、5 |
| GPU 3 | Expert 6、7 |

具体映射可能因负载均衡而不同，并不要求专家编号一定连续。

### 5.2 一次 MoE 计算的过程

1. Router 为 Token 选择专家。
2. Dispatch 将 Token 数据送到对应专家所在的设备。
3. 各设备计算自己的专家。
4. Combine 将结果送回并按路由权重汇总。

常用通信模式是 All-to-All 或针对这种分发、汇合模式优化的实现。性能既受通信影响，也受专家负载是否均衡影响。

### 5.3 vLLM 中的配置及 DP、TP 关系

```bash
vllm serve /path/to/moe-model \
    --tensor-parallel-size 2 \
    --data-parallel-size 4 \
    --enable-expert-parallel
```

在 PP=1、PCP=1 的这个例子中：

$$
N_{\mathrm{devices}} = TP \times DP = 2 \times 4 = 8
$$

官方部署说明中的专家并行规模为：

$$
EP = TP \times DP = 8
$$

| 层类型 | 这个例子中的执行方式 |
| --- | --- |
| Attention | 4 个 DP 组各自处理请求，每组内部 TP=2 |
| MoE Experts | 专家分布在由全部 8 张设备组成的 EP 组中 |

因此，EP 不是在这 8 张设备之外再乘一个独立设备倍数，也不能把这个配置画成四套各自完整独立的专家副本。

参考：[Expert Parallel Deployment][4]。涉及 PCP 的组合还应结合对应版本的 [ParallelConfig][6] 检查。

## 6. CP：上下文并行，包括 PCP 与 DCP

上下文并行处理的是：**同一个请求的上下文很长，怎样让多个设备协作完成相关计算和存储？**

需要先区分两个阶段：

| 阶段 | 典型工作 |
| --- | --- |
| Prefill | 处理输入 Prompt，计算大量新 Token 的表示并建立 KV Cache |
| Decode | 逐步生成新 Token，对较少的新 Query 使用不断增长的历史 KV Cache |

两者的计算形态不同，所以 PCP 与 DCP 的设计目标也不同。

### 6.1 PCP：预填充上下文并行

假设一个请求的输入有 128K Token，使用 4 路 PCP。下面给出概念上的输入分配，区间按左闭右开理解：

| PCP Rank | 负责的新输入 Token |
| --- | --- |
| Rank 0 | [0, 32K) |
| Rank 1 | [32K, 64K) |
| Rank 2 | [64K, 96K) |
| Rank 3 | [96K, 128K) |

各设备可以分担一部分 Query、Key、Value 的投影与 Attention 工作。

但是，**把 Token 切开，不代表 Attention 也能在各块内完全独立完成。** 对于因果 Attention，后面的 Query 仍可能需要访问前面块的 Key、Value。

可采用的算法思路包括：

- 每张设备计算部分 Query，收集所需的 Key、Value。
- 每张设备持有部分 Query、Key、Value，通过分块传输协作计算，例如 Ring Attention 类方法。

上述方法是算法选择，不能据此认定任一 vLLM 安装版本已经支持所有路径。

对应配置项为：

```text
--prefill-context-parallel-size N
```

PCP 主要用于分摊长输入的 Prefill 计算，目标之一是降低 TTFT（Time To First Token，首 Token 延迟）。具体 Token 布局、负载均衡、支持模型及后端需要单独确认。

参考：[Context Parallel Deployment][5]、[ParallelConfig][6]。

### 6.2 DCP：解码上下文并行

Decode 时，当前新 Token 的 Query 需要访问历史 KV Cache。DCP 让上下文 KV Cache 分布在多个设备上，各设备计算局部 Attention，再通过通信完成正确的全局合并。

用四个 KV 分片作概念说明：

| DCP Rank | 本地数据 | 计算任务 |
| --- | --- | --- |
| Rank 0 | 历史 KV 的一部分 | 对本地 KV 计算局部 Attention |
| Rank 1 | 历史 KV 的一部分 | 对本地 KV 计算局部 Attention |
| Rank 2 | 历史 KV 的一部分 | 对本地 KV 计算局部 Attention |
| Rank 3 | 历史 KV 的一部分 | 对本地 KV 计算局部 Attention |

实际 KV 分片可以采用交错布局，不一定是四段连续的 Token 区间。

局部结果不能直接做普通平均：Attention 的 Softmax 归一化依赖全部相关 Key，需要结合局部归一化统计量等信息完成合并。

### 6.3 为什么 TP 之后还需要 DCP？

TP 可以沿 KV Head 维度分片，但 KV Head 数量有限。当 TP 的规模超过可独立分配的 KV Head 数量时，部分 KV Cache 可能在多个 TP Rank 上重复。

DCP 可以进一步沿上下文 Token 维度分片，减少这种重复。这个思路对 MLA 以及部分 GQA 配置尤其有意义。

以下命令用于说明支持该配置的模型如何组合 TP 与 DCP：

```bash
vllm serve /path/to/supported-mla-model \
    --tensor-parallel-size 4 \
    --decode-context-parallel-size 4
```

在 PCP=1、PP=1、DP=1 的情况下，这个例子仍然使用 4 张设备，**不是 4×4=16 张**。DCP 复用已有的设备组；合法大小受到 TP、模型和后端等约束。

增加 DCP 能节省某些配置中的 KV Cache 空间，但会引入通信成本，不保证单 Token 延迟一定下降。

参考：[Context Parallel Deployment][5]、[ParallelConfig][6]。

### 6.4 PCP 与 DCP 对照

| 维度 | PCP | DCP |
| --- | --- | --- |
| 重点阶段 | Prefill | Decode |
| 主要分担对象 | 长输入对应的计算 | 上下文 KV 存储与 Attention 计算 |
| 典型目标 | 降低长输入 TTFT | 减少 KV 重复、提高上下文或批次容量 |
| 是否需要通信 | 需要跨块获取信息 | 需要聚合局部 Attention 结果 |
| 是否独立增加设备规模 | 当前支持的 PCP 配置可以作为额外设备轴 | 不作为额外设备倍数 |
| 主要代价 | 跨块通信、负载均衡 | 聚合通信、布局与调度开销 |

PCP 与 DCP 可以共享上下文切分的思想，但不能把二者当作相同开关。PCP 本身也不意味着 KV Cache 一定按照相同倍数减少。

## 7. SP：序列并行

### 7.1 基本原理

SP 沿 Token 或序列维度分片部分中间激活值。

例如：

$$
X \in \mathbb{R}^{B \times S \times H}
$$

可以沿 $S$ 维度分配不同 Token 的激活值，让某些逐 Token 算子在不同设备上分别执行。

### 7.2 vLLM 中的一个具体例子

vLLM 的编译优化包含 Sequence Parallelism Pass。一个示意性的变换是：

$$
\operatorname{AllReduce}
\rightarrow
\operatorname{RMSNorm}
$$

变为：

$$
\operatorname{ReduceScatter}
\rightarrow
\operatorname{LocalRMSNorm}
\rightarrow
\operatorname{AllGather}
$$

意思是：归约之后先保留 Token 分片，各设备只处理自己的那部分 RMSNorm，再在需要的位置恢复后续计算所需的布局。

相关设置包括：

```text
compilation_config.pass_config.enable_sp
```

它通常与 TP、编译模式、算子融合及硬件条件有关；并非一个对所有模型通用、独立扩展设备数量的 `--sequence-parallel-size` 开关。

### 7.3 SP 与 CP 的区别

| 对比项 | SP | CP |
| --- | --- | --- |
| 关注重点 | 部分算子的激活值与计算布局 | 同一请求的长上下文 Attention 与 KV |
| 典型操作 | 分片后执行 RMSNorm 等逐 Token 算子 | 跨设备协作完成上下文 Attention |
| 是否天然需要跨 Token 信息 | 逐 Token 算子通常不需要 | Attention 需要访问相关上下文 |
| 在 vLLM 中的形态 | 可体现为编译变换和 TP 优化 | PCP、DCP 等上下文并行机制 |

参考：[Fusion torch.compile passes][7]。

## 8. DBO：双批次重叠

### 8.1 基本原理

DBO 将一个 Batch 拆成两个 Micro-batch，在保持依赖正确的前提下，让一个 Micro-batch 的通信与另一个 Micro-batch 的计算重叠。

概念上，一个可重叠时段可能是：

| 执行资源 | 同一时段正在处理的工作 |
| --- | --- |
| 计算资源 | Micro-batch A 的某段计算 |
| 通信资源 | Micro-batch B 的某段数据交换 |

随后两个 Micro-batch 继续交错推进。真实执行顺序由依赖关系和调度实现决定。

DBO 的目标是减少设备等待通信的时间，而不是额外复制模型或引入新的设备维度。

### 8.2 配置及版本区别

相关参数包括：

```text
--enable-dbo
--dbo-decode-token-threshold
--dbo-prefill-token-threshold
```

截至本文核对时，官方不同文档分支的描述有所区别：

| 文档分支 | 主要说明 |
| --- | --- |
| stable | 主要面向 DP>1 + EP 的 MoE 部署，重叠专家 All-to-All 通信与计算，文档要求 DeepEP |
| latest | 还描述了单 DP Rank、启用 TP 且未启用 EP 时的 `all-reduce-overlap` 路径 |

因此，不能无版本限定地认为 `--enable-dbo` 在任意 TP 或 EP 配置中都可直接使用。

官方文档还列出了组合限制，包括 LoRA、PP、PCP 等。对于多 LoRA 推理，应先核对目标版本及 vllm-ascend 对应实现的兼容性。

参考：[DBO stable 文档][8]、[DBO latest 文档][9]。

## 9. 混合并行与设备数量

### 9.1 TP + PP

```bash
vllm serve /path/to/model \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 2
```

在 DP=PCP=1 时：

$$
N_{\mathrm{devices}} = TP \times PP = 4 \times 2 = 8
$$

每个 Stage 内用 TP 分担同一层，Stage 之间用 PP 串联不同层。

### 9.2 Dense 模型的 TP + DP

```bash
vllm serve /path/to/dense-model \
    --tensor-parallel-size 4 \
    --data-parallel-size 2
```

在 PP=PCP=1 时：

$$
N_{\mathrm{devices}} = TP \times DP = 4 \times 2 = 8
$$

两个副本分别接收请求，每个副本内部使用 4 路 TP。

### 9.3 MoE 模型的 TP + DP + EP

```bash
vllm serve /path/to/moe-model \
    --tensor-parallel-size 2 \
    --data-parallel-size 4 \
    --enable-expert-parallel
```

在 PP=PCP=1 时，总设备数为 8；Attention 使用 DP=4、每组 TP=2，专家层使用跨 8 张设备的 EP。

### 9.4 不要把所有缩写都直接相乘

在普通单设备 Worker 部署中，不启用 PCP 时，常见设备数量关系为：

$$
N_{\mathrm{devices}} = DP \times PP \times TP
$$

当前 `ParallelConfig` 的初始化还将 PCP 计入模型执行的进程规模。对于实际支持这种组合的部署，包含 DP 后可以写为：

$$
N_{\mathrm{devices}} = DP \times PP \times TP \times PCP
$$

**不要再机械地乘 EP、DCP、SP、DBO：**

- EP 的组规模由既有并行配置和专家分布方式决定。
- DCP 使用已有设备形成上下文并行组。
- SP 通常在已有 TP 设备间改变激活值布局。
- DBO 改变批次执行的重叠方式。

这些关系用于解释资源规模，不能代替模型、平台和版本的兼容性检查。

参考：[Parallelism and Scaling][2]、[Expert Parallel Deployment][4]、[ParallelConfig][6]。

## 10. 参数速查

| 机制 | 主要参数或设置 | 说明 |
| --- | --- | --- |
| TP | `--tensor-parallel-size N` | 同一层内部的分片规模 |
| PP | `--pipeline-parallel-size N` | 流水线 Stage 数量 |
| DP | `--data-parallel-size N` | 数据并行规模；MoE 与 Dense 的权重布局需区分 |
| EP | `--enable-expert-parallel` | 对 MoE 专家层启用专家并行 |
| PCP | `--prefill-context-parallel-size N` | 预填充上下文并行规模，须确认支持范围 |
| DCP | `--decode-context-parallel-size N` | 解码上下文并行规模，受组结构和模型约束 |
| SP | `compilation_config.pass_config.enable_sp` 等 | 与 TP 和编译优化相关 |
| DBO | `--enable-dbo` | 批次计算、通信重叠优化 |

上述配置入口来自官方 [ParallelConfig][6]、[编译优化说明][7] 和 [DBO 说明][8]；具体组合需要按安装版本确认。

Ray、Multiprocessing 等属于分布式执行后端，负责进程组织和执行；TP、PP、DP 等描述模型或数据如何分配。两类概念应分别理解。参见 [Parallelism and Scaling][2]。

## 11. 学习重点与源码阅读方向

下面是根据各机制依赖关系安排的学习顺序，不代表所有项目的性能优化优先级。

| 顺序 | 机制 | 建议重点 |
| --- | --- | --- |
| 1 | TP | 列并行、行并行、局部矩阵计算、AllReduce |
| 2 | PP | 分层、PP Rank、中间张量、Send/Recv、流水线气泡 |
| 3 | DP | 请求分配、负载均衡、Dense 与 MoE 的差别 |
| 4 | EP | Router、Top-K、Dispatch、专家布局、Combine |
| 5 | DCP | KV Cache 分片、MLA/GQA、局部 Attention 合并 |
| 6 | PCP | 长输入分块、跨块 KV 访问、负载均衡 |
| 7 | SP | RMSNorm 等逐 Token 算子、ReduceScatter/AllGather、编译变换 |
| 8 | DBO | Micro-batch 切分、计算通信重叠、同步和兼容性 |

对于 DeepSeek、vllm-ascend 和多 LoRA 推理，可以围绕以下问题阅读代码：

1. 当前线性层的基座权重和 LoRA 的 A、B 矩阵如何匹配 TP 分片？
2. 一个 Token 被 Router 选中后，会发往哪个 EP Rank？
3. 本地专家编号与全局专家编号如何映射？
4. 当前 KV Cache 是沿 Head 分片、沿上下文分片，还是存在重复？
5. 多 LoRA 元数据如何随 Token 的分发与重排保持一致？
6. 某项重叠优化是否支持当前 LoRA、PP、图执行模式和通信后端？

这些问题是阅读和验证方向，不预先假定所有组合都已受支持。

## 12. 官方参考资料

以下链接的 `stable`、`latest` 内容会随项目更新。部署时应切换到与安装版本对应的文档，并检查目标后端的说明。

1. [Optimization and Tuning][1]
2. [Parallelism and Scaling][2]
3. [Data Parallel Deployment][3]
4. [Expert Parallel Deployment][4]
5. [Context Parallel Deployment][5]
6. [ParallelConfig API][6]
7. [Fusion torch.compile passes][7]
8. [Dual Batch Overlap — stable][8]
9. [Dual Batch Overlap — latest][9]

[1]: https://docs.vllm.ai/en/stable/configuration/optimization/
[2]: https://docs.vllm.ai/en/stable/serving/parallelism_scaling/
[3]: https://docs.vllm.ai/en/stable/serving/data_parallel_deployment/
[4]: https://docs.vllm.ai/en/stable/serving/expert_parallel_deployment/
[5]: https://docs.vllm.ai/en/stable/serving/context_parallel_deployment/
[6]: https://docs.vllm.ai/en/stable/api/vllm/config/parallel/
[7]: https://docs.vllm.ai/en/stable/design/fusions/
[8]: https://docs.vllm.ai/en/stable/design/dbo/
[9]: https://docs.vllm.ai/en/latest/design/dbo/

