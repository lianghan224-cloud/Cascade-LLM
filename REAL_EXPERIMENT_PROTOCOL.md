# Llama-3.1-8B 真实权重实验协议

本文件定义“真实模型实验”的最低证据要求。合成Tensor只用于硬件校准，
不得写成模型推理结果。

## 0. 固定实验对象

- 模型：`meta-llama/Llama-3.1-8B`
- 权重：官方BF16 safetensors
- revision：下载时解析为不可变commit SHA
- 本地目录：`/ssd/cascade-llm/models/Llama-3.1-8B`
- 收据：`real_results/llama31_8b_checkpoint.json`
- 可选镜像：`AI-ModelScope/Meta-Llama-3.1-8B`，必须固定commit并逐文件
  校验SHA-256，且根目录BF16 safetensor大小须与官方仓库清单一致

checkpoint、Token和Hugging Face缓存禁止提交Git。

## 1. 环境验收

```bash
source scripts/activate_env.sh
python scripts/check_real_environment.py
```

必须记录：

- Git commit；
- Python、PyTorch、Transformers、CUDA和Driver版本；
- GPU型号与空闲显存；
- CPU RAM与NVMe剩余空间；
- checkpoint revision和文件SHA-256。

## 2. 权重完整性

```bash
python scripts/download_llama31_8b.py
# 或使用固定revision、逐文件校验的ModelScope镜像：
python scripts/download_llama31_8b_modelscope.py
```

下载后必须验证：

- config与Llama-3.1-8B的维度一致；
- safetensors index引用的分片全部存在；
- 每个Tensor的名称、shape、dtype可枚举；
- Decoder 32层的权重不能用同一个合成slab重复代替；
- `full_pinned`必须真实分配并填充整个checkpoint，而非只重复单层。

## 3. 正确性基线

在报告任何速度前，先产生固定输入的参考结果：

1. 使用官方Transformers实现和真实权重运行CPU参考；
2. 保存input IDs、首Token logits摘要、top-k token及概率；
3. 使用流式运行时运行同一输入；
4. 比较最大绝对/相对误差、top-k重合和贪心Token；
5. 至少覆盖一次Prefill和连续8个Decode Token。

任一层输出或最终Token不一致时，不得进入性能结论。

## 4. 内存模式

分别验证：

- `full_pinned`：完整真实checkpoint驻留Pinned CPU内存；
- `pinned_staging`：完整真实checkpoint驻留pageable RAM，两个Pinned
  staging slot循环复用；
- 词表流式模式下，Embedding和LM Head只存在于CPU权重Arena；
- Embedding只传输当前Token对应的行；
- LM Head按词表行分块，并验证在线Top-k等价于完整logits Top-k；
- GPU常驻区只包含当前计划声明的Norm等小型权重；
- 两个device slot的峰值显存符合当前粒度计划；
- slot覆写发生在对应H2D/Compute Event完成之后。

## 5. 性能测量

每个模式至少记录：

- checkpoint加载时间；
- TTFT；
- Decode ms/token与token/s；
- H2D总字节和CUDA Event时间；
- GPU计算时间；
- CPU staging memcpy时间；
- Host提交时间；
- 峰值CPU RAM、Pinned RAM与GPU显存；
- Nsight Systems时间线。

对照组：

1. 官方CPU参考，仅用于正确性；
2. 单缓冲串行真实权重；
3. 整层双缓冲真实权重；
4. 矩阵双缓冲真实权重；
5. 矩阵组加词表常驻真实权重；
6. 矩阵组加Embedding按行和LM Head词表分块；
7. `full_pinned`和`pinned_staging`分别报告。

结果必须标明实测、推算或合成校准，三者不得混用。
