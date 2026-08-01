# Cascade-LLM 基础用户测试手册

> 适用版本：M6 第一轮，2026-08-01
> 测试目标：确认本地 checkpoint 可加载、BF16 最低显存路径可运行、W8A16 fused 路径可运行、GPU 常驻策略有效、运行报告字段可信。
> 本手册只测试单机、单 GPU、单请求，不包含服务端、batch、多 GPU 或正式 M6A 长基准。

## 1. 测试前准备

推荐环境：

- Linux；
- NVIDIA RTX 3080 Ti 或其他受支持 GPU；当前 CUTLASS provider 只支持 SM86；
- 可用 CPU RAM 至少 20 GiB；
- BF16 checkpoint 约需 15 GiB 磁盘，当前 W8A16 checkpoint 约需 8.5 GiB；
- 项目虚拟环境 `.venv` 已安装完成。

进入项目并设置本次测试路径：

```bash
cd /disk2/home/guest/lianghan/repos/Cascade-LLM
source scripts/activate_env.sh

export CASCADE_TEST_BF16_MODEL=/ssd/cascade-llm/models/Llama-3.1-8B-Instruct
export CASCADE_TEST_INT8_MODEL=/ssd/cascade-llm/models/Llama-3.1-8B-Instruct-W8A16
export CASCADE_TEST_OUTPUT=/tmp/cascade-user-test
mkdir -p "${CASCADE_TEST_OUTPUT}"
```

如果模型位于其他目录，只修改以上两个模型变量，不要修改源码中的默认路径。

## 2. 五分钟快速冒烟测试

### 2.1 检查 GPU 和 Python 环境

```bash
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv

.venv/bin/python - <<'PY'
import torch
import layer_streaming

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("sm:", torch.cuda.get_device_capability(0))
print("Cascade import: OK")
PY
```

通过标准：

- `cuda available: True`；
- 目标 GPU 名称正确；
- CUTLASS W8A16 测试机应显示 `sm: (8, 6)`；
- 最后一行显示 `Cascade import: OK`。

### 2.2 检查 fused provider

```bash
.venv/bin/python tools/check_backends.py \
  --device cuda:0 \
  --provider cutlass \
  --require fused_w8a16 \
  --output "${CASCADE_TEST_OUTPUT}/backends.json"
```

通过标准：命令退出码为 0，报告中 `fused_w8a16.available` 为 `true`、`supported_sms` 包含 `86`，并且 provider 字段非空。生成报告中的实际 provider 名称应为 `cutlass_sm86_w8a16`。如果 GPU 不是 SM86，此项允许失败，但不能继续执行本手册的 fused 测试。

### 2.3 验证 checkpoint 元数据

先验证 BF16：

```bash
.venv/bin/python tools/accept_checkpoint.py \
  --checkpoint "${CASCADE_TEST_BF16_MODEL}" \
  --metadata-only \
  --backend bf16_linear \
  --ignore-memlock-limit \
  --output "${CASCADE_TEST_OUTPUT}/bf16-metadata.json"
```

再验证 W8A16：

```bash
.venv/bin/python tools/accept_checkpoint.py \
  --checkpoint "${CASCADE_TEST_INT8_MODEL}" \
  --metadata-only \
  --backend int8_dequant_bf16_fallback \
  --ignore-memlock-limit \
  --output "${CASCADE_TEST_OUTPUT}/int8-metadata.json"
```

通过标准：两个命令退出码均为 0；报告没有 missing key、shape conflict、dtype conflict 或 shard missing。

### 2.4 运行推荐的 W8A16 单卡模式

```bash
.venv/bin/python tools/run_llama31.py \
  --checkpoint "${CASCADE_TEST_INT8_MODEL}" \
  --prompt '请用一句话介绍重庆。' \
  --max-new-tokens 8 \
  --weight-format auto \
  --backend fused_w8a16 \
  --provider cutlass \
  --weight-store pinned_staging \
  --granularity matrix_group \
  --embedding-placement streamed \
  --lm-head-placement resident \
  --slots 2 \
  --prefetch-depth 2 \
  --gpu-resident-weight-budget 8GiB \
  --ignore-memlock-limit \
  --device cuda:0 \
  --output "${CASCADE_TEST_OUTPUT}/w8a16-generation.json"
```

本机已验证该配置约占用 7.7～7.9 GiB GPU allocated memory。不同 CUDA context 和驱动版本会造成小幅变化。

检查报告：

```bash
.venv/bin/python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["CASCADE_TEST_OUTPUT"]) / "w8a16-generation.json"
report = json.loads(path.read_text(encoding="utf-8"))
summary = {
    "checkpoint_ok": report["checkpoint_validation"]["ok"],
    "preflight_ok": report["memory_preflight"]["ok"],
    "backends": report["pipeline"]["backends"],
    "fallback_backends": report["pipeline"]["fallback_backends"],
    "providers": report["pipeline"]["backend_providers"],
    "resident_hit_ratio": report["transformer_placement"]["resident_hit_ratio"],
    "streamed_weight_bytes": report["transformer_placement"]["streamed_weight_bytes"],
    "generated_text": report["generation"]["generated_text"],
    "gpu_peak_gib": report["memory"]["gpu_peak_memory_bytes"] / 1024 ** 3,
}
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
```

通过标准：

- `checkpoint_ok` 和 `preflight_ok` 为 `true`；
- `backends` 只包含 `fused_w8a16`；
- `fallback_backends` 为空；
- provider 包含 `cutlass_sm86_w8a16`；
- `resident_hit_ratio` 为 `1.0`，`streamed_weight_bytes` 为 `0`；
- `generated_text` 非空；
- GPU peak 低于物理显存。

完成以上四项即可判定“基础冒烟测试通过”。

## 3. 完整基础用户验收

### UT-01：BF16 最低显存路径

该测试验证原始 BF16 checkpoint 在单张 12 GB GPU 上能够全流式运行。

```bash
.venv/bin/python tools/run_llama31.py \
  --checkpoint "${CASCADE_TEST_BF16_MODEL}" \
  --prompt 'The capital of France is' \
  --max-new-tokens 4 \
  --weight-format auto \
  --backend bf16_linear \
  --weight-store pinned_staging \
  --granularity matrix_group \
  --embedding-placement streamed \
  --lm-head-placement streamed \
  --slots 2 \
  --prefetch-depth 2 \
  --gpu-resident-weight-budget 0 \
  --ignore-memlock-limit \
  --device cuda:0 \
  --output "${CASCADE_TEST_OUTPUT}/bf16-min-memory.json"
```

通过标准：命令成功；backend 为 `bf16_linear`；resident hit ratio 为 0；Transformer H2D 不为 0；GPU peak 应明显低于完整 BF16 checkpoint 大小。

预计耗时：3080 Ti 上约 1.3～1.6 秒/token，首次 checkpoint 加载时间另计。

### UT-02：W8A16 全流式路径

```bash
.venv/bin/python tools/run_llama31.py \
  --checkpoint "${CASCADE_TEST_INT8_MODEL}" \
  --prompt 'The capital of France is' \
  --max-new-tokens 4 \
  --weight-format auto \
  --backend fused_w8a16 \
  --provider cutlass \
  --weight-store pinned_staging \
  --granularity matrix_group \
  --embedding-placement streamed \
  --lm-head-placement streamed \
  --slots 2 \
  --prefetch-depth 2 \
  --gpu-resident-weight-budget 0 \
  --ignore-memlock-limit \
  --device cuda:0 \
  --output "${CASCADE_TEST_OUTPUT}/w8a16-streamed.json"
```

通过标准：无 fallback；Transformer H2D 约为 BF16 的一半；GPU peak 低于 BF16 fallback workspace 路径。

### UT-03：显存预算与速度曲线

使用统一 benchmark，分别测试 0、4、8 GiB：

```bash
for CASCADE_TEST_BUDGET in 0 4GiB 8GiB; do
  .venv/bin/python tools/benchmark.py \
    --checkpoint "${CASCADE_TEST_INT8_MODEL}" \
    --output "${CASCADE_TEST_OUTPUT}/benchmark-${CASCADE_TEST_BUDGET}.json" \
    --device cuda:0 \
    --preset smoke \
    --backend fused_w8a16 \
    --provider cutlass \
    --slots 2 \
    --prefetch-depth 2 \
    --prefill-tokens 8 \
    --decode-tokens 8 \
    --warmup 1 \
    --repeats 3 \
    --embedding-placement streamed \
    --lm-head-placement streamed \
    --gpu-resident-weight-budget "${CASCADE_TEST_BUDGET}" \
    --ignore-memlock-limit
done
```

查看曲线：

```bash
.venv/bin/python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["CASCADE_TEST_OUTPUT"])
for budget in ("0", "4GiB", "8GiB"):
    report = json.loads(
        (root / ("benchmark-" + budget + ".json")).read_text(encoding="utf-8")
    )
    result = report["results"][0]
    metrics = result["median"]
    print(
        budget,
        "status=", result["status"],
        "decode_ms=", round(metrics["decode_ms_per_token"], 2),
        "H2D_GiB=", round(
            metrics["transformer_weight_h2d_bytes_per_forward"] / 1024 ** 3, 3
        ),
        "GPU_GiB=", round(metrics["gpu_peak_allocated_bytes"] / 1024 ** 3, 3),
        "hit=", metrics["resident_hit_ratio"],
        "CV=", round(result["variability"]["max_latency_cv"], 4),
    )
PY
```

通过标准：

- 三个 case 的 `status` 均为 `ok`；
- resident budget 增加时，hit ratio 单调增加；
- Transformer H2D 单调下降；
- 8 GiB 对当前 W8A16 8B checkpoint 应达到 hit ratio 1.0 和 Transformer H2D 0；
- decode 延迟总体下降；
- 三次短跑的 latency CV 应低于 5%。

### UT-04：LM Head placement

在 UT-03 的 8 GiB 命令中，将：

```text
--lm-head-placement streamed
```

改成：

```text
--lm-head-placement resident
```

并将输出写入 `benchmark-8GiB-lm-resident.json`。

通过标准：`lm_head_h2d_bytes_per_forward` 从约 0.98 GiB 变为 0；GPU peak 增加约 1 GiB；decode 延迟进一步下降；Top-k/生成不能因为 placement 改变。

### UT-05：显存不足的启动前拒绝

这是预期失败测试：

```bash
.venv/bin/python tools/run_llama31.py \
  --checkpoint "${CASCADE_TEST_BF16_MODEL}" \
  --max-new-tokens 1 \
  --backend bf16_linear \
  --weight-store pinned_staging \
  --embedding-placement resident \
  --lm-head-placement resident \
  --gpu-resident-weight-budget 20GiB \
  --ignore-memlock-limit \
  --device cuda:0
```

通过标准：命令返回非零退出码，并在 checkpoint 权重加载和实际生成前报告 `estimated GPU peak` 超过可用显存。不能出现进程卡死、CUDA illegal memory access 或后台线程残留。

## 4. Memlock 说明

先查看当前限制：

```bash
ulimit -l
```

`pinned_staging` 仍需要锁页 staging buffer。本机限制约为 64 MiB，而双 slot 测试约需 224 MiB，因此示例显式使用了 `--ignore-memlock-limit`。

这个选项只忽略 `RLIMIT_MEMLOCK` 这一项预检，不会忽略 GPU OOM、CPU RAM、shape、dtype、backend 或 checkpoint 错误。只有已经确认当前 PyTorch/CUDA 环境能够成功分配 pinned tensor 时才能使用。如果出现 pinned allocation 错误，应停止使用该选项，并由管理员提高 memlock，或减少 slot、词表 chunk 和 staging 深度。

不要把 `full_pinned` 作为普通用户测试默认值：BF16 8B 会尝试锁定接近完整 checkpoint 大小的 CPU RAM。

## 5. 结果解释

生成报告重点字段：

```text
checkpoint_validation.ok
memory_preflight.ok
runtime_config.linear_backend_requested
pipeline.backends
pipeline.fallback_backends
pipeline.backend_providers
transformer_placement.resident_hit_ratio
transformer_placement.streamed_weight_bytes
timings.time_to_first_token_ms
timings.decode_token_latencies_ms
memory.gpu_peak_memory_bytes
generation.generated_text
```

benchmark 报告重点字段：

```text
results[0].status
results[0].median.decode_ms_per_token
results[0].median.transformer_weight_h2d_bytes_per_forward
results[0].median.lm_head_h2d_bytes_per_forward
results[0].median.resident_hit_ratio
results[0].median.gpu_peak_allocated_bytes
results[0].variability.max_latency_cv
```

短 prompt 的文本内容只用于冒烟，不能代替模型质量验收。INT8 与 BF16 也不要求逐 token 文本完全相同；需要数值质量结论时，应使用固定输入运行 `tools/compare_cascade_paths.py`，并记录 logits 误差和 Top-k 交集率。

## 6. 常见故障

| 现象 | 原因与处理 |
|---|---|
| `CUDA is unavailable` | 检查驱动、`CUDA_VISIBLE_DEVICES` 和 PyTorch CUDA 安装。 |
| `SMxx is not in supported SMs (86,)` | 当前 CUTLASS provider 只支持 SM86；改用 fallback backend，或在受支持机器测试。 |
| `RLIMIT_MEMLOCK` | 参考第 4 节；不要盲目忽略实际 pinned allocation 失败。 |
| `estimated GPU peak ... above available` | 降低 resident budget，stream LM Head/Embedding，减少 context 或 slots。 |
| missing shard/key/scale | checkpoint 不完整或格式不匹配；重新下载/量化，不要绕过校验。 |
| `actual backends` 与 requested 不同 | 属于错误；框架禁止静默 fallback，应保存报告并停止测试。 |
| 输出为空 | 检查 generated token IDs、EOS、prompt 和 tokenizer；短输出不一定代表 runtime 失败。 |
| 第一次运行慢 | checkpoint 读盘、OS page cache 和 CUDA 初始化所致；性能测试必须 warmup。 |

## 7. 用户测试记录模板

每次提交问题时建议同时提供：

```text
测试日期：
Git revision / 工作区状态：
GPU 与数量：
驱动 / CUDA / PyTorch：
CPU RAM：
Checkpoint 路径与格式：
测试命令：
是否使用 --ignore-memlock-limit：
测试结果：通过 / 失败
TTFT：
Decode ms/token：
GPU peak：
Resident hit ratio：
Transformer H2D/forward：
错误信息：
报告 JSON 路径：
```

请保留完整 JSON 报告；终端最后几行不足以判断 checkpoint、fallback、placement 和内存预算是否符合预期。
