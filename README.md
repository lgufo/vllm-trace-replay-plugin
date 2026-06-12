# vLLM Trace Replay Platform Plugin

This plugin replays pre-sampled request traces during vLLM serving without
executing real model kernels on accelerator hardware.

**详细操作步骤与调用链说明**（采集 intermediate states、在线 replay、SchedulerOutput、环境变量与排错）：见 [docs/TRACE_REPLAY_OPERATIONS_AND_CALL_CHAIN.md](docs/TRACE_REPLAY_OPERATIONS_AND_CALL_CHAIN.md)。

## 1) hidden states 和 intermediate states 的关系

- 在 LLM 语境里，`intermediate states` 是一个泛化概念，指前向过程中的中间表示。
- `hidden states` 是 intermediate states 的一个子集，通常指每层输出（尤其最后一层）。
- 对于本插件，离线采集重点是 request 对齐的 `hidden_states`，并可进一步映射成每步 `step_logits` 供在线 replay。

## 2) 目录结构

- `trace_replay_plugin/my_dummy_platform.py`: OOT platform 实现
- `trace_replay_plugin/my_dummy_worker.py`: WorkerBase replay worker
- `trace_replay_plugin/trace_db.py`: trace 数据加载/步进回放
- `trace_replay_plugin/schedule_reporter.py`: 每 step 调度报告（logger + JSONL）
- `scripts/collect_intermediate_states.py`: 离线采集中间状态与 logits
- `scripts/project_hidden_to_logits.py`: hidden -> logits 投影
- `scripts/demo_run_vllm.py`: 最小端到端 replay（三条请求，对应 trace 键 `0`/`1`/`2`）

## 3) 安装

```bash
cd /fact_home/lizhang/project/vllm-trace-replay-plugin
pip install -e .
```

### 若 `pip install -e .` 报找不到 `vllm>=...`

常见原因：

1. **PyPI 镜像同步滞后**：你看到的可用版本列表只到 `0.11.x`，而旧版 `setup.py` 若写 `vllm>=0.13.0` 会直接解析失败。当前仓库已把依赖放宽为 `vllm>=0.6.0`，一般可装。
2. **Python / 平台 wheel 不全**：某些 Python 版本下 PyPI 没有对应 vLLM wheel，也会装不上。需换 Python 版本或从官方 wheel / 源码安装 vLLM。

**是否必须 editable？** 不必须。editable 只是开发方便（改代码立刻生效）。你也可以：

```bash
pip install .          # 普通安装
# 或 vLLM 已从别处装好，只注册 entry point、不拉依赖：
pip install --no-deps -e .
```

若不用 `pip install`，则需自行保证 `trace_replay_plugin` 在 `PYTHONPATH` 里，且 **entry point 能被 vLLM 发现**（通常仍建议至少 `pip install --no-deps -e .` 注册 `vllm.platform_plugins`）。

### 最小端到端 demo（三条请求）

需已 `pip install -e .`，且与 vLLM 同一环境。脚本会在导入 vLLM 前设置
`VLLM_PLUGINS=trace_replay` 与 `TRACE_REPLAY_DB_PATH`；若未提供有效 DB，会写入
含请求键 `0`、`1`、`2` 的默认 trace（与 `LLM.generate` 的前三个 request id 一致）。

```bash
python scripts/demo_run_vllm.py --model gpt2 --max-tokens 8
# 或指定已有 trace：
python scripts/demo_run_vllm.py --db-path artifacts/trace_db.pt --model gpt2
```

## 4) 离线采集 intermediate states

输入 JSONL 格式：

```json
{"request_id":"req-1","prompt":"Hello"}
{"request_id":"req-2","prompt":"What is AI?"}
```

运行采集：

```bash
python scripts/collect_intermediate_states.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --input-jsonl requests.jsonl \
  --output-pt artifacts/trace_db.pt \
  --num-decode-steps 4 \
  --dtype bfloat16 \
  --device cuda
```

可选：仅用 hidden states 重新投影 logits：

```bash
python scripts/project_hidden_to_logits.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --input-pt artifacts/trace_db.pt \
  --output-pt artifacts/trace_db_projected.pt \
  --dtype bfloat16 \
  --device cuda
```

## 5) 在线 replay（不跑真实硬件）

设置环境变量：

```bash
export TRACE_REPLAY_DB_PATH=artifacts/trace_db.pt
export TRACE_REPLAY_REPORT_PATH=artifacts/schedule_report.jsonl
export TRACE_REPLAY_STRICT_MISSING=0
export TRACE_REPLAY_FALLBACK_TOKEN_ID=0
```

启动 vLLM 时加载本 plugin 后，worker 在 `execute_model` 中按每个 step 的
`scheduler_output` 提取被调度 request，并从 trace DB 按 `request_id + step_cursor`
返回 token 结果。

## 6) 为什么 platform plugin 需要实现这些 method

- `_enum` / `device_type` / `device_name`
  - 作用：统一平台识别、设备语义与日志标签。
  - 生效阶段：平台装载、设备配置、日志输出。
- `check_and_update_config`
  - 作用：初始化早期修正配置；最关键是设置 `parallel_config.worker_cls` 指向自定义 worker。
  - 生效阶段：`VllmConfig` 初始化后、executor 创建 worker 前。
- `get_attn_backend_cls`
  - 作用：满足 attention backend 选择与类解析链路，即使 replay 模式不真正执行 attention 也要可解析。
  - 生效阶段：模型执行图准备、attention backend 选择期。
- `get_device_communicator_cls`
  - 作用：分布式通信抽象选择点；单机 replay 也要返回合法 communicator 类，保证执行路径完整。
  - 生效阶段：分布式初始化/collective 通信路径构建期。

### Worker 关键方法在推理流程中的作用

- `init_device`: worker 设备初始化阶段调用（这里是 replay no-op）。
- `load_model`: 模型加载阶段调用（这里不加载真实权重）。
- `get_kv_cache_spec` + `determine_available_memory` + `initialize_from_config`
  - 作用：参与 KV cache 容量规划与初始化流程；replay 模式返回最小可运行结果即可。
- `execute_model`
  - 作用：每 step 核心执行入口；本插件在这里完成“读取调度请求 -> 查 trace -> 返回结果 -> 记录报告”。

## 7) 哪些 method 可以不实现

对本插件目标（trace replay）而言，以下可不支持：
- LoRA 方法（`add_lora/remove_lora/pin_lora/list_loras`）: 返回 false/空集合。
- sleep/wakeup、graph mode、spec decode draft token 等高级能力：可不实现或 no-op。

前提是：主路径 `init_device/load_model/get_kv_cache_spec/determine_available_memory/initialize_from_config/execute_model`
必须能稳定跑通。

## 8) 调度报告内容

每 step 输出：
- `step_idx`
- `scheduled_req_ids`
- `new_req_ids`
- `cached_req_ids`
- `finished_req_ids`
- `num_scheduled_tokens`
- `timestamp`

并同时写入：
- logger 实时日志
- `TRACE_REPLAY_REPORT_PATH` 指定 JSONL
