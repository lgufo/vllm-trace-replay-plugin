# vLLM + trace-replay-plugin 推理执行调用链路详解

本文档以 `demo_run_vllm.py` 为基础，以 3 条请求（req "0"/"1"/"2"）、`max_tokens=4` 为例，
完整追踪从 `LLM.generate()` 入口到最终输出的每一次函数调用，包含 vLLM 自身的执行框架与
本插件的介入点，并对关键行为进行注释说明。

**约定：**
- `# →` 表示返回值或状态变化
- `# ←` 表示入参来源
- `# ★ PLUGIN` 标记本插件介入的位置
- `vllm/` 路径为 vLLM 源码内的相对位置（基于 vLLM v1 架构）

---

## 目录

0. [vLLM 完整执行框架（总览）](#零vllm-完整执行框架总览)
1. [启动阶段：插件注册与 Worker 初始化](#一启动阶段插件注册与-worker-初始化)
2. [vLLM 初始化序列](#二vllm-初始化序列)
3. [Step 0 — Prefill（3 条请求首次进入）](#三step-0--prefill3-条请求首次进入)
4. [Step 1–3 — Decode（循环推理）](#四step-13--decode循环推理)
5. [Step 4 — Finish（max_tokens 到达，清理）](#五step-4--finishmax_tokens-到达清理)
6. [全流程状态追踪表](#六全流程状态追踪表)

---

## 零、vLLM 完整执行框架（总览）

在深入逐步调用链之前，先从全局视角了解 vLLM v1 的分层架构，以及本插件在哪些层面
改变了默认行为。

### 0.1 vLLM v1 核心组件层次

```
┌─────────────────────────────────────────────────────────────────┐
│  用户代码 / API                                                   │
│    LLM.generate()  /  vllm.AsyncLLMEngine  /  OpenAI Server     │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│  Engine 层                                                       │
│    LLMEngine (vllm/v1/engine/llm_engine.py)                     │
│      └─ EngineCoreClient → EngineCore (vllm/v1/engine/core.py)  │
│           每次 step() = schedule + execute + update              │
└──────────┬────────────────────┬────────────────────────────────-┘
           │                    │
┌──────────▼──────────┐  ┌──────▼──────────────────────────────────┐
│  Scheduler 层        │  │  Executor 层                             │
│  vllm/v1/core/sched/│  │  vllm/v1/executor/                       │
│  scheduler.py       │  │    UniProcExecutor (单进程)               │
│                     │  │    RayGPUExecutor  (多卡/多机)            │
│  职责：              │  │                                           │
│  · 请求队列管理       │  │  职责：将 SchedulerOutput 分发给 Worker  │
│  · prefill/decode   │  └──────────────────┬──────────────────────┘
│    决策              │                     │
│  · KV Cache 分配     │  ┌──────────────────▼──────────────────────┐
│  · 生成停止检测       │  │  Worker 层  ← ★ PLUGIN 替换此层          │
└─────────────────────┘  │                                           │
                         │  [普通 vLLM]  GPUWorker                  │
                         │    vllm/v1/worker/gpu_worker.py          │
                         │    └─ GPUModelRunner                     │
                         │         · _prepare_inputs()              │
                         │         · model.forward()  ← GPU kernel  │
                         │         · sample(logits)                 │
                         │                                           │
                         │  [本插件 ★]  MyDummyWorker               │
                         │    trace_replay_plugin/my_dummy_worker.py│
                         │    └─ TraceDatabase.get_next_token()     │
                         │         · 无 GPU 计算                    │
                         │         · 直接返回预录 token              │
                         └───────────────────────────────────────────┘
```

### 0.2 插件共介入 4 个位置

| 位置 | 时机 | 插件做什么 | 对应代码 |
|------|------|-----------|---------|
| **① 平台发现** | vLLM import 时 | 注册 `MyDummyPlatform` 为 OOT 平台 | `__init__.py:6` |
| **② 配置注入** | `VllmConfig` 初始化后 | 替换 `worker_cls`，关闭缓存特性 | `my_dummy_platform.py:26` |
| **③ Worker 创建** | Executor 构建 Worker 时 | 创建 `MyDummyWorker`，加载 `TraceDatabase` | `my_dummy_worker.py:39` |
| **④ 每步执行** | Scheduler 每次调度后 | `execute_model()` 查 trace 返回 token | `my_dummy_worker.py:172` |

---

### 0.3 vLLM 完整调用链（含插件介入点）

下面是从 `LLM.generate()` 到最终输出的完整函数调用树。
`★` 标注的行是本插件替换或注入的位置。

```
═══════════════════════════════════════════════════════
  Phase A: 进程启动 & 插件加载（import 阶段）
═══════════════════════════════════════════════════════

import vllm
│
├─ vllm/platforms/__init__.py: _init_plugins()
│   │   # 扫描 entry_points["vllm.platform_plugins"]
│   └─ ★ trace_replay_platform_plugin()              # __init__.py:6
│           └─ return "...MyDummyPlatform"
│               # vLLM 将 current_platform 设为 MyDummyPlatform
│
└─ vllm/config.py: VllmConfig.__post_init__()
    └─ current_platform.check_and_update_config(self) # my_dummy_platform.py:26
        │   # ★ 修改 VllmConfig 字段，注入 MyDummyWorker
        ├─ if parallel_config.worker_cls == "auto":      # 仅在未显式指定时替换
        │      parallel_config.worker_cls = "...MyDummyWorker"
        ├─ scheduler_config.async_scheduling = False
        ├─ cache_config.enable_prefix_caching = False
        ├─ scheduler_config.disable_hybrid_kv_cache_manager = True
        └─ compilation_config.custom_ops = []            # 关闭自定义算子编译


═══════════════════════════════════════════════════════
  Phase B: LLM 对象构建 & 引擎初始化
═══════════════════════════════════════════════════════

LLM.__init__(model="gpt2", load_format="dummy", ...)
│   # vllm/entrypoints/llm.py
│
├─ LLMEngine.__init__(vllm_config)
│   # vllm/v1/engine/llm_engine.py
│   │
│   ├─ Tokenizer.from_config(...)                    # 加载 tokenizer（真实加载）
│   │
│   ├─ EngineCoreClient.make_client(...)             # vllm/v1/engine/core_client.py
│   │   # 进程内模式下直接构建 EngineCore（多进程模式则经 ZMQ 通信）
│   │   └─ EngineCore.__init__(vllm_config)
│   │   # vllm/v1/engine/core.py
│   │   │
│   │   ├─ self.model_executor = UniProcExecutor(vllm_config)
│   │   │   # vllm/v1/executor/uniproc_executor.py（在 EngineCore 内创建）
│   │   │   └─ _init_executor():                     # 注意：是 _init_executor 而非 __init__
│   │   │       ├─ self.driver_worker = WorkerWrapperBase(rpc_rank=0)
│   │   │       └─ driver_worker.init_worker(all_kwargs=[{
│   │   │               "vllm_config": vllm_config,   #   worker_cls 已被插件替换
│   │   │               "local_rank": 0, "rank": 0,
│   │   │               "is_driver_worker": True,
│   │   │               # distributed_init_method 由 get_distributed_init_method() 生成（tcp://...）
│   │   │           }])
│   │   │           # WorkerWrapperBase 内部实例化 MyDummyWorker  ★ my_dummy_worker.py:39
│   │   │           │   # ★ 插件在此完成所有初始化
│   │   │           ├─ TraceDatabase.from_file(...)   # 加载 trace_db.pt
│   │   │           ├─ self.req_last_token = {}
│   │   │           └─ self.reporter = ScheduleReporter(...)
│   │   │
│   │   │       # _init_executor() 紧接着调用（见第二章详解）：
│   │   │       ├─ driver_worker.init_device()        # ★ no-op
│   │   │       └─ driver_worker.load_model()         # ★ no-op
│   │   │
│   │   ├─ self._initialize_kv_caches(vllm_config)    # EngineCore 初始化 KV Cache
│   │   │   ├─ model_executor.get_kv_cache_specs()    # ★ return [{}]
│   │   │   ├─ model_executor.determine_available_memory()  # 仅当存在 KV spec 时；此处跳过
│   │   │   └─ model_executor.initialize_from_config(...)   # ★ 保存 kv_cache_config + warmup
│   │   │           # 内部触发 worker.compile_or_warm_up_model()  # ★ return 0.0
│   │   │
│   │   └─ Scheduler.__init__(vllm_config)           # vllm/v1/core/sched/scheduler.py
│   │       │   # 初始化请求队列、KV Cache 管理器（因插件已禁用 prefix cache，
│   │       │   # KV Cache 管理器走最简路径）
│   │       ├─ self.waiting = create_request_queue(policy)  # 等待 prefill 的请求队列（RequestQueue）
│   │       ├─ self.running: list[Request] = []     # 正在 decode 的请求列表
│   │       └─ self.finished_req_ids: set[str] = set()
│   │
│   └─ # LLM 对象就绪，等待 generate() 调用
└─ # （注：以上初始化顺序为 v1 实际调用关系，第二章仍按 Worker 方法逐个展开）


═══════════════════════════════════════════════════════
  Phase C: LLM.generate() — 请求提交
═══════════════════════════════════════════════════════

LLM.generate(
    prompts=["Hello trace replay.", "Second request.", "Third request."],
    sampling_params=SamplingParams(max_tokens=4, temperature=0.0)
)
│   # vllm/entrypoints/llm.py
│
├─ for i, prompt in enumerate(prompts):
│   │
│   ├─ tokenizer.encode(prompt) → prompt_token_ids
│   │       # "Hello trace replay." → [15496, 12645, 24788, 13]（示意）
│   │
│   └─ LLMEngine.add_request(
│           request_id=str(i),          # "0", "1", "2"
│           prompt=prompt,
│           sampling_params=sampling_params,
│           prompt_token_ids=prompt_token_ids
│       )
│       └─ EngineCore.add_request(request)        # 经 EngineCoreClient 转发
│           └─ Scheduler.add_request(request)      # vllm/v1/core/sched/scheduler.py
│               └─ self.waiting.add_request(request)
│                       # 3 条请求依次进入等待队列
│
└─ LLM._run_engine(use_tqdm=False)
    │   # 驱动引擎循环直到所有请求完成
    └─ while True:
           outputs = LLMEngine.step()        # ← 推理主循环
           if all_requests_done: break


═══════════════════════════════════════════════════════
  Phase D: 推理主循环 — 每次 step() 的完整执行
═══════════════════════════════════════════════════════

LLMEngine.step()
│   # vllm/v1/engine/llm_engine.py
│   # 经 self.engine_core.get_output()（EngineCoreClient）取回输出；
│   # 进程内模式下实际驱动 EngineCore.step()
│
└─ EngineCore.step()                         # vllm/v1/engine/core.py
    │   # 实现：schedule → execute_model(non_block=True) → future.result() → update_from_output
    │
    │   ┌─────────────────────────────────────────────────────────┐
    │   │  子步骤 D-1：调度决策                                     │
    │   └─────────────────────────────────────────────────────────┘
    ├─ scheduler_output = Scheduler.schedule()
    │   # vllm/v1/core/sched/scheduler.py
    │   │
    │   ├─ # 处理上一步标记为 finished 的请求
    │   │   for req_id in self.finished_req_ids:
    │   │       self.running.remove(requests[req_id])
    │   │       → finished_req_ids 字段写入 SchedulerOutput
    │   │
    │   ├─ # 调度等待队列中的请求进行 prefill
    │   │   for req in self.waiting:
    │   │       if self._can_schedule(req):       # 检查 token budget
    │   │           scheduled_new_reqs.append(req)
    │   │           self.running.append(req)
    │   │           self.waiting.remove(req)
    │   │           num_scheduled_tokens[req.req_id] = len(req.prompt_token_ids)
    │   │
    │   ├─ # 调度正在运行的请求进行 decode
    │   │   for req in self.running:
    │   │       if req not in scheduled_new_reqs:  # 非本步新加入的
    │   │           scheduled_cached_reqs.req_ids.append(req.req_id)
    │   │           num_scheduled_tokens[req.req_id] = 1
    │   │
    │   └─ return SchedulerOutput(
    │           scheduled_new_reqs=scheduled_new_reqs,
    │           scheduled_cached_reqs=CachedRequestData(req_ids=...),
    │           num_scheduled_tokens=num_scheduled_tokens,
    │           finished_req_ids=finished_req_ids
    │       )
    │
    │   ┌─────────────────────────────────────────────────────────┐
    │   │  子步骤 D-2：模型执行                                     │
    │   └─────────────────────────────────────────────────────────┘
    ├─ model_output = UniProcExecutor.execute_model(scheduler_output)
    │   # vllm/v1/executor/uniproc_executor.py
    │   │
    │   └─ self.worker.execute_model(scheduler_output)
    │       │
    │       │   ┌────────────────────────────────────────────────┐
    │       │   │  [普通 vLLM — 不使用插件时的路径]               │
    │       │   └────────────────────────────────────────────────┘
    │       │   GPUWorker.execute_model(scheduler_output)
    │       │   # vllm/v1/worker/gpu_worker.py
    │       │   └─ GPUModelRunner.execute_model(scheduler_output)
    │       │       # vllm/v1/worker/gpu_model_runner.py
    │       │       ├─ _prepare_inputs(scheduler_output)   # v1 内部方法（旧称 prepare_model_input）
    │       │       │   ├─ 构建 input_ids、position_ids tensor
    │       │       │   ├─ 构建 attention_metadata（KV Cache 地址）
    │       │       │   └─ 构建 sampling_metadata
    │       │       ├─ model.forward(input_ids, positions, attn_metadata)
    │       │       │   ├─ Embedding(input_ids) → hidden_states
    │       │       │   ├─ for layer in transformer_layers:
    │       │       │   │   ├─ LayerNorm → Q, K, V
    │       │       │   │   ├─ PagedAttention(Q,K,V, kv_cache) → attn_out
    │       │       │   │   └─ FFN(attn_out) → hidden_states
    │       │       │   └─ lm_head(hidden_states[-1]) → logits [vocab_size]
    │       │       └─ sampler.forward(logits, sampling_metadata)
    │       │           └─ argmax(logits) / top-p sampling → sampled_token_ids
    │       │
    │       │   ┌────────────────────────────────────────────────┐
    │       │   │  ★ [本插件路径 — 替换 GPUWorker]               │
    │       │   └────────────────────────────────────────────────┘
    │       └─ MyDummyWorker.execute_model(scheduler_output)      # ★ my_dummy_worker.py:172
    │               # 无任何 GPU 操作，直接查 TraceDatabase
    │               # 详细调用链见第三～五章
    │               └─ return ModelRunnerOutput(sampled_token_ids=[[tok0],[tok1],[tok2]])
    │
    │   ┌─────────────────────────────────────────────────────────┐
    │   │  子步骤 D-3：调度器更新（处理模型输出）                    │
    │   └─────────────────────────────────────────────────────────┘
    ├─ engine_core_outputs = Scheduler.update_from_output(
    │       scheduler_output, model_output
    │   )
    │   # vllm/v1/core/sched/scheduler.py
    │   │
    │   ├─ for req_id, token_ids in zip(
    │   │       model_output.req_ids,
    │   │       model_output.sampled_token_ids
    │   │   ):
    │   │   req = self.requests[req_id]
    │   │   token_id = token_ids[0]              # 每条请求每步一个 token
    │   │   req.output_token_ids.append(token_id) # 追加到已生成序列
    │   │   req.num_computed_tokens += 1
    │   │   │
    │   │   ├─ # 检查停止条件
    │   │   │   if token_id == tokenizer.eos_token_id:
    │   │   │       req.status = FINISHED_STOPPED   # EOS token
    │   │   │   elif len(req.output_token_ids) >= max_tokens:
    │   │   │       req.status = FINISHED_LENGTH    # 达到 max_tokens
    │   │   │   elif token_id in stop_token_ids:
    │   │   │       req.status = FINISHED_STOPPED   # 自定义停止 token
    │   │   │
    │   │   └─ if req.is_finished():
    │   │           self.finished_req_ids.add(req_id)
    │   │           # 下一次 schedule() 时会放入 SchedulerOutput.finished_req_ids
    │   │           # Worker 的 _cleanup_finished() 将据此清理 TraceDatabase 状态
    │   │
    │   └─ return EngineCoreOutputs(
    │           outputs=[EngineCoreOutput(request_id, new_token_ids, finish_reason, ...), ...]
    │       )   # 注意：此处是 EngineCoreOutput（增量 token），RequestOutput 在 D-4 才构建
    │
    │   ┌─────────────────────────────────────────────────────────┐
    │   │  子步骤 D-4：输出后处理                                   │
    │   └─────────────────────────────────────────────────────────┘
    └─ # LLMEngine.step() 通过 OutputProcessor.process_outputs() 将 EngineCoreOutput
       # 转换为用户可见的 RequestOutput（vllm/v1/engine/output_processor.py）
        for output in engine_core_outputs.outputs:
            token_ids = output.new_token_ids
            text = detokenizer.decode_incremental(token_ids)  # 增量解码为文本
            request_outputs.append(
                RequestOutput(
                    request_id=output.request_id,
                    prompt=prompts[i],
                    outputs=[CompletionOutput(text=text, token_ids=token_ids)]
                )
            )


═══════════════════════════════════════════════════════
  Phase E: 循环结束，返回结果
═══════════════════════════════════════════════════════

# 当所有请求状态变为 FINISHED 后，_run_engine() 退出循环
# LLM.generate() 返回：

outputs = [
    RequestOutput(request_id="0", outputs=[CompletionOutput(
        text="...",               # tokenizer.decode([1000,1001,1002,1003])
        token_ids=(1000,1001,1002,1003)
    )]),
    RequestOutput(request_id="1", outputs=[CompletionOutput(
        text="...",               # tokenizer.decode([1010,1011,1012,1013])
        token_ids=(1010,1011,1012,1013)
    )]),
    RequestOutput(request_id="2", outputs=[CompletionOutput(
        text="...",               # tokenizer.decode([1020,1021,1022,1023])
        token_ids=(1020,1021,1022,1023)
    )]),
]
```

---

### 0.4 每次 step() 的数据流向图

```
                    ┌──────────────────────────────┐
                    │  Scheduler.schedule()         │
                    │                               │
  waiting queue ───►│  waiting → scheduled_new_reqs │
  running list  ───►│  running → scheduled_cached   │──► SchedulerOutput
  finished set  ───►│  finished → finished_req_ids  │
                    └──────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │  Executor.execute_model()     │
                    │                               │
  SchedulerOutput ──►│  [普通] GPUModelRunner       │
                    │    GPU forward → logits       │──► ModelRunnerOutput
                    │  [★插件] MyDummyWorker        │    sampled_token_ids
                    │    TraceDatabase 查表         │    = [[tok]×N]
                    └──────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │  Scheduler.update_from_output()│
                    │                               │
  ModelRunnerOutput─►│  追加 output_token_ids        │
                    │  检查 EOS / max_tokens         │──► EngineCoreOutputs
                    │  标记 finished_req_ids         │    (每步新完成的请求)
                    └──────────────────────────────┘
                                   │
                                   ▼
                         tokenizer.decode()
                              │
                              ▼
                        RequestOutput（用户可见）
```

---

### 0.5 普通 vLLM vs 本插件的核心差异

```
                普通 vLLM（GPU 推理）           本插件（Trace 回放）
                ─────────────────────           ─────────────────────
Worker 类        GPUWorker                      ★ MyDummyWorker

execute_model()  _prepare_inputs()              _cleanup_finished()
内部调用          → model.forward() [GPU]        → _extract_req_ids()
                 → sampler(logits)              → get_next_token_by_last_token()
                                                 （纯内存查表，无 GPU 操作）

KV Cache         PagedAttention 管理            ★ 返回 {} 空规格，无 KV 分配

token 来源        GPU 计算的 logits argmax       ★ trace_db.pt 中预录的 token id

运行时延迟        取决于模型大小 + GPU 性能       极低（纯 Python 字典查找）

prefix cache     可启用（复用历史 KV）           ★ 强制关闭（无真实 KV）

Attention 后端   FlashAttention / FlashInfer    ★ CPUAttentionBackend（占位）
```

---

## 一、启动阶段：插件注册与 Worker 初始化

> **对应总览位置：** Phase A（进程启动 & 插件加载）和 Phase B（LLM 对象构建）中
> `worker = MyDummyWorker(...)` 一行，即插件介入点 ①②③。
> 本章展开这两个阶段内部每行函数调用的完整细节。

vLLM 启动时会通过 Python 的 **setuptools entry points** 机制发现并加载外部平台插件。
`setup.py` 中注册的入口点 `trace_replay` 指向 `__init__.py` 中的工厂函数，
该函数只返回一个类路径字符串，vLLM 再用它动态导入并实例化平台类。

```
vLLM 进程启动
│
├─ 读取 entry_points["vllm.platform_plugins"]
│   │   # setuptools 扫描已安装包，找到 trace_replay 入口点
│   └─ trace_replay_platform_plugin()                       # __init__.py:6
│       └─ return "trace_replay_plugin.my_dummy_platform.MyDummyPlatform"
│           # 只返回类路径字符串，vLLM 负责实际导入
```

平台类加载后，vLLM 会立即调用 `check_and_update_config` 来完成配置注入。
这是插件能够"劫持" vLLM 内部行为的核心挂载点：在这里我们把 Worker 类替换为
我们自己的 `MyDummyWorker`，同时关闭所有依赖真实 KV Cache 的特性，
避免 vLLM 在初始化阶段触发不必要的 GPU 内存分配路径。

```
├─ MyDummyPlatform.check_and_update_config(vllm_config)     # my_dummy_platform.py:26
│   │   # vllm_config 是 vLLM 全局配置对象，此处直接修改其字段
│   ├─ vllm_config.parallel_config.worker_cls
│   │       = "trace_replay_plugin.my_dummy_worker.MyDummyWorker"
│   │       # 把默认 GPU Worker 替换为我们的回放 Worker
│   ├─ vllm_config.scheduler_config.async_scheduling = False
│   │       # 关闭异步调度，保证每步执行顺序确定性
│   ├─ vllm_config.cache_config.enable_prefix_caching = False
│   │       # 关闭 prefix cache，避免 KV Cache 协调器初始化
│   └─ vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = True
│           # 禁用混合 KV Cache 管理器，回放模式无真实 KV 分配需求
│
├─ MyDummyPlatform.get_attn_backend_cls(...)                # my_dummy_platform.py:43
│   └─ return "vllm.v1.attention.backends.cpu_attn.CPUAttentionBackend"
│       # vLLM 要求每个平台提供一个 Attention Backend 类，此处返回 CPU 版占位符
│       # 回放 Worker 实际上从不执行真正的 Attention 计算
```

Worker 初始化时完成所有运行时依赖的创建。最关键的三个成员是：
`trace_db`（回放数据源）、`req_last_token`（跨步骤状态）、`reporter`（观测输出）。

```
└─ MyDummyWorker.__init__(vllm_config, local_rank=0, ...)   # my_dummy_worker.py:39
    │
    ├─ self.device = torch.device("cpu")                    # 声明为 CPU 设备
    ├─ self.model_runner = _NoOpModel()                     # 占位模型，调用 forward() 会直接抛异常
    │
    ├─ trace_path = os.environ.get("TRACE_REPLAY_DB_PATH")  # 必填，未设置则抛 ValueError
    ├─ strict_missing = False   # TRACE_REPLAY_STRICT_MISSING=0，缺失请求用 fallback token
    ├─ fallback_token_id = 0    # 找不到 trace 时返回 token id=0
    │
    ├─ TraceDatabase.from_file("trace_db.pt", ...)          # trace_db.py:51
    │   │   # 一次性加载整个 trace 文件到内存
    │   ├─ torch.load("trace_db.pt", map_location="cpu")
    │   │       # map_location="cpu" 确保 GPU 环境生成的 trace 也能在 CPU 上读取
    │   │       → payload = {"meta": {...}, "records": {"0":..., "1":..., "2":...}}
    │   │
    │   ├─ _normalize_payload(payload)                      # trace_db.py:83
    │   │       # 兼容两种格式：
    │   │       # Format A: {"meta":..., "records":{...}}（标准格式）
    │   │       # Format B: {"req-1": ..., "req-2": ...}（平铺格式）
    │   │       → meta = {"demo": True, ...}
    │   │         records = {"0": {...}, "1": {...}, "2": {...}}
    │   │
    │   ├─ for req_id="0":  # 逐条解析 record，构建 RequestTrace 对象
    │   │   ├─ _extract_step_token_ids(record)              # trace_db.py:112
    │   │   │       # 优先取 record["step_token_ids"]；无则对 logits 做 argmax
    │   │   │       → [1000, 1001, 1002, 1003]
    │   │   ├─ _extract_step_logits(record)                 # trace_db.py:96
    │   │   │       → []  # demo trace 未存 logits
    │   │   ├─ _extract_prompt_last_token(record)           # trace_db.py:126
    │   │   │       → None  # demo trace 未存 prompt_last_token
    │   │   ├─ _extract_transitions(record, ...)            # trace_db.py:132
    │   │   │       # 解析 transition map（current_token → [next_token, ...]）
    │   │   │       → {}  # demo trace 无 transitions，使用顺序 cursor 回放
    │   │   └─ RequestTrace(                                # trace_db.py:13
    │   │           request_id="0",
    │   │           step_token_ids=[1000,1001,1002,1003],
    │   │           transitions={},
    │   │           cursor=0          # 游标初始为 0
    │   │       )
    │   ├─ for req_id="1": RequestTrace(id="1", step_token_ids=[1010,1011,1012,1013], cursor=0)
    │   ├─ for req_id="2": RequestTrace(id="2", step_token_ids=[1020,1021,1022,1023], cursor=0)
    │   └─ self.traces = {"0": <trace0>, "1": <trace1>, "2": <trace2>}
    │
    ├─ self.req_last_token = {}
    │       # 运行时状态字典：记录每个请求上一步返回的 token id
    │       # 用于 transition map 查找；首次调用时为空
    │
    └─ self.reporter = ScheduleReporter(                    # schedule_reporter.py:16
            jsonl_path=report_path,   # 默认 "trace_replay_schedule_report.jsonl"
            enabled=True
       )
            # 构造函数内部将 self.step_idx 初始化为 0（非构造参数，每次 report() 后自增）
```

---

## 二、vLLM 初始化序列

> **对应总览位置：** Phase B 末尾，`LLMEngine.__init__()` 在构建完 `UniProcExecutor`
> 后立即调用的 Worker 生命周期方法序列（`init_device` → `compile_or_warm_up_model`）。
> 本章展示每个方法在插件中的具体实现（均为 no-op）及其存在的必要性。

Worker 注册完成后，vLLM 会按固定顺序调用一系列初始化方法。
在回放模式下，这些方法全部是 **no-op**——它们什么都不做，只打印日志或返回占位值。
这是因为回放 Worker 不需要真正分配 GPU 内存、编译 CUDA kernel 或执行模型预热。

```
MyDummyWorker.init_device()                                 # my_dummy_worker.py:83
    └─ logger.info("MyDummyWorker(rank=0) initialized on cpu")
        # 仅打印日志，无实际设备初始化

MyDummyWorker.load_model(load_dummy_weights=False)          # my_dummy_worker.py:94
    └─ logger.info("MyDummyWorker load_model no-op (replay mode).")
        # 不加载任何模型权重

MyDummyWorker.get_kv_cache_spec()                           # my_dummy_worker.py:100
    └─ return {}
        # 返回空字典，告知 vLLM 此 Worker 不需要 KV Cache 块

MyDummyWorker.determine_available_memory()                  # my_dummy_worker.py:104
    └─ return 0
        # 返回 0，告知 vLLM 无可用硬件内存（不触发内存分析）

MyDummyWorker.initialize_from_config(kv_cache_config)       # my_dummy_worker.py:108
    └─ self.kv_cache_config = kv_cache_config
        # 仅保存引用，不做实际初始化

MyDummyWorker.compile_or_warm_up_model()                    # my_dummy_worker.py:112
    └─ return 0.0
        # 跳过模型编译和 warmup，直接返回 0 耗时
```

---

## 三、Step 0 — Prefill（3 条请求首次进入）

> **对应总览位置：** Phase D 第一次循环，子步骤 D-1（Scheduler 将 3 条请求从
> `waiting` 移入 `scheduled_new_reqs`）→ 子步骤 D-2（★ `MyDummyWorker.execute_model()`
> 介入点 ④ 首次被调用）→ 子步骤 D-3（Scheduler 追加第一个 output token，
> 3 条请求均未到达 `max_tokens`，继续运行）。

Prefill 阶段是 3 条请求的"第一次出现"。vLLM Scheduler 将它们放入
`scheduled_new_reqs`，并附带各自的完整 prompt token 序列。
Worker 需要为每条请求查一次 Trace DB，返回第一个 decode token。

由于这是请求的第一步，`req_last_token` 中还没有该请求的记录。
Worker 会尝试从 `new_req_last_tokens`（prompt 末尾 token）中获取初始上下文，
但在 demo trace 中没有配置 transition map，因此最终总是走**顺序 cursor 回放**路径，
直接取 `step_token_ids[0]`。

```
vLLM Scheduler 生成 SchedulerOutput:
    scheduled_new_reqs         = [Req("0", prompt_token_ids=[...]),
                                  Req("1", prompt_token_ids=[...]),
                                  Req("2", prompt_token_ids=[...])]
        # 3 条请求同时进入 prefill，在一个 batch 中处理
    scheduled_cached_reqs.req_ids = []                      # prefill 阶段无缓存请求
    num_scheduled_tokens = {"0": 6, "1": 4, "2": 3}        # 各自 prompt 的 token 数量
    finished_req_ids     = []                               # 无完成请求

MyDummyWorker.execute_model(scheduler_output)               # my_dummy_worker.py:172
│
├─ _cleanup_finished(scheduler_output)                      # my_dummy_worker.py:136
│   ├─ finished_req_ids = []                                # 没有需要清理的请求
│   └─ return []
│
├─ _extract_req_ids(scheduler_output)                       # my_dummy_worker.py:122
│   │   # 合并三个来源的请求 id，以 num_scheduled_tokens 的顺序为主
│   ├─ new_req_ids     = ["0", "1", "2"]                    # 来自 scheduled_new_reqs
│   ├─ cached_req_ids  = []                                 # 来自 scheduled_cached_reqs
│   ├─ scheduled_keys  = ["0", "1", "2"]                    # 来自 num_scheduled_tokens 的键
│   ├─ ordered = ["0", "1", "2"]                            # 去重后的执行顺序
│   └─ return (["0","1","2"], ["0","1","2"], [])
│
├─ _extract_new_req_last_tokens(scheduler_output)           # my_dummy_worker.py:160
│   │   # 提取每条新请求 prompt 的最后一个 token，作为 transition 查找的初始上下文
│   ├─ req "0": prompt_token_ids[-1] = 9
│   ├─ req "1": prompt_token_ids[-1] = 7
│   ├─ req "2": prompt_token_ids[-1] = 5
│   └─ return {"0": 9, "1": 7, "2": 5}
│
```

下面是对三条请求逐一进行 token 查找的过程。
每条请求都经历相同的逻辑：先确定"当前上下文 token"，再查 Trace DB 取下一个 token。

```
├─ for req_id = "0":
│   │
│   ├─ _resolve_trace_req_id("0")                          # my_dummy_worker.py:143
│   │   │   # 将调度器使用的 req_id 映射到 trace_db 中的 key
│   │   │   # vLLM 有时会给 req_id 加上运行时后缀，如 "req-001-93fdb26d"
│   │   ├─ "0" in self.trace_db.traces → True              # 直接命中，无需剥离后缀
│   │   └─ return "0"
│   │
│   ├─ current_last_token = self.req_last_token.get("0")
│   │       → None  # 首步，req_last_token 中尚无此请求的记录
│   │
│   ├─ current_last_token = new_req_last_tokens.get("0")
│   │       → 9     # 使用 prompt 末尾 token 作为上下文
│   │       # （若仍为 None，则第三级回退到 trace.prompt_last_token）
│   │
│   ├─ trace_db.get_next_token_by_last_token("0", 9)       # trace_db.py:168
│   │   │   # 优先尝试 transition map 查找，失败则顺序 cursor 回放
│   │   ├─ trace = self.traces["0"]
│   │   ├─ 9 in trace.transitions?
│   │   │       → False  # demo trace 无 transition 配置
│   │   └─ fallback → get_next_token_id("0")               # trace_db.py:153
│   │       └─ trace.next_token_id()                       # trace_db.py:23
│   │           ├─ token = step_token_ids[cursor=0] = 1000  # 读取游标位置的 token
│   │           ├─ cursor += 1  → cursor=1                  # 游标前进一位
│   │           └─ return 1000
│   │
│   ├─ sampled_token_ids.append([1000])                    # 加入本次 batch 的输出列表
│   └─ self.req_last_token["0"] = 1000                     # 保存，供下一步使用
│
├─ for req_id = "1":
│       # 逻辑完全相同，current_last_token=7 → transitions 未命中 → cursor=0 → token=1010
│       → sampled_token_ids.append([1010])
│       → self.req_last_token["1"] = 1010
│
├─ for req_id = "2":
│       # current_last_token=5 → transitions 未命中 → cursor=0 → token=1020
│       → sampled_token_ids.append([1020])
│       → self.req_last_token["2"] = 1020
│
├─ reporter.report(                                         # schedule_reporter.py:23
│       scheduled_req_ids=["0","1","2"],
│       new_req_ids=["0","1","2"],
│       cached_req_ids=[],
│       finished_req_ids=[],
│       num_scheduled_tokens={"0":6,"1":4,"2":3}
│   )
│   │   # 将本步调度信息序列化为 JSON 行追加写入 JSONL 文件
│   ├─ record = {                                           # 构建记录字典
│   │       "timestamp": "2026-06-08T...",                  # UTC 时间戳
│   │       "step_idx": 0,                                  # 当前全局步骤号
│   │       "scheduled_req_ids": ["0","1","2"],
│   │       "new_req_ids": ["0","1","2"],
│   │       "cached_req_ids": [],
│   │       "finished_req_ids": [],
│   │       "num_scheduled_tokens": {"0":6,"1":4,"2":3}
│   │   }
│   ├─ logger.info("trace-replay step=0 scheduled=...")     # 同时输出到日志
│   ├─ jsonl_path.open("a").write(json.dumps(record) + "\n")  # 追加写入文件
│   └─ self.step_idx = 1                                    # 步骤计数器自增
│
└─ return ModelRunnerOutput(                                # my_dummy_worker.py:205
       req_ids=["0","1","2"],
       req_id_to_index={"0":0, "1":1, "2":2},             # vLLM 用此映射找到每条请求的结果
       sampled_token_ids=[[1000], [1010], [1020]],         # 每条请求一个 token
       logprobs=None,                                       # 回放模式不计算对数概率
       prompt_logprobs_dict={}                              # 同上
   )
```

**Step 0 结束后的状态快照：**

```
req_last_token = {"0": 1000,  "1": 1010,  "2": 1020}
cursors        = {"0": 1,     "1": 1,     "2": 1   }
reporter.step_idx = 1
```

---

## 四、Step 1–3 — Decode（循环推理）

> **对应总览位置：** Phase D 第 2～4 次循环。每次循环中，Scheduler 将 3 条请求
> 从 `running` 列表放入 `scheduled_cached_reqs`（子步骤 D-1），★ Worker 再次从
> TraceDatabase 取下一个 token（子步骤 D-2），Scheduler 追加 token 并检查停止条件
> （子步骤 D-3）。Step 3 执行完毕后，3 条请求的 `output_token_ids` 均达到
> `max_tokens=4`，被标记为 `FINISHED`，下一次 schedule() 时放入 `finished_req_ids`。

Decode 阶段是 vLLM 的主循环部分。每一步，Scheduler 会将所有还在生成中的请求
放入 `scheduled_cached_reqs`，表示它们已经完成了 Prefill，正在逐 token 续写。

与 Prefill 的关键区别在于：
- `req_last_token` 中已有记录，直接用作 transition 查找的上下文
- cursor 从上步结束的位置继续向前推进
- 由于三条请求进度完全同步，每步 token 差值保持固定（req "1" 比 "0" 大 10，"2" 比 "0" 大 20）

下面以 Step 1 为例展示完整调用链，Step 2、Step 3 以相同逻辑循环执行。

```
vLLM Scheduler 生成 SchedulerOutput:
    scheduled_new_reqs         = []                        # 无新请求
    scheduled_cached_reqs.req_ids = ["0", "1", "2"]        # 3 条请求继续 decode
    num_scheduled_tokens       = {"0": 1, "1": 1, "2": 1}  # 每条请求本步只处理 1 个 token
    finished_req_ids           = []

MyDummyWorker.execute_model(scheduler_output)               # my_dummy_worker.py:172
│
├─ _cleanup_finished(...)
│       → []   # 无完成请求，跳过
│
├─ _extract_req_ids(...)
│   │   # scheduled_cached_reqs 的请求会合并进 ordered 列表
│   └─ return (["0","1","2"], [], ["0","1","2"])
│           # (ordered, new_req_ids, cached_req_ids)
│
├─ _extract_new_req_last_tokens(...)
│       → {}   # 无新请求，返回空字典
│
├─ for req_id = "0":
│   ├─ _resolve_trace_req_id("0")  → "0"                  # 直接命中
│   │
│   ├─ current_last_token = self.req_last_token.get("0")
│   │       → 1000  # 上一步（Step 0）保存的值，直接使用
│   │
│   ├─ trace_db.get_next_token_by_last_token("0", 1000)
│   │   ├─ 1000 in trace.transitions? → False              # 仍无 transition
│   │   └─ get_next_token_id("0")
│   │       └─ trace.next_token_id()
│   │           ├─ token = step_token_ids[cursor=1] = 1001  # cursor 从上步的 1 开始
│   │           ├─ cursor += 1 → cursor=2
│   │           └─ return 1001
│   │
│   ├─ sampled_token_ids.append([1001])
│   └─ self.req_last_token["0"] = 1001                     # 更新，供 Step 2 使用
│
├─ for req_id = "1":  → token=1011, cursor 1→2, req_last_token["1"]=1011
├─ for req_id = "2":  → token=1021, cursor 1→2, req_last_token["2"]=1021
│
├─ reporter.report(step_idx=1, cached_req_ids=["0","1","2"], ...)
│       → step_idx = 2
│
└─ return ModelRunnerOutput(sampled_token_ids=[[1001],[1011],[1021]], ...)
```

**Step 2、Step 3 依此类推：**

```
Step 2: cursor 2→3, tokens = [1002, 1012, 1022]
Step 3: cursor 3→4, tokens = [1003, 1013, 1023]
        # Step 3 是最后一个 decode 步骤，cursor 到达 step_token_ids 末尾
```

---

## 五、Step 4 — Finish（max_tokens 到达，清理）

> **对应总览位置：** Phase D 第 5 次循环（也是最后一次）。此时 Scheduler 的
> `finished_req_ids = {"0","1","2"}`，子步骤 D-1 生成一个 `scheduled_new_reqs=[]`、
> `scheduled_cached_reqs=[]`、`finished_req_ids=["0","1","2"]` 的 SchedulerOutput。
> 子步骤 D-2 中 ★ Worker 的 `_cleanup_finished()` 调用 `trace.reset()`，
> `req_ids` 为空，直接返回空 `ModelRunnerOutput`。子步骤 D-3 无新 token 可处理，
> `_run_engine()` 检测到所有请求完成后退出循环，进入 Phase E。

当所有请求的 decode token 数量达到 `max_tokens=4` 后，vLLM Scheduler 将它们
标记为完成，并在下一次 `execute_model()` 调用时通过 `finished_req_ids` 通知 Worker。

Worker 在此步只做清理工作：
- 调用 `trace.reset()` 将 cursor 归零，以便同一 request_id 下次可以重用
- 从 `req_last_token` 中移除该请求的记录，释放内存
- 因为 `req_ids` 为空（没有需要生成 token 的请求），直接返回空输出

这一步体现了插件的**无状态设计**：完成清理后，TraceDatabase 回到初始状态，
可以接受下一批相同 request_id 的请求。

```
vLLM Scheduler 生成 SchedulerOutput:
    scheduled_new_reqs         = []
    scheduled_cached_reqs.req_ids = []
    num_scheduled_tokens       = {}                        # 无需处理 token
    finished_req_ids           = ["0", "1", "2"]           # 三条请求全部完成

MyDummyWorker.execute_model(scheduler_output)               # my_dummy_worker.py:172
│
├─ _cleanup_finished(scheduler_output)                      # my_dummy_worker.py:136
│   │   # 对每条完成的请求执行清理
│   ├─ finished_req_ids = ["0", "1", "2"]
│   │
│   ├─ for req_id = "0":
│   │   ├─ _resolve_trace_req_id("0") → "0"
│   │   ├─ trace_db.finish_request("0")                    # trace_db.py:195
│   │   │   └─ traces["0"].reset()                         # trace_db.py:31
│   │   │       ├─ self.cursor = 0                          # 游标归零，可重用
│   │   │       └─ self.transition_cursor = {}              # 清空 transition 使用记录
│   │   └─ self.req_last_token.pop("0")                    # 移除运行时状态
│   │           # req_last_token 中不再有 "0" 的记录
│   │
│   ├─ for req_id = "1": finish_request("1"), req_last_token.pop("1")
│   ├─ for req_id = "2": finish_request("2"), req_last_token.pop("2")
│   └─ return ["0", "1", "2"]                              # 返回已清理的 id 列表
│
├─ _extract_req_ids(...)
│       → ([], [], [])  # 所有列表均为空
│
├─ req_ids = []
│       → 触发提前返回逻辑（my_dummy_worker.py:177-178）
│
└─ return ModelRunnerOutput(req_ids=[], req_id_to_index={})
        # 空输出，vLLM 不会将此结果分发给任何请求
```

**Step 4 结束后的状态快照（全部归零）：**

```
req_last_token = {}                   # 所有请求已清理
cursors        = {"0": 0, "1": 0, "2": 0}  # trace.reset() 后归零
reporter.step_idx = 5                 # 继续累加，即使是空步骤也会记录
```

---

## 六、全流程状态追踪表

> **对应总览位置：** Phase D 子步骤 D-2 执行完毕后，★ Worker 内部的
> `req_last_token`、`RequestTrace.cursor` 与 `ScheduleReporter.step_idx`
> 在各 step 后的变化。

每列展示一步执行完成后的系统状态。
`cursor` 表示 `RequestTrace.cursor`（下一次 `next_token_id()` 会读取的位置索引）。

```
                Step 0        Step 1        Step 2        Step 3        Step 4
                (Prefill)     (Decode 1)    (Decode 2)    (Decode 3)    (Finish)
─────────────────────────────────────────────────────────────────────────────────────
req_last_token:
  "0"           1000          1001          1002          1003          (已 pop)
  "1"           1010          1011          1012          1013          (已 pop)
  "2"           1020          1021          1022          1023          (已 pop)

cursor（step 后）:
  "0"           0 → 1         1 → 2         2 → 3         3 → 4         reset → 0
  "1"           0 → 1         1 → 2         2 → 3         3 → 4         reset → 0
  "2"           0 → 1         1 → 2         2 → 3         3 → 4         reset → 0

reporter.step_idx（step 后）:
                0 → 1         1 → 2         2 → 3         3 → 4         4 → 5

sampled_token_ids:
  [req "0"]     [1000]        [1001]        [1002]        [1003]        —
  [req "1"]     [1010]        [1011]        [1012]        [1013]        —
  [req "2"]     [1020]        [1021]        [1022]        [1023]        —

SchedulerOutput 来源:
  "0"           new           cached        cached        cached        finished
  "1"           new           cached        cached        cached        finished
  "2"           new           cached        cached        cached        finished
```

---

## 附：关键数据结构速查

### RequestTrace（trace_db.py:13）

```python
@dataclass
class RequestTrace:
    request_id: str           # trace 的唯一标识，对应 scheduler 中的 req_id
    step_token_ids: list[int] # 预录 token 序列，顺序回放的数据源
    step_logits: list[Tensor] # 可选，logits 列表（未存则为空）
    prompt_last_token: int | None  # 可选，prompt 末尾 token（用于初始 transition 查找）
    transitions: dict[int, list[int]]  # token 转移图：cur_token → [next_token, ...]
    transition_cursor: dict[int, int]  # 记录每个 cur_token 已使用了几次 transition
    hidden_states: torch.Tensor | None = None  # 可选，预录隐藏状态（未存则为 None）
    cursor: int = 0           # 顺序回放游标，指向下一个待返回的 token 位置
```

### Token 查找优先级（get_next_token_by_last_token）

```
输入: request_id, current_last_token
  │
  ├─ 1. traces[request_id] 不存在?
  │       strict_missing=True  → 抛 KeyError
  │       strict_missing=False → 返回 fallback_token_id (默认 0)
  │
  ├─ 2. current_last_token in transitions?
  │       是 → 取 transitions[current_last_token][transition_cursor[token]]
  │            transition_cursor[token] += 1（防止重复返回同一后继）
  │            越界时取最后一个（饱和处理）
  │
  └─ 3. Fallback：顺序 cursor 回放
          step_token_ids[cursor]，cursor += 1
          cursor 越界 → 返回 step_token_ids[-1]（最后一个 token 循环）
          step_token_ids 为空 → 返回 fallback_token_id
```

### 生成的 JSONL 报告格式（每步一行）

```json
{
  "timestamp": "2026-06-08T07:30:00.123456+00:00",
  "step_idx": 0,
  "scheduled_req_ids": ["0", "1", "2"],
  "new_req_ids": ["0", "1", "2"],
  "cached_req_ids": [],
  "finished_req_ids": [],
  "num_scheduled_tokens": {"0": 6, "1": 4, "2": 3}
}
```
