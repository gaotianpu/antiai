# Python 常驻进程设计开发原则

> 适用场景：asyncio 长期运行的后台服务（WebSocket 数据管道、消息消费、定时调度）
> 经验来源：`src/pipeline/realtime.py` 设计、开发、审计全过程

---

## 原则速查

| # | 原则 | 阶段 | 一句话 |
|---|------|------|--------|
| 1 | 启动即崩溃 | 启动 | 初始化失败必须 raise，不允许带病运行 |
| 2 | 故障自愈 | 运行 | 每个 I/O 失败都有重试路径，每条数据都有最终归宿 |
| 3 | 数据完整 | 运行 | 不静默丢数据；写入幂等；断线可补缺 |
| 4 | 背压控制 | 运行 | 生产快于消费时，有明确的丢弃/降级策略 |
| 5 | 资源有界 | 长期运行 | 所有集合/队列/缓存设上限，长期运行不泄漏 |
| 6 | 监督重启 | 进程容错 | 任务崩溃自动恢复，窗口内超限则放弃 |
| 7 | 可观测性 | 贯穿全程 | 日志充分可追溯，告警及时可行动 |
| 8 | 优雅关闭 | 关闭 | 信号 → 排空 → 清理，不丢正在处理的数据 |

---

## 原则一：启动即崩溃 — 初始化失败不允许带病运行

生命起点。配置错误、连接失败等不可恢复问题在启动阶段必须终止进程，禁止静默绕过。

```python
# 违宪：初始化失败用 log + return 绕过
def __init__(self):
    try:
        self.config = load_config()
    except Exception:
        logger.error("config load failed")
        return  # 进程继续运行，状态残缺

# 合宪：crash early, crash loud
def __init__(self):
    try:
        self.config = load_config()
    except Exception as e:
        raise ConfigError(f"failed to load config: {e}") from e
```

- `__init__` / `_initialize` 调用链中的不可恢复故障 → `raise`
- 运行时方法中的可恢复故障 → `log + 降级 + 告警`
- 数据库连接失败在初始化阶段 → 必须 raise

---

## 原则二：故障自愈 — 每条 I/O 路径都有退路

常驻进程不能因为一次网络抖动就永久卡死或退出。

### 2.1 重试层级

```
                      ┌─ 成功 → 结束
一次 I/O 失败 ──┤
                      └─ 失败 → 重试队列 ──┬─ 重试成功 → 结束
                                           │
                                           └─ 耗尽重试 → 死信持久化
```

**正例**（`RetryWorker`）：
```python
# 独立重试队列，不阻塞消费热循环
class RetryWorker:
    async def _retry(self, df, attempt):
        if attempt > 1:
            await asyncio.sleep(RETRY_BACKOFF[attempt])
        try:
            await repo.insert(df, timeframe)
        except Exception:
            if attempt < RETRY_MAX_RETRIES:
                await self._retry(df, attempt + 1)  # 递归重试
            else:
                await self._write_dead_letter(df, "max_retries_exhausted")
```

**反例**：
```python
# 一次失败就丢弃，数据静默丢失
try:
    await repo.insert(df)
except Exception:
    pass  # 违宪
```

### 2.2 死信队列

当所有重试耗尽后，数据必须落盘而不是丢弃：

- 格式：JSONL（每行一条，追加写，便于恢复和人工排查）
- 路径：`data/dead_letter_{topic}.jsonl`
- 恢复：见 §3.4 数据回放

### 2.3 什么该重试

| 场景 | 策略 |
|------|------|
| 网络超时、连接断开 | 重试 + 指数退避 |
| 服务端 503/429 | 重试 + Retry-After |
| 业务逻辑错误、认证失败 | 不重试，直接告警 |

---

## 原则三：数据完整 — 不静默丢数据

故障自愈保证了"路径存在"，数据完整保证"过程中数据不丢失"。

### 3.1 写入必须检查返回值

```python
# 违宪：忽略返回值
await repo.insert(df)

# 合宪：检查结果
ret, msg = await repo.insert(df)
if not ret:
    raise WriteError(msg, df=df)
```

### 3.2 幂等写入

使用 `INSERT ... ON DUPLICATE KEY UPDATE`（upsert），支持安全重放。

### 3.3 断线补缺

- WS 断线重连后，标记 `needs_gap_fill`
- 消费者从 DB 查 `MAX(ts)` 确定缺口起点
- REST 分页补全缺失数据
- 缺口过大（> 24h）时告警，建议手动跑 history pipeline

### 3.4 数据回放

死信文件在进程启动时自动回放：读取 JSONL → 逐行 upsert → 成功则删除文件，部分失败则保留失败行。

---

## 原则四：背压控制 — 生产快于消费时有明确策略

流量过载保护。不因外部数据涌入导致内存撑爆或消费积压。

### 4.1 队列溢出

```python
# 方案 A：丢弃最旧（实时数据场景）
try:
    queue.put_nowait(msg)
except asyncio.QueueFull:
    queue.get_nowait()  # 丢弃最旧
    queue.put_nowait(msg)

# 方案 B：丢弃最新（不可丢数据场景）
try:
    queue.put_nowait(msg)
except asyncio.QueueFull:
    logger.warning("queue_full_drop_new")
```

### 4.2 消费慢检测

不应仅依赖单一条件。独立检测消费延迟：

```python
# 连续 N 次采样队列深度 > 阈值 → 告警
if consecutive_high_water >= 3:
    alert("slow_consumer")
```

### 4.3 并发控制

- 外部 API 调用用 `asyncio.Semaphore` 控制并发度
- DB 写入用 `asyncio.Semaphore` + 线程池
- 聚合任务用 `BoundedSemaphore` 防止创建过多 task

---

## 原则五：资源有界 — 长期运行不泄漏

进程跑上数周后，内存曲线应保持平坦。

### 5.1 集合/缓存

| 类型 | 风险 | 防护 |
|------|------|------|
| `dict` 缓存 | 无界增长 | maxsize + TTL + 定期淘汰 |
| `asyncio.Queue` | 内存撑爆 | 必须设 `maxsize` |
| `set` 任务引用 | hung task 永不回收 | `add_done_callback` + 定期清理 |

**正例**：
```python
# 有界缓存
class Cache:
    MAXSIZE = 10000
    TTL = 1800

    def set(self, key, value):
        self._data[key] = (value, time.time() + self.TTL)
        if len(self._data) > self.MAXSIZE:
            self._evict_expired()

# 有界队列
queue = asyncio.Queue(maxsize=4096)
```

**反例**：
```python
queue = asyncio.Queue()  # maxsize=0 = 无上限 → 内存泄露
```

### 5.2 连接/会话

- `aiohttp.ClientSession` / DB 连接池在进程生命周期内复用，不在每次请求时创建
- 关闭必须在同一 event loop 中执行（不能用 `asyncio.run(client.close())` 跨 loop 关闭）
- SQLAlchemy `pool_pre_ping=True` + `pool_recycle=3600` 防止连接被服务端断开后复用

### 5.3 同步 I/O 隔离

长时间运行的协程中，同步文件 I/O 会阻塞整个事件循环：

```python
# 违宪：同步写入阻塞事件循环
with open(path, "a") as f:
    f.write(data)

# 合宪：投递到线程池
await asyncio.to_thread(_sync_write, path, data)
```

---

## 原则六：监督重启 — 任务崩溃不导致进程退出

进程级的最后防线。单个 task 崩了不应该拖垮整个进程。

### 6.1 监督器模式

每个后台任务包装在监督器中：崩溃 → 等待 → 重启，窗口内超限则放弃。

```python
async def _supervised(self, coro, name):
    crashes = []
    while not self._shutdown_requested.is_set():
        try:
            await coro()
            return  # 正常退出
        except asyncio.CancelledError:
            raise
        except Exception:
            now = time.monotonic()
            crashes = [t for t in crashes if now - t < WINDOW]
            crashes.append(now)
            logger.error(f"task_crashed, name={name}")
            if len(crashes) >= MAX_CRASHES:
                alert.send_critical(f"crash_limit_reached, name={name}")
                raise  # 放弃，让上层决策
            await asyncio.sleep(RESTART_DELAY)
```

### 6.2 参数建议

| 参数 | 建议值 | 理由 |
|------|--------|------|
| 重启延迟 | 5s | 避免瞬时故障时疯狂重启刷日志 |
| 崩溃窗口 | 300s (5min) | 短窗口内多次崩溃说明非瞬时故障 |
| 窗口内上限 | 5 次 | 超过则放弃，等待外部干预（systemd/supervisor 进程级重启） |

### 6.3 监督覆盖

所有后台 task 都应监督，包括：
- WebSocket 连接循环
- 消息消费循环
- 定时维护/检测循环
- 重试 worker

---

## 原则七：可观测性 — 出问题能快速定位

贯穿整个生命周期。没有可观测性的常驻进程是盲飞。

### 7.1 日志原则

| 级别 | 用途 | 必须包含 |
|------|------|---------|
| `error` | 需要人工介入 | traceback + 关键上下文 |
| `warning` | 可恢复异常 | 重试次数 + 降级行为 |
| `info` | 关键决策点 | 输入 → 决策 → 结果 |
| `debug` | 排查细节 | 循环中间状态、API 原始返回 |

- 日志用 `key=value` 格式，不用散文
- 每条日志携带可定位上下文（coin、timeframe、label）
- 高频循环日志降级为 `debug`，仅状态变更时 `info`

### 7.2 告警原则

| 级别 | 通道 | 触发条件 |
|------|------|---------|
| CRITICAL | TG critical_id | 资金/数据安全、进程即将退出 |
| WARNING | TG warning_id | 异常但可继续运行 |
| INFO | TG heartbeat_id | 例行状态同步 |

- 同场景同级别必须防抖（300s 冷却）
- 消息格式：`server:场景标识, key1=value1, ...`
- 告警内容需支持独立排查（无需翻阅日志即可决策）
- 通知和日志分离：通知 = 结论，日志 = 完整上下文

### 7.3 健康检查

周期性自检并暴露状态：

```python
async def health(self):
    return await self._db.execute("SELECT 1")  # DB 连通性
```

- DB 写入失败累计超阈值 → CRITICAL
- 队列深度持续 > 80% → WARNING
- 所有 WS 无消息 + REST 不可达 → WARNING（维护检测）

---

## 原则八：优雅关闭 — 不丢正在处理的数据

生命终点。信号 → 排空 → 清理，确保进程退出时数据完整落盘。

### 8.1 标准关闭序列

```
SIGTERM/SIGINT
  │
  ├─ 1. 设置关闭标志 → 生产者停止接收新数据
  ├─ 2. 排空处理中队列（设 timeout，超时则写死信）
  ├─ 3. 等待异步任务完成（设 timeout）
  ├─ 4. 关闭连接池（DB、HTTP session）
  └─ 5. 退出
```

**正例**（`RealtimePipeline`）：
```python
async def shutdown(self, sig):
    self._shutdown_requested.set()          # 通知所有循环退出
    for stream in self.streams.values():
        stream.running = False              # 停止接收新消息
    for worker in self._retry_workers.values():
        worker._running = False             # 停止重试

async def _cleanup(self):
    for stream in self.streams.values():
        await asyncio.wait_for(
            self._drain_queue(stream),      # 排空残留消息
            timeout=GRACE_PERIOD
        )
    await self.repo.close()                  # 关闭连接
    await self.rest.close()
```

### 8.2 关闭标志设计

- 生产循环：`while self.running` — 控制外部数据流入
- 定时循环：`while not self._shutdown_requested.is_set()` — 控制内部定时任务
- 注意 `asyncio.sleep()` 后的二次检查（sleep 期间可能已设置标志）

### 8.3 取消传播

```python
# CancelledError 必须 re-raise，不能吞掉
try:
    await coro()
except asyncio.CancelledError:
    cleanup()
    raise  # 必须！
```

---

## 开发检查清单

### 设计阶段

- [ ] 每条数据的完整生命周期：接收 → 处理 → 持久化 → 失败回退？
- [ ] 每个 I/O 调用：有无 timeout？有无重试？重试耗尽后去哪里？
- [ ] 每个后台 task：是否被监督？崩溃后能否自动重启？
- [ ] 关闭路径：是否有信号处理？是否排空队列？是否关闭连接？
- [ ] 所有集合/队列/缓存：是否设了上限？

### 实现阶段

- [ ] 同步 I/O 已包裹 `asyncio.to_thread`？
- [ ] `CancelledError` 被捕获后 re-raise？
- [ ] `asyncio.gather` 用了 `return_exceptions=True`？
- [ ] 硬编码数值常量已提取到 `constants.py`？
- [ ] 函数 < 50 行，文件 < 300 行？

### 测试阶段

- [ ] 故障注入：DB 断开能否自愈？WS 断线能否重连补缺？
- [ ] 背压测试：高速生产下队列不撑爆内存？
- [ ] 关闭测试：SIGTERM 后数据是否落盘？连接是否释放？
- [ ] 长期运行：24h 内存无增长？task 数量稳定？

---

## 关联文档

- `message_guid.md` — 通知/告警原则
- `async_guid.md` — asyncio 最佳实践
- `logging_guid.md` — 日志原则
- `memory_optimization_guid.md` — 内存优化
- `unix_pythonic_guid.md` — 架构宪法
- `unit_test_guid.md` — 测试原则
- `docs/specs/fault_recovery_design.md` — 本次 realtime 故障自愈设计
- `docs/analysis/fault_recovery_code_audit.md` — 本次实现合规审查
