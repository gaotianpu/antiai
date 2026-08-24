# Python Async/Await 生产级最佳实践

> 适用场景：asyncio 高并发交易系统、网络服务、数据管道
> 合宪性声明：本文遵循"策略与机制分离"原则 — 模式是机制，具体业务逻辑选择哪种模式是策略

---

## 1. 核心模式

### 1.1 async/await 基础 — 何时使用

**用 async/await 的场景：**
- 网络 I/O（HTTP API 调用、WebSocket、gRPC）
- 文件 I/O（aiofiles）
- 数据库查询（asyncpg、databases）
- 并发执行多个 I/O 操作

**不要用 async 的场景：**
- CPU 密集型计算（用 `concurrent.futures.ProcessPoolExecutor` 或移到独立进程）
- 纯内存操作（同步函数即可，强行 async 徒增开销）

```python
# 好：I/O 密集，并发发 HTTP 请求
async def fetch_all(urls: list[str]) -> list[dict]:
    async with httpx.AsyncClient() as client:
        tasks = [client.get(url) for url in urls]
        return [r.json() for r in await asyncio.gather(*tasks)]

# 坏：CPU 计算阻塞事件循环
async def compute_heavy(data: list[int]) -> int:
    return sum(x ** 2 for x in data)  # 应改用 run_in_executor
```

**判断准则：** 如果你在函数体内看到了 `await`，那它必须是 `async def`。如果你需要 `asyncio.sleep()` 或网络 I/O，用 async。数据结构转换、配置解析、纯计算 — 用同步。

### 1.2 Task 管理

三种创建并发的核心 API 各有适用场景：

| API | 使用时机 | 异常行为 | 返回顺序 |
|-----|---------|---------|---------|
| `asyncio.create_task()` | 启动"即发即忘"后台任务 | 未捕获异常在 task 被 await/gather 时抛出 | N/A |
| `asyncio.gather()` | 并发执行一组协程，需要全部结果 | `return_exceptions=True` 时异常对象放入结果列表 | 保持传入顺序 |
| `asyncio.wait()` | 需精细控制（FIRST_COMPLETED / FIRST_EXCEPTION） | 异常在 done/pending 集合中 | 需要手动取结果 |

```python
# Pattern A：create_task — 结构化并发（参考 6.3）
async def run_strategies(strategies: list[Strategy]) -> None:
    async with asyncio.TaskGroup() as tg:
        for s in strategies:
            tg.create_task(s.run())

# Pattern B：gather — 并发获取批量数据（当前项目: apps/cta.py:44-49）
async def execute_all_symbols(symbols: dict) -> None:
    tasks = [execute_symbol(sym, cfg) for sym, cfg in symbols.items()]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for sym, result in zip(symbols, results):
        if isinstance(result, Exception):
            logger.error(f"{sym} failed: {result}")

# Pattern C：wait — 等待最快完成者
async def race_fetch(urls: list[str]) -> str:
    tasks = {asyncio.create_task(fetch(u)): u for u in urls}
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    return await done.pop()
```

**注意：** `asyncio.create_task()` 创建的 task 持有对协程的强引用。如果 task 不被 `await` 也不被存储，GC 可能会意外取消它。对于"即发即忘"的后台任务，务必保存 task 引用（如 `TaskScheduler._background_tasks`，见 `common/task_scheduler.py:11`）。

### 1.3 事件循环管理 — 入口点铁律

```python
# 铁律：asyncio.run() 只在进程入口调用一次
def main():
    asyncio.run(run_app())    # Python 3.7+

# 禁止：嵌套的事件循环
async def bad_nested():
    asyncio.run(inner())      # RuntimeError: asyncio.run() cannot be called from a running event loop

# 禁止：手动创建事件循环（除非你清楚后果）
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
loop.run_until_complete(main())
```

当前项目 `trade_os.py:142,171` 已正确遵守此铁律。

**循环策略：** 仅在 Windows + `ProactorEventLoop` 或需要 uvloop 时设置策略，Linux 默认 `SelectorEventLoop` 足够。

```python
# 性能优化（可选）：Linux 上替换为 uvloop
import uvloop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
```

### 1.4 优雅关闭

生产系统必须处理 SIGTERM（K8s pod 驱逐）和 SIGINT（Ctrl+C）。

```python
import signal
import asyncio

async def shutdown(loop, tasks: dict[str, asyncio.Task], timeout: float = 30.0):
    """优雅关闭：取消任务 → 等待完成 → 清理资源"""
    logger.info("Shutting down...")

    # Step 1: 发送取消信号给所有后台任务
    for name, task in tasks.items():
        task.cancel()
        logger.info(f"Cancelled {name}")

    # Step 2: 等待任务响应取消（给它们清理时间）
    pending = list(tasks.values())
    if pending:
        await asyncio.wait(pending, timeout=timeout)
        # Step 3: 超时未完成的强制回收（不做额外处理，让 loop 关闭）
        still_running = [t for t in pending if not t.done()]
        if still_running:
            logger.warning(f"{len(still_running)} tasks did not finish in time")

    # Step 4: 关闭连接池
    await session.aclose()

async def main():
    loop = asyncio.get_running_loop()
    tasks: dict[str, asyncio.Task] = {}

    # 注册信号处理器
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(
            sig,
            lambda: asyncio.create_task(shutdown(loop, tasks))
        )

    # 启动业务任务
    tasks["worker"] = asyncio.create_task(run_worker())
    await asyncio.gather(*tasks.values())
```

**当前项目存在风险点：**
- `trade_os.py` 没有信号处理器，Ctrl+C 会触发 `KeyboardInterrupt` 经 `asyncio.run()` 传播，导致所有 task 被销毁且跳过 `finally` 块（Python < 3.11）
- Python 3.11+ 中 `asyncio.run()` 已改进，但显式信号处理仍是生产最佳实践
- `TaskScheduler.cancel_all_tasks()` 存在但未在关闭路径中调用

**推荐改进：** 在 `TradeOs.run()` 中注册信号 handler，调用 `TaskScheduler.cancel_all_tasks()` → `wait_for_tasks()` → 关闭 AccountManager。

---

## 2. 常见陷阱

### 2.1 阻塞事件循环 — 头号杀手

```python
# 致命：在 async 函数中调用同步 I/O
async def bad_fetch():
    data = requests.get("https://api.example.com")  # 阻塞整个事件循环！
    return data.json()

# 修复 A：用异步库
async def good_fetch():
    async with httpx.AsyncClient() as client:
        r = await client.get("https://api.example.com")
        return r.json()

# 修复 B：扔进线程池（仅当没有异步替代时）
async def ok_fetch():
    loop = asyncio.get_running_loop()
    r = await loop.run_in_executor(None, requests.get, "https://api.example.com")
    return r.json()

# 修复 C：CPU 密集型扔进进程池
async def heavy_compute(data):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, cpu_bound_func, data)
```

**常见阻塞源：**
- `time.sleep()` → 改用 `await asyncio.sleep()`
- `open()` / `read()` → 改用 `aiofiles`
- `requests` → 改用 `httpx` 或 `aiohttp`
- `sqlite3` → 改用 `aiosqlite`
- `pandas.read_csv()` → 用 `run_in_executor`

**诊断工具：**
```python
# 开启 asyncio debug 模式检测慢回调
import asyncio
asyncio.get_event_loop().slow_callback_duration = 0.1  # 超过 100ms 打印警告
```

**当前项目信号：** `signals/fetcher.py:173` 的 `fetch_and_cache_all_signals()` 是同步方法（可能内部有网络 I/O），在 `_fetch_loop` 中被直接调用，若其内部有阻塞 I/O 会阻塞事件循环。应改为异步或 `run_in_executor`。

### 2.2 asyncio.Lock 死锁

```python
# 致命：在持有锁期间 await，形成死锁
lock = asyncio.Lock()

async def deadlock():
    async with lock:
        result = await another_coro()  # another_coro 也试图获取 lock？
        # 或者 await 了一个永不返回的 future
```

**问题根因：**
1. 同一个协程重入同一个锁（asyncio.Lock 不可重入）
2. 持有锁时 `await` 了一个需要获取同一锁的操作
3. 持有锁时 `CancelledError` 被吞掉，锁永不被释放

```python
# 安全模式：最小化锁持有时长
async def safe():
    data = await fetch_data()           # 锁外做 I/O
    async with lock:
        shared_state.update(data)       # 锁内只做内存操作

# 超时获取锁
try:
    await asyncio.wait_for(lock.acquire(), timeout=5.0)
    try:
        # 临界区
        ...
    finally:
        lock.release()
except asyncio.TimeoutError:
    logger.warning("Could not acquire lock within 5s")

# Python 3.11+ 可以用 async with asyncio.timeout(5.0):
```

**当前项目：** 未使用 asyncio.Lock，使用了文件锁互斥（FRA alignment 和 fix_exposure 之间）。文件锁在 async 上下文中安全，因为它是 `fcntl.flock` / `msvcrt.locking` 级别的 OS 锁，不做跨协程等待。

### 2.3 CancelledError 处理

```python
# 致命：吞掉 CancelledError
async def bad_worker():
    try:
        while True:
            await asyncio.sleep(1)
    except Exception:       # CancelledError 继承自 BaseException，不是 Exception！
        pass                # 这句话永远不会执行（Python < 3.9）
    # Python 3.9+：可以捕获 Exception，但仍不推荐

# 正确：明确处理 CancelledError
async def good_worker():
    try:
        while True:
            await do_work()
    except asyncio.CancelledError:
        logger.info("Worker cancelled, cleaning up...")
        await cleanup()
        raise   # 必须重新抛出！否则任务不会被标记为已取消
    finally:
        await release_resources()

# 反模式：在 finally 中做重量级操作
async def bad_finally():
    try:
        await work()
    finally:
        await asyncio.sleep(5)  # 阻止取消传播，导致关闭延迟
```

**当前项目正确示例：** `signals/fetcher.py:192-193` 正确捕获并处理了 `CancelledError`。

```python
except asyncio.CancelledError:
    break   # 退出循环，允许关闭
```

### 2.4 未被观察的 Task 异常 — 静默失败

```python
# 危险：task 异常无人观察
async def main():
    task = asyncio.create_task(buggy_worker())  # buggy_worker 抛出异常
    await asyncio.sleep(10)                     # 异常在 task 里"闷烧"
    # 直到 task 被 GC 时打印 "Task exception was never retrieved"

# 修复 A：gather 观察所有异常
async def main():
    tasks = [asyncio.create_task(work()) for _ in range(5)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for r in results:
        if isinstance(r, Exception):
            logger.error(f"Task failed: {r}")

# 修复 B：注册回调
task = asyncio.create_task(work())
task.add_done_callback(lambda t: logger.error(t.exception()) if t.exception() else None)

# 修复 C：Python 3.11+ TaskGroup（最推荐）
async def main():
    try:
        async with asyncio.TaskGroup() as tg:
            for _ in range(5):
                tg.create_task(work())  # 任何异常立即取消所有兄弟任务
    except* ExceptionGroup as eg:
        logger.error(f"Tasks failed: {eg}")
```

**当前项目状态：** `orchestration/app_runner.py:63` 使用 `return_exceptions=True`，这是正确的防护措施。

### 2.5 协程饥饿

```python
# 饥饿场景：一个协程 CPU 密集、从不 await
async def hog():
    while True:
        x = sum(range(10**7))  # CPU 密集，无 await，永不交出控制权

# 修复：主动让出控制权
async def polite():
    for i in range(10):
        x = sum(range(10**6))
        await asyncio.sleep(0)  # 在每个小批量后让出

# 或者在 executor 中运行
async def offloaded():
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, cpu_heavy)
```

### 2.6 同步/异步代码混用

```python
# 致命：从同步代码调用 async 函数
def sync_caller():
    result = async_function()  # 返回 coroutine object，不是结果！
    # 静默 bug：result 是 <coroutine object>，永远不是期望的值

# 修复 A：确保调用链全程 async
async def entry():
    result = await async_function()

# 修复 B：在同步代码中创建临时 loop（仅在测试/脚本中）
def sync_caller():
    return asyncio.run(async_function())

# 修复 C：对象构造时不做异步初始化（反模式）
class BadService:
    def __init__(self):
        self.data = None

    async def init(self):  # 两步初始化，调用者容易忘记
        self.data = await fetch_data()

# 用工厂函数代替
async def create_service() -> Service:
    data = await fetch_data()
    return Service(data)
```

### 2.7 协程 vs 协程函数的混淆

```python
# 陷阱：await 了一个函数调用而不是协程
async def fetch():
    return "data"

async def main():
    # result = await fetch   # 错误：fetch 是函数，不是协程
    # result = await fetch() # 正确：fetch() 返回协程
    pass

# 陷阱：asyncio.create_task 接收协程，不小心传入函数
async def main():
    # task = asyncio.create_task(fetch)      # 错误
    task = asyncio.create_task(fetch())      # 正确
```

---

## 3. 性能与并发控制

### 3.1 asyncio.Semaphore — 速率限制

```python
# 无限制并发：一次发 10000 个请求 → 被 API 限流
semaphore = asyncio.Semaphore(10)  # 最多 10 个并发

async def rate_limited_fetch(url: str) -> dict:
    async with semaphore:
        return await client.get(url)

# 批量并发 + 限流
async def fetch_all(urls: list[str], concurrency: int = 10) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    async def _fetch(url):
        async with sem:
            return await client.get(url)
    tasks = [_fetch(u) for u in urls]
    return await asyncio.gather(*tasks)

# 动态限流：根据 API 响应头调整
class DynamicRateLimiter:
    def __init__(self, max_concurrency: int = 10):
        self._sem = asyncio.Semaphore(max_concurrency)

    async def __aenter__(self):
        await self._sem.acquire()

    async def __aexit__(self, *args):
        self._sem.release()
```

**当前项目应用场景：** CTA 多交易对并行下单、信号拉取对 gRPC 服务器的并发控制。当前无限流，若 symbols 数量暴增可能打爆对端。

### 3.2 asyncio.Queue — 生产者-消费者

```python
async def producer(queue: asyncio.Queue, source):
    """从信号源读取，放入队列"""
    while True:
        item = await source.next()
        await queue.put(item)
        if done:
            break

async def consumer(queue: asyncio.Queue, handler):
    """从队列取出，处理交易"""
    while True:
        item = await queue.get()
        try:
            await handler.process(item)
        finally:
            queue.task_done()

async def pipeline():
    queue = asyncio.Queue(maxsize=100)
    producers = [asyncio.create_task(producer(queue, s)) for s in sources]
    consumers = [asyncio.create_task(consumer(queue, h)) for h in handlers]
    await asyncio.gather(*producers)
    await queue.join()      # 等待队列清空
    for c in consumers:
        c.cancel()
```

### 3.3 as_completed vs gather

```python
# gather：必须等所有完成才能拿到第一个结果
async def with_gather(urls):
    results = await asyncio.gather(*(fetch(u) for u in urls))
    return results[0]   # 即使第一结果 100ms 返回，也要等最慢的

# as_completed：谁先完成先用谁
async def with_as_completed(urls: list[str]) -> dict:
    tasks = [asyncio.create_task(fetch(u)) for u in urls]
    for coro in asyncio.as_completed(tasks):
        result = await coro
        process(result)   # 立即处理，不等其他
```

**选型指南：**
| 场景 | API |
|------|-----|
| 需要所有结果、order 重要 | `gather` |
| 需要所有结果、order 不重要 | `gather` + dict key |
| 先到先处理 | `as_completed` |
| 超时等待任意一个完成 | `wait(FIRST_COMPLETED)` |

### 3.4 连接池：aiohttp / httpx

```python
# aiohttp：高性能、连接池自动管理
async with aiohttp.ClientSession(
    connector=aiohttp.TCPConnector(
        limit=50,                # 总连接数上限
        limit_per_host=10,       # 每主机连接数上限
        ttl_dns_cache=300,       # DNS 缓存 TTL
        enable_cleanup_closed=True,
    ),
    timeout=aiohttp.ClientTimeout(total=30),
) as session:
    async with session.get(url) as resp:
        data = await resp.json()

# httpx：API 更友好、同步/异步双模
limits = httpx.Limits(
    max_keepalive_connections=20,
    max_connections=100,
    keepalive_expiry=30.0,
)
async with httpx.AsyncClient(
    limits=limits,
    timeout=httpx.Timeout(30.0),
) as client:
    r = await client.get(url)
    data = r.json()
```

**选型：** aiohttp 在纯异步、大量连接场景下性能更好；httpx 的 API 与 requests 兼容，开发体验更好。本项目若需 HTTP 通信（非 gRPC 场景），任选其一即可。

---

## 4. 测试

### 4.1 pytest-asyncio 模式

```python
# conftest.py 或 pyproject.toml 中设置模式
# [tool.pytest.ini_options]
# asyncio_mode = "auto"          # 自动检测 async def test_* 函数
# asyncio_default_fixture_loop_scope = "function"  # 每个测试独立循环

# 简单异步测试
async def test_fetch_price():
    result = await fetch_price("BTC/USDT")
    assert result > 0

# 带 fixture 的异步测试
@pytest.fixture
async def exchange_client():
    client = await create_test_client()
    yield client
    await client.aclose()

async def test_order(exchange_client):
    order = await exchange_client.create_order("BTC/USDT", "buy", 0.001)
    assert order["status"] == "open"
```

### 4.2 异步 Fixture

```python
# Fixture 作用域：function（默认）> class > module > session
# async fixture 不兼容 scope="session"（跨事件循环问题）

# 共享连接池：scope="module" 配合 async fixture
@pytest.fixture(scope="module")
async def shared_client():
    async with httpx.AsyncClient() as client:
        yield client

# 使用 event_loop fixture 控制循环
@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
```

### 4.3 Mock 异步函数

```python
# 基础：使用 AsyncMock (Python 3.8+)
from unittest.mock import AsyncMock, patch

async def test_with_mock():
    mock_fetch = AsyncMock(return_value={"price": 42000})
    with patch("myapp.exchange.fetch_price", mock_fetch):
        result = await get_price("BTC/USDT")
        assert result == 42000

# 带 side_effect 的 mock
mock_fetch = AsyncMock(side_effect=[{"price": 100}, {"price": 200}, TimeoutError])

# 模拟 CancelledError
mock_worker = AsyncMock(side_effect=asyncio.CancelledError)

# 模拟 Semaphore（不需要 mock，直接注入）
async def test_rate_limit():
    sem = asyncio.Semaphore(2)
    tasks = [rate_limited_op(sem, i) for i in range(5)]
    await asyncio.gather(*tasks)
```

---

## 5. 库生态选型

### 5.1 HTTP 客户端

| 库 | 适用场景 | 特点 |
|---|---------|------|
| `aiohttp` | 高吞吐 HTTP 服务/客户端 | 性能最优，连接池丰富，WebSocket 原生支持 |
| `httpx` | 通用 HTTP 客户端 | 同步/异步双模，API 兼容 requests，HTTP/2 |
| `grpclib` | gRPC 客户端 | 纯 Python asyncio gRPC |

**当前项目：** 使用 gRPC + grpclib 做信号通信，无需额外 HTTP 客户端。

### 5.2 数据库

| 库 | 存储 | 特点 |
|---|------|------|
| `asyncpg` | PostgreSQL | 最高性能，连接池，binary protocol |
| `databases` | PG/MySQL/SQLite | ORM 友好，Starlette 生态 |
| `redis-py (asyncio)` | Redis | 官方库，4.2+ 内置异步支持 |
| `redis[hiredis]` | Redis | 更快的解析器 |

```python
# redis-py 异步用法
import redis.asyncio as redis

async with redis.Redis(
    host="localhost",
    port=6379,
    max_connections=20,
) as r:
    await r.set("key", "value")
    value = await r.get("key")
```

### 5.3 anyio / trio — 结构化并发替代方案

```python
# anyio：跨框架抽象（支持 asyncio/trio 后端）
import anyio

async def main():
    async with anyio.create_task_group() as tg:
        tg.start_soon(worker1)
        tg.start_soon(worker2)
    # 任何子任务异常 → 所有兄弟任务被取消 → 异常传播

# trio：原生结构化并发，语义更严格
import trio

async def main():
    async with trio.open_nursery() as nursery:
        nursery.start_soon(worker1)
        nursery.start_soon(worker2)
    # 任何任务异常 → 整个 nursery 取消
```

**选型建议：**
- 纯 asyncio 项目（如当前项目）：Python 3.11+ 直接用 `asyncio.TaskGroup`，无需引入 anyio/trio
- 需要同时支持 asyncio 和 trio：用 anyio
- 新项目追求结构化并发极致：trio

---

## 6. 生产加固

### 6.1 Timeout — 一切 I/O 都要设超时

```python
# 方式 A：wait_for 包裹单个协程
try:
    result = await asyncio.wait_for(
        exchange.fetch_order(order_id),
        timeout=10.0
    )
except asyncio.TimeoutError:
    logger.error(f"Order fetch timed out: {order_id}")
    result = None

# 方式 B：httpx/aiohttp 连接级别超时（更精确）
client = httpx.AsyncClient(timeout=httpx.Timeout(
    connect=5.0,   # 连接超时
    read=10.0,     # 读取超时
    write=5.0,     # 写入超时
    pool=2.0,      # 连接池等待超时
))

# 方式 C：Python 3.11+ asyncio.timeout（上下文管理器）
async with asyncio.timeout(10.0):  # 内的所有 await 合计不超过 10s
    await asyncio.gather(
        fetch1(),  # 如果 fetch1 耗时 6s
        fetch2(),  # fetch2 只剩 4s
    )

# 方式 D：总时间和单次时间不同
try:
    result = await asyncio.wait_for(
        retry_fetch(order_id),   # 内部可能有 3 次重试
        timeout=30.0              # 总超时 30s
    )
except asyncio.TimeoutError:
    ...
```

**当前项目风险：** `apps/cta.py:104-127` 交易执行没有 timeout，`business/orders/twap.py:18` 下单也没有 timeout。如果交易所 API 挂死，整个任务永久卡住。

### 6.2 重试与指数退避

```python
import random

async def retry_with_backoff(
    coro_factory,       # 协程工厂（每次重试创建新的，避免连接状态污染）
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: bool = True,
):
    last_exc = None
    for attempt in range(max_retries):
        try:
            return await coro_factory()
        except (ConnectionError, TimeoutError, asyncio.TimeoutError) as e:
            last_exc = e
            if attempt == max_retries - 1:
                raise
            delay = min(base_delay * (2 ** attempt), max_delay)
            if jitter:
                delay = delay * (0.5 + random.random())  # ±50% 抖动
            logger.warning(f"Retry {attempt+1}/{max_retries} in {delay:.1f}s: {e}")
            await asyncio.sleep(delay)

# 用法：传入工厂而不是协程
result = await retry_with_backoff(
    lambda: exchange.fetch_order(order_id),  # 每次重试创建新协程
    max_retries=3,
    base_delay=1.0,
)
```

**什么错误该重试：** 网络错误、超时、503/429（配合 Retry-After 头）、临时故障。
**什么错误不该重试：** 400（请求错误）、401/403（认证失败）、业务逻辑错误、幂等性不保证的写操作（下单）。

### 6.3 结构化并发

```python
# 非结构化（传统 asyncio）：任务独立漂移
task_a = asyncio.create_task(worker_a())
task_b = asyncio.create_task(worker_b())
await asyncio.sleep(10)
task_a.cancel()  # 手动管理生命周期，容易遗漏

# 结构化并发（Python 3.11+ asyncio.TaskGroup）
async def main():
    async with asyncio.TaskGroup() as tg:
        tg.create_task(worker_a())
        tg.create_task(worker_b())
        tg.create_task(worker_c())
    # TaskGroup __aexit__ 自动：
    # 1. 任何子任务异常 → 取消所有兄弟任务
    # 2. 所有子任务完成后才退出
    # 3. 未被捕获的异常打包为 ExceptionGroup

# Python 3.10 及以下：用 asyncio.gather + try/finally 模拟
async def main():
    tasks = [
        asyncio.create_task(worker_a()),
        asyncio.create_task(worker_b()),
    ]
    try:
        await asyncio.gather(*tasks)
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
```

### 6.4 信号处理 (SIGTERM/SIGINT)

完整模式见 1.4 节。Docker/K8s 环境要点：

```python
# Docker 要点：
# 1. ENTRYPOINT 使用 exec 形式：ENTRYPOINT ["python", "trade_os.py"]
# 2. 不要用 shell 形式：ENTRYPOINT python trade_os.py（信号不会转发）
# 3. STOPSIGNAL 默认 SIGTERM，K8s 先 SIGTERM → 30s → SIGKILL

# K8s: terminationGracePeriodSeconds 应大于你的 shutdown timeout
```

---

## 7. AI Agent 特异性陷阱

### 7.1 生成 `async def` 但不用 `await`

最常见的 AI 生成错误 — 把函数声明为 async 但内部全是同步代码：

```python
# AI 常见错误
async def calculate_position(price: float, qty: float) -> float:
    return price * qty  # 无 await，不应是 async

# 应该
def calculate_position(price: float, qty: float) -> float:
    return price * qty
```

### 7.2 `asyncio.gather` 传了协程函数而不是协程对象

```python
# AI 常见错误
tasks = [fetch_data for url in urls]        # 传了函数，不是协程
await asyncio.gather(*tasks)                # 静默失败 — 无任何执行

# 正确
tasks = [fetch_data(url) for url in urls]   # 调用函数，得到协程
await asyncio.gather(*tasks)
```

### 7.3 在 `__init__` 中调用 `asyncio.create_task`

```python
# AI 常见错误
class Strategy:
    def __init__(self):
        self._task = asyncio.create_task(self._loop())  # 此时事件循环可能未运行！

# 正确：两步初始化或工厂
class Strategy:
    async def start(self):
        self._task = asyncio.create_task(self._loop())
```

### 7.4 忽略了 `return_exceptions` 的重要性

```python
# AI 常见错误
tasks = [asyncio.create_task(work(sym)) for sym in symbols]
await asyncio.gather(*tasks)  # 一个失败 → 全部取消，其他成功的结果丢失

# 交易系统必须用 return_exceptions=True
await asyncio.gather(*tasks, return_exceptions=True)
```

### 7.5 在循环中串行 await 而非并发

```python
# AI 常见错误（串行，总耗时 = 所有耗时之和）
for symbol in symbols:
    await process_symbol(symbol)   # N 秒 * M 个交易对

# 正确（并发，总耗时 = max(所有耗时)）
tasks = [process_symbol(sym) for sym in symbols]
await asyncio.gather(*tasks, return_exceptions=True)
```

### 7.6 忘记在 Task 的 `finally` 中做清理

```python
# AI 常见错误 — 任务被 cancel 后连接泄露
async def worker():
    client = await connect()
    while True:
        await client.poll()

# 正确 — finally 保证清理
async def worker():
    client = await connect()
    try:
        while True:
            await client.poll()
    except asyncio.CancelledError:
        raise
    finally:
        await client.close()
```

### 7.7 滥用 `asyncio.sleep(0)` 和 `asyncio.sleep(0.001)`

```python
# 坏模式 — 用 sleep 做同步/等待
while not data_ready:
    await asyncio.sleep(0.001)     # 忙等待，CPU 空转
    # 应用 asyncio.Event 代替

# 好模式
event = asyncio.Event()
# 生产者
data_ready = True
event.set()
# 消费者
await event.wait()
```

### 7.8 混用 `asyncio.wait_for` 和 `async with`

```python
# AI 常见错误
try:
    async with client.get(url) as resp:     # async with 在 wait_for 里面
        data = await resp.json()
except asyncio.TimeoutError:
    # resp 的 __aexit__ 可能没有被调用，连接泄露

# 正确：需要额外清理
try:
    resp = await asyncio.wait_for(client.get(url), timeout=5)
    async with resp:
        data = await resp.json()
except asyncio.TimeoutError:
    # resp 是 None 或未定义
    pass
```

### 7.9 缺少超时让 bug 永远是静默的

AI 倾向于生成没有任何超时的 async 代码。没有 timeout，任何网络问题都会导致任务永久挂起。**每条 `await` 都应该被 timeout 包裹**，或者使用连接级别的 timeout（httpx/aiohttp）。

### 7.10 健康检查清单

- [ ] 所有 I/O await 有 timeout
- [ ] `asyncio.gather()` 使用了 `return_exceptions=True`
- [ ] `CancelledError` 被正确捕获并重新抛出
- [ ] 后台 task 有保存引用（不被 GC 提前回收）
- [ ] 没有在 `__init__` 中 `create_task`
- [ ] 没有 sync I/O 在 async 函数中（或已 `run_in_executor`）
- [ ] 关闭路径：cancel → await gather → close connections
- [ ] 信号处理：SIGTERM/SIGINT 触发优雅关闭
- [ ] 并发有限流（Semaphore）
- [ ] 重试有指数退避 + 抖动
