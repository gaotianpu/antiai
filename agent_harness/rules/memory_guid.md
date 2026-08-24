# Python Async 内存优化最佳实践

> 适用场景：asyncio 长期运行服务、WebSocket 数据管道
> 依据：unix_pythonic_guid.md 第11条 — 小而专注

---

## 一、集合无界增长 — 头号内存泄露源

长期运行的 asyncio 进程中，任何 `dict`/`set`/`list` 只增不减是内存泄露第一嫌疑人。

### 1.1 TTL 缓存必须配 maxsize

```python
# 泄露：缓存只增不减
class Cache:
    _data: dict = {}

    def set(self, key, value, ttl=1800):
        self._data[key] = (value, time.time() + ttl)

    def get(self, key):
        entry = self._data.get(key)
        if entry and entry[1] > time.time():
            return entry[0]
        # 过期的删了，没访问过的永远不删

# 安全：TTL + maxsize + 被动驱逐
class Cache:
    _data: dict = {}
    _maxsize = 10000

    def _evict_expired(self):
        now = time.time()
        stale = [k for k, v in self._data.items() if v[1] <= now]
        for k in stale:
            del self._data[k]

    def set(self, key, value, ttl=1800):
        self._data[key] = (value, time.time() + ttl)
        if len(self._data) > self._maxsize:
            self._evict_expired()
```

**本项目实例：** `OhlcvRepo._exists_cache` — 聚合窗口检查每次写入新 key，窗口一次性（检查后不再访问），每天 ~4000 条，30 天 ~10MB。已修复：`CACHE_MAXSIZE=10000` + `_evict_expired()`。

### 1.2 无上限优先级队列

```python
# 泄露：无界队列
queue: asyncio.Queue = asyncio.Queue()  # maxsize=0 = 无上限

# 安全：有界队列 + 溢出策略
queue: asyncio.Queue = asyncio.Queue(maxsize=4096)

async def put(msg):
    try:
        queue.put_nowait(msg)
    except asyncio.QueueFull:
        try:
            queue.get_nowait()   # 丢弃最旧
        except asyncio.QueueEmpty:
            pass
        queue.put_nowait(msg)
```

---

## 二、Task 引用泄露

### 2.1 即发即忘 task 必须保存引用但不超额累积

```python
# 泄露：task hang 住后永久占引用
self._tasks: set[asyncio.Task] = set()

async def fire_and_forget(coro):
    task = asyncio.create_task(coro)
    self._tasks.add(task)
    task.add_done_callback(self._tasks.discard)  # 正常完成会移除
    # 但如果 task 死锁/挂起 → 永远不移除

# 安全：定期清理已完成/异常 task
async def _cleanup(self):
    while True:
        await asyncio.sleep(120)
        done = {t for t in self._tasks if t.done()}
        self._tasks -= done
```

**本项目实例：** `RealtimePipeline._agg_tasks` — 聚合 task 通过 `done_callback` 清理，但 hung task 永不触发 callback。已修复：`_cleanup_agg_tasks()` 每 120s 执行。

### 2.2 避免在循环中 `create_task` 不保存

```python
# 泄露：GC 可能提前回收
for item in items:
    asyncio.create_task(process(item))  # 无引用，可能被 GC 取消

# 安全：保存引用或用 gather
tasks = [asyncio.create_task(process(item)) for item in items]
await asyncio.gather(*tasks, return_exceptions=True)
```

---

## 三、连接与会话泄露

### 3.1 aiohttp.ClientSession 复用不当

```python
# 泄露：每次请求创建新 session
async def fetch(url):
    async with aiohttp.ClientSession() as session:  # 每次建连/拆连
        return await session.get(url)

# 正确：会话级复用
class Client:
    def __init__(self):
        self._session = None

    def _session_ctx(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
```

### 3.2 close() 必须在同一 event loop 上执行

```python
# 泄露：跨 loop 关闭，连接永不释放
def shutdown():
    asyncio.run(client.close())  # 创建新 loop 去关闭旧 loop 的资源

# 安全：同一 loop 中 await close
async def run():
    try:
        client = Client()
        await client.run()
    finally:
        await client.close()  # 同 loop
```

---

## 四、诊断清单

运行 24 小时后检查：

| 指标 | 检测命令 | 异常阈值 |
|------|---------|---------|
| 内存增长率 | `ps aux \| grep python` 监控 RSS | > 10MB/天 |
| `asyncio.Task` 数量 | `len(asyncio.all_tasks())` | 持续增长 |
| 自定义缓存大小 | `len(self._cache)` | 增长不止 |
| `asyncio.Queue` 积压 | `queue.qsize()` 持续 > 80% maxsize | 消费跟不上 |
| 文件描述符数 | `ls /proc/<pid>/fd \| wc -l` | 持续增长 |
