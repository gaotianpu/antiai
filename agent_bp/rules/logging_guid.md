# 日志原则

### 原则一：每条日志必须携带可追溯的上下文

日志必须包含足够信息，能在不翻阅代码的情况下定位问题。

**正例：**
```python
self.logger.info(
    f"Fixed exposure for {asset}: "
    f"idle=${idle:.2f}, short_qty={short_qty:.6f}, order_id={order.order_id}"
)
```

**反例：**
```python
self.logger.info(f"amount={amount},side={side}")  # 缺少资产标识，无法区分
```

---

### 原则二：区分日志级别，不滥用 `info`

| 级别 | 适用场景 |
|------|---------|
| `debug` | TWAP 拆单中间变量、API 原始返回、循环每次迭代的细节、无信号时静默等待 |
| `info` | 一次完整的交易决策（信号到达、仓位变更、订单提交、状态文件写入） |
| `warning` | 可恢复的异常（API 重试、价格获取失败但跳过、信号延迟但可容忍、配置缺失） |
| `error` | 不可恢复的异常（下单失败、配置校验失败、进程退出），必须带 `traceback` |
| `critical` | 进程即将终止，需立即人工介入 |

**正例：**
```python
# info — 一次完整的仓位对齐决策
self.logger.info("Validated %d allocation assets", valid)

# debug — 循环中间状态，排查时才需要
self.logger.debug("No assets need exposure fix (no current == last)")

# error — 必须带上下文
self.logger.error(f"Position alignment failed: {e}")
```

**反例：**
```python
# 每分钟循环都用 info
self.logger.info(f"waiting for signal, current_ts={current_ts}")  # 应改为 debug
```

---

### 原则三：用 `key=value` 结构化格式，不用散文句式

结构化日志可直接被 `grep` 匹配、被脚本解析。

**正例：**
```python
self.logger.info(f"idle=${idle:.2f}, short_qty={short_qty:.6f}, order_id={order.order_id}")
self.logger.info("Validated %d allocation assets", valid)
```

**反例：**
```python
# 散文句式，grep 困难
self.logger.info(f"The idle amount is {idle} and the short quantity is {short_qty}")
```

> **注意**：`%s` 风格格式（如 `"Validated %d allocation assets", valid`）具备**惰性求值**优势，日志级别不过滤时避免字符串构造开销，推荐使用。

---

### 原则四：异常日志必须输出完整上下文

`error` 及以上级别的异常日志必须包含问题来源和关键变量，才能快速定位。

**正例：**
```python
except Exception as e:
    self.logger.error(
        f"Position alignment failed for {asset}: "
        f"expected={expected}, actual={actual}, err={e}"
    )
```

**反例：**
```python
except Exception as e:
    self.logger.error(f"failed: {str(e)}")  # 没有上下文，无法定位
```

---

### 原则五：敏感信息禁止写入日志

> **本项目已通过 `SensitiveDataFilter` 自动脱敏**，无需在每个调用处手动处理。但开发者仍需避免将敏感字段明文拼入日志消息。

`SensitiveDataFilter` 自动匹配并脱敏的字段：

```
apiKey, api_key, api_key_secret, apiSecret, api_secret,
secret, password, passwd, signature, sign_ts,
access_token, refresh_token, private_key
```

匹配 `key=value` 或 `key:value` 格式，替换 value 为 `***`。

**正例（安全——即使 API Key 意外拼入日志，过滤器也会拦截）：**
```python
# apiKey=abc123def → 日志中变为 apiKey=***
self.logger.info(f"load api from server, key={api_key_tips}")
```

**反例：**
```python
# 不应依赖脱敏器兜底，主动避免拼入敏感字段
self.logger.debug(f"apiKey={apiKey},secret={secret}")
```

---

### 原则六：高频率循环日志采用采样或降级

1 分钟或秒级循环中，每次迭代都打 `info` 会淹没磁盘。

**正例：**
```python
# 只有状态变更时才 info
if diff_min != last_logged_diff:
    self.logger.info(f"diff_min={diff_min}, expected={expected_signal_ts}")
    last_logged_diff = diff_min
# 每次迭代仅 debug
self.logger.debug(f"current_ts={current_ts}, still waiting...")
```

**反例：**
```python
while True:
    self.logger.info(f"waiting for signal, current_ts={current_ts}")
    await asyncio.sleep(60)
```

---

### 原则七：关键决策点打印"输入→输出"完整链路

每次下单或调仓，上下文必须包含：输入数据 → 决策过程 → 执行结果。

**正例（`FRAFixExposureApp` 模式）：**
```python
# 输入：读取状态
state = read_state(state_path)

# 计算：判断逻辑
idle = spot_value - contract_short_value

# 执行：下单
order = await self.exchange.place_order(...)

# 结果：统一输出
log.info(
    f"Fixed exposure for {asset}: "
    f"idle=${idle:.2f}, short_qty={short_qty:.6f}, "
    f"order_id={order.order_id}"
)
```

**反例：**
```python
# 只打下单前信号，没有执行结果
self.exchange.place_order(symbol, side, qty, "MARKET")
```

---

### 原则八：数字精度对标业务语义

金额、仓位数量应四舍五入到有意义的精度，避免输出 `0.0010000000000001` 这类浮点噪声。

**正例：**
```python
self.logger.info(f"idle=${idle:.2f}, short_qty={short_qty:.6f}")
round(current_vom, 3)  # U 本位保留 3 位小数
```

**反例：**
```python
self.logger.info(f"idle={idle}")  # 可能输出 0.0010000000000001
```

---

### 原则九：异步/并行任务日志必须携带协程标识

多 Portfolio/App 并发运行时，日志中必须有区分标识。

**本项目实践：**
- Logger 名用 `tradeos.{coroutine_id}` 天然隔离
- 日志文件名按 `coroutine_id` 区分：`logs/fra_cta.log`、`logs/trade_os.log`
- 每条日志头的 `%(name)s` 字段印出完整层级名

**正例：**
```python
self.logger.info(f"Starting app {self.portfolio_name}, execution mode: {self.execute_way}")
```

**反例：**
```python
self.logger.info(f"Starting app")  # 没有 portfolio 名称
```

---

### 原则十：`warning` 应注明重试次数和最终处理

重试逻辑的日志应注明当前是第几次、最终成功还是失败。

**正例：**
```python
for retry in range(self.retry_times):
    try:
        order = self.create_order(...)
        self.logger.debug(f"order success, retry={retry}, order_id={order.get('id')}")
        return order
    except Exception as e:
        self.logger.warning(f"create_order failed, retry={retry}/{self.retry_times}, err={e}")
        if retry == self.retry_times - 1:
            self.logger.error(f"create_order finally failed after {self.retry_times} retries")
```

**反例：**
```python
for retry in range(3):
    try:
        return create_order(...)
    except:
        pass  # 无日志，静默失败
```



---
