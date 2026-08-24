# TradeOS 消息通知原则

消息通知的目的是**在正确的时间、用正确的通道、把正确的信息告诉正确的人**。以下原则帮助开发者在所有通知场景中做出一致的判定。

---

## 原则一：三级分级，消息按级路由

所有通知必须归属于且仅归属于一个级别，不同级别走不同的通道和 chat_id。

| 级别 | 判定准则 | 通道策略 |
|------|---------|---------|
| **CRITICAL** | 资金安全受威胁，或系统无法继续运行 | TG `warning_id`（可配电话备选） |
| **WARNING** | 异常已发生但系统可继续运行，需关注 | TG `warning_id`，仅消息 |
| **INFO** | 例行状态同步，不需要人立即处理 | TG `heartbeat_id` 或 `report_id` |

**判定示例：**
- 下单失败 → CRITICAL（资金操作失败）
- 信号超时 → WARNING（策略延迟，但系统仍在运行）
- 日终净值报告 → INFO（例行汇总）

**反例：**
```python
# 不应将所有异常都打为 CRITICAL
self.notifier.send_alert("price fetch failed")       # → WARNING（可降级继续）
self.notifier.send_alert("order execution failed")    # → CRITICAL（资金操作失败）
```

---

## 原则二：不多发 — 同场景同级别必须防抖

同一异常场景在短时间内反复触发时，只发一条通知，不刷屏。

**正例：**
```python
# 同一个 symbol 仓位检查失败，5 分钟内不再重复报警
if self._last_alert.get(symbol, 0) + 300 < time.time():
    self.notifier.send_alert(f"position_check_failed, {symbol}, ...")
    self._last_alert[symbol] = time.time()
```

**反例：**
```python
# 循环每次迭代都发，1 分钟刷几十条
while True:
    if position_mismatch:
        self.notifier.send_alert(...)  # 无防抖
```

**防抖建议间隔：**

| 场景类型 | 建议冷却 |
|---------|---------|
| 下单失败 | 120s |
| 仓位校验异常 | 300s |
| 保证金/风控告警 | 300s |
| API 连接失败 | 60s |
| 认证失败 | 0s（立即通知） |
| 心跳 | 按发送周期（不额外限制） |

---

## 原则三：不漏发 — 关键路径必须有通知

以下场景**必须**发送通知，不允许静默失败：

| 场景 | 级别 | 原因 |
|------|------|------|
| 下单执行失败（所有 retry 耗尽后） | CRITICAL | 资金操作失败，需人工介入 |
| 仓位校验发现持续偏差 | CRITICAL | 仓位不一致可能造成损失 |
| 配置文件加载失败 | CRITICAL | 系统无法启动 |
| API 认证失败 | CRITICAL | 无法继续交易 |
| API 限流达上限 | WARNING | 需关注，但系统会自愈 |
| 信号延迟超过配置阈值 | WARNING | 策略可能落后于市场 |
| 策略进程异常退出 | CRITICAL | 需人工恢复 |

**反例：**
```python
# 静默失败 — 违反原则三
try:
    await exchange.place_order(symbol, side, qty)
except Exception:
    pass  # 下单失败不通知，仓位偏差了都不知道
```

---

## 原则四：消息内容必须支持独立排查

每条通知应包含足够信息，收信人不需要翻阅日志就能决定下一步动作。

**正例：**
```
tpserver:order_failed, account=fra_cta, symbol=BTC/USDT:USDT, side=buy, qty=0.5, err=insufficient_balance
```

**反例：**
```
tpserver:something went wrong on fra_cta  # 完全无法判断是什么失败
```

**必含字段规则：**
- 涉及交易 → `account` + `symbol` + 具体操作
- 涉及系统 → 组件名 + 具体错误
- 所有通知 → `server`（配置自动注入）+ 具体场景标识

---

## 原则五：`key=value` 结构，不写散文

通知消息必须用 `key=value` 格式，方便脚本解析和多语言统一处理。

**正例：**
```python
msg = f"order_failed, account={account}, symbol={symbol}, side={side}, qty={qty}, err={e}"
```

**反例：**
```python
msg = f"Failed to place order on {account} for {symbol}. The error is: {e}"
```

---

## 原则六：通知与日志分离

通知用于让人知道，日志用于让程序排查。一条信息**必须同时写日志**，但两者内容侧重点不同：

| | 通知 | 日志 |
|--|------|------|
| 目的 | 让人快速决策 | 让程序排查根因 |
| 包含 | 关键字段 + 结论 | 完整上下文 + traceback |
| 长度 | 1–2 行 | 不限 |

**正例：**
```python
self.logger.error(f"order_failed, symbol={symbol}, qty={qty}, side={side}, "
                  f"price={price}, err={traceback.format_exc()}")
self.notifier.send_alert(f"order_failed, account={account}, symbol={symbol}, "
                         f"side={side}, qty={qty}, err={e}")
```

**反例：**
```python
# 只用通知代替日志 — 丢了堆栈信息
self.notifier.send_alert(f"order_failed, err={e}")
```

---

## 原则七：敏感信息禁止进入消息体

通知走外部通道（Telegram API），API Key、Secret、密码等永远不能出现在消息中。

> 本项目日志层已有 `SensitiveDataFilter` 自动脱敏，但通知通道**没有**这层保护，开发者必须主动确保。

---

## 原则八：通道可靠性意识

| 约束 | 应对 |
|------|------|
| TG 可能断连/限频 | CRITICAL 通知必须有日志冗余（至少保证日志可追溯） |
| 通知通道本身故障 | 关键操作失败必须写日志 + 本地文件，不能只依赖外部通道 |
| 电话通知成本高 | 仅 CRITICAL 且在静默时段使用；Phase 1 不实现电话 |

---

## 原则九：全局静默开关

当 `msg_open=0` 时，所有通知通道静默。通知调用方**不需要**检查此开关——通知基础设施层自动处理。但开发者应确保：即使通知被关闭，日志必须正常写入，不依赖通知替代日志。

---

## 原则十：通知是"做了"的标记，不是"正在做"的日志

通知应发送**已完成或已失败**的确定性结果，而不是过程状态。

**正例：**
```python
# 有确定结果后再通知
order = await exchange.place_order(...)
self.notifier.send_alert(f"order_placed, symbol={symbol}, order_id={order.order_id}")
```

**反例：**
```python
# 还没执行就通知，如果后续失败还需要补一条
self.notifier.send_alert(f"about to place order, symbol={symbol}")
order = await exchange.place_order(...)
```

---

## 快速参考

**什么时候发？**
```
CRITICAL ← 资金/系统不可用 → 必发 + 日志
WARNING  ← 异常但可运行 → 必发 + 日志（防抖）
INFO     ← 例行同步 → 按周期发 + 日志
```

**什么时候不发？**
- 同场景同级别在冷却期内 → 不发
- `msg_open=0` → 不发
- 正在执行中、尚未有结果 → 不发
- 已有日志可完整回溯、人不需要知道 → 不发

**每条的格式：**
```
{server}:{场景标识}, key1=value1, key2=value2, ...
```

---
