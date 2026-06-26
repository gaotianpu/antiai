# Python 单元测试最佳实践

**适用项目**: TradeOS (pytest + pytest-asyncio, Python ≥3.11)
**整合来源**: `unix_pythonic_guid.md` §5 测试宪法 + `async_best_practices.md` §4 + pytest 官方文档

---

## 一、测试哲学（宪法第 17 条）

```
纯函数优先                    适配器集成测试                  避免模拟过度
──────────                    ──────────────                  ──────────
domain/ 纯数据、utils/        exchanges/ 外部 API           只 mock 网络 / 文件系统 / 数据库
计算逻辑 → 零 mock 单元测试    → 真实连接或 mock 适配器       不对 domain 逻辑 mock
```

> **核心原则**: 测试金字塔 — 底层纯函数测试大量、快速；上层集成测试少而精。

---

## 二、测试范围界定

为每个模块的测试规格文档提供统一的"测什么 / 不测什么"决策依据。与 §三"优先级分级"（测多深）互补。

### 2.1 纳入三问

一个行为必须满足**全部三条**才纳入单元测试：

| 问题 | 含义 |
|------|------|
| **1. 是我自己的逻辑吗？** | 行为由本模块代码实现，不是委托给下游类/库/平台 |
| **2. 失败后有破坏性吗？** | 行为出错 → 调用方拿假数据、功能静默失效、或核心路径崩溃 |
| **3. 能稳定验证吗？** | 不依赖实时网络、时间、随机数等不可控外部因素（可用 mock 隔离） |

### 2.2 排除四类

以下四类不应出现在单元测试中，不论优先级高低：

| 排除类型 | 判断方法 | 示例 |
|---------|---------|------|
| **语言/平台行为** | 去掉本模块代码，测试仍然有意义吗？ | `assert obj.portfolio_name == "test"` — 测的是 Python 赋值语法 |
| **委托调用** | 本方法是否只做了"调用 A → 返回 A 的结果"？ | `__aenter__` → 调 `initialize()`；应测 `initialize`，不测 `__aenter__` |
| **实现细节** | 修改了实现方式但不改变对外行为，测试是否会破坏？ | `get_all_exchanges` 返回 `.copy()` 还是 `dict()` — 调用方不依赖这个选择 |
| **非功能需求** | 是否可以通过静态检查或 Code Review 保证？ | logger 输出格式、异常消息措辞 — 改一行文案不应导致 CI 红 |

### 2.3 集成测试标记

外部依赖（网络、文件系统、真实 API）不纳入单元测试，但需要时放在集成测试：

```python
@pytest.mark.integration
async def test_real_exchange_connect():
    ...
```

运行：`pytest -m "not integration"` 跳过后不影响 CI 速度。

### 2.4 纳入/排除对照表

| 行为 | 判断 | 归类 |
|------|------|:---:|
| `_prepare_account_configs` 抛出 `ValueError` | 自己的逻辑 + 破坏静默错误 + 可 mock | ✅ 纳入 |
| `get_exchange("unknown")` 抛出 `ValueError` | 自己的逻辑 + 破坏调用链 + 可 mock | ✅ 纳入 |
| `am.portfolio_name == "test"` | 平台语法 | ❌ 排除 |
| `am.__aenter__` 调了 `initialize` | 委托 | ❌ 排除 |
| 日志包含 `"Account manager initialized"` | 非功能需求 | ❌ 排除 |
| 真实连接 KuCoin 并查余额 | 外部依赖 | ➡️ 集成测试 |

---

## 三、测试用例优先级分级

为所有模块的测试规格文档提供统一的 P0/P1/P2 分级依据。

### 3.1 三维评分

每个测试用例从三个维度独立评分：

| 维度 | 含义 | 评分标准 |
|------|------|---------|
| **隐蔽性** | 出问题时调用方能否立刻感知？ | **高**: 静默错误 — 返回假成功/假数据/None，调用方无感知继续运行 |
| | | **中**: 抛异常但信息模糊（如 `RuntimeError` 无 message）|
| | | **低**: 直接崩溃或抛明确异常，调用方立刻发现 |
| **影响面** | 有多少 App / 路径依赖这个行为？ | **高**: 所有 App 启动或执行路径必经 |
| | | **中**: 部分 App 或部分执行路径使用 |
| | | **低**: 仅边界场景触发，正常流程不走 |
| **回退难度** | 出错后能否降级运行？ | **高**: 无替代路径，核心功能完全不可用 |
| | | **中**: 可手动绕过或降级运行 |
| | | **低**: 不影响核心功能，有兜底逻辑 |

### 3.2 分级规则

#### P0 — 模块契约

两个触发条件，任一命中即为 P0：

1. **隐蔽性 = 高** — 行为破坏后静默返回假数据，调用方无感知继续运行
2. **回退难度 = 高** — 无替代路径，核心功能完全不可用

> P0 不包含「影响面 = 高」：影响面大但抛明确异常的错误，调用方立刻能感知，不会带错运行。这类错误属 P1。

P0 测试失败 → 模块对外承诺的核心行为已破坏，**发布前必须修复**。

#### P1 — 重要路径

隐蔽性 ≤ 中，影响面 ≥ 中。

- 出错时会抛异常，不会静默返回假数据
- 正常业务流程会经过，影响面大
- 有一定降级可能

P1 测试失败 → 影响生产可靠性，**当前迭代应修复**。

#### P2 — 防御边界

隐蔽性低 + 影响面低。

- 仅边界场景、防御性逻辑触发
- 即使行为异常，核心流程不受影响

P2 测试失败 → 技术债，**排入下个迭代**。

### 3.3 判定流程图

```
测试用例
  │
  ├─ 隐蔽性 = 高？ ────────── YES ──▶ P0
  ├─ 回退难度 = 高？ ──────── YES ──▶ P0
  │
  ├─ 影响面 ≥ 中 且 隐蔽性 ≤ 中？ ── YES ──▶ P1
  │
  └─ 都不是 ──────────────────────────────▶ P2
```

> 影响面 = 高不单独触发 P0：所有 App 都用的方法抛明确异常，调用方立刻崩溃 → 不会静默带错运行 → 属 P1。

（待补充：P0 最小门槛策略 — Code Review 只检查 P0，P1/P2 按迭代优先级挑选 — 见 dup2 §3.3）

### 3.4 使用示例

在模块测试规格文档中，对每个用例标注三个维度：

```markdown
| 用例 | 隐蔽性 | 影响面 | 回退难度 | 等级 |
|------|:---:|:---:|:---:|:---:|
| 缺失配置抛异常 | 高 | 高 | 高 | **P0** |
| is_connected 状态 | 中 | 高 | 中 | **P1** |
| 重复初始化幂等 | 低 | 低 | 低 | **P2** |
```

---

## 四、测试布局

### 4.1 推荐结构 (Tests outside application code)

```
trader/
├── pyproject.toml          ← pytest 配置在此
├── apps/                   ← 应用代码
├── business/               ← 业务代码
├── domain/                 ← 领域模型
├── tests/
│   ├── conftest.py         ← 共享 fixtures
│   ├── test_domain.py      ← 纯数据模型
│   ├── test_exchange.py    ← Exchange 接口契约
│   ├── test_errors.py      ← 异常层级
│   ├── test_binance.py     ← 币安适配器
│   ├── test_okx.py         ← OKX 适配器
│   ├── test_kucoin.py      ← KuCoin 适配器
│   ├── test_factory.py     ← get_exchange() 工厂
│   ├── test_signals/       ← 信号模块
│       ├── test_client.py
│       ├── test_fetcher.py
│       └── test_cache_manager.py
└── ...
```

**关键规则**:
- 文件名: `test_*.py` 或 `*_test.py`
- 函数名: `test_` 前缀
- 类名: `Test` 前缀（无 `__init__`）
- 目录需要 `__init__.py` 才会被识别为包

### 4.2 当前 `pyproject.toml` 配置

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"          # 自动检测 async def test_*
testpaths = ["tests"]          # 测试搜索路径
```

运行方式:
```bash
pytest                          # 运行全部
pytest tests/test_domain.py     # 单个文件
pytest -k "test_position"       # 按名称过滤
pytest -v --tb=short            # 详细输出 + 短回溯
```

---

## 五、测试四步法 (Arrange-Act-Assert-Cleanup)

```python
# 模板
async def test_<what>_<condition>_<expected>():
    # 1. Arrange — 准备数据、依赖、状态
    position = Position(symbol="BTC/USDT", side="long", quantity=1.0, entry_price=42000)

    # 2. Act — 执行被测试行为
    result = calculate_pnl(position, current_price=43000)

    # 3. Assert — 验证结果
    assert result == 1000.0

    # 4. Cleanup — pytest fixtures 自动处理
```

**命名约定**: `test_<被测对象>_<条件>_<预期结果>`
```python
def test_position_pnl_positive_when_price_rises(): ...
def test_account_total_balance_zero_with_empty_positions(): ...
def test_get_exchange_unknown_name_raises_value_error(): ...
```

---

## 六、纯函数 / 纯数据测试（最高优先级）

domain 层是纯数据结构，测试零依赖、零 mock — 最先写、最多写。

```python
import pytest
from domain.models import Position, Order, Account, Ticker

class TestPosition:
    def test_default_values(self):
        p = Position(symbol="BTC/USDT", side="long", quantity=0.1, entry_price=42000)
        assert p.unrealized_pnl == 0.0
        assert p.margin == 0.0
        assert p.leverage == 1.0

    def test_position_id_is_none_by_default(self):
        p = Position(symbol="ETH/USDT", side="short", quantity=1.0, entry_price=3000)
        assert p.position_id is None

    @pytest.mark.parametrize("field", [
        "symbol", "side", "quantity", "entry_price"
    ])
    def test_required_fields_raise_on_missing(self, field):
        kwargs = {"symbol": "BTC/USDT", "side": "long", "quantity": 0.1, "entry_price": 42000}
        del kwargs[field]
        with pytest.raises(TypeError):
            Position(**kwargs)


class TestOrder:
    @pytest.mark.parametrize("status", ["NEW", "PARTIALLY_FILLED", "FILLED", "CANCELLED"])
    def test_valid_status_values(self, status):
        order = Order(order_id="123", symbol="BTC/USDT", side="buy", quantity=0.1, status=status)
        assert order.status == status

    def test_filled_quantity_not_exceeding_total(self):
        # 业务规则：filled 不能超过 quantity
        order = Order(order_id="123", symbol="BTC/USDT", side="buy",
                      quantity=0.1, filled_quantity=0.05)
        assert order.filled_quantity <= order.quantity
```

---

## 七、异步测试

### 7.1 基础模式

```python
import pytest
import asyncio

# pytest-asyncio auto 模式 — 直接 async def
async def test_fetch_signal():
    fetcher = SignalFetcher(server_params, signal_configs)
    signal = await fetcher.fetch_and_cache_signal("product", "BTC/USDT:USDT")
    assert signal is not None
    assert "price" in signal
```

### 7.2 异步 Fixture

```python
# tests/conftest.py
import pytest
from business.account_manager import AccountManager

@pytest.fixture
def accounts_config():
    """共享的测试账户配置 — 纯数据 fixture"""
    return {
        "kctest": {"exchange": "kucoin", "apiKey": "test_key",
                   "secret": "test_secret", "password": "test_pw"}
    }

@pytest.fixture
def active_accounts():
    return ["kctest"]

@pytest.fixture
def account_manager(accounts_config, active_accounts):
    """创建未初始化的 AccountManager"""
    return AccountManager("test_portfolio",
                          accounts_config=accounts_config,
                          active_accounts=active_accounts)
```

### 7.3 Mock 异步函数

```python
from unittest.mock import AsyncMock, patch

async def test_fetcher_with_mock_signal():
    mock_client = AsyncMock()
    mock_client.read_signal.return_value = [{
        "strategy_name": "product",
        "symbol": "BTC/USDT:USDT",
        "signal_ts": 1700000000000,
        "price": 42000.0,
        "current_vom": 0.5,
    }]

    with patch("signals.fetcher.SignalClient", return_value=mock_client):
        fetcher = SignalFetcher(server_params, signal_configs)
        signal = await fetcher.fetch_and_cache_signal("product", "BTC/USDT:USDT")
        assert signal["price"] == 42000.0


async def test_gather_with_exceptions():
    """宪法要求: asyncio.gather 必须 return_exceptions=True"""
    async def fail_later():
        await asyncio.sleep(0.1)
        raise ValueError("expected")

    results = await asyncio.gather(
        asyncio.sleep(0),       # 成功
        fail_later(),            # 失败
        return_exceptions=True,
    )
    assert isinstance(results[1], ValueError)
```

---

（待补充：I/O 依赖隔离 — 文件系统用 pytest `tmp_path` fixture，外部 API 用 `unittest.mock` — 见 dup2 §七）

## 八、Parametrize — 减少重复

```python
import pytest
from exchanges import get_exchange

# 一个测试覆盖 3 个交易所
@pytest.mark.parametrize("name,expected_class", [
    ("binance", "BinanceExchange"),
    ("okx", "OKXExchange"),
    ("kucoin", "KuCoinExchange"),
])
def test_get_exchange_returns_correct_type(name, expected_class):
    exchange = get_exchange(name, api_key="k", api_secret="s",
                            passphrase="p", testnet=True)
    assert exchange.__class__.__name__ == expected_class


# 一个测试覆盖多个非法输入
@pytest.mark.parametrize("name", [
    "unknown", "BINANCE_EXCHANGE", "", "bitmex"
])
def test_get_exchange_invalid_name_raises(name):
    with pytest.raises(ValueError, match="不支持的交易所"):
        get_exchange(name, api_key="k", api_secret="s")


# 异常层级测试
@pytest.mark.parametrize("exc_class,parent", [
    (AuthenticationError, ExchangeError),
    (RateLimitError, ExchangeError),
    (InsufficientFundsError, ExchangeError),
    (OrderError, ExchangeError),
])
def test_exception_hierarchy(exc_class, parent):
    assert issubclass(exc_class, parent)
```

---

## 九、Mock 指南（避免过度模拟 — 宪法第 17 条）

### 9.1 可以 Mock 的

| 层级 | 可以 Mock | 不能 Mock |
|------|----------|----------|
| 网络 I/O | `aiohttp.ClientSession`, gRPC stub | Exchange 接口本身 |
| 文件系统 | `open()`, `load_config()` | `FileCache` (直接实例化测试) |
| 时间 | `time.time()`, `datetime.now()` | — |
| 数据库 | DB driver | Repository 接口 |

### 9.2 模式

```python
# 正确: Mock 网络层，测试业务逻辑
async def test_position_manager_calculates_correctly():
    mock_exchange = AsyncMock()
    mock_exchange.get_positions.return_value = [
        Position(symbol="BTC/USDT", side="long", quantity=0.1, entry_price=42000)
    ]
    mock_exchange.get_balance.return_value = {"BTC": 0.05, "USDT": 1000}
    mock_exchange.get_ticker.return_value = Ticker(
        symbol="BTC/USDT", bid=41900, ask=42100, last=42000,
        volume=1000, timestamp=datetime.now()
    )

    pm = PositionManager(mock_exchange)
    actual = await pm.get_actual_positions({"BTC": 10000.0})
    assert "BTC" in actual


# 错误: Mock domain 逻辑
# ❌ mock_calculate = AsyncMock(return_value={"BTC": 5000})
# ✅ 不 mock，直接测试纯函数
def test_calculate_adjustments():
    pm = PositionManager(exchange=None)
    adjustments = pm.calculate_adjustments(
        actual={"BTC": 9500}, target={"BTC": 10000}
    )
    assert len(adjustments) == 1
    assert adjustments[0]["side"] == "buy"
```

---

## 十、依赖隔离策略

### 10.1 决策流程

拿到一个被测试模块，先画依赖图，对每条下游依赖做判断：

```
SUT 的下游依赖
  │
  ├─ 是本模块自己的逻辑吗？
  │     └─ NO → 不测（属 §二 排除范围）
  │
  ├─ 需要真实连接才能验证行为吗？
  │     └─ YES → @pytest.mark.integration（网络、文件系统、数据库）
  │
  └─ 可以 mock 吗？
        ├─ YES → mock，验证点：
        │       1. SUT 传给下游的参数是否正确
        │       2. SUT 是否正确处理了下游的返回值/异常
        │       3. 下游是否被调用了期望的次数
        └─ NO  → 不 mock，直接注入真实对象（纯数据、已测模块的类）
```

### 10.2 验证边界

| SUT 应该验证 | 不应该验证 |
|-------------|-----------|
| 传给了 `get_exchange()` 正确的 `passphrase` 参数 | `get_exchange()` 内部如何创建 Exchange 实例 |
| `exchange.connect()` 被调用了 1 次 | `connect()` 之后的网络握手细节 |
| `exchange.disconnect()` 在 `close_all` 中被调用 | `disconnect()` 是否真的关闭了 TCP 连接 |

> **原则**: 单元测试验证 SUT 与下游的**交互契约**（参数、调用次数、返回值处理），不验证下游的**内部实现**。下游的实现在其自己的单元测试中覆盖。

### 10.3 文档化要求

每个模块测试规格文档的「依赖隔离策略」节应包含：

1. **依赖图** — SUT → 哪些下游模块/方法
2. **隔离方式表** — 每条依赖：patch 还是 AsyncMock？验证什么？
3. **共享归属** — mock 对象是否应提取到 `test-fixtures-spec.md`

（待补充：测试规格文档编写规范 — 每个模块的测试规格文档应包含：模块定位、测试范围、用例列表（P0/P1/P2 分级）、依赖隔离；用例表格格式见 dup2 §十）

---

## 十一、模块测试规格

各模块的完整测试用例规格见 `docs/analysis/` 下的独立文档，按 §二（范围界定）和 §三（优先级分级）编写。

| 模块 | 文档 |
|------|------|
| AccountManager | [`account-manager-test-spec.md`](../analysis/account-manager-test-spec.md) |
| （后续模块） | — |

---

## 十二、运行与 CI

### 12.1 本地运行

```bash
# 全量
pytest

# 单文件 + 详细输出
pytest tests/test_account_manager.py -v

# 只跑特定标记
pytest -m "not slow"

# 失败即停 + 显示局部变量
pytest -x --tb=long -l
```

### 12.2 `pyproject.toml` 扩展配置

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
addopts = ["-v", "--tb=short", "--strict-markers"]
markers = [
    "slow: slow tests (deselect with '-m \"not slow\"')",
    "integration: tests requiring real exchange connection",
]
```

### 12.3 Pre-commit 验证

```bash
ruff check . && ruff format --check .
mypy apps/ business/ orchestration/ common/
python -m pytest tests/
```

---

## 十三、常见反模式

以下模式应避免：

```python
# ❌ 反模式 1: 手动调度 asyncio
if __name__ == "__main__":
    asyncio.run(main())

# ✅ 正确: 让 pytest-asyncio 接管
async def test_something(): ...


# ❌ 反模式 2: print 代替 assert
if result == expected:
    print("✓ passed")
else:
    print("✗ failed")

# ✅ 正确: 用 assert
assert result == expected


# ❌ 反模式 3: 手写 for 循环参数化
for name in ["binance", "okx", "kucoin"]:
    test_get_exchange(name)

# ✅ 正确: @pytest.mark.parametrize
@pytest.mark.parametrize("name", ["binance", "okx", "kucoin"])
def test_get_exchange(name): ...


# ❌ 反模式 4: 多个测试挤在一个函数里，共享可变状态
async def test_everything():
    am = AccountManager(...)
    await am.initialize()
    # ... 10 个检查 ...
    await am.close_all()

# ✅ 正确: 每个测试函数只验证一件事；fixture 管理状态
class TestAccountManager:
    async def test_initialize_sets_flag(self, am): ...
    async def test_close_clears_exchanges(self, initialized_am): ...


# ❌ 反模式 5: Mock 纯函数或 domain 对象
mock_position = AsyncMock(spec=Position)    # 不需要 mock 纯数据

# ✅ 正确: 直接实例化
position = Position(symbol="BTC/USDT", side="long", quantity=0.1, entry_price=42000)


# ❌ 反模式 6: 测试依赖真实外部连接（不含显式标记）
async def test_real_exchange():
    exchange = BinanceExchange(real_key, real_secret)
    await exchange.connect()
    # ... 网络抖动 → 测试随机失败

# ✅ 正确: mock 网络层，或标记为 integration
@pytest.mark.integration
async def test_real_exchange(): ...
```

---

**关联文档**:
- [`unix_pythonic_guid.md`](../rules/unix_pythonic_guid.md) §5 — 测试宪法
- [`async_best_practices.md`](../rules/async_best_practices.md) §4 — 异步测试

---
