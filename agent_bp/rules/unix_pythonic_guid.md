# Unix + Pythonic 架构宪法

**本宪法是项目所有文档规范和代码实现的最高指导原则。任何规范类文档和代码必须首先通过本宪法审查。**

---

## Agent 速查清单

审查代码时逐条对照以下信号，任一命中即触发审查：

| # | 信号 | 对应条款 | 动作 |
|---|------|---------|------|
| 1 | `except Exception: pass` 或 `except: pass` | 第8条 | 违宪：吞错 |
| 2 | `return None` 代替抛出异常 | 第8条 | 违宪：静默失败 |
| 3 | 初始化方法中 `log.error(...); return` 而非 `raise` | 第8.1条 | 违宪：Crash on Boot |
| 4 | `isinstance(obj, XxxBase)` 用于逻辑分发 | 第9条 | 用 Protocol/hasattr |
| 5 | 仅有 1 个子类的抽象基类 | 第9条 | 冗余 ABC，删除 |
| 6 | 单个方法 `if/elif/else` 超过 5 个分支 | 第10条 | 改用 dict 映射 |
| 7 | 函数内硬编码数值常量（非 0/1/-1/100） | 第10条 | 提取到配置 |
| 8 | 类级 `_instance = None` + `__new__` 返回单例 | 第13条 | 违宪：单例 |
| 9 | 类名含 `Factory` + `@abstractmethod` | 第13条 | 违宪：过度工厂 |
| 10 | 文件中同时出现 `visit_` 和 `accept()` 方法 | 第13条 | 违宪：访问者 |
| 11 | 类 `__init__` 注入 5+ 组件，方法仅做转发 `self.x.do()` | 第13条 | 违宪：中介者 |
| 12 | 文件 >300 行 / 函数 >50 行 / 类 >10 方法 | 第11条 | 拆分 |
| 13 | `from module import *` | 第12条 | 违宪：污染命名空间 |

> 项目特定的层边界违宪检测（机制层 import 应用层等）见项目 `architecture.md` §层间边界规则。

### 审查报告规范

审查报告只罗列有问题的条款（违宪、待商榷、改进建议），完全合宪的条款**不用列出**。报告结构如下：

```
## Agent 速查清单扫描
（只列出命中的条目，未命中的跳过）

## 审查结论
（违宪条目汇总 + 整改建议，合宪部分只字不提）
```

**合宪 = 默认状态，不需要证明。** 只有在发现违反时才需要记录。

> 正例：
> ```
> ## Agent 速查清单扫描
> | # | 信号 | 命中 | 说明 |
> | 6 | 函数内硬编码数值常量 | ✅ 命中 | `1000.0` 硬编码 |
> ```
> 其余 11 项未命中 → 不出现。

> 反例（禁止）：
> ```
> ## 逐条宪法审查
> ### 第1条：简单 — ✅ 合宪
> 代码很简洁...
> ### 第2条：透明 — ✅ 合宪
> 逻辑流很清晰...
> ### 第3条：组合 — ✅ 合宪
> ...
> ```
> 合宪条款逐条罗列属于噪音，禁止。

---

## 第1章：核心原则

### 第1条：简单

如无必要，勿增实体。复杂性是成本，不是资产。每个新增的类、函数、模块都必须证明其必要性。能用简单方法解决的，绝不用复杂方法。有最好的，就不提供其他的选项。

> **反面案例：argparse 参数别名**
> ```python
> # 反模式：同时提供了多个可用选项 --timeframe 和 --tf，用户困惑该用哪个
> parser.add_argument('--timeframe', '--tf', ...)
> 
> # 合宪：只保留一种符合惯例的 --timeframe
> parser.add_argument('--timeframe', ...)
> ```
> argparse 惯例：长参数用 `--` + 全名，短参数才用单字母 `-t`。`--tf` 既不是标准短名又不是全名，属于不必要的选择。

### 第2条：透明

代码逻辑流必须是线性的、可追踪的。隐式行为、全局状态、深层嵌套的魔法方法均被禁止。错误和状态必须明确可见。

### 第3条：组合

小工具拼出大系统。每个组件都是独立的过滤器，通过接口连接，而非通过继承耦合。运行时组合优于编译时绑定。策略模式、适配器模式、装饰器模式 — 推荐。单例、抽象工厂、访问者、中介者 — 禁止。

---

## 第2章：代码设计规则

### 第4条：策略与机制分离

机制（技术实现）应稳定通用；策略（业务逻辑）可灵活变化。机制层不得包含业务逻辑。

> 具体边界规则（每层"可以/禁止"行为、违宪检测信号）见项目 `architecture.md` §层间边界规则。

### 第5条：纯函数优先

纯函数：输入确定 → 输出确定，不修改外部状态，不依赖参数外的任何状态。

```python
# 反模式：有状态方法 — 结果依赖 self 历史
class Calculator:
    def __init__(self):
        self._cache = {}

    def calc(self, data: list) -> float:
        key = hash(str(data))
        if key in self._cache:     # 依赖外部可变状态
            return self._cache[key]
        result = sum(data)
        self._cache[key] = result  # 副作用
        return result

# 合宪：纯函数
def calc(data: list) -> float:
    return sum(data)  # 同样输入永远同样输出
```

```python
# 反模式：构建器模式 — 链式修改 self
class Builder:
    def set_x(self, v): self._x = v; return self
    def set_y(self, v): self._y = v; return self
    def build(self): return Thing(self._x, self._y)

# 合宪：dataclass — 数据即数据
from dataclasses import dataclass

@dataclass(frozen=True)
class Params:
    x: int
    y: int

def build(p: Params) -> Thing:
    return Thing(p.x, p.y)
```

**Agent 检查点**:
- [ ] 方法是否同时读取和写入 `self.xxx`？（→ 拆为纯函数）
- [ ] 类是否有 5+ 个可变 `self._xxx` 属性？（→ 状态过多）
- [ ] 是否存在链式 `.set_xxx()` 返回 self 的构建器？（→ dataclass）
- [ ] 函数是否可以移出类成为模块级函数？（→ 如果只用 `__init__` 参数而不依赖可变状态）

### 第6条：显式依赖

依赖通过构造函数注入，禁止隐式依赖和全局状态。

```python
# 反模式
class Processor:
    def __init__(self):
        self.config = load_config()   # 隐式依赖
        self.logger = setup_logger()  # 隐式依赖
        self.db = connect_db()        # 隐式依赖

# 合宪
class Processor:
    def __init__(self, config: dict, logger: Logger, db: Database):
        self.config = config   # 显式注入
        self.logger = logger
        self.db = db
```

```python
# 反模式：模块级全局状态
_cache = {}
_config = {}

# 合宪：显式传递
def process(data: dict, cache: dict, config: dict) -> dict:
    ...
```

### 第7条：禁止行为清单

以下行为违反宪法，必须立即修正：

**上帝基类（严重违宪）**：
```python
# 反模式：全能基类
class BaseProcessor:
    def __init__(self):
        self.config = load_config()   # 隐式依赖
        self.logger = setup_logger()  # 隐式依赖
        self.db = connect_db()        # 隐式依赖
```

**复杂工厂模式（违宪）**：
```python
# 反模式：过度抽象的工厂
class AbstractProcessorFactory(ABC):
    @abstractmethod
    def create(self): ...

# 合宪：简单映射
PROCESSORS = {"data": DataProcessor, "report": ReportProcessor}
```

**全局状态（违宪）**：模块级可变变量、模块顶层实例化后隐式使用。

### 第8条：失败透明

```python
# 反模式 1：静默吞错
async def fetch():
    try:
        return await api.get()
    except Exception:
        pass  # ← 违宪

# 反模式 2：返回 None 代替异常
def calc(target: float, actual: float) -> float | None:
    if target <= 0:
        return None  # ← 违宪

# 反模式 3：raise 无上下文
raise ConfigError  # ← 违宪：不知道哪个文件、为什么失败

# 合宪
async def fetch():
    try:
        return await api.get()
    except ApiError as e:
        raise DataError(f"Failed to fetch: {e}") from e

def calc(target: float, actual: float) -> float:
    if target <= 0:
        raise ValueError(f"target must be positive, got {target}")
    return target - actual

raise ConfigError(f"Config file not found: {path}")
```

**Agent 检查点**:
- [ ] 存在 `except: pass` 或 `except Exception: pass`？（→ 严重违宪）
- [ ] 存在 `return None` 代替抛异常？（→ 改为 raise）
- [ ] 存在 `raise XxxError` 不带消息字符串？（→ 补上下文）
- [ ] `except` 块是否有日志记录或异常转换？（→ 不能静默吞）
- [ ] 初始化方法中有 `log.error(...); return` 而非 `raise`？（→ 违宪：Crash on Boot）

### 第8.1条：初始化即崩溃 (Crash on Boot)

系统初始化阶段遇到配置错误、缺失关键依赖等不可恢复故障时，必须抛出异常终止启动，禁止 `log.error + return` 或 `log.error + continue` 静默绕过。

```python
# 反模式 4：初始化阶段 log.error + return（继续运行残缺系统）
def _setup_component(self):
    config = self.main_config.get("component")
    if not config:
        self.logger.error("component not configured in main.yaml")
        return  # ← 违宪：启动继续，组件缺失，运行时再出问题难排查

# 合宪：crash early, crash loud
def _setup_component(self):
    config = self.main_config.get("component")
    if not config:
        raise ConfigError("component is required but not configured in main.yaml")
```

**判断标准**：`__init__` → `_initialize_system()` 调用链中的初始化方法，一旦确认无法正常运转，必须 raise。运行时方法（从 `run()` 下发）可以用 log + return 优雅降级。

**Agent 检查点**:
- [ ] 初始化方法中有 `log.error(...); return` 而非 `raise`？（→ 违宪：Crash on Boot）

### 第9条：协议优于继承

不要为"类型统一"强行引入抽象基类。Python 判断标准是对象能否满足行为契约，而非在类层级中的血缘。

```python
# 反模式：单子类 ABC
class Channel(ABC):
    @abstractmethod
    async def send(self, msg: str) -> None: ...

class TelegramChannel(Channel):
    async def send(self, msg: str) -> None: ...

# 合宪：鸭子类型 — 有 send 方法就能用
class TelegramChannel:
    async def send(self, msg: str) -> None: ...

class FeishuChannel:
    async def send(self, msg: str) -> None: ...

async def notify(channel, msg: str):
    await channel.send(msg)  # 不要求继承任何基类
```

```python
# 反模式：isinstance 类型分发
def process(thing):
    if isinstance(thing, list):
        ...
    elif isinstance(thing, dict):
        ...

# 合宪：hasattr 或 Protocol
from typing import Protocol

class Multipliable(Protocol):
    def __mul__(self, other): ...

def process(thing: Multipliable):
    return thing * 2
```

**Agent 检查点**:
- [ ] 仅 1 个子类的 ABC？（→ 删除 ABC）
- [ ] `isinstance(obj, SomeBase)` 用于分发逻辑？（→ Protocol/hasattr）

### 第10条：数据驱动

用数据结构（dict、dataclass）描述系统行为变化，而非类层级或 if/elif 分支。

```python
# 反模式：硬编码分支
def get_algo(name: str):
    if name == "twap":    return TWAPAlgo()
    elif name == "market": return MarketAlgo()
    elif name == "limit":  return LimitAlgo()

# 合宪：数据驱动映射
REGISTRY = {"twap": TWAPAlgo, "market": MarketAlgo, "limit": LimitAlgo}

def get_algo(name: str):
    cls = REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown: {name}")
    return cls()
```

```python
# 反模式：硬编码配置值
class Strategy:
    def __init__(self):
        self.max_slippage = 0.02   # 硬编码

# 合宪：配置注入
class Strategy:
    def __init__(self, config: dict):
        self.max_slippage = config["max_slippage"]
```

```python
# 反模式：类层级描述数据变体（类爆炸）
class BTCSpotOrder: ...
class BTCFutureOrder: ...
class ETHSpotOrder: ...

# 合宪：数据字段描述变体
@dataclass
class Order:
    symbol: str
    market_type: str  # "spot" | "future"
    side: str
    quantity: float
```

**Agent 检查点**:
- [ ] 5+ 分支的 if/elif/else 用于分发逻辑？（→ dict 映射）
- [ ] 新增变体需要修改现有函数？（→ 违反开闭原则）
- [ ] 数值常量硬编码在函数体内？（→ 提取到配置或参数）
- [ ] 以类型名区分类层级（`XxxType1`, `XxxType2`）？（→ 用数据字段）

### 第11条：小而专注

| 指标 | 上限 | 超标时动作 |
|------|------|-----------|
| 文件 | 300 行 | 拆分为多个模块 |
| 函数 | 50 行 | 提取子函数 |
| 类方法数 | 10 个 | 拆分为多个类 |
| 嵌套深度 | 3 层 | 提取函数/early return |

### 第12条：显式导入

```python
# 合宪
from typing import Dict, List
from .models import DataModel
from .services import process_data

# 违宪
from . import *          # 污染命名空间，禁止追踪
from module import *     # 同上
```

---

## 第3章：设计模式判定

### 第13条：违宪模式（禁止使用）

**单例模式** — 违反透明原则
- 特征签名：类级 `_instance = None` + `__new__` 返回缓存 / 模块顶层实例化后隐式使用
- 检测：grep `_instance`、`__new__` 中的返回缓存逻辑

**抽象工厂模式** — 违反简洁原则
- 特征签名：类名含 `Factory` + `@abstractmethod` 返回对象 / 三层以上工厂创建链
- 检测：grep 类名含 `Factory` + 有 `@abstractmethod`

**访问者模式** — 违反简单原则
- 特征签名：`visit_xxx` 方法分发 + `accept(visitor)` 方法
- 检测：文件中同时出现 `visit_` 和 `accept()`

**中介者模式** — 违反透明原则
- 特征签名：单个类持有所有"同事"引用 + 同事间不直接通信 + 方法仅做转发无自身逻辑
- 检测：`__init__` 注入 5+ 其他组件 + 方法体 `self.x.do_y()` 纯转发

**常见混淆**：
- 简单字典映射（`STRATEGY = {key: Class}`）不是工厂模式 → 合宪
- 抽象接口用于外部系统适配（ABC + 多个子类适配不同实现）不是过度抽象 → 合宪
- 多组件注入 + 由构造函数接收不是中介者 → 合宪（是依赖注入，第6条要求）

---

## 第4章：代码组织

### 第14条：模块组织

模块按功能正交组织，而非按类型。依赖方向：应用层 → 业务层 → 基础设施层 → 领域层。下层不 import 上层。领域层不包含 I/O。

### 第15条：错误类型

```python
# 合宪：标准错误层级
class AppError(Exception):
    """基础错误"""
    pass

class ConfigError(AppError):
    """配置错误"""
    pass

class DataError(AppError):
    """数据错误"""
    pass

class ValidationError(AppError):
    """验证错误"""
    pass
```

### 第16条：异步模式

```python
# 合宪
async def process_batch(items: list) -> dict:
    tasks = [get_item(item) for item in items]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return {
        item: result
        for item, result in zip(items, results)
        if not isinstance(result, Exception)
    }
```

---

## 第5章：测试

### 第17条：测试原则

- **纯函数优先**：数据处理和计算的单元测试
- **适配器集成测试**：外部 API 适配的集成测试
- **避免模拟过度**：只 mock 外部依赖（网络、文件系统、数据库）

```python
# 合宪：纯函数测试 — 无 mock
def test_calc():
    assert calc([1, 2, 3]) == 6
```

### 第18条：优化原则

1. 先写清晰代码，再优化热点 — 避免过早优化
2. 使用异步 I/O 处理网络请求 — 不阻塞事件循环
3. 批量处理减少外部调用

---
