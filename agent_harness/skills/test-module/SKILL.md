---
name: test-module
description: 为模块编写测试。提示词：为X模块写单测，跑X模块的测试，给X模块补测试，整理X模块测试，写单元测试，写测试，补测试，补充单元测试，进行单元测试，给X模块写测试，进行集成测试，做验收测试
---

# test-module — 综合测试指南

用户指定一个模块，为之编写三类测试。全部规则自包含。

---

## 0. 测试类型总览

| 类型 | 前缀约定 | 外部依赖 | 运行频率 |
|------|---------|---------|---------|
| **单元测试** | `utest_` | mock 所有外部依赖 | CI 每次提交 |
| **集成测试** | `itest_` | 真实外部服务 | CI 每日/手动 |
| **验收测试** | `mantest_` | mock 或真实，按需选择 | 功能完成后 |

三类测试按功能模块分目录存放，共享 `tests/<模块>/conftest.py`。

---

## 0a. 文件命名与目录规范

```
<project_root>/tests/
├── conftest.py                       # 全局 fixtures
├── <模块A>/                          # 按功能模块分目录
│   ├── conftest.py                   # 模块级 fixtures
│   ├── utest_<组件>.py
│   ├── itest_<组件>.py
│   └── mantest_<验收场景>.py
├── <模块B>/
│   ├── utest_<组件>.py
│   └── mantest_<场景>.py
└── ...
```

**规则**：
- 三类前缀：`utest_` / `itest_` / `mantest_`
- `__test__ = False` 标记手动测试文件（`mantest_` 中的纯脚本类）
- 测试子目录 **不设 `__init__.py`**，防止与源码目录同名冲突
- `conftest.py` 只放 pytest fixture，不涉及测试逻辑

对应 pytest 配置（以 `pyproject.toml` 为例）：

```toml
[tool.pytest.ini_options]
python_files = ["utest_*.py", "itest_*.py", "mantest_*.py", "test_*.py"]
```

---

## 0b. 各类型编写要旨

### 单元测试 (`utest_`)
- mock 所有外部依赖（网络、文件、时间）
- 纯函数零 mock
- 遵循 §1-§5 完整流程
- 运行：`pytest tests/<模块>/utest_<组件>.py -v`

### 集成测试 (`itest_`)
- 连接真实外部服务
- 使用公开端点（无需认证）或配置文件中的凭据
- 标记 `@pytest.mark.integration`
- 独立 fixture 封装连接生命周期
- 运行：`pytest tests/<模块>/itest_<组件>.py -v -m integration`

### 验收测试 (`mantest_`)
- 基于验收文档逐一验证
- 外部依赖按需 mock（mock 模式）或连真实（联调模式）
- 验收文档与测试脚本一一对应
- 运行：`pytest tests/<模块>/mantest_<组件>.py -v`

---

## 1. 摸清家底

不做任何决策，先搞清楚模块里有什么。

1. **列出目录结构** — 几个 `.py` 文件，文件大小，类名
2. **列出 public 方法** — 每个类的 public 方法签名、参数、返回类型
3. **列出 import 依赖** — 本模块 import 了哪些其他模块（排除标准库和第三方）
4. **设计文档对照** — 如果项目有设计文档，快速通读提取关键断言
5. **需求场景提取** — 如果项目有需求文档，提取场景矩阵，标记所有 P0 场景为测试强制覆盖目标

### 1.1 需求→测试映射表（强制）

摸清家底后，先输出一张映射表再往下走：

| 需求场景 | 设计对应 | 测试用例 | 状态 |
|---------|---------|---------|:----:|
| S1 正常流程 | §3.1 核心逻辑 | test_normal_flow | 待编写 |
| S2 边界场景 | §3.2 边界处理 | test_edge_case | 待编写 |
| ... | ... | ... | ... |

**P0 场景必须全覆盖，不允许遗漏。**

## 2. 拆分决策

基于步骤 1 的扫描结果，决定拆不拆、拆几个：

- **目录含 ≥2 个 `.py` 文件** → 每个 `.py` 一个独立 spec
- **单文件但 public 方法 ≥10 或外部依赖 ≥3** → 按功能组拆分
- **其他** → 不拆，一个 spec

> 输出命名按项目约定。目标：单 spec ≤ 300 行。已存在的超限 spec 按此规则回拆。

## 3. 依赖链检测

对步骤 1 列出的每个下游依赖，查是否已有测试覆盖。

如果存在未测试的底层依赖，提醒用户：

```
⚠️ {module} 的下游依赖中，以下尚无测试，建议自底向上：

  1. 纯数据/纯函数层    ← 应最先
  2. 中间业务层         ← 其次
  3. {目标模块}         ← 最后

纯数据/纯函数先测，中间业务层其次，App 编排最后。
底层未测时 mock 它测上层 → mock 的可能是 buggy 行为。

是否仍继续，还是先从底层开始？
```

## 4. 编写测试规格文档

对每个拆分后的 sub-module，产出测试规格文档。

### 4.1 深入源码（仅当前 sub-module）

- 读方法体逻辑、分支、异常路径
- 对照步骤 1 的设计期望做 diff：

  | 设计断言 | 源码 | 结论 |
  |---------|------|------|
  | ... | 符合 | ✅ |
  | ... | 不符 | ❌→P0 用例 |
  | 未覆盖 | 源码有 | ⚠️ 与用户确认 |

- 追踪调用方：每个方法被谁调、什么场景调

### 4.1a 数据格式一致性检查

测试中使用的 mock 数据格式必须与需求文档的数据格式严格一致：

- ✅ **字段名一致**：字段名、大小写、数组/对象
- ✅ **类型一致**：`string[]` vs `string`、`int` vs `float`
- ✅ **必填字段完整**：mock 数据不遗漏必填列

> 需求文档改了数据格式，mock 数据必须同步更新。测试绿但数据格式不对 = 测试虚假通过。

### 4.2 划定范围

```
是我自己的逻辑？ → NO → 排除（委托/平台行为/实现细节/非功能需求）
失败有破坏性？   → NO → 排除
能稳定验证？     → NO → 标记 @pytest.mark.integration
全部 YES        → 纳入
```

产出「纳入」表 +「排除」表 + 依赖隔离图（谁 mock、谁不 mock、谁走集成测试）。

### 4.3 优先级分级

| 维度 | 含义 |
|------|------|
| 隐蔽性 | 出错调用方能立刻感知？高=静默错误/返回假数据 |
| 影响面 | 多少 App/路径依赖？ |
| 回退难度 | 出错有替代方案？ |

```
隐蔽性=高 OR 回退难度=高 → P0（发布前必须修）
影响面≥中 AND 隐蔽性≤中   → P1（当前迭代修）
都不是                   → P2（排入下个迭代）
```

产出用例等级分布表。

### 4.4 定义 Fixture

需要的 fixture 标注来源：复用全局已有，或在本文件新定义。

### 4.5 列出测试用例

```markdown
#### P0-1: {描述}
> **覆盖方法**: xxx
> **风险**: xxx
​```python
async def test_xxx(): ...
​```
```

### 4.5a 测试命名规范

```
# 格式：test_<方法/场景>_<条件>_<预期>
def test_normalize_qty_step_down_aligns_correctly(): ...
def test_validate_qty_below_min_returns_false(): ...
def test_signal_processing_stale_timestamp_skipped(): ...
```

### 4.5b AsyncMock 使用规范

mock 外部依赖时遵循以下模式：

```python
# ✅ 正确：AsyncMock + return_value
mock_service = AsyncMock()
mock_service.get_data.return_value = {"status": "ok", "value": 42}

# ✅ side_effect 模拟多次调用不同返回值
mock_service.get_data.side_effect = [
    None,                          # 第一次调用：不存在
    {"status": "ok", "value": 1},  # 第二次调用：正常
]

# ✅ 验证调用参数
mock_service.process.assert_awaited_once_with(
    key="abc", option="xyz"
)
mock_service.process.assert_not_called()
```

### 4.5c Fixture 作用域选择

| 作用域 | 适用场景 | async 兼容 |
|-------|---------|:---------:|
| `function`（默认） | 大多数情况，每个测试独立状态 | ✅ |
| `module` | 共享连接池、配置对象（只读） | ✅ 用 `scope="module"` |
| `session` | 极少用 | ❌ 不兼容 async fixture |

```python
# ✅ function 级 — 默认，每个测试拿新实例
@pytest.fixture
def app():
    return MyApp("test", config)

# ✅ module 级 — 共享只读配置
@pytest.fixture(scope="module")
def base_config():
    return {"mode": "default", "timeout": 30}
```

### 4.5d 并发/并行测试验证

当设计中有并行逻辑时，测试需验证：

```python
# ✅ 验证部分成功、部分失败
async def test_gather_partial_failure(app, mock_service):
    mock_service.check.side_effect = [
        {"valid": True},   # 第一条成功
        None,              # 第二条不存在 → 跳过
    ]

    await app.process_batch([
        {"id": "A"}, {"id": "B"},
    ])

    # A 处理成功，B 被跳过
    assert mock_service.process.call_count == 1
```

### 4.5e 纯函数测试必须用 parametrize

所有纯函数测试禁止手写 `for` 循环，必须用 `@pytest.mark.parametrize`：

```python
# ✅ 正确：parametrize 覆盖所有边界
@pytest.mark.parametrize("n,expected", [
    (0, 0.0),    # 零值边界
    (1, 0.3),    # 最小值
    (5, 0.5),    # 中间值
    (10, 1.5),   # 上限饱和
])
def test_allocation(n, expected):
    assert compute_allocation(n) == expected

# ❌ 禁止：手写 for 循环
def test_bad():
    for n, exp in [(0, 0.0), (1, 0.3)]:
        assert compute_allocation(n) == exp
```

### 4.6 保存

**铁律：写完立即保存为文件**。后续调整在该文件上追加，不新建。

### 4.7 用户确认

保存后停止，将 spec 文档的摘要展示给用户，等待确认：

```
📄 {module}-test-spec.md 已保存

   纳入: X 个行为
   排除: Y 个行为
   用例: P0×A / P1×B / P2×C
   依赖隔离: mock(下游1, 下游2) / 不mock(下游3) / 集成测试(下游4)

是否确认，还是需要调整？
```

用户可能追问的方向：
- "P0 多了/少了某个用例" → 在 spec 中增删
- "这个方法是委托调用，应该排除" → 移到排除表
- "某个边界场景没覆盖" → 补充 P2 用例
- "依赖太多，再拆一个 spec" → 回到步骤 2

每次调整后更新 spec 文档，再次确认。直到用户说"可以"或"开始写代码"。

---

## 5. 实现测试代码

### 5.1 fixture 落位

- 共享 fixture → 追加到全局 `conftest.py`
- 本模块专用 fixture → 放在本模块测试目录的 `conftest.py`

### 5.2 测试文件命名规范

见 §0a 目录规范。按优先级分组：

```python
class TestP0: ...
class TestP1Xxx: ...
class TestP2: ...
```

### 5.3 运行验证

```bash
# 单类测试
pytest tests/<模块>/utest_<组件>.py -v             # 单元
pytest tests/<模块>/itest_<组件>.py -v -m integration  # 集成
pytest tests/<模块>/mantest_<组件>.py -v            # 验收

# 全量
pytest tests/
```

---

## 6. 收尾

1. 清理被取代的旧测试文件
2. 更新测试索引文档（按项目约定）
3. 追加操作摘要到工作日志（按项目约定）

---

## 禁止事项

- 不测委托调用、平台行为、实现细节
- 不测 logger 输出、异常消息措辞
- 不自行提取通用规则到 `docs/rules/`
- 不修改被测模块代码

---

## 附录：spec 文档模板

```markdown
# {Module} 测试规格

**需求文档**: `{project_specs}/{module}.requirements.md`
**设计文档**: `{project_specs}/{module}.design.md`
**基准代码**: `path/to/module.py`
**测试框架**: pytest + pytest-asyncio (auto mode)

---

## 一、测试范围

### 纳入
| 被测试行为 | 原因 |
|-----------|------|

### 排除
| 不测试 | 归类 |
|--------|------|

### 依赖隔离策略
（依赖图 + 隔离方式）

---

## 二、优先级定义

| 用例 | 隐蔽性 | 影响面 | 回退难度 | 等级 |
|------|:---:|:---:|:---:|:---:|

---

## 三、公共 Fixtures
（引用共享 + 定义专用）

---

## 四、测试用例

### P0 — 模块完整性
#### P0-1: ...
### P1 — 重要路径
#### P1-1: ...
### P2 — 边界场景
#### P2-1: ...

---

## 五、自检清单

| 检查项 | 结论 |
|--------|:----:|
| 所有 P0 场景有对应的测试用例 | ✅/❌ |
| mock 数据格式与需求文档一致 | ✅/❌ |
| 纯函数使用 `@pytest.mark.parametrize` 覆盖所有边界值 | ✅/❌ |
| 并发逻辑测试验证了部分成功/部分失败 | ✅/❌ |
| 配置校验测试覆盖了新增参数（缺失/类型错误/边界值） | ✅/❌ |
| 无手写 `for` 循环代替 parametrize | ✅/❌ |
| 无 print 代替 assert | ✅/❌ |
| 无手写 `asyncio.run`（由 pytest-asyncio 接管） | ✅/❌ |
| pytest 全量通过 | ✅/❌ |
```
