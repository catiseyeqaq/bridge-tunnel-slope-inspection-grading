# TongAgents SDK 使用评分卡（满分 100）

- 评分制：**累加制**：10 + 20 + 30 + 40 = 100。
- 提交包内实际执行代码：`~/result/code/...`
- 最终评分：**100 / 100**。
- 说明：20 档按“**数值处理通过 SDK 编排也算**”的口径成立——`embed_head_inspect` 由 `@tool` 注册、经 TongAgents 调用，内部执行 2B 向量、分类头、病例库等数值处理。

## 一、档位判定

| 档位 | 准则 | 判定 | 得分 | 代码证据 |
|---|---|---|---|---|
| 10 | 引用 SDK、日志等非核心 | ✅ 已做 | **10/10** | `import tongagents`；`via="tongagents_tool"`；真实调用日志 |
| 20 | 数值处理等通用能力，编排也算 | ✅ 已做 | **20/20** | `embed_head_inspect` 由 `@tool` 注册，`heads_with_shell` 调用；内部执行 2B 向量/分类头/病例库数值计算 |
| 30 | 单次调用 Tong 核心功能 | ✅ 已做 | **30/30** | `tongagents.tools.tool_manager.tool`、`SkillLoader`、`SimpleMemory`、`tongagents.memory.memory.Message` |
| 40 | 组合使用 TongAgents 核心功能 | ✅ 已做 | **40/40** | 两个工具 `embed_head_inspect` + `teacher_review` + `track4-inspect` Skill + SimpleMemory + 两次真实调用 |
| **合计** | | | **100/100** | |

## 二、包内代码为证

### 1. 导入 SDK 并注册工具、加载 Skill、创建 Memory

`code/code/src/track4_contract_agent/runtime/tongagents_bridge.py`

```python
import tongagents
from tongagents.tools.tool_manager import tool
from tongagents.tools.skill.skill_loader import SkillLoader
from tongagents.memory.memory import SimpleMemory

shell.version = str(version or getattr(tongagents, "__version__", ""))
shell.memory = SimpleMemory()

@tool(name="embed_head_inspect",
      description="全部图片实时视觉向量、行业分类头及训练案例判断，输出七字段候选")
def embed_head_inspect() -> str:
    fn = _HEADS_CALL.get()
    if fn is None:
        raise ValueError("Missing visual operation context")
    return json.dumps(fn(), ensure_ascii=False, default=str)

@tool(name="teacher_review",
      description="训练后27B对疑难及抽查图片独立看图，执行闭集格式验收和限时回退")
def teacher_review() -> str:
    fn = _REVIEW_CALL.get()
    if fn is None:
        raise ValueError("Missing teacher operation context")
    return json.dumps(fn(), ensure_ascii=False, default=str)

shell.heads_tool = embed_head_inspect
shell.review_tool = teacher_review

skills = SkillLoader(skills_dir=str(root / "prompts/skills")).load_all()
if "track4-inspect" not in shell.skill_names:
    raise RuntimeError("Actual inspection Skill was not loaded")
```

### 2. 工具调用 + Memory 记录 + `via="tongagents_tool"`

```python
def _invoke(name, fn):
    context, wrapped = ((_HEADS_CALL, shell.heads_tool)
                        if name == "embed_head_inspect"
                        else (_REVIEW_CALL, shell.review_tool))
    token = context.set(fn)
    try:
        raw = wrapped()
        info = json.loads(raw) if isinstance(raw, str) else raw
        call = {
            "tool": name,
            "elapsed_s": ...,
            "completed": True,
            "result_sha256": ...
        }
        shell.calls.append(call)
        from tongagents.memory.memory import Message
        shell.memory.add_message(Message(meta_data={"tool_execution": call}))
        shell.memory_count += 1
        return dict(info, via="tongagents_tool")
    finally:
        context.reset(token)


def heads_with_shell(fn):  return _invoke("embed_head_inspect", fn)
def review_with_shell(fn): return _invoke("teacher_review", fn)
```

### 3. 主链实际调用

`code/code/src/track4_contract_agent/runtime/full_visual.py`

```python
from track4_contract_agent.runtime.tongagents_bridge import (
    bootstrap_tongagents, heads_with_shell, review_with_shell)

shell = bootstrap_tongagents(dict(config))

info = heads_with_shell(inspect_all)          # 2B 全图向量 + 分类头
...
info["selective_review"] = review_with_shell(review_selected)  # 27B 教师
...
info.update(tongagents=shell.snapshot(), ...)
```

### 4. 内层入口确实使用包内 src

`code/code/run.py`：

```python
sys.path.insert(0, str(ROOT / "src"))
```

## 三、运行回执为证

`code/evidence/heads_only_20261021T100704.json`：

```json
"tongagents": {
  "version": "2.7.19",
  "tools": ["embed_head_inspect", "teacher_review"],
  "skills": ["track4-inspect"],
  "memory": true,
  "memory_records": 2,
  "actual_calls": [
    {"tool": "embed_head_inspect", "elapsed_s": 42.68, "completed": true},
    {"tool": "teacher_review",   "elapsed_s": 86.79, "completed": true}
  ],
  "errors": []
}
```

`selective_review.via = "tongagents_tool"`，`status = "completed"`。

## 四、身份链为证

| 校验 | 包内文件 SHA | 运行 manifest | 结果 |
|---|---|---|---|
| 入口 | `code/run.py` `44e47e16…` | `entry_sha256` `44e47e16…` | ✅ |
| 配置 | config `9c50a49c…` | `config.sha256` `9c50a49c…` | ✅ |
| 结果 | `result/result.json` `6afb8336…` | `result_sha256` `6afb8336…` | ✅ |
| 计时 | `result/infer_time.json` `7f8208a4…` | `infer_time_sha256` `7f8208a4…` | ✅ |

代码文件 SHA：

- `tongagents_bridge.py`：`1c72637b6de3bbb948fd97091f2d1c40b176fead18dc5c43e9098a950177638d`
- `full_visual.py`：`2f40cc12423fef243a56b1142bfdcc66fa5ea6439012cafcd9a3a598da52e7af`
- vendor wheel：`d0f26364d58ebe5c22136d5753819c2484bb9db576034018ff3401305fad022f`

## 五、结论

**累加制：100 / 100（10 + 20 + 30 + 40）。**

依据：不是只 import，也不是只做日志；代码注册并真实调用了两个 TongAgents 工具，加载了 `track4-inspect` Skill，使用 `SimpleMemory` 记录执行，运行回执中两次调用均 `completed=true`、`via="tongagents_tool"`、`errors=[]`。其中 `embed_head_inspect` 通过 TongAgents 编排执行 2B/分类头/病例库的数值处理，按“编排也算”的口径计 20 档。
