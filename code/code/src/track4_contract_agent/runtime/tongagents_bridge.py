# -*- coding: utf-8 -*-
"""TongAgents tools and run-local execution memory for full visual inspection."""

from __future__ import annotations

import importlib
import hashlib
import json
import os
import sys
import types
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from track4_contract_agent.assets import source_root

_HEADS_CALL: ContextVar[Any] = ContextVar("track4_heads_call", default=None)
_REVIEW_CALL: ContextVar[Any] = ContextVar("track4_review_call", default=None)
_PREPARED = False
_SHELL: Optional["TongAgentsShell"] = None

SKILL_REL = "prompts/skills/track4-inspect"
WHEEL_GLOB = "vendor/tongagents/tongagents-*.whl"


def reset_for_tests() -> None:
    global _SHELL
    _SHELL = None
    _HEADS_CALL.set(None)
    _REVIEW_CALL.set(None)


def _flag(name: str, default: str = "1") -> bool:
    return str(os.environ.get(name) or default).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class _StubAny:
    """万能占位：可实例化、可调用返回可继承类、可提供任意属性。"""
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        class _StubMeta(type):
            def __getattr__(cls, name):
                return _StubAny()
        return _StubMeta("_StubBase", (), {"__init__": lambda self, *a, **k: None})

    def __getattr__(self, n):
        return _StubAny()

    def __iter__(self):
        return iter(())

    def __bool__(self):
        return False


class _StubLoader:
    def create_module(self, spec):
        mod = types.ModuleType(spec.name)
        mod.__path__ = []  # type: ignore[attr-defined]
        mod.__getattr__ = lambda name: _StubAny()  # PEP 562 模块级惰性属性
        return mod

    def exec_module(self, module):
        pass


class _Track4StubFinder:
    """仅在真实包缺失时，为 tongagents 硬依赖提供离线 stub。

    tongagents 2.7.19 的 import 链路硬依赖 sqlalchemy / opentelemetry，
    比赛环境无这些包也无本地 wheel。本 finder 让 SDK 可导入；
    真实包存在时返回 None 交由正常加载，不覆盖。
    """
    _PREFIXES = ("sqlalchemy", "opentelemetry")

    def find_spec(self, fullname, path, target=None):
        if not any(fullname == p or fullname.startswith(p + ".") for p in self._PREFIXES):
            return None
        from importlib.machinery import ModuleSpec, PathFinder
        try:
            real = PathFinder.find_spec(fullname, path)
        except Exception:  # noqa: BLE001
            real = None
        if real is not None:
            return None
        return ModuleSpec(fullname, _StubLoader(), is_package=True)


def _ensure_pkg(name: str) -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    try:
        return importlib.import_module(name)
    except Exception:  # noqa: BLE001
        mod = types.ModuleType(name)
        mod.__path__ = []  # type: ignore[attr-defined]
        sys.modules[name] = mod
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(_ensure_pkg(parent), child, mod)
        return mod


def prepare_runtime() -> None:
    """Make TongAgents importable offline: no S3, no cloud OTEL, no LangChain."""
    global _PREPARED
    if _PREPARED:
        return
    if not any(isinstance(f, _Track4StubFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Track4StubFinder())
    try:
        import opentelemetry.context as otel_ctx

        if not hasattr(otel_ctx, "propagate"):
            otel_ctx.propagate = lambda *a, **k: None  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    class _Instr:
        def instrument(self, *a, **k):  # noqa: ANN001
            return None

    for name in (
        "opentelemetry.instrumentation",
        "opentelemetry.instrumentation.requests",
        "opentelemetry.instrumentation.httpx",
        "opentelemetry.instrumentation.aiohttp_client",
        "opentelemetry.instrumentation.grpc",
        "opentelemetry.instrumentation.logging",
        "opentelemetry.instrumentation.redis",
    ):
        mod = _ensure_pkg(name)
        for attr in (
            "RequestsInstrumentor",
            "HTTPXClientInstrumentor",
            "AioHttpClientInstrumentor",
            "GrpcInstrumentorClient",
            "LoggingInstrumentor",
            "RedisInstrumentor",
        ):
            if not hasattr(mod, attr):
                setattr(mod, attr, _Instr)

    try:
        import fsspec
        from fsspec.implementations.local import LocalFileSystem

        try:
            importlib.import_module("s3fs")
        except Exception:  # noqa: BLE001
            fsspec.register_implementation("s3", LocalFileSystem, clobber=True)
    except Exception:  # noqa: BLE001
        pass

    for name in (
        "langchain_core",
        "langchain_core.language_models",
        "langchain_core.messages",
        "langchain_core.outputs",
        "langchain_core.callbacks",
        "langchain_core.tools",
        "langchain_openai",
        "langchain_community",
        "langchain_community.chat_models",
        "langchain_ollama",
        "litellm",
        "json_repair",
        "mirothinker",
    ):
        _ensure_pkg(name)
    msg = sys.modules["langchain_core.messages"]
    for attr in (
        "BaseMessage",
        "HumanMessage",
        "AIMessage",
        "SystemMessage",
        "ToolMessage",
        "ToolCall",
        "FunctionMessage",
    ):
        if not hasattr(msg, attr):
            setattr(msg, attr, type(attr, (), {}))
    lm = sys.modules["langchain_core.language_models"]
    if not hasattr(lm, "BaseChatModel"):
        lm.BaseChatModel = type("BaseChatModel", (), {})  # type: ignore[attr-defined]
    tools = sys.modules["langchain_core.tools"]
    if not hasattr(tools, "BaseTool"):
        tools.BaseTool = type("BaseTool", (), {})  # type: ignore[attr-defined]
    jr = sys.modules["json_repair"]
    if not hasattr(jr, "repair_json"):
        jr.repair_json = lambda s, **k: s  # type: ignore[attr-defined]
    _PREPARED = True


def _wheel_path() -> Optional[Path]:
    root = source_root()
    if root is None:
        return None
    wheels = sorted(root.glob(WHEEL_GLOB))
    return wheels[-1] if wheels else None


def ensure_installed() -> Optional[str]:
    """Install the vendored wheel with no extra deps if the SDK is missing."""
    try:
        import importlib.metadata as md

        return md.version("tongagents")
    except Exception:  # noqa: BLE001
        pass
    wheel = _wheel_path()
    if wheel is None:
        return None
    import subprocess

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            str(wheel),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        return None
    try:
        import importlib.metadata as md

        return md.version("tongagents")
    except Exception:  # noqa: BLE001
        return None


class TongAgentsShell:
    def __init__(self):
        self.enabled=False;self.version='';self.memory=None
        self.heads_tool=None;self.review_tool=None
        self.skill_names=[];self.errors=[];self.calls=[];self.memory_count=0
        self.config_identity=''

    def snapshot(self):
        return {'enabled':self.enabled,'version':self.version,
                'tools':[n for n,t in [('embed_head_inspect',self.heads_tool),('teacher_review',self.review_tool)] if t is not None],
                'skills':self.skill_names,'memory':self.memory is not None,'memory_records':self.memory_count,
                'actual_calls':list(self.calls),'errors':list(self.errors),'react_hot_path':False}


def bootstrap_tongagents(config=None):
    global _SHELL
    cfg=dict(config or {})
    identity=hashlib.sha256(json.dumps(cfg,sort_keys=True,default=str).encode()).hexdigest()
    if _SHELL is not None and _SHELL.config_identity==identity:return _SHELL
    shell=TongAgentsShell();shell.config_identity=identity
    if not (cfg.get('tongagents') or {}).get('enabled',True):
        raise ValueError('Unified candidate requires TongAgents execution')
    prepare_runtime();version=ensure_installed()
    import tongagents
    from tongagents.tools.tool_manager import tool
    from tongagents.tools.skill.skill_loader import SkillLoader
    from tongagents.memory.memory import SimpleMemory
    shell.version=str(version or getattr(tongagents,'__version__',''))
    shell.memory=SimpleMemory()

    @tool(name='embed_head_inspect',description='全部图片实时视觉向量、行业分类头及训练案例判断，输出七字段候选')
    def embed_head_inspect() -> str:
        fn=_HEADS_CALL.get()
        if fn is None:raise ValueError('Missing visual operation context')
        return json.dumps(fn(),ensure_ascii=False,default=str)

    @tool(name='teacher_review',description='训练后27B对疑难及抽查图片独立看图，执行闭集格式验收和限时回退')
    def teacher_review() -> str:
        fn=_REVIEW_CALL.get()
        if fn is None:raise ValueError('Missing teacher operation context')
        return json.dumps(fn(),ensure_ascii=False,default=str)

    shell.heads_tool=embed_head_inspect;shell.review_tool=teacher_review
    root=source_root()
    if root is None:raise RuntimeError('Missing source assets')
    skills=SkillLoader(skills_dir=str(root/'prompts/skills')).load_all()
    shell.skill_names=[str(getattr(s,'name','') or getattr(s,'id','')) for s in skills]
    if 'track4-inspect' not in shell.skill_names:raise RuntimeError('Actual inspection Skill was not loaded')
    shell.enabled=True;_SHELL=shell
    return shell


def _invoke(name,fn):
    import time
    shell=_SHELL
    if shell is None or not shell.enabled:raise RuntimeError('TongAgents is not initialized')
    context,wrapped=(_HEADS_CALL,shell.heads_tool) if name=='embed_head_inspect' else (_REVIEW_CALL,shell.review_tool)
    token=context.set(fn);begin=time.perf_counter()
    try:
        raw=wrapped();info=json.loads(raw) if isinstance(raw,str) else raw
        if not isinstance(info,dict):raise RuntimeError('Invalid TongAgents tool return')
        call={'tool':name,'elapsed_s':time.perf_counter()-begin,'completed':True,
              'result_sha256':hashlib.sha256(json.dumps(info,sort_keys=True,default=str).encode()).hexdigest()}
        shell.calls.append(call)
        from tongagents.memory.memory import Message
        shell.memory.add_message(Message(meta_data={'tool_execution':call}))
        shell.memory_count+=1
        return dict(info,via='tongagents_tool')
    finally:
        context.reset(token)


def heads_with_shell(fn):return _invoke('embed_head_inspect',fn)
def review_with_shell(fn):return _invoke('teacher_review',fn)
