from __future__ import annotations

import builtins
import importlib
import sys

import numereng


def test_api_import_does_not_require_agentic_research(monkeypatch) -> None:
    original_import = builtins.__import__
    original_api_module = sys.modules.get("numereng.api")
    original_api_attr = getattr(numereng, "api", None)

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "numereng.api._agentic_research" or name.startswith("numereng.features.agentic_research"):
            raise ImportError("agentic_research_broken")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    sys.modules.pop("numereng.api", None)
    sys.modules.pop("numereng.api._agentic_research", None)
    for key in tuple(sys.modules):
        if key.startswith("numereng.features.agentic_research"):
            sys.modules.pop(key, None)

    try:
        api_module = importlib.import_module("numereng.api")

        assert callable(api_module.store_doctor)
        assert callable(api_module.cloud_aws_train_submit)
        assert callable(api_module.remote_train_launch)
    finally:
        if original_api_module is not None:
            sys.modules["numereng.api"] = original_api_module
            setattr(numereng, "api", original_api_attr)
