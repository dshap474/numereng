from __future__ import annotations

import importlib.util
import sys
from contextlib import ExitStack
from importlib.resources import as_file, files


def _load_module():
    root = files("numereng.assets").joinpath("shipped_skills/numerai-api-ops/scripts/numerai_api_ops.py")
    with ExitStack() as stack:
        path = stack.enter_context(as_file(root))
        spec = importlib.util.spec_from_file_location("numerai_api_ops_skill", path)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


def test_run_create_model_verifies_against_authenticated_account_models(monkeypatch) -> None:
    module = _load_module()
    calls: list[tuple[str, dict[str, object] | None, bool]] = []

    def fake_graphql_request(query, variables=None, *, auth_required, auth_header):
        calls.append((query, variables, auth_required))
        if query == module.ADD_MODEL_MUTATION:
            return {"addModel": {"id": "model-1", "name": "lgbm_cross_scope", "tournament": 8}}
        if query == module.ACCOUNT_MODELS_QUERY:
            return {
                "account": {
                    "models": [
                        {
                            "id": "model-1",
                            "name": "lgbm_cross_scope",
                            "tournament": 8,
                            "username": "lgbm_cross_scope",
                            "computePickleUpload": None,
                        }
                    ]
                }
            }
        raise AssertionError(f"Unexpected query: {query}")

    monkeypatch.setattr(module, "graphql_request", fake_graphql_request)

    result = module.run_create_model({"name": "lgbm_cross_scope"}, "Token abc", False)

    assert result["result"]["id"] == "model-1"
    assert result["verification"]["id"] == "model-1"
    assert calls == [
        (module.ADD_MODEL_MUTATION, {"name": "lgbm_cross_scope", "tournament": 8}, True),
        (module.ACCOUNT_MODELS_QUERY, None, True),
    ]


def test_run_assign_compute_pickle_verifies_assignment_from_account_model_state(monkeypatch) -> None:
    module = _load_module()
    calls: list[tuple[str, dict[str, object] | None, bool]] = []

    def fake_graphql_request(query, variables=None, *, auth_required, auth_header):
        calls.append((query, variables, auth_required))
        if query == module.ASSIGN_PICKLE_MUTATION:
            return {"assignPickleToModel": True}
        if query == module.ACCOUNT_MODELS_QUERY:
            return {
                "account": {
                    "models": [
                        {
                            "id": "slot-2",
                            "name": "lgbm_cross_scope",
                            "tournament": 8,
                            "username": "lgbm_cross_scope",
                            "computePickleUpload": {
                                "id": "pickle-7",
                                "filename": "model.pkl",
                                "modelId": "slot-1",
                                "assignedModelSlots": ["slot-2"],
                                "validationStatus": "validated",
                                "diagnosticsStatus": "succeeded",
                                "triggerStatus": "submission_succeeded",
                                "insertedAt": "2026-04-14T00:00:00Z",
                                "updatedAt": "2026-04-14T00:00:00Z",
                            },
                        }
                    ]
                }
            }
        if query == module.COMPUTE_PICKLES_QUERY:
            return {
                "computePickles": [
                    {
                        "id": "pickle-7",
                        "filename": "model.pkl",
                        "modelId": "slot-1",
                        "assignedModelSlots": ["slot-2"],
                        "validationStatus": "validated",
                        "diagnosticsStatus": "succeeded",
                        "triggerStatus": "submission_succeeded",
                        "insertedAt": "2026-04-14T00:00:00Z",
                        "updatedAt": "2026-04-14T00:00:00Z",
                    }
                ]
            }
        raise AssertionError(f"Unexpected query: {query}")

    monkeypatch.setattr(module, "graphql_request", fake_graphql_request)

    result = module.run_assign_compute_pickle(
        {"model_id": "slot-2", "pickle_id": "pickle-7"},
        "Token abc",
        False,
    )

    assert result["result"] is True
    assert result["verification"]["model"]["computePickleUpload"]["id"] == "pickle-7"
    assert result["verification"]["pickles"][0]["assignedModelSlots"] == ["slot-2"]
    assert calls == [
        (module.ASSIGN_PICKLE_MUTATION, {"modelId": "slot-2", "pickleId": "pickle-7"}, True),
        (module.ACCOUNT_MODELS_QUERY, None, True),
        (
            module.COMPUTE_PICKLES_QUERY,
            {"id": "pickle-7", "modelId": None, "unassigned": False},
            True,
        ),
    ]
