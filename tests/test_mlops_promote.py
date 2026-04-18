from __future__ import annotations

import pytest

from src.mlops.promote_model import metric_gates_pass
from src.serving.mlflow_model_loader import parse_models_uri


def test_parse_models_uri_alias():
    name, alias, ver = parse_models_uri("models:/fraud_xgb_classifier@champion")
    assert name == "fraud_xgb_classifier"
    assert alias == "champion"
    assert ver is None


def test_parse_models_uri_version():
    name, alias, ver = parse_models_uri("models:/fraud_xgb_classifier/12")
    assert name == "fraud_xgb_classifier"
    assert alias is None
    assert ver == 12


def test_parse_models_uri_invalid():
    with pytest.raises(ValueError):
        parse_models_uri("runs:/abc/model")


def test_metric_gates_pass_empty():
    ok, errs = metric_gates_pass({"auprc": 0.9, "f1": 0.5}, None, None)
    assert ok and not errs


def test_metric_gates_pass_fail_auprc():
    ok, errs = metric_gates_pass({"auprc": 0.1}, min_auprc=0.5, min_f1=None)
    assert not ok
    assert any("auprc=" in e for e in errs)


def test_metric_gates_pass_missing_metric():
    ok, errs = metric_gates_pass({}, min_auprc=0.5, min_f1=None)
    assert not ok
    assert any("missing metric 'auprc'" in e for e in errs)
