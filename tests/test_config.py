"""Testes da configuração central."""

from pathlib import Path

from triage.config import SEED, Settings, settings


def test_caminhos_derivam_de_data_dir(tmp_path):
    custom = Settings(data_dir=tmp_path)

    assert custom.raw_dir == tmp_path / "raw"
    assert custom.processed_dir == tmp_path / "processed"


def test_artefatos_derivam_de_models_dir(tmp_path):
    custom = Settings(models_dir=tmp_path)

    assert custom.sklearn_model_path == tmp_path / "model.pkl"
    assert custom.onnx_model_path == tmp_path / "model.onnx"
    assert custom.metrics_path == tmp_path / "metrics.json"


def test_variaveis_de_ambiente_sobrescrevem_defaults(monkeypatch):
    monkeypatch.setenv("TRIAGE_MODEL_BACKEND", "onnx")

    assert Settings().model_backend == "onnx"


def test_defaults_do_projeto():
    assert SEED == 42
    assert settings.kaggle_dataset == "saharalaa/medical-abstracts-tc-corpus"
    assert isinstance(settings.data_dir, Path)
    assert 0 < settings.test_size < 1
