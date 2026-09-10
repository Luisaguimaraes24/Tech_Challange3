"""Testes do medidor de latência."""

import pytest

from triage.benchmark.latency import measure, summarize


def test_summarize_calcula_percentis():
    amostras = list(range(1, 101))  # 1..100 ms

    stats = summarize(amostras)

    assert stats["n"] == 100
    assert stats["min_ms"] == 1
    assert stats["max_ms"] == 100
    assert stats["p50_ms"] == pytest.approx(50, abs=1)
    assert stats["p95_ms"] == pytest.approx(95, abs=1)
    assert stats["p99_ms"] == pytest.approx(99, abs=1)


def test_summarize_ordena_amostras_desordenadas():
    assert summarize([9.0, 1.0, 5.0])["min_ms"] == 1.0


def test_summarize_com_amostra_unica():
    stats = summarize([4.0])

    assert stats["p50_ms"] == stats["p99_ms"] == 4.0


def test_summarize_rejeita_serie_vazia():
    with pytest.raises(ValueError, match="Nenhuma amostra"):
        summarize([])


def test_throughput_e_o_inverso_da_media():
    stats = summarize([2.0, 2.0, 2.0])  # 2 ms por chamada -> 500 req/s

    assert stats["throughput_rps"] == pytest.approx(500.0)


def test_measure_descarta_o_aquecimento():
    chamadas = []

    def call(text):
        chamadas.append(text)

    stats = measure(call, ["laudo"], iterations=10, warmup=3)

    assert stats["n"] == 10  # só as 10 medidas entram nas estatísticas
    assert len(chamadas) == 13  # mas o aquecimento realmente rodou


def test_measure_faz_rodizio_entre_os_textos():
    vistos = []

    stats = measure(vistos.append, ["a", "b"], iterations=4, warmup=0)

    assert vistos == ["a", "b", "a", "b"]
    assert stats["n"] == 4
