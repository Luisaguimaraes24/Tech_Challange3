"""Comparativo de latência entre os motores de inferência.

A Etapa 3 mediu que três execuções consecutivas do mesmo binário variam até 40% no p95.
Isso condiciona a metodologia aqui: **um número por backend não separa ganho de ruído.**
Cada backend é medido em várias execuções independentes e o que se reporta é a mediana
das execuções, com o intervalo observado ao lado — sem o intervalo, o leitor não tem como
julgar se a diferença é real.

Os backends são intercalados em vez de medidos em bloco. Se o backend A rodasse inteiro
e depois o B, qualquer variação de carga da máquina ao longo do tempo — outro processo,
escalonamento térmico — seria atribuída à diferença entre eles.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from triage.config import settings

logger = logging.getLogger(__name__)

BACKENDS = ("sklearn", "onnx")
"""Motores comparados. `onnx-int8` fica de fora: ver `export_onnx.quantize`."""

DEFAULT_REPEATS = 5
DEFAULT_ITERATIONS = 500
WARMUP = 50


def _percentil(ordenadas: Sequence[float], fracao: float) -> float:
    """Percentil de uma série já ordenada."""
    indice = min(int(round(fracao * len(ordenadas) + 0.5)) - 1, len(ordenadas) - 1)
    return ordenadas[max(indice, 0)]


def _uma_execucao(call: Callable[[str], object], textos: Sequence[str], n: int) -> dict:
    """Mede uma execução isolada.

    Args:
        call: Função de inferência.
        textos: Laudos em rodízio.
        n: Quantidade de chamadas medidas.

    Returns:
        Percentis da execução, em milissegundos.
    """
    for indice in range(WARMUP):
        call(textos[indice % len(textos)])

    amostras = []
    for indice in range(n):
        inicio = time.perf_counter()
        call(textos[indice % len(textos)])
        amostras.append((time.perf_counter() - inicio) * 1_000)

    amostras.sort()
    return {
        "p50": _percentil(amostras, 0.50),
        "p95": _percentil(amostras, 0.95),
        "p99": _percentil(amostras, 0.99),
    }


def comparar(
    textos: Sequence[str],
    backends: Sequence[str] = BACKENDS,
    repeticoes: int = DEFAULT_REPEATS,
    iteracoes: int = DEFAULT_ITERATIONS,
) -> dict:
    """Mede cada backend em execuções repetidas e intercaladas.

    Args:
        textos: Laudos usados na medição.
        backends: Motores a comparar.
        repeticoes: Execuções independentes por backend.
        iteracoes: Chamadas medidas em cada execução.

    Returns:
        Estatísticas por backend e o ganho relativo ao primeiro da lista.
    """
    from triage.api.predictor import Predictor

    preditores = {nome: Predictor.load(backend_name=nome) for nome in backends}
    execucoes: dict[str, list[dict]] = {nome: [] for nome in backends}

    for rodada in range(repeticoes):
        for nome in backends:
            resultado = _uma_execucao(preditores[nome].predict, textos, iteracoes)
            execucoes[nome].append(resultado)
            logger.info(
                "rodada %d/%d | %-8s p50=%.3fms p95=%.3fms",
                rodada + 1,
                repeticoes,
                nome,
                resultado["p50"],
                resultado["p95"],
            )

    resumo = {}
    for nome, rodadas in execucoes.items():
        resumo[nome] = {
            metrica: {
                "mediana": round(statistics.median(r[metrica] for r in rodadas), 3),
                "min": round(min(r[metrica] for r in rodadas), 3),
                "max": round(max(r[metrica] for r in rodadas), 3),
            }
            for metrica in ("p50", "p95", "p99")
        }
        resumo[nome]["execucoes"] = len(rodadas)

    referencia = backends[0]
    for nome in backends[1:]:
        for metrica in ("p50", "p95", "p99"):
            base = resumo[referencia][metrica]["mediana"]
            atual = resumo[nome][metrica]["mediana"]
            resumo[nome].setdefault("ganho", {})[metrica] = round(base / atual, 3)

    return {
        "referencia": referencia,
        "iteracoes_por_execucao": iteracoes,
        "repeticoes": repeticoes,
        "n_laudos": len(textos),
        "backends": resumo,
    }


def main() -> int:
    """Roda o comparativo e grava o resultado.

    Returns:
        0 em caso de sucesso, 1 se algum artefato estiver ausente.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    from triage.benchmark.latency import load_sample_texts

    try:
        textos = load_sample_texts()
        resultado = comparar(textos, repeticoes=args.repeats, iteracoes=args.iterations)
    except (FileNotFoundError, ValueError) as erro:
        logger.error("%s", erro)
        return 1

    print()
    print(f"{'backend':<10s} {'p50 (ms)':>22s} {'p95 (ms)':>22s} {'ganho p50':>10s}")
    print("-" * 68)
    for nome, dados in resultado["backends"].items():
        ganho = dados.get("ganho", {}).get("p50")
        faixa = {
            metrica: f"{d['mediana']:.3f} [{d['min']:.3f}-{d['max']:.3f}]"
            for metrica, d in dados.items()
            if metrica in ("p50", "p95")
        }
        print(
            f"{nome:<10s} {faixa['p50']:>22s} {faixa['p95']:>22s} "
            f"{(f'{ganho:.2f}x' if ganho else '-'):>10s}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(resultado, indent=2), encoding="utf-8")
        logger.info("Resultado gravado em %s", args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
