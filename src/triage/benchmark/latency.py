"""Medição de latência da triagem.

Mede em dois pontos, porque eles respondem a perguntas diferentes:

* **`model`** — só a inferência, em processo. É o número que a otimização da Etapa 4
  ataca e o único comparável entre `sklearn` e `onnx`.
* **`http`** — a requisição completa contra a API em execução. Inclui rede,
  serialização e validação Pydantic, e é o que o usuário do serviço realmente sente.

A diferença entre os dois mostra quanto do tempo de resposta é modelo e quanto é
overhead de serviço — sem isso, otimizar o modelo pode render ganho invisível na ponta.

Toda medição descarta um aquecimento inicial: a primeira inferência paga import
preguiçoso e alocação de buffers, e contaminaria os percentis.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path

from triage.config import settings

logger = logging.getLogger(__name__)

DEFAULT_ITERATIONS = 1_000
DEFAULT_WARMUP = 50


def summarize(samples_ms: Sequence[float]) -> dict[str, float]:
    """Resume uma série de latências em estatísticas comparáveis.

    Args:
        samples_ms: Latências individuais, em milissegundos.

    Returns:
        Dicionário com média, mediana, percentis 95 e 99, mínimo, máximo e vazão.

    Raises:
        ValueError: Se a série estiver vazia.
    """
    if not samples_ms:
        raise ValueError("Nenhuma amostra de latência coletada.")

    ordered = sorted(samples_ms)
    mean_ms = statistics.fmean(ordered)

    return {
        "n": len(ordered),
        "mean_ms": round(mean_ms, 3),
        "p50_ms": round(_percentile(ordered, 0.50), 3),
        "p95_ms": round(_percentile(ordered, 0.95), 3),
        "p99_ms": round(_percentile(ordered, 0.99), 3),
        "min_ms": round(ordered[0], 3),
        "max_ms": round(ordered[-1], 3),
        "throughput_rps": round(1_000 / mean_ms, 1) if mean_ms else 0.0,
    }


def measure(
    call: Callable[[str], object],
    texts: Sequence[str],
    iterations: int = DEFAULT_ITERATIONS,
    warmup: int = DEFAULT_WARMUP,
) -> dict[str, float]:
    """Executa uma função de inferência repetidas vezes e resume a latência.

    Args:
        call: Função que recebe o texto do laudo e devolve o resultado.
        texts: Laudos usados no rodízio das chamadas.
        iterations: Quantidade de chamadas medidas.
        warmup: Chamadas descartadas antes de começar a medir.

    Returns:
        Estatísticas de latência, no formato de `summarize`.
    """
    for index in range(warmup):
        call(texts[index % len(texts)])

    samples_ms: list[float] = []
    for index in range(iterations):
        text = texts[index % len(texts)]
        started = time.perf_counter()
        call(text)
        samples_ms.append((time.perf_counter() - started) * 1_000)

    return summarize(samples_ms)


def benchmark_model(
    texts: Sequence[str],
    backend_name: str | None = None,
    iterations: int = DEFAULT_ITERATIONS,
) -> dict:
    """Mede a latência de inferência em processo, sem camada HTTP.

    Args:
        texts: Laudos usados na medição.
        backend_name: Backend a carregar (`sklearn` ou `onnx`).
        iterations: Quantidade de chamadas medidas.

    Returns:
        Estatísticas de latência acrescidas do nome do backend medido.
    """
    from triage.api.predictor import Predictor

    predictor = Predictor.load(backend_name=backend_name)
    stats = measure(predictor.predict, texts, iterations=iterations)
    stats["backend"] = predictor.backend.name
    stats["mode"] = "model"
    return stats


def benchmark_http(
    texts: Sequence[str],
    url: str,
    iterations: int = DEFAULT_ITERATIONS,
) -> dict:
    """Mede a latência ponta a ponta contra a API em execução.

    Args:
        texts: Laudos usados na medição.
        url: Endereço completo do endpoint `/predict`.
        iterations: Quantidade de requisições medidas.

    Returns:
        Estatísticas de latência acrescidas da URL medida.

    Raises:
        RuntimeError: Se a API não estiver acessível.
    """

    def call(text: str) -> object:
        request = urllib.request.Request(
            url,
            data=json.dumps({"texto": text}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.read()

    try:
        call(texts[0])
    except urllib.error.URLError as exc:
        raise RuntimeError(f"API inacessível em {url}: {exc}. Suba o serviço antes.") from exc

    stats = measure(call, texts, iterations=iterations)
    stats["url"] = url
    stats["mode"] = "http"
    return stats


def load_sample_texts(limit: int = 100) -> list[str]:
    """Carrega laudos reais do split de teste para alimentar o benchmark.

    Medir com texto sintético curto distorceria o resultado: o custo do TF-IDF cresce
    com o tamanho do documento, e os abstracts do corpus têm ~1.200 caracteres.

    Args:
        limit: Quantidade de laudos a carregar.

    Returns:
        Lista de textos de laudo.

    Raises:
        FileNotFoundError: Se o corpus não estiver disponível em `data/raw/`.
    """
    from triage.data.loader import TEXT_COLUMN, load_split

    frame = load_split("test")
    return frame[TEXT_COLUMN].head(limit).tolist()


def main() -> int:
    """Roda o benchmark pela linha de comando e grava o resultado em JSON.

    Returns:
        0 em caso de sucesso, 1 se a medição não puder ser realizada.
    """
    parser = argparse.ArgumentParser(description="Mede a latência da triagem.")
    parser.add_argument(
        "--mode",
        choices=["model", "http", "both"],
        default="both",
        help="O que medir: inferência em processo, requisição HTTP ou ambos.",
    )
    parser.add_argument("--backend", default=None, help="Backend de inferência a medir.")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--url", default="http://localhost:8000/predict")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Arquivo JSON de saída. Sem isso, apenas imprime no log.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    try:
        texts = load_sample_texts()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    results: list[dict] = []
    try:
        if args.mode in ("model", "both"):
            results.append(benchmark_model(texts, args.backend, args.iterations))
        if args.mode in ("http", "both"):
            results.append(benchmark_http(texts, args.url, args.iterations))
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    for stats in results:
        logger.info(
            "%-6s | n=%d | p50=%.3fms p95=%.3fms p99=%.3fms | %.1f req/s",
            stats["mode"],
            stats["n"],
            stats["p50_ms"],
            stats["p95_ms"],
            stats["p99_ms"],
            stats["throughput_rps"],
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info("Resultados gravados em %s", args.output)

    return 0


def _percentile(ordered: Sequence[float], fraction: float) -> float:
    """Devolve o percentil de uma série já ordenada.

    Args:
        ordered: Amostras em ordem crescente.
        fraction: Percentil desejado, entre 0 e 1.

    Returns:
        O valor no percentil pedido.
    """
    index = min(int(round(fraction * len(ordered) + 0.5)) - 1, len(ordered) - 1)
    return ordered[max(index, 0)]


if __name__ == "__main__":
    sys.exit(main())
