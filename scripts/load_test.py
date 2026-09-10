"""Gerador de carga para a API de triagem.

Existe por dois motivos. O primeiro é a demonstração: um dashboard sem tráfego mostra
linhas retas em zero, e não dá para avaliar painel nenhum assim. O segundo é medir sob
concorrência — a linha de base da Etapa 1 foi medida com um cliente por vez, e é sob
carga simultânea que a diferença entre motores de inferência aparece.

O tráfego usa laudos reais do split de teste, em rodízio, porque o custo do TF-IDF cresce
com o tamanho do documento e uma frase curta sintética daria números que não se sustentam.

Uso:
    python scripts/load_test.py --duration 60 --concurrency 8
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

from triage.config import SEED, settings

logger = logging.getLogger("load_test")

# Uma fração das requisições é inválida de propósito: um dashboard em que a taxa de erro
# nunca sai de zero não prova que o painel de erro funciona.
INVALID_RATIO = 0.05
INVALID_PAYLOAD = {"texto": "curto"}


class Resultado:
    """Acumula o desfecho das requisições de forma segura entre threads."""

    def __init__(self) -> None:
        """Inicializa contadores e trava de acesso."""
        self.latencias_ms: list[float] = []
        self.status: Counter[int] = Counter()
        self.urgencias: Counter[str] = Counter()
        self.falhas = 0
        self._lock = threading.Lock()

    def registrar(self, codigo: int, latencia_ms: float, urgencia: str | None) -> None:
        """Registra uma resposta.

        Args:
            codigo: Status HTTP devolvido.
            latencia_ms: Tempo da requisição, em milissegundos.
            urgencia: Nível classificado, quando houver.
        """
        with self._lock:
            self.latencias_ms.append(latencia_ms)
            self.status[codigo] += 1
            if urgencia:
                self.urgencias[urgencia] += 1

    def registrar_falha(self) -> None:
        """Registra uma requisição que nem chegou a receber resposta."""
        with self._lock:
            self.falhas += 1


def carregar_laudos(limite: int = 200) -> list[str]:
    """Carrega laudos reais do split de teste.

    Args:
        limite: Quantidade de laudos a carregar.

    Returns:
        Lista de textos de laudo.

    Raises:
        FileNotFoundError: Se o corpus não estiver em `data/raw/`.
    """
    from triage.data.loader import TEXT_COLUMN, load_split

    return load_split("test")[TEXT_COLUMN].head(limite).tolist()


def disparar(url: str, laudos: list[str], fim: float, resultado: Resultado, rng: random.Random):
    """Envia requisições até o instante de término.

    Args:
        url: Endpoint `/predict`.
        laudos: Laudos usados no rodízio.
        fim: Instante (monotônico) em que a thread deve parar.
        resultado: Acumulador compartilhado.
        rng: Gerador aleatório próprio da thread.
    """
    while time.monotonic() < fim:
        invalida = rng.random() < INVALID_RATIO
        corpo = INVALID_PAYLOAD if invalida else {"texto": rng.choice(laudos)}
        requisicao = urllib.request.Request(
            url,
            data=json.dumps(corpo).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        inicio = time.perf_counter()
        try:
            with urllib.request.urlopen(requisicao, timeout=10) as resposta:
                corpo_resposta = json.loads(resposta.read())
                resultado.registrar(
                    resposta.status,
                    (time.perf_counter() - inicio) * 1_000,
                    corpo_resposta.get("urgencia"),
                )
        except urllib.error.HTTPError as erro:
            # 422 é resposta legítima do serviço, não falha do teste de carga.
            resultado.registrar(erro.code, (time.perf_counter() - inicio) * 1_000, None)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            resultado.registrar_falha()


def main() -> int:
    """Executa o teste de carga e imprime o resumo.

    Returns:
        0 em caso de sucesso, 1 se a API estiver inacessível ou o corpus ausente.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000/predict")
    parser.add_argument("--duration", type=int, default=60, help="Duração em segundos.")
    parser.add_argument("--concurrency", type=int, default=8, help="Clientes simultâneos.")
    parser.add_argument("--output", type=Path, default=None, help="Arquivo JSON de saída.")
    args = parser.parse_args()

    logging.basicConfig(level=settings.log_level, format="%(asctime)s | %(message)s")

    try:
        laudos = carregar_laudos()
    except FileNotFoundError as erro:
        logger.error("%s", erro)
        return 1

    try:
        urllib.request.urlopen(args.url.replace("/predict", "/health"), timeout=5)
    except urllib.error.URLError as erro:
        logger.error("API inacessível em %s: %s. Suba a stack com `make up`.", args.url, erro)
        return 1

    logger.info(
        "Gerando carga por %ds com %d clientes simultâneos em %s",
        args.duration,
        args.concurrency,
        args.url,
    )

    resultado = Resultado()
    fim = time.monotonic() + args.duration
    threads = [
        threading.Thread(
            target=disparar,
            args=(args.url, laudos, fim, resultado, random.Random(SEED + indice)),
            daemon=True,
        )
        for indice in range(args.concurrency)
    ]

    inicio = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    decorrido = time.monotonic() - inicio

    ordenadas = sorted(resultado.latencias_ms)
    total = len(ordenadas)
    if not total:
        logger.error("Nenhuma requisição concluída.")
        return 1

    resumo = {
        "requisicoes": total,
        "duracao_s": round(decorrido, 2),
        "concorrencia": args.concurrency,
        "vazao_rps": round(total / decorrido, 1),
        "p50_ms": round(ordenadas[int(total * 0.50)], 3),
        "p95_ms": round(ordenadas[min(int(total * 0.95), total - 1)], 3),
        "p99_ms": round(ordenadas[min(int(total * 0.99), total - 1)], 3),
        "media_ms": round(statistics.fmean(ordenadas), 3),
        "status": dict(resultado.status),
        "urgencias": dict(resultado.urgencias),
        "falhas_de_conexao": resultado.falhas,
    }

    logger.info(
        "%d requisições em %.1fs | %.1f req/s | p50=%.2fms p95=%.2fms p99=%.2fms",
        resumo["requisicoes"],
        resumo["duracao_s"],
        resumo["vazao_rps"],
        resumo["p50_ms"],
        resumo["p95_ms"],
        resumo["p99_ms"],
    )
    logger.info("status: %s", resumo["status"])
    logger.info("urgências classificadas: %s", resumo["urgencias"])

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(resumo, indent=2), encoding="utf-8")
        logger.info("Resumo gravado em %s", args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
