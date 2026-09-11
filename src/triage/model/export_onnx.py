"""Exportação do classificador para ONNX.

## Por que só o classificador, e não o pipeline inteiro

A exportação óbvia seria converter o pipeline completo — `TfidfVectorizer` mais
`LogisticRegression` — num único grafo. Ela é mais rápida: **2,0× a 2,5×** contra o
scikit-learn, porque elimina a travessia Python inteira. **E foi rejeitada.**

O motivo está na tokenização. O `TfidfVectorizer` tokeniza em Python com uma expressão
regular; o operador equivalente do ONNX reimplementa isso em C++, e as duas
implementações não coincidem. Medido no split de teste completo, o grafo completo
classifica **98,1%** dos laudos igual ao modelo treinado — ou seja, **56 laudos em 2.888
recebem urgência diferente**. Testamos as saídas prováveis e nenhuma resolveu:

| Configuração | Paridade | Ganho |
|---|---|---|
| (1,2) gramas, padrão original | 0,9806 | 2,48× |
| (1,2) gramas, `token_pattern` que o ONNX digere | 0,9841 | 2,26× |
| Apenas unigramas | 0,9965 | 4,20× |
| Vocabulário podado a 5.000 | 0,9868 | 7,98× |

Nem trocar o padrão de tokenização, nem abandonar bigramas, nem podar o vocabulário
levam a paridade ao limiar de 0,999. Um ganho de latência que muda a classificação de 2%
dos laudos não é otimização, é troca de modelo — e num sistema de triagem, é troca de
modelo sem validação clínica.

## O que foi feito

Exportamos **apenas o classificador**: a vetorização continua no scikit-learn e o ONNX
Runtime recebe o vetor de features já pronto. A paridade passa a ser exata (diferença
máxima de 1,7e-07, que é arredondamento de float32), e o ganho é menor, porém real.

O que torna essa escolha viável é a poda do vocabulário para 10.000 termos, feita no
treino por validação cruzada. Com os 50.000 originais, densificar o vetor esparso para
alimentar o ONNX custava mais do que o grafo economizava, e a "otimização" ficava **mais
lenta** que o original.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np

from triage.config import settings

logger = logging.getLogger(__name__)

OPSET = 18
"""Versão do conjunto de operadores. Fixada para que a exportação seja reprodutível."""

PARITY_THRESHOLD = 0.999
"""Fração mínima de predições idênticas para a exportação poder substituir o original.

Definido **antes** de medir, de propósito: um limiar escolhido depois do resultado é uma
justificativa, não um critério.
"""


def export(
    model_path: Path | None = None,
    onnx_path: Path | None = None,
    quantized_path: Path | None = None,
) -> dict[str, Path]:
    """Exporta o classificador do pipeline treinado para ONNX, e sua versão quantizada.

    Args:
        model_path: Pipeline scikit-learn de origem. Usa `settings.sklearn_model_path`.
        onnx_path: Destino do grafo. Usa `settings.onnx_model_path`.
        quantized_path: Destino do grafo INT8. Usa `settings.onnx_quantized_path`.

    Returns:
        Caminhos dos artefatos gerados, por nome.

    Raises:
        FileNotFoundError: Se o pipeline de origem não existir.
    """
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import FloatTensorType

    model_path = model_path or settings.sklearn_model_path
    onnx_path = onnx_path or settings.onnx_model_path
    quantized_path = quantized_path or settings.onnx_quantized_path

    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado em {model_path}. Rode `make train`.")

    pipeline = joblib.load(model_path)
    classificador = pipeline.named_steps["clf"]
    n_features = len(pipeline.named_steps["tfidf"].vocabulary_)

    # `zipmap=False` faz o grafo devolver um tensor de probabilidades em vez de uma lista
    # de dicionários. O ZipMap existe por compatibilidade e cobra alocação de dicionário
    # por predição — exatamente o custo que a exportação pretende eliminar.
    modelo = to_onnx(
        classificador,
        initial_types=[("features", FloatTensorType([None, n_features]))],
        target_opset=OPSET,
        options={id(classificador): {"zipmap": False}},
    )

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path.write_bytes(modelo.SerializeToString())
    logger.info(
        "Grafo ONNX salvo em %s (%d features, %.1f KB)",
        onnx_path,
        n_features,
        onnx_path.stat().st_size / 1024,
    )

    artefatos = {"onnx": onnx_path}
    if (quantizado := quantize(onnx_path, quantized_path)) is not None:
        artefatos["onnx_int8"] = quantizado

    return artefatos


def quantize(onnx_path: Path, quantized_path: Path) -> Path | None:
    """Tenta gerar a versão INT8 do grafo por quantização dinâmica.

    **Não se aplica a este modelo, e o motivo é estrutural.** O `skl2onnx` exporta a
    regressão logística como um único operador `LinearClassifier` do domínio
    `ai.onnx.ml`; o quantizador dinâmico do ONNX Runtime opera sobre `MatMul`, `Conv`,
    `Gather`, `LSTM` e afins, e não tem substituto INT8 para `LinearClassifier`. Não há
    o que quantizar.

    A tentativa fica no código de propósito, com o diagnóstico no log: quantização é uma
    técnica que paga em modelos dominados por multiplicações de matriz grandes, e uma
    camada linear de 3 × 10.000 não é um desses. Silenciar isso daria a impressão de que
    a técnica foi esquecida, quando na verdade foi avaliada e descartada com evidência.

    Args:
        onnx_path: Grafo de origem, em float32.
        quantized_path: Destino do grafo quantizado.

    Returns:
        O caminho do grafo INT8, ou `None` quando a quantização não é aplicável.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    try:
        quantize_dynamic(
            model_input=str(onnx_path),
            model_output=str(quantized_path),
            weight_type=QuantType.QInt8,
        )
    except Exception as erro:  # pragma: no cover - depende da versão do onnxruntime
        logger.warning(
            "Quantização dinâmica não se aplica: %s. O grafo é um LinearClassifier do "
            "domínio ai.onnx.ml, para o qual não existe operador INT8 equivalente.",
            erro,
        )
        return None

    logger.info(
        "Grafo INT8 salvo em %s (%.1f KB, %.0f%% do original)",
        quantized_path,
        quantized_path.stat().st_size / 1024,
        100 * quantized_path.stat().st_size / onnx_path.stat().st_size,
    )
    return quantized_path


def check_parity(
    texts: list[str],
    model_path: Path | None = None,
    onnx_path: Path | None = None,
) -> dict:
    """Compara as saídas do pipeline original e do caminho servido por ONNX.

    Args:
        texts: Laudos usados na comparação.
        model_path: Pipeline scikit-learn de referência.
        onnx_path: Grafo ONNX a verificar.

    Returns:
        Fração de predições idênticas, maior divergência de probabilidade e
        exemplos de discordância, quando houver.
    """
    import onnxruntime

    model_path = model_path or settings.sklearn_model_path
    onnx_path = onnx_path or settings.onnx_model_path

    pipeline = joblib.load(model_path)
    sessao = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    proba_sklearn = pipeline.predict_proba(texts)
    features = pipeline.named_steps["tfidf"].transform(texts).toarray().astype(np.float32)
    proba_onnx = np.asarray(sessao.run(None, {sessao.get_inputs()[0].name: features})[1])

    classes_sklearn = proba_sklearn.argmax(axis=1)
    classes_onnx = proba_onnx.argmax(axis=1)
    concordam = classes_sklearn == classes_onnx

    exemplos = [
        {
            "indice": int(indice),
            "sklearn": int(pipeline.classes_[classes_sklearn[indice]]),
            "onnx": int(pipeline.classes_[classes_onnx[indice]]),
            "trecho": texts[indice][:120],
        }
        for indice in np.flatnonzero(~concordam)[:3]
    ]

    return {
        "n": len(texts),
        "concordancia": round(float(concordam.mean()), 6),
        "divergencias": int((~concordam).sum()),
        "max_diff_probabilidade": float(f"{np.abs(proba_sklearn - proba_onnx).max():.3g}"),
        "exemplos_divergentes": exemplos,
        "opset": OPSET,
        "n_features": int(features.shape[1]),
    }


def main() -> int:
    """Exporta o classificador e verifica a paridade no split de teste.

    Returns:
        0 se a paridade atingir o limiar, 1 caso contrário.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Quantidade de laudos na verificação. 0 usa o split de teste inteiro.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Arquivo JSON onde gravar o relatório de paridade.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    from triage.data.loader import TEXT_COLUMN, load_split

    try:
        artefatos = export()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    textos = load_split("test")[TEXT_COLUMN]
    if args.sample:
        textos = textos.head(args.sample)

    relatorio = check_parity(textos.tolist())
    relatorio["artefatos_kb"] = {
        "model.pkl": round(settings.sklearn_model_path.stat().st_size / 1024, 1),
        **{nome: round(caminho.stat().st_size / 1024, 1) for nome, caminho in artefatos.items()},
    }

    logger.info(
        "Paridade: %.4f%% em %d laudos | %d divergências | maior diferença: %.1e",
        relatorio["concordancia"] * 100,
        relatorio["n"],
        relatorio["divergencias"],
        relatorio["max_diff_probabilidade"],
    )
    for exemplo in relatorio["exemplos_divergentes"]:
        logger.warning(
            "  divergência no índice %d: sklearn=%s onnx=%s | %s...",
            exemplo["indice"],
            exemplo["sklearn"],
            exemplo["onnx"],
            exemplo["trecho"],
        )

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False), "utf-8")
        logger.info("Relatório gravado em %s", args.report)

    if relatorio["concordancia"] < PARITY_THRESHOLD:
        logger.error(
            "Paridade %.4f abaixo do limiar %.4f: o grafo exportado classifica "
            "diferente do modelo treinado e não pode substituí-lo.",
            relatorio["concordancia"],
            PARITY_THRESHOLD,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
