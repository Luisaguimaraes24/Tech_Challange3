"""DAG de treino e publicação do classificador de urgência.

O ciclo de vida do modelo é a única parte assíncrona da arquitetura: a inferência é
síncrona, mas ingestão, treino e avaliação são cargas em lote, com perfil de recurso e
de falha completamente diferentes do serviço de API.

A DAG separa **treinar** de **publicar**. O treino escreve em `models/staging/`, que
ninguém serve; só o `publish_model` promove os artefatos para `models/`, que é o volume
que a API lê. Entre os dois há um portão de qualidade. A consequência prática é que um
retreino ruim falha antes de tocar em produção, em vez de substituir silenciosamente um
modelo bom por um pior — que é o modo mais comum de um pipeline de retreino automático
degradar um serviço ao longo de meses.

Ordem das tarefas:

    ingest_data -> prepare_dataset -> train_model -> validate_model -> publish_model
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from airflow.sdk import dag, task
from airflow.sdk.exceptions import AirflowFailException

logger = logging.getLogger(__name__)

SERVING_DIR = Path("/opt/airflow/models")
"""Volume compartilhado com a API: o que está aqui é o que está em produção."""

STAGING_DIR = SERVING_DIR / "staging"
"""Área de preparo: o treino escreve aqui e nada é servido a partir daqui."""

PUBLISHED_ARTIFACTS = ("model.pkl", "decision.json", "metrics.json")
"""Artefatos promovidos juntos. O modelo e sua regra de decisão precisam vir do mesmo
treino: um limiar calibrado para um modelo não vale para outro."""

DEFAULT_ARGS = {
    "owner": "mlet-fase3",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


@dag(
    dag_id="triage_training",
    description="Ingestão, treino, validação e publicação do classificador de urgência",
    schedule="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["mlet", "fase3", "treino"],
    doc_md=__doc__,
)
def triage_training():
    """Pipeline semanal de retreino do classificador de urgência."""

    @task
    def ingest_data() -> str:
        """Garante que o corpus esteja disponível no volume de dados.

        Returns:
            Caminho do diretório com os CSVs brutos.
        """
        from triage.data.loader import download_dataset

        raw_dir = download_dataset()
        logger.info("Corpus disponível em %s", raw_dir)
        return str(raw_dir)

    @task
    def prepare_dataset(raw_dir: str) -> dict:
        """Carrega os splits, aplica a rotulagem de urgência e registra a distribuição.

        A distribuição por classe é registrada de propósito: uma mudança brusca aqui
        entre execuções é o sinal mais barato de que a fonte de dados mudou.

        Args:
            raw_dir: Diretório com os CSVs brutos.

        Returns:
            Contagem de amostras e distribuição de urgência por split.
        """
        from triage.config import settings
        from triage.data.loader import URGENCY_COLUMN, load_split

        resumo: dict[str, dict] = {}
        settings.processed_dir.mkdir(parents=True, exist_ok=True)

        for split in ("train", "test"):
            frame = load_split(split, raw_dir=Path(raw_dir))
            destino = settings.processed_dir / f"{split}.parquet"
            frame.to_parquet(destino, index=False)

            resumo[split] = {
                "n": len(frame),
                "urgencias": frame[URGENCY_COLUMN].value_counts().to_dict(),
            }
            logger.info("%s: %d amostras -> %s", split, len(frame), destino)

        return resumo

    @task
    def train_model(raw_dir: str) -> dict:
        """Treina o classificador e grava os artefatos na área de preparo.

        Args:
            raw_dir: Diretório com os CSVs brutos.

        Returns:
            Métricas da avaliação no split de teste.
        """
        from triage.model.train import train

        STAGING_DIR.mkdir(parents=True, exist_ok=True)
        metrics = train(
            raw_dir=Path(raw_dir),
            model_path=STAGING_DIR / "model.pkl",
            metrics_path=STAGING_DIR / "metrics.json",
            decision_path=STAGING_DIR / "decision.json",
        )

        logger.info(
            "Treino concluído | recall_urgente=%.4f | f1_macro=%.4f | acurácia=%.4f",
            metrics["recall_urgente"],
            metrics["f1_macro"],
            metrics["accuracy"],
        )
        # XCom carrega só o que é pequeno e útil adiante; o relatório completo fica
        # no metrics.json em disco.
        return {
            "recall_urgente": metrics["recall_urgente"],
            "f1_macro": metrics["f1_macro"],
            "accuracy": metrics["accuracy"],
            "urgent_threshold": metrics["decision_rule"]["urgent_threshold"],
        }

    @task
    def validate_model(metrics: dict) -> dict:
        """Aplica os portões de qualidade antes de qualquer publicação.

        Args:
            metrics: Métricas devolvidas pelo treino.

        Returns:
            As mesmas métricas, quando aprovadas.

        Raises:
            AirflowFailException: Se o modelo ficar abaixo de algum piso configurado.
        """
        from triage.config import settings

        reprovacoes = []
        if metrics["recall_urgente"] < settings.min_recall_urgente:
            reprovacoes.append(
                f"recall de urgente {metrics['recall_urgente']:.4f} "
                f"< {settings.min_recall_urgente:.4f}"
            )
        if metrics["f1_macro"] < settings.min_f1_macro:
            reprovacoes.append(f"f1-macro {metrics['f1_macro']:.4f} < {settings.min_f1_macro:.4f}")

        if reprovacoes:
            # Falhar aqui é o comportamento desejado: o modelo em produção continua
            # intocado e a falha aparece no Airflow, em vez de degradar o serviço.
            raise AirflowFailException(
                "Modelo reprovado no portão de qualidade: " + "; ".join(reprovacoes)
            )

        logger.info("Modelo aprovado nos portões de qualidade.")
        return metrics

    @task
    def publish_model(metrics: dict) -> list[str]:
        """Promove os artefatos da área de preparo para o volume servido.

        Args:
            metrics: Métricas já validadas, usadas apenas para registro.

        Returns:
            Caminhos dos artefatos publicados.

        Raises:
            AirflowFailException: Se algum artefato esperado não estiver na área de preparo.
        """
        faltando = [nome for nome in PUBLISHED_ARTIFACTS if not (STAGING_DIR / nome).exists()]
        if faltando:
            raise AirflowFailException(f"Artefatos ausentes na área de preparo: {faltando}")

        publicados = []
        for nome in PUBLISHED_ARTIFACTS:
            destino = SERVING_DIR / nome
            shutil.copy2(STAGING_DIR / nome, destino)
            publicados.append(str(destino))
            logger.info("Publicado %s", destino)

        logger.info(
            "Modelo em produção atualizado | recall_urgente=%.4f | limiar=%s",
            metrics["recall_urgente"],
            metrics["urgent_threshold"],
        )
        return publicados

    raw_dir = ingest_data()
    dataset = prepare_dataset(raw_dir)
    metrics = train_model(raw_dir)

    # O treino não depende do parquet, mas não deve começar antes que a preparação
    # tenha confirmado que os dados carregam e estão íntegros.
    dataset >> metrics

    publish_model(validate_model(metrics))


dag_instance = triage_training()
