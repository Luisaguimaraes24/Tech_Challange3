"""Valida a DAG sem subir o Airflow.

Serve ao CI: carregar a pasta de DAGs pega erro de import, ciclo de dependência e
tarefa órfã em segundos, sem banco de dados, scheduler ou container. Não substitui
executar a DAG — substitui descobrir que ela nem carrega só depois do deploy.

Uso:
    python scripts/validate_dag.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DAGS_DIR = PROJECT_ROOT / "dags"

ESPERADAS = {
    "ingest_data",
    "prepare_dataset",
    "train_model",
    "validate_model",
    "publish_model",
}


def main() -> int:
    """Carrega a pasta de DAGs e confere a estrutura esperada.

    Returns:
        0 se a DAG carregar e tiver a estrutura esperada, 1 caso contrário.
    """
    # O import do Airflow é local: este script roda num ambiente isolado, criado só
    # para a validação, e não deve influenciar o resto do projeto.
    from airflow.models import DagBag

    dag_bag = DagBag(dag_folder=str(DAGS_DIR), include_examples=False)

    if dag_bag.import_errors:
        for arquivo, erro in dag_bag.import_errors.items():
            print(f"ERRO ao importar {arquivo}:\n{erro}", file=sys.stderr)
        return 1

    # `dag_bag.dags` vem do parse dos arquivos; `get_dag()` consultaria o banco de
    # metadados, que não existe num ambiente de validação sem `airflow db migrate`.
    dag = dag_bag.dags.get("triage_training")
    if dag is None:
        print(
            f"DAG 'triage_training' não encontrada. Carregadas: {list(dag_bag.dags)}",
            file=sys.stderr,
        )
        return 1

    tarefas = {task.task_id for task in dag.tasks}
    if faltando := ESPERADAS - tarefas:
        print(f"Tarefas ausentes na DAG: {sorted(faltando)}", file=sys.stderr)
        return 1

    # Um ciclo impede o scheduler de montar a ordem de execução; o Airflow detecta
    # isso na validação da topologia.
    dag.validate()

    print(f"DAG 'triage_training' carregada com {len(tarefas)} tarefas: {sorted(tarefas)}")
    agendamento = getattr(dag.timetable, "description", type(dag.timetable).__name__)
    print(f"Agendamento: {agendamento} | catchup: {dag.catchup}")
    for task in sorted(dag.tasks, key=lambda t: t.task_id):
        downstream = sorted(task.downstream_task_ids)
        print(f"  {task.task_id} -> {downstream or '(fim)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
