"""Testes do dashboard provisionado.

O modo como um dashboard quebra é traiçoeiro: renomear uma métrica no código não gera
erro nenhum: o painel simplesmente para de desenhar, e ninguém percebe até precisar dele
num incidente. Estes testes amarram o JSON do painel às métricas que a API realmente
expõe.
"""

import json
import re
from pathlib import Path

import pytest
import yaml

RAIZ = Path(__file__).resolve().parents[1]
DASHBOARD = RAIZ / "monitoring" / "grafana" / "provisioning" / "dashboards" / "triage.json"
PROVEDOR = RAIZ / "monitoring" / "grafana" / "provisioning" / "dashboards" / "dashboards.yml"
DATASOURCE = RAIZ / "monitoring" / "grafana" / "provisioning" / "datasources" / "prometheus.yml"
PROMETHEUS = RAIZ / "monitoring" / "prometheus.yml"


@pytest.fixture(scope="module")
def painel() -> dict:
    """Dashboard versionado, já desserializado."""
    return json.loads(DASHBOARD.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def metricas_expostas() -> set[str]:
    """Séries que a API pode publicar, derivadas dos coletores declarados.

    Raspar `/metrics` seria a leitura mais direta, mas engana: uma métrica com rótulos
    só aparece na exposição depois de observada ao menos uma vez. O teste passaria a
    depender de gerar exatamente o tráfego certo, e um esquecimento viraria falso
    negativo. Os coletores declarados são a fonte completa.
    """
    from prometheus_client import Counter, Gauge, Histogram

    from triage.api import metrics as modulo

    sufixos = {
        Counter: ("_total", "_created"),
        Histogram: ("_bucket", "_sum", "_count", "_created"),
        Gauge: (),
    }

    nomes = set()
    for objeto in vars(modulo).values():
        for tipo, extras in sufixos.items():
            if isinstance(objeto, tipo):
                base = objeto.describe()[0].name
                nomes.add(base)
                nomes.update(f"{base}{sufixo}" for sufixo in extras)
    return nomes


def _consultas(painel: dict) -> list[tuple[str, str]]:
    """Pares (título do painel, expressão PromQL) de todo alvo do dashboard."""
    return [
        (p["title"], alvo["expr"])
        for p in painel["panels"]
        for alvo in p.get("targets", [])
        if "expr" in alvo
    ]


def test_dashboard_tem_pelo_menos_tres_paineis(painel):
    # O enunciado do projeto exige três; o dashboard entrega mais.
    assert len(painel["panels"]) >= 3


def test_todo_painel_tem_titulo_e_descricao(painel):
    """Painel sem descrição obriga quem olha a adivinhar o que a linha significa."""
    for p in painel["panels"]:
        assert p.get("title"), f"painel {p['id']} sem título"
        assert p.get("description"), f"painel {p['title']!r} sem descrição"


def test_todo_painel_aponta_para_a_fonte_de_dados_provisionada(painel):
    uid = yaml.safe_load(DATASOURCE.read_text(encoding="utf-8"))["datasources"][0]["uid"]

    for p in painel["panels"]:
        assert p["datasource"]["uid"] == uid, f"painel {p['title']!r} usa outra fonte"


def test_toda_metrica_consultada_existe_na_api(painel, metricas_expostas):
    """A consulta do painel precisa casar com o nome que a API publica."""
    referenciadas = {
        nome for _, expr in _consultas(painel) for nome in re.findall(r"\btriage_[a-z_]+", expr)
    }

    ausentes = {nome for nome in referenciadas if nome not in metricas_expostas}
    assert not ausentes, f"o dashboard consulta métricas que a API não expõe: {ausentes}"


def test_toda_consulta_tem_legenda(painel):
    """Série sem legenda vira uma linha anônima no gráfico."""
    for titulo, _ in _consultas(painel):
        alvos = next(p for p in painel["panels"] if p["title"] == titulo)["targets"]
        for alvo in alvos:
            assert alvo.get("legendFormat"), f"alvo sem legenda em {titulo!r}"


def test_series_com_significado_de_estado_usam_a_paleta_de_status(painel):
    """Urgência é estado, não identidade: verde, âmbar e vermelho, nessa ordem."""
    urgencias = next(p for p in painel["panels"] if "Urgências" in p["title"])
    cores = {
        override["matcher"]["options"]: override["properties"][0]["value"]["fixedColor"]
        for override in urgencias["fieldConfig"]["overrides"]
    }

    assert cores == {"normal": "#0ca30c", "atencao": "#fab219", "urgente": "#d03b3b"}


def test_paineis_de_multiplas_series_mostram_legenda(painel):
    for p in painel["panels"]:
        if p["type"] == "timeseries":
            assert p["options"]["legend"]["showLegend"] is True, f"{p['title']!r} sem legenda"


def test_nenhum_painel_usa_dois_eixos(painel):
    """Dois eixos y no mesmo gráfico permitem sugerir qualquer correlação."""
    for p in painel["panels"]:
        for override in p["fieldConfig"].get("overrides", []):
            ids = {prop["id"] for prop in override["properties"]}
            assert "custom.axisPlacement" not in ids, f"{p['title']!r} tem eixo secundário"


def test_intervalo_do_grafana_acompanha_o_scrape_do_prometheus():
    """Um `timeInterval` maior que o scrape achata os picos que o painel deve mostrar."""
    scrape = yaml.safe_load(PROMETHEUS.read_text(encoding="utf-8"))["global"]["scrape_interval"]
    grafana = yaml.safe_load(DATASOURCE.read_text(encoding="utf-8"))["datasources"][0]["jsonData"][
        "timeInterval"
    ]

    assert scrape == grafana


def test_prometheus_coleta_a_api_pelo_nome_do_servico():
    config = yaml.safe_load(PROMETHEUS.read_text(encoding="utf-8"))
    alvos = [
        alvo
        for job in config["scrape_configs"]
        for static in job.get("static_configs", [])
        for alvo in static["targets"]
    ]

    # Precisa ser o nome do serviço no compose, não localhost: dentro da rede do
    # Docker, localhost é o próprio container do Prometheus.
    assert "api:8000" in alvos


def test_dashboard_nao_e_editavel_pela_interface(painel):
    """A versão que vale é a do git; edição pela interface divergiria em silêncio."""
    provedor = yaml.safe_load(PROVEDOR.read_text(encoding="utf-8"))["providers"][0]

    assert painel["editable"] is False
    assert provedor["allowUiUpdates"] is False
