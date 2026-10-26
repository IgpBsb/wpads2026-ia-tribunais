"""Modelo multicritério (TOPSIS) para seleção de infraestrutura de cargas de IA em tribunais.

Alternativas: A1 própria, A2 IaaS nacional, A3 API no exterior, A4 híbrida (A1 + A3).
Critérios: C1 energia, C2 emissões, C3 custo, C4 latência, C5 soberania, C6 exposição, C7 carga de TI.
Cenários: Justiça Estadual por porte (medianas do Justiça em Números 2025).

Análises:
  1. matriz de decisão e ranking determinístico por perfil de pesos;
  2. frequência de 1º lugar sob pesos aleatórios (Dirichlet) — só pesos, e pesos + parâmetros;
  3. sensibilidade ao número de tarefas por caso.

Uso: python mcda.py  (lê parametros.yaml; grava em ../resultados/mcda/)
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
SAIDA = AQUI.parent / "resultados" / "mcda"
ALT = ["A1", "A2", "A3", "A4"]
CRIT = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
HORAS_ANO = 8760


def carregar():
    p = yaml.safe_load((AQUI / "parametros.yaml").read_text(encoding="utf-8"))
    jn = pd.read_csv(RAIZ / "dados" / "jn_ultimo_ano_tribunais.csv")
    jn = jn[jn.justica == p["cenarios"]["ramo"]]
    cen = jn.groupby("porte")[["cnelet", "tptic"]].median().loc[p["cenarios"]["portes"]]
    return p, cen


def sortear(p, rng=None):
    """Achata o YAML em {grupo.param: valor}; com rng, sorteia uniforme em [min, max]."""
    v = {}

    def visita(no, prefixo):
        for k, x in no.items():
            nome = f"{prefixo}{k}"
            if isinstance(x, dict) and "base" in x:
                v[nome] = x["base"] if rng is None else rng.uniform(x["min"], x["max"])
            elif isinstance(x, dict):
                visita(x, nome + ".")

    for g in ["cenarios", "carga", "infraestrutura_dedicada", "A1_propria", "A2_iaas_br",
              "A3_api_ext", "A4_hibrida", "fatores_emissao_tCO2_MWh"]:
        visita(p[g], g + ".")
    c = p["cambio_brl_usd"]
    v["cambio"] = c["base"] if rng is None else rng.uniform(c["min"], c["max"])
    return v


def dedicada(v, tarefas, fator_lat, pue):
    """Energia (kWh/ano) e nº de servidores de uma infraestrutura dedicada sempre ligada."""
    dur = v["carga.duracao_s_por_tarefa"] * fator_lat
    horas_demanda = tarefas * dur / 3600
    n = math.ceil(horas_demanda / (HORAS_ANO * v["infraestrutura_dedicada.utilizacao_maxima_gpu"])) \
        + round(v["infraestrutura_dedicada.redundancia_servidores"])
    p_ocioso_kw = (v["carga.potencia_ociosa_gpu_W"]
                   + v["infraestrutura_dedicada.potencia_ociosa_resto_servidor_W"]) / 1000
    e_dyn_kwh = (v["carga.energia_dinamica_gpu_J_por_tarefa"]
                 * v["infraestrutura_dedicada.fator_sistema_dinamico"] / 3.6e6)
    energia = pue * (n * p_ocioso_kw * HORAS_ANO + tarefas * e_dyn_kwh)
    return energia, n


def api(v, tarefas):
    a = "A3_api_ext."
    energia = tarefas * v[a + "energia_Wh_por_consulta"] / 1000
    custo = tarefas * (v["carga.tokens_entrada_por_tarefa"] * v[a + "preco_entrada_USD_por_Mtok"]
                       + v["carga.tokens_saida_por_tarefa"] * v[a + "preco_saida_USD_por_Mtok"]) \
        / 1e6 * v["cambio"]
    lat = v[a + "tempo_primeiro_token_s"] + v["carga.tokens_saida_por_tarefa"] / v[a + "vazao_tok_s"]
    return energia, custo, lat


def propria(v, tarefas):
    a = "A1_propria."
    energia, n = dedicada(v, tarefas, 1.0, v[a + "pue"])
    capex = n * v[a + "capex_servidor_gpu_R"]
    custo = capex / v[a + "vida_util_anos"] + capex * v[a + "om_fracao_capex_ano"] \
        + energia * v[a + "tarifa_energia_R_kWh"]
    return energia, custo, v["carga.duracao_s_por_tarefa"]


def matriz(v, casos, pessoal_ti):
    """Matriz de decisão 4x7 (linhas A1..A4, colunas C1..C7)."""
    t = casos * v["cenarios.tarefas_por_caso"]
    f_br, f_ext = v["fatores_emissao_tCO2_MWh.brasil_sin"], v["fatores_emissao_tCO2_MWh.exterior_api"]

    e1, c1, l1 = propria(v, t)

    a = "A2_iaas_br."
    e2, n2 = dedicada(v, t, v[a + "latencia_fator"], v[a + "pue"])
    c2 = n2 * HORAS_ANO * v[a + "preco_instancia_gpu_USD_h"] * v["cambio"]
    l2 = v["carga.duracao_s_por_tarefa"] * v[a + "latencia_fator"]

    e3, c3, l3 = api(v, t)

    s = v["A4_hibrida.fracao_sensivel"]
    e4a, c4a, _ = propria(v, s * t)
    e4b, c4b, _ = api(v, (1 - s) * t)
    l4 = s * l1 + (1 - s) * l3

    linhas = [
        (e1, e1 * f_br, c1, l1, "A1_propria"),
        (e2, e2 * f_br, c2, l2, "A2_iaas_br"),
        (e3, e3 * f_ext, c3, l3, "A3_api_ext"),
        (e4a + e4b, e4a * f_br + e4b * f_ext, c4a + c4b, l4, "A4_hibrida"),
    ]
    m = np.array([[e / 1000, co2 / 1000, custo / 1000, lat,
                   v[g + ".soberania"], v[g + ".exposicao"],
                   100 * v[g + ".fte_operacao"] / pessoal_ti]
                  for e, co2, custo, lat, g in linhas])
    return m


def topsis(m, w, sentidos):
    """Coeficiente de proximidade (0-1, maior = melhor) com normalização vetorial."""
    r = m / np.sqrt((m ** 2).sum(axis=0))
    pond = r * w
    melhor = np.where(sentidos == "max", pond.max(axis=0), pond.min(axis=0))
    pior = np.where(sentidos == "max", pond.min(axis=0), pond.max(axis=0))
    d_mais = np.sqrt(((pond - melhor) ** 2).sum(axis=1))
    d_menos = np.sqrt(((pond - pior) ** 2).sum(axis=1))
    return d_menos / (d_mais + d_menos)


def frequencia_primeiro(p, cen, sentidos, variar_parametros):
    mc = p["monte_carlo"]
    rng = np.random.default_rng(mc["semente"])
    v_base = sortear(p)
    linhas = []
    for porte, (casos, pti) in cen.iterrows():
        cont = np.zeros(len(ALT))
        m_base = matriz(v_base, casos, pti)
        for _ in range(mc["n_sorteios"]):
            w = rng.dirichlet(np.full(len(CRIT), mc["dirichlet_alpha"]))
            m = matriz(sortear(p, rng), casos, pti) if variar_parametros else m_base
            cont[np.argmax(topsis(m, w, sentidos))] += 1
        linhas.append({"porte": porte, **dict(zip(ALT, 100 * cont / mc["n_sorteios"]))})
    return pd.DataFrame(linhas)


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    p, cen = carregar()
    sentidos = np.array([p["criterios"][c]["sentido"] for c in CRIT])
    nomes_c = [p["criterios"][c]["nome"] for c in CRIT]
    siglas = [p["alternativas"][a]["sigla"] for a in ALT]
    v = sortear(p)
    pd.set_option("display.width", 220)

    print("Cenários (mediana JN 2025, Justiça Estadual):")
    print(cen.to_string(), "\n")

    # 1. Matriz de decisão e ranking por perfil
    mats, rank = [], []
    for porte, (casos, pti) in cen.iterrows():
        m = matriz(v, casos, pti)
        mats.append(pd.DataFrame(m, columns=nomes_c, index=siglas).assign(porte=porte))
        for perfil, w in p["perfis_de_peso"].items():
            cc = topsis(m, np.array(w), sentidos)
            rank.append({"porte": porte, "perfil": perfil, **dict(zip(siglas, cc.round(3))),
                         "vencedora": siglas[int(np.argmax(cc))]})
    mats = pd.concat(mats)
    mats.to_csv(SAIDA / "matriz_decisao_base.csv")
    rank = pd.DataFrame(rank)
    rank.to_csv(SAIDA / "ranking_perfis.csv", index=False)
    print(mats.round(3).to_string(), "\n")
    print(rank.to_string(index=False), "\n")

    # 2. Frequência de 1º lugar (dominance frequency)
    fq_w = frequencia_primeiro(p, cen, sentidos, variar_parametros=False)
    fq_wp = frequencia_primeiro(p, cen, sentidos, variar_parametros=True)
    fq_w.to_csv(SAIDA / "freq_primeiro_so_pesos.csv", index=False)
    fq_wp.to_csv(SAIDA / "freq_primeiro_pesos_e_parametros.csv", index=False)
    print("Frequência de 1º lugar (%) — só pesos:\n", fq_w.round(1).to_string(index=False))
    print("Frequência de 1º lugar (%) — pesos + parâmetros:\n", fq_wp.round(1).to_string(index=False), "\n")

    # 3. Sensibilidade a tarefas por caso (perfil equilibrado)
    sens = []
    for k in [1, 3, 5]:
        vk = dict(v, **{"cenarios.tarefas_por_caso": k})
        for porte, (casos, pti) in cen.iterrows():
            cc = topsis(matriz(vk, casos, pti), np.array(p["perfis_de_peso"]["equilibrado"]), sentidos)
            sens.append({"tarefas_por_caso": k, "porte": porte, **dict(zip(siglas, cc.round(3)))})
    sens = pd.DataFrame(sens)
    sens.to_csv(SAIDA / "sensibilidade_tarefas_por_caso.csv", index=False)
    print(sens.to_string(index=False))

    # Figura: frequência de 1º lugar (pesos + parâmetros); mesma tipografia de experimento/figuras.py
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(6.3, 2.0))
    x = np.arange(len(cen))
    larg = 0.2
    for i, (a, s) in enumerate(zip(ALT, siglas)):
        ax.bar(x + (i - 1.5) * larg, fq_wp[a], larg, label=s)
    ax.set_xticks(x, [f"Porte {pt.lower()}" for pt in cen.index])
    ax.set_ylabel("Frequência de 1º lugar (%)")
    ax.set_ylim(0, 100)
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(SAIDA / "fig_freq_primeiro.pdf")
    fig.savefig(SAIDA / "fig_freq_primeiro.png", dpi=200)


if __name__ == "__main__":
    main()
