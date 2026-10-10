"""Figuras do artigo a partir dos resultados consolidados.

Saídas em resultados/figuras/:
    fig_fluxo.pdf     Figura 1: encadeamento metodológico
    fig_energia.pdf   Figura 2: (a) energia x tokens de saída na matriz; (b) energia por tarefa concluída
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

BASE = Path(__file__).resolve().parent / "resultados"
SAIDA = BASE / "figuras"
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False})
CORES = {"llama3.2": "#1f77b4", "gemma4": "#d62728"}
ROTULO = {"llama3.2": "Llama 3.2 (3B)", "gemma4": "Gemma 4 (8B)"}


def fig_fluxo():
    fig, ax = plt.subplots(figsize=(6.3, 1.55))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")
    caixas = [
        (1, 16, "Medição em GPU\n(Ollama + NVML)\nJ/token, J/tarefa"),
        (1, 1, "Parâmetros públicos\nPUE, preços, fatores\nde emissão, câmbio"),
        (26, 8.5, "Cenários de demanda\nJustiça em Números\n(porte, k tarefas/caso)"),
        (51, 8.5, "Matriz de decisão\n4 alternativas ×\n7 critérios"),
        (76, 16, "TOPSIS\n4 perfis de pesos"),
        (76, 1, "Monte Carlo\n(Dirichlet)\nfreq. de 1º lugar"),
    ]
    for x, y, txt in caixas:
        ax.add_patch(FancyBboxPatch((x, y), 22, 12.5, boxstyle="round,pad=0.4",
                                    fc="#f2f2f2", ec="#444444", lw=0.8))
        ax.text(x + 11, y + 6.25, txt, ha="center", va="center", fontsize=6.6)
    seta = dict(arrowstyle="->", lw=0.8, color="#444444")
    for (x0, y0), (x1, y1) in [((23.8, 22), (26, 17)), ((23.8, 7.5), (26, 12)),
                               ((48.8, 14.75), (51, 14.75)), ((73.8, 17), (76, 21)),
                               ((73.8, 12.5), (76, 8))]:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=seta)
    fig.savefig(SAIDA / "fig_fluxo.pdf", bbox_inches="tight")
    fig.savefig(SAIDA / "fig_fluxo.png", dpi=200, bbox_inches="tight")


def fig_energia():
    m = pd.read_csv(BASE / "matriz_consolidado" / "execucoes_todas.csv")
    a = pd.read_csv(BASE / "matriz_consolidado" / "modelo_energia.csv").set_index("modelo")
    t = pd.read_csv(BASE / "tarefa" / "execucoes_todas.csv")
    t = t[t.motivo_fim == "stop"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.3), gridspec_kw={"width_ratios": [1.25, 1]})
    marcadores = {1000: "o", 4000: "s", 12000: "^"}
    for mod in ["llama3.2", "gemma4"]:
        x = m[m.modelo == mod]
        for ch, mk in marcadores.items():
            y = x[x.chars_entrada == ch]
            ax1.scatter(y.tokens_saida, y.energia_J, s=9, marker=mk, color=CORES[mod],
                        alpha=0.6, lw=0)
        n_ent = x.tokens_entrada.median()
        xs = np.linspace(100, 540, 50)
        c = a.loc[mod]
        ax1.plot(xs, c.J_fixo + c.J_por_token_entrada * n_ent + c.J_por_token_saida * xs,
                 color=CORES[mod], lw=0.9,
                 label=f"{ROTULO[mod]}: {c.J_por_token_saida:.2f} J/token (R²={c.R2:.3f})")
    for ch, mk in marcadores.items():
        ax1.scatter([], [], marker=mk, color="gray", s=9, label=f"entrada {ch // 1000} mil car.")
    ax1.set_xlabel("Tokens gerados")
    ax1.set_ylabel("Energia da GPU por execução (J)")
    ax1.set_title("(a) Custo marginal por token", fontsize=8)
    ax1.legend(fontsize=5.8, frameon=False, loc="upper left")

    cfgs = [("llama3.2_think-padrao", "Llama 3.2"), ("gemma4_think-nao", "Gemma 4\nsem raciocínio"),
            ("gemma4_think-sim", "Gemma 4\ncom raciocínio")]
    med = [t[t.cfg == c].energia_J.mean() for c, _ in cfgs]
    dp = [t[t.cfg == c].energia_J.std() for c, _ in cfgs]
    tok = [t[t.cfg == c].tokens_saida.mean() for c, _ in cfgs]
    cores = [CORES["llama3.2"], CORES["gemma4"], "#8c1515"]
    ax2.bar(range(3), med, yerr=dp, color=cores, capsize=3, width=0.6, error_kw={"lw": 0.8})
    for i, (e, k) in enumerate(zip(med, tok)):
        ax2.text(i, e + dp[i] + 120, f"{e / 1000:.2f} kJ\n{k:.0f} tokens", ha="center", fontsize=6.3)
    ax2.set_xticks(range(3), [r for _, r in cfgs], fontsize=6.6)
    ax2.set_ylabel("Energia da GPU por tarefa (J)")
    ax2.set_ylim(0, max(med) * 1.35)
    ax2.set_title("(b) Energia por tarefa concluída", fontsize=8)
    fig.tight_layout()
    fig.savefig(SAIDA / "fig_energia.pdf", bbox_inches="tight")
    fig.savefig(SAIDA / "fig_energia.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    SAIDA.mkdir(parents=True, exist_ok=True)
    fig_fluxo()
    fig_energia()
    print("Figuras em", SAIDA)
