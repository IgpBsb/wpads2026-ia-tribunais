"""Sensibilidade: incluir a geração descontrolada nas projeções (pedido do Revisor 3).

Na bateria de tarefas, 1 das 10 execuções do Llama 3.2 entrou em repetição e só parou no limite de
4.096 tokens. O cenário-base usa a média das 9 execuções concluídas; aqui a média inclui as 10.
Ajusta energia dinâmica, duração e tokens de saída por tarefa e compara, por porte:
  - a vencedora do TOPSIS com pesos iguais;
  - a frequência de 1º lugar só com pesos aleatórios (Dirichlet, alpha = 1).

Uso: python sensibilidade_falhas.py  (grava em ../resultados/mcda/sensibilidade_falhas.csv)
"""
from pathlib import Path

import numpy as np
import pandas as pd

from mcda import ALT, CRIT, SAIDA, carregar, dedicada, matriz, sortear, topsis

AQUI = Path(__file__).resolve().parent
N_SORTEIOS = 20000


def medias_llama():
    t = pd.read_csv(AQUI.parent / "resultados" / "tarefa" / "execucoes_todas.csv")
    llama = t[t.cfg == "llama3.2_think-padrao"]
    return llama[llama.motivo_fim == "stop"], llama


def avaliar(p, cen, v, sentidos, w_iguais, rng):
    linhas = []
    for porte, (casos, pti) in cen.iterrows():
        m = matriz(v, casos, pti)
        venc = int(np.argmax(topsis(m, w_iguais, sentidos)))
        cont = np.zeros(len(ALT))
        for _ in range(N_SORTEIOS):
            w = rng.dirichlet(np.full(len(CRIT), p["monte_carlo"]["dirichlet_alpha"]))
            cont[np.argmax(topsis(m, w, sentidos))] += 1
        n_a1 = dedicada(v, casos * v["cenarios.tarefas_por_caso"], 1.0, v["A1_propria.pue"])[1]
        linhas.append({"porte": porte, "vencedora_pesos_iguais": ALT[venc],
                       "energia_A1_MWh": m[0, 0], "servidores_A1": n_a1,
                       **{f"{a}_1o_lugar_%": 100 * c / N_SORTEIOS for a, c in zip(ALT, cont)}})
    return pd.DataFrame(linhas)


def main():
    p, cen = carregar()
    sentidos = np.array([p["criterios"][c]["sentido"] for c in CRIT])
    w_iguais = np.array(p["perfis_de_peso"]["equilibrado"])
    ok, todas = medias_llama()
    print(f"Energia dinâmica por tarefa: {ok.energia_dinamica_J.mean():.1f} J (9 concluídas) -> "
          f"{todas.energia_dinamica_J.mean():.1f} J (10, com a descontrolada); "
          f"+{100 * (todas.energia_dinamica_J.mean() / ok.energia_dinamica_J.mean() - 1):.0f}%\n")

    base = sortear(p)
    com_falhas = dict(base, **{
        "carga.energia_dinamica_gpu_J_por_tarefa": todas.energia_dinamica_J.mean(),
        "carga.duracao_s_por_tarefa": todas.dur_s.mean(),
        "carga.tokens_saida_por_tarefa": todas.tokens_saida.mean(),
    })
    pd.set_option("display.width", 200)
    saida = []
    for cenario, nome, v in [("base", "Cenário-base (9 concluídas)", base),
                             ("com_descontrolada", "Com a geração descontrolada (10)", com_falhas)]:
        r = avaliar(p, cen, v, sentidos, w_iguais, np.random.default_rng(p["monte_carlo"]["semente"]))
        r.insert(0, "energia_dinamica_J_por_tarefa", v["carga.energia_dinamica_gpu_J_por_tarefa"])
        r.insert(0, "cenario", cenario)
        print(nome)
        print(r.round(1).to_string(index=False), "\n")
        saida.append(r)
    SAIDA.mkdir(parents=True, exist_ok=True)
    pd.concat(saida).round(2).to_csv(SAIDA / "sensibilidade_falhas.csv", index=False)
    print("Gravado:", SAIDA / "sensibilidade_falhas.csv")


if __name__ == "__main__":
    main()
