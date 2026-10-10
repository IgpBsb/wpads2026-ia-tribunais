"""Consolida as rodadas da matriz (resultados/matriz/*/execucoes.csv) em tabelas-resumo.

Saídas em resultados/matriz_consolidado/:
    execucoes_todas.csv   todas as repetições, com as colunas de configuração
    resumo.csv            média, dp e IC95% por configuração (modelo x entrada x saída)
    modelo_energia.csv    ajuste linear E = a + b*tokens_entrada + c*tokens_saida por modelo
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent / "resultados"
SAIDA = BASE / "matriz_consolidado"
METRICAS = ["energia_J", "energia_dinamica_J", "J_por_token_saida", "dur_s",
            "prefill_s", "decode_s", "tok_s", "tokens_entrada", "tokens_saida"]


def carregar():
    partes = []
    for csv in sorted((BASE / "matriz").glob("*/execucoes.csv")):
        modelo, ent, sai = csv.parent.name.rsplit("_", 2)
        d = pd.read_csv(csv)
        d["chars_entrada"] = int(ent.removeprefix("in"))
        d["max_saida"] = int(sai.removeprefix("out"))
        partes.append(d)
    return pd.concat(partes, ignore_index=True)


def resumir(d):
    g = d.groupby(["modelo", "chars_entrada", "max_saida"])
    linhas = []
    for chave, x in g:
        linha = dict(zip(["modelo", "chars_entrada", "max_saida"], chave), n=len(x))
        for m in METRICAS:
            media, dp = x[m].mean(), x[m].std(ddof=1)
            linha[f"{m}_media"] = media
            linha[f"{m}_dp"] = dp
            linha[f"{m}_ic95_pct"] = 100 * 1.96 * dp / np.sqrt(len(x)) / media if media else np.nan
        linhas.append(linha)
    return pd.DataFrame(linhas)


def ajustar(d):
    """Mínimos quadrados de E (J) em função de tokens de entrada e de saída, por modelo."""
    linhas = []
    for modelo, x in d.groupby("modelo"):
        A = np.column_stack([np.ones(len(x)), x["tokens_entrada"], x["tokens_saida"]])
        coef, *_ = np.linalg.lstsq(A, x["energia_J"], rcond=None)
        pred = A @ coef
        r2 = 1 - ((x["energia_J"] - pred) ** 2).sum() / ((x["energia_J"] - x["energia_J"].mean()) ** 2).sum()
        linhas.append({"modelo": modelo, "J_fixo": coef[0], "J_por_token_entrada": coef[1],
                       "J_por_token_saida": coef[2], "R2": r2, "n": len(x)})
    return pd.DataFrame(linhas)


if __name__ == "__main__":
    SAIDA.mkdir(parents=True, exist_ok=True)
    d = carregar()
    d.to_csv(SAIDA / "execucoes_todas.csv", index=False)
    r = resumir(d)
    r.to_csv(SAIDA / "resumo.csv", index=False)
    a = ajustar(d)
    a.to_csv(SAIDA / "modelo_energia.csv", index=False)
    pd.set_option("display.width", 200)
    cols = ["modelo", "chars_entrada", "max_saida", "n", "tokens_entrada_media", "tokens_saida_media",
            "energia_J_media", "energia_J_ic95_pct", "J_por_token_saida_media", "dur_s_media", "tok_s_media"]
    print(r[cols].round(2).to_string(index=False))
    print()
    print(a.round(4).to_string(index=False))
