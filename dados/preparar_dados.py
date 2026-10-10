"""Reconstrói os CSVs de dados a partir das planilhas oficiais (memória de cálculo).

Entradas (baixar das fontes oficiais):
  --jn     Base-JN-2026.xlsx, Justiça em Números/CNJ, aba "JN_23-Jun-2026" (série 2009-2025)
  --mcti   Inventario_2026_janago.xlsx, MCTI, fatores médios de emissão do SIN para inventários

Saídas (nesta pasta):
  jn_ultimo_ano_tribunais.csv                    ano mais recente, um tribunal por linha
  jn_serie_historica_variaveis_selecionadas.csv  mesmas variáveis, 2009-2025
  mcti_fator_medio_SIN_inventario.csv            fator anual e mensal (tCO2/MWh), 2006-2026

Uso: python preparar_dados.py --jn <Base-JN-2026.xlsx> --mcti <Inventario_2026_janago.xlsx>
"""
import argparse
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parent
COLS_JN = ["ano", "justica", "sigla", "porte", "idporte", "ranking", "cnelet", "cnnjud", "cp",
           "tbaix", "sent", "procel", "dinf1", "dinf2", "tptic", "f5", "inf1"]


def preparar_jn(caminho):
    d = pd.read_excel(caminho, sheet_name="JN_23-Jun-2026", usecols=lambda c: c in COLS_JN)
    for c in COLS_JN[6:]:  # numéricas: a planilha usa vírgula decimal em parte das células
        d[c] = pd.to_numeric(d[c].astype(str).str.replace(",", "."), errors="coerce")
    d[d.ano == d.ano.max()].to_csv(AQUI / "jn_ultimo_ano_tribunais.csv", index=False, encoding="utf-8")
    d[COLS_JN].to_csv(AQUI / "jn_serie_historica_variaveis_selecionadas.csv", index=False, encoding="utf-8")


def preparar_mcti(caminho):
    """Cada ano ocupa um bloco: linha com o ano na col. 0, fator anual na col. 14 da linha seguinte
    e os 12 fatores mensais nas col. 2-13 duas linhas abaixo."""
    d = pd.read_excel(caminho, header=None)
    linhas = []
    for i in range(len(d)):
        v = d.iat[i, 0]
        if isinstance(v, (int, float)) and not pd.isna(v) and 2000 < v < 2100:
            mensais = list(d.iloc[i + 2, 2:14])
            linhas.append({"ano": int(v), "fator_anual": d.iat[i + 1, 14],
                           **{f"m{j + 1:02d}": m for j, m in enumerate(mensais)}})
    pd.DataFrame(linhas).to_csv(AQUI / "mcti_fator_medio_SIN_inventario.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jn", required=True)
    ap.add_argument("--mcti", required=True)
    a = ap.parse_args()
    preparar_jn(a.jn)
    preparar_mcti(a.mcti)
    print("CSVs gravados em", AQUI)
