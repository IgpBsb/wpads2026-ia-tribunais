# Dados do artigo

## Justiça em Números (CNJ): recorte para os cenários

**Origem:** base do Justiça em Números publicada pelo CNJ (`Base-JN-2026.xlsx`, aba `JN_23-Jun-2026`, versão de 23/06/2026; série de 2009 a 2025, 1.493 linhas, 93 órgãos por ano).
Os decimais com vírgula foram convertidos para ponto. As variáveis são as do dicionário (aba `Variaveis`).
Os CSVs desta pasta são reconstruídos a partir das planilhas oficiais por `preparar_dados.py` (reprodução verificada: arquivos idênticos).

```
python preparar_dados.py --jn Base-JN-2026.xlsx --mcti Inventario_2026_janago.xlsx
```

| Arquivo | Conteúdo |
|---|---|
| `jn_ultimo_ano_tribunais.csv` | Ano-base 2025, um tribunal por linha |
| `jn_serie_historica_variaveis_selecionadas.csv` | Mesmas variáveis, de 2009 a 2025 (para projeção de demanda, se necessário) |

| Variável | Descrição (dicionário JN) | Papel no artigo |
|---|---|---|
| `cnelet` | Casos Novos Eletrônicos | **Volume anual de demanda de inferência** (k inferências por caso) |
| `cnnjud`, `cp`, `tbaix`, `sent` | Casos novos (exceto execuções), pendentes, baixados, sentenças | Volumes alternativos para a análise de sensibilidade |
| `porte`, `idporte`, `ranking` | Porte do tribunal (Pequeno, Médio, Grande; `nd` para Federal, Superior e Militar) | Definição dos cenários |
| `dinf1`, `dinf2` | Despesa com aquisições e com custeio de TIC (R$) | Capacidade de investimento e de custeio |
| `tptic`, `f5` | Pessoal de TI (absoluto e proporção do total de servidores) | Critério de capacidade operacional da alternativa própria |
| `inf1` | Despesa com TIC em relação à despesa total | Contexto |
| `procel` | Índice de processos eletrônicos | Verificação (≈1 em quase todos os tribunais) |

### Medianas de 2025 por ramo e porte (referência rápida)

| Ramo | Porte | Casos novos eletrônicos | Pessoal de TI | Aquisição TIC (R$ mi) | Custeio TIC (R$ mi) |
|---|---|---|---|---|---|
| Estadual | Grande | 2.122.763 | 173,5 | 64,5 | 160,2 |
| Estadual | Médio | 694.167 | 122 | 17,2 | 59,0 |
| Estadual | Pequeno | 380.012 | 87,5 | 4,5 | 17,2 |
| Trabalho | Grande | 521.070 | 137 | 7,5 | 19,8 |
| Trabalho | Médio | 185.277 | 78 | 1,1 | 10,1 |
| Trabalho | Pequeno | 64.216 | 41 | 0,8 | 6,9 |

Total nacional de casos novos eletrônicos em 2025: **40.831.497**.

> A Justiça Estadual é o recorte natural para os cenários: tem os três portes e o maior contraste de volume (cerca de 5,6 vezes entre a mediana do porte pequeno e a do grande).

## Fator de emissão do SIN (MCTI)

**Origem:** planilha oficial do MCTI com os fatores médios de emissão de CO₂ do Sistema Interligado Nacional para inventários (`Inventario_2026_janago.xlsx`, série 2006 a agosto/2026), obtida em gov.br/mcti → SIRENE → Dados e ferramentas → Fatores de emissão.

| Arquivo | Conteúdo |
|---|---|
| `mcti_fator_medio_SIN_inventario.csv` | Fator anual e mensal (tCO₂/MWh) por ano. **2025 = 0,0461**; faixa anual 2019–2025 de 0,0385 (2023) a 0,1263 (2021, crise hídrica) |

Use o fator **médio** (para inventários), não os de margem de operação ou de construção (para projetos de MDL).
