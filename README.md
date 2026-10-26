# Energia, carbono e soberania na inferência de IA em tribunais

Dados, memória de cálculo e código do experimento do artigo aceito no **I Workshop de Processamento de Alto Desempenho Sustentável (WPADS 2026)**, evento satélite do SSCAD 2026 (SBC).

O trabalho combina dois componentes:

1. **Medição** da energia de inferência de modelos de linguagem abertos em GPU (Ollama + contador de energia da NVML), por *token* e por tarefa concluída.
2. **Análise multicritério** (TOPSIS com análise de robustez por Monte Carlo) de quatro alternativas de infraestrutura para cargas de IA em tribunais (própria, IaaS nacional, API no exterior e híbrida), com cenários de demanda construídos a partir do Justiça em Números (CNJ).

## Estrutura

```
corpus/                    textos usados como entrada das tarefas (5 resoluções do CNJ e 1 relatório do TCU)
dados/
  preparar_dados.py        reconstrói os CSVs a partir das planilhas oficiais (CNJ e MCTI)
  jn_*.csv                 recortes do Justiça em Números (2009-2025)
  mcti_*.csv               fator médio de emissão do SIN (MCTI, 2006-2026)
  LEIA-ME.md               origem e dicionário das variáveis
experimento/
  medir_inferencia.py      medição de energia e latência (Ollama + NVML)
  PROMPT.md                prompt exato e regra de extração dos trechos do corpus
  rodar_matriz.sh          bateria 1: custo por token (2 modelos x 3 entradas x 2 saídas x 10 repetições)
  rodar_tarefa.sh          bateria 2: energia por tarefa concluída (com e sem modo de raciocínio)
  consolidar.py            consolida a bateria 1 e ajusta E = a + b*n_entrada + c*n_saida
  mcda/parametros.yaml     parâmetros do modelo, com valor-base, faixa, status e fonte
  mcda/mcda.py             TOPSIS, perfis de peso, Monte Carlo (Dirichlet) e sensibilidade
  mcda/sensibilidade_falhas.py  efeito de incluir a geração descontrolada nas projeções
  resultados/              saídas brutas e consolidadas de todas as rodadas
```

## Como reproduzir

Requisitos: Python 3.11 ou superior e `pip install -r requirements.txt`. A medição exige GPU NVIDIA e o [Ollama](https://ollama.com) com os modelos `llama3.2` e `gemma4`.

```bash
# 1. Dados (opcional; os CSVs já estão versionados)
python dados/preparar_dados.py --jn Base-JN-2026.xlsx --mcti Inventario_2026_janago.xlsx

# 2. Medição (requer GPU; os resultados usados no artigo estão em experimento/resultados/)
python experimento/medir_inferencia.py --modelos llama3.2 --reps 10   # usa corpus/ por padrão
bash experimento/rodar_matriz.sh
bash experimento/rodar_tarefa.sh
python experimento/consolidar.py

# 3. Análise multicritério (não requer GPU)
python experimento/mcda/mcda.py
python experimento/mcda/sensibilidade_falhas.py
```

**Corpus da carga de trabalho.** As tarefas de inferência resumem trechos de cinco resoluções do CNJ (n. 370/2021, 332/2020, 335/2020, 396/2021 e 615/2025) e de um relatório do TCU (iESGo 2024), convertidos de PDF para Markdown. Os arquivos estão em `corpus/`, exatamente como foram usados nas medições; o *prompt* e a regra de extração dos trechos estão em `experimento/PROMPT.md`.

## Ambiente de medição

GPU NVIDIA GeForce RTX 3060 (12 GB, 170 W), CPU AMD Ryzen 5 5500, 32 GB de RAM, Windows 11 Pro, driver NVIDIA 596.36 (CUDA 13.2), Ollama 0.35.1. Modelos: Llama 3.2 (3,2B, Q4_K_M) e Gemma 4 (8,0B, Q4_K_M).

Notas de método: a energia é a diferença do contador acumulado da GPU (NVML); cada repetição usa um trecho distinto, para que o cache de *prompt* do servidor não elimine o *prefill*; a potência da CPU não é medida (RAPL indisponível para AMD no Windows) e entra no modelo por um fator de sistema.

## Fontes de dados

- **CNJ.** Justiça em Números 2026 (ano-base 2025), base de dados por tribunal.
- **MCTI.** Fatores médios de emissão de CO₂ do Sistema Interligado Nacional para inventários.
- Os demais parâmetros (PUE, preços de nuvem e de API, energia por consulta, câmbio) estão em `experimento/mcda/parametros.yaml`, cada um com a sua fonte e data de consulta.

## Licença

Código sob a licença MIT (ver `LICENSE`). Dados derivados e resultados de medição sob [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.pt-br). Os dados de origem e os textos do corpus pertencem às respectivas instituições (CNJ, MCTI e TCU).

## Como citar

Pedreira, I. G. (2026). Energia, Carbono e Soberania na Inferência de IA em Tribunais: Medição em GPU e Análise Multicritério. Aceito para publicação nos *Anais Estendidos do XXVII Simpósio em Sistemas Computacionais de Alto Desempenho (SSCAD 2026)*, WPADS. Sociedade Brasileira de Computação.
