# Prompt usado nas medições

Cada execução enviou ao Ollama a instrução abaixo seguida de um trecho do corpus (`corpus/`). O texto é o de `montar_prompts`, em `experimento/medir_inferencia.py`.

```
Você é assistente de um tribunal. Leia o trecho normativo abaixo e produza um resumo estruturado em tópicos, identificando obrigações, vedações e prazos.

<trecho do corpus>
```

## Como os trechos são extraídos

1. Os arquivos `.md` de `corpus/` são lidos em ordem alfabética e concatenados com quebra de linha.
2. Todos os espaços em branco são normalizados para um espaço simples (182.704 caracteres no total).
3. Para `n` trechos de `c` caracteres, o passo é `max(c, (182704 - c) // n)`; o trecho `i` vai da posição `i * passo` até `i * passo + c`.
4. `n` = repetições + aquecimentos (10 + 2). Os 10 primeiros trechos são as repetições medidas; os 2 últimos, os aquecimentos descartados.

Com isso, cada repetição recebe um trecho distinto de mesmo tamanho, o que impede o servidor de reaproveitar o processamento de um *prompt* idêntico.

## Parâmetros da geração

| Parâmetro | Valor |
|---|---|
| `temperature` | 0 |
| `seed` | 42 |
| `num_predict` | 128 ou 512 (bateria 1); 4096 (bateria 2) |
| `num_ctx` | 8192 |
| `think` | não enviado (padrão do modelo), exceto Gemma 4 na bateria 2: `false` e `true` |
| Tamanhos de trecho | 1.000, 4.000 e 12.000 caracteres (bateria 1); 4.000 caracteres (bateria 2) |

## Observação sobre o relatório do TCU

O corpus inclui um relatório do TCU (iESGo 2024) além das cinco resoluções do CNJ. Pela regra acima, ele só aparece em uma repetição: a 3ª das configurações com trechos de 12.000 caracteres, cujo trecho vai da posição 28.450 à 40.450 e cobre a parte final da Resolução 370 e cerca de 6 mil caracteres do relatório.
