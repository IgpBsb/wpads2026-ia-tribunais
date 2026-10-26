"""Medição de energia e latência de inferência de LLM na GPU local (Ollama + NVML).

Energia principal: diferença do contador acumulado da GPU (nvmlDeviceGetTotalEnergyConsumption, mJ).
Energia dinâmica: energia total menos a potência ociosa média (linha de base) x duração.
Amostragem de potência (50 ms) é registrada em paralelo para pico/média e checagem cruzada.

Uso:
    python medir_inferencia.py --modelos llama3.2 gemma4 --reps 10 --num-predict 256 --saida resultados/piloto
"""
import argparse
import json
import statistics
import threading
import time
import urllib.request
from pathlib import Path

import pynvml

# 127.0.0.1 em vez de localhost: no Windows, localhost tenta IPv6 antes e soma ~2 s ociosos a cada chamada
OLLAMA = "http://127.0.0.1:11434"
RAIZ = Path(__file__).resolve().parent.parent
# mesmos arquivos de Referencias/04_Normativos-Setor-Publico usados nas medições de 07/10/2026
CORPUS_PADRAO = RAIZ / "corpus"


def ollama(rota, payload=None, timeout=600):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(OLLAMA + rota, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class Amostrador(threading.Thread):
    """Amostra potência (W) e utilização (%) da GPU em intervalo fixo."""

    def __init__(self, h, intervalo=0.05):
        super().__init__(daemon=True)
        self.h, self.intervalo = h, intervalo
        self.amostras, self._parar = [], threading.Event()

    def run(self):
        while not self._parar.is_set():
            p = pynvml.nvmlDeviceGetPowerUsage(self.h) / 1000
            u = pynvml.nvmlDeviceGetUtilizationRates(self.h).gpu
            self.amostras.append((time.perf_counter(), p, u))
            time.sleep(self.intervalo)

    def parar(self):
        self._parar.set()
        self.join()

    def janela(self, t0, t1):
        return [(p, u) for t, p, u in self.amostras if t0 <= t <= t1]


def linha_de_base(h, segundos=15):
    e0, t0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h), time.perf_counter()
    time.sleep(segundos)
    e1, t1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h), time.perf_counter()
    return (e1 - e0) / 1000 / (t1 - t0)  # W


def montar_prompts(pasta, n_chars, n):
    """Gera n prompts com trechos distintos e de mesmo tamanho do corpus normativo.

    Trechos distintos evitam que o cache de prompt do Ollama elimine a etapa de prefill.
    """
    corpus = "\n".join(p.read_text(encoding="utf-8") for p in sorted(Path(pasta).glob("*.md")))
    corpus = " ".join(corpus.split())
    passo = max(n_chars, (len(corpus) - n_chars) // n)
    if passo * (n - 1) + n_chars > len(corpus):
        raise ValueError(f"Corpus de {len(corpus)} caracteres insuficiente para {n} trechos de {n_chars}")
    instrucao = ("Você é assistente de um tribunal. Leia o trecho normativo abaixo e "
                 "produza um resumo estruturado em tópicos, identificando obrigações, "
                 "vedações e prazos.\n\n")
    return [instrucao + corpus[i * passo: i * passo + n_chars] for i in range(n)]


def resumo(xs):
    m = statistics.mean(xs)
    dp = statistics.stdev(xs) if len(xs) > 1 else 0.0
    ic = 1.96 * dp / len(xs) ** 0.5
    return {"media": m, "dp": dp, "cv_pct": 100 * dp / m if m else 0.0,
            "ic95": ic, "ic95_pct": 100 * ic / m if m else 0.0}


def medir_modelo(h, amostrador, modelo, prompts, args, p_ocioso):
    opcoes = {"temperature": 0, "seed": 42, "num_predict": args.num_predict,
              "num_ctx": args.num_ctx}
    extra = {} if args.think == "padrao" else {"think": args.think == "sim"}
    # aquecimento: carrega o modelo na VRAM e estabiliza clocks (trechos fora da amostra)
    for prompt in prompts[args.reps:]:
        ollama("/api/generate", {"model": modelo, "prompt": prompt, "stream": False,
                                 "options": opcoes, "keep_alive": "10m"} | extra)
    ps = ollama("/api/ps")
    info = next((m for m in ps.get("models", []) if m["name"].startswith(modelo)), {})
    frac_gpu = info.get("size_vram", 0) / info["size"] if info.get("size") else None

    linhas = []
    for rep in range(1, args.reps + 1):
        time.sleep(args.pausa)
        e0, t0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h), time.perf_counter()
        r = ollama("/api/generate", {"model": modelo, "prompt": prompts[rep - 1], "stream": False,
                                     "options": opcoes, "keep_alive": "10m"} | extra)
        e1, t1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h), time.perf_counter()
        dur = t1 - t0
        e_tot = (e1 - e0) / 1000
        e_dyn = e_tot - p_ocioso * dur
        jan = amostrador.janela(t0, t1)
        n_out, n_in = r.get("eval_count", 0), r.get("prompt_eval_count", 0)
        linhas.append({
            "modelo": modelo, "rep": rep, "dur_s": dur,
            "tokens_entrada": n_in, "tokens_saida": n_out,
            "tok_s": n_out / (r["eval_duration"] / 1e9) if r.get("eval_duration") else None,
            "ttft_s": (r.get("load_duration", 0) + r.get("prompt_eval_duration", 0)) / 1e9,
            "dur_servidor_s": r.get("total_duration", 0) / 1e9,
            "prefill_s": r.get("prompt_eval_duration", 0) / 1e9,
            "decode_s": r.get("eval_duration", 0) / 1e9,
            "energia_J": e_tot, "energia_dinamica_J": e_dyn,
            "J_por_token_saida": e_tot / n_out if n_out else None,
            "pot_media_W": statistics.mean(p for p, _ in jan) if jan else None,
            "pot_max_W": max(p for p, _ in jan) if jan else None,
            "util_media_pct": statistics.mean(u for _, u in jan) if jan else None,
            "fracao_vram": frac_gpu,
            # eval_count inclui tokens de raciocínio; os tamanhos abaixo mostram como se dividem
            "chars_raciocinio": len(r.get("thinking") or ""),
            "chars_resposta": len(r.get("response") or ""),
            "motivo_fim": r.get("done_reason"),
        })
        print(f"  {modelo} rep {rep:2d}: {dur:6.2f} s  {e_tot:7.1f} J  "
              f"{n_out} tok  {linhas[-1]['J_por_token_saida']:.3f} J/tok")
    return linhas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", nargs="+", default=["llama3.2", "gemma4"])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--aquecimento", type=int, default=2)
    ap.add_argument("--num-predict", type=int, default=256)
    ap.add_argument("--num-ctx", type=int, default=4096)
    ap.add_argument("--chars-entrada", type=int, default=4000)
    ap.add_argument("--pausa", type=float, default=2.0)
    ap.add_argument("--corpus", default=str(CORPUS_PADRAO))
    ap.add_argument("--think", choices=["padrao", "sim", "nao"], default="padrao",
                    help="modo de raciocínio (modelos com 'thinking'); padrao = não envia o parâmetro")
    ap.add_argument("--saida", default=str(RAIZ / "experimento" / "resultados" / "piloto"))
    args = ap.parse_args()

    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    prompts = montar_prompts(args.corpus, args.chars_entrada, args.reps + args.aquecimento)

    # descarrega modelos residentes para medir a linha de base ociosa
    for m in ollama("/api/ps").get("models", []):
        ollama("/api/generate", {"model": m["name"], "keep_alive": 0})
    time.sleep(5)
    p_ocioso = linha_de_base(h)
    print(f"Potência ociosa da GPU: {p_ocioso:.2f} W")

    amostrador = Amostrador(h)
    amostrador.start()
    linhas = []
    try:
        for modelo in args.modelos:
            print(f"Modelo {modelo}")
            linhas += medir_modelo(h, amostrador, modelo, prompts, args, p_ocioso)
            ollama("/api/generate", {"model": modelo, "keep_alive": 0})
            time.sleep(3)
    finally:
        amostrador.parar()

    import csv
    with open(saida / "execucoes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)

    sumario = {
        "ambiente": {
            "gpu": pynvml.nvmlDeviceGetName(h),
            "limite_potencia_W": pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000,
            "driver": pynvml.nvmlSystemGetDriverVersion(),
            "ollama": ollama("/api/version")["version"],
            "potencia_ociosa_W": p_ocioso,
            # caminhos relativos à raiz do projeto: não registrar a estrutura de pastas local
            "parametros": {k: (Path(x).resolve().relative_to(RAIZ).as_posix()
                               if k in ("corpus", "saida") else x) for k, x in vars(args).items()},
        },
        "modelos": {},
    }
    for modelo in args.modelos:
        ls = [l for l in linhas if l["modelo"] == modelo]
        sumario["modelos"][modelo] = {
            k: resumo([l[k] for l in ls]) for k in
            ["dur_s", "dur_servidor_s", "prefill_s", "decode_s", "energia_J",
             "energia_dinamica_J", "J_por_token_saida", "tok_s", "pot_media_W",
             "tokens_entrada", "tokens_saida"]
        } | {"fracao_vram": ls[0]["fracao_vram"]}
    (saida / "sumario.json").write_text(json.dumps(sumario, indent=2, ensure_ascii=False),
                                        encoding="utf-8")
    print(json.dumps(sumario["modelos"], indent=2, ensure_ascii=False))
    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
