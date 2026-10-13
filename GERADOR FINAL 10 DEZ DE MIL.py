!pip install -q numpy pandas scikit-learn

import json, urllib.request, random, ssl
import numpy as np
from collections import Counter
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURACOES
# ============================================================
CUSTO_JOGO = 3.50
N_JOGOS = 5
JANELA = 20

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0 Safari/537.36"}
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def baixar_dados():
    url = "https://loteriascaixa-api.herokuapp.com/api/lotofacil"
    print("Baixando dados oficiais...")
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=120, context=ctx) as r:
        dados = json.loads(r.read().decode("utf-8"))
    concursos = []
    datas = []
    for item in sorted(dados, key=lambda x: x["concurso"]):
        concursos.append(sorted(int(d) for d in item["dezenas"]))
        datas.append(item.get("data", ""))
    print(f"OK! {len(concursos)} concursos. Ultimo: {len(concursos)} ({datas[-1]})")
    return concursos


# ============================================================
# MOTORES
# ============================================================

def motor_original(janela):
    f = Counter()
    for j in janela: f.update(j)
    a = {}
    for n in range(1, 26):
        a[n] = 0
        for j in reversed(janela):
            if n in j: break
            a[n] += 1
    cooc = np.zeros((26, 26))
    for j in janela:
        for x in j:
            for y in j:
                if x != y: cooc[x][y] += 1
    return {n: f[n]*2.0 + a[n]*1.5 + cooc[n].sum()*0.1 for n in range(1, 26)}


def motor_feller(janela):
    scores = {}
    for n in range(1, 26):
        pos = [i for i, j in enumerate(janela) if n in j]
        if not pos: scores[n] = 0; continue
        gaps = np.diff(pos).tolist() if len(pos) > 1 else [len(janela)]
        gap_at = len(janela) - 1 - pos[-1]
        gap_m = np.mean(gaps)
        gap_s = np.std(gaps) if len(gaps) > 1 else 1.0
        z = abs(gap_at - gap_m) / (gap_s + 0.001) if gap_s > 0 else 0
        scores[n] = 10 / (1 + z)
    return scores


def motor_fourier(janela):
    scores = {}
    for n in range(1, 26):
        serie = np.array([1.0 if n in j else 0.0 for j in janela])
        if serie.sum() < 2: scores[n] = 0; continue
        fft = np.abs(np.fft.rfft(serie - serie.mean()))
        scores[n] = fft[1:].sum() + serie[-5:].sum() * 2
    return scores


def motor_associacao(janela, ultimo):
    cooc = np.zeros((26, 26))
    for i, j in enumerate(janela):
        peso = 1.0 + i / len(janela)
        for x in j:
            for y in j:
                if x != y: cooc[x][y] += peso
    scores = {}
    for n in range(1, 26):
        if n in ultimo:
            scores[n] = sum(cooc[n][u] for u in ultimo if u != n) * 0.1
        else:
            scores[n] = sum(cooc[n][u] for u in ultimo) * 0.15
    return scores


def motor_multi_janela(cs):
    scores = {}
    for n in range(1, 26):
        def f(j):
            if len(j) == 0: return 0
            return sum(1 for c in j if n in c) / len(j)
        s = (f(cs[-5:]) * 0.30 + f(cs[-15:]) * 0.25 + f(cs[-30:]) * 0.20 +
             f(cs[-100:]) * 0.15 + f(cs) * 0.10)
        scores[n] = s
    return scores


class MotorMLP:
    def __init__(self): self.model = None; self.scaler = None
    def _feat(self, jan, ult, n):
        f5 = sum(1 for j in jan[-5:] if n in j)
        f10 = sum(1 for j in jan[-10:] if n in j)
        f20 = sum(1 for j in jan if n in j)
        a = 0
        for j in reversed(jan):
            if n in j: break
            a += 1
        return [f5, f10, f20, a, 1 if n in ult else 0]
    def treinar(self, cs):
        X, y = [], []
        for t in range(20, len(cs)-1):
            jan = cs[max(0, t-20):t]
            alvo = set(cs[t])
            for n in range(1, 26):
                X.append(self._feat(jan, cs[t-1], n))
                y.append(1 if n in alvo else 0)
        X, y = np.array(X), np.array(y)
        self.scaler = StandardScaler().fit(X)
        self.model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=100, random_state=42)
        self.model.fit(self.scaler.transform(X), y)
    def pontuar(self, jan, ult):
        if self.model is None: return {n: 0 for n in range(1, 26)}
        X = np.array([self._feat(jan, ult, n) for n in range(1, 26)])
        p = self.model.predict_proba(self.scaler.transform(X))[:, 1]
        return {n: p[n-1] * 100 for n in range(1, 26)}


# ============================================================
# FILTROS
# ============================================================

PRIMOS = {2,3,5,7,11,13,17,19,23}
FIBO = {1,2,3,5,8,13,21}

def passou_filtros(c, ult, hs):
    if tuple(sorted(c)) in hs: return False
    p = sum(1 for n in c if n % 2 == 0)
    if p not in (7, 8): return False
    if not (175 <= sum(c) <= 215): return False
    o = sorted(c); run = 1; mx = 1
    for i in range(1, len(o)):
        if o[i] - o[i-1] == 1:
            run += 1; mx = max(mx, run)
        else: run = 1
    if mx > 4: return False
    lin = [0]*5; col = [0]*5
    for n in c:
        lin[(n-1)//5] += 1; col[(n-1)%5] += 1
    if not (all(x >= 1 for x in lin) and all(x >= 1 for x in col)): return False
    if not (8 <= len(set(c) & set(ult)) <= 10): return False
    if not (5 <= sum(1 for n in c if n in PRIMOS) <= 7): return False
    if not (3 <= sum(1 for n in c if n in FIBO) <= 6): return False
    q = [0,0,0,0]
    for n in c:
        i = (n-1)//5; j = (n-1)%5
        if i < 2 and j < 3: q[0] += 1
        elif i < 2: q[1] += 1
        elif j < 3: q[2] += 1
        else: q[3] += 1
    if not all(x >= 2 for x in q): return False
    return True


def norm(s):
    v = np.array(list(s.values()))
    if v.max() == v.min(): return {k: 0.5 for k in s}
    return {k: (x - v.min()) / (v.max() - v.min()) for k, x in s.items()}


def score_final(jan, cs, ult, mlp):
    m1 = norm(motor_original(jan))
    m2 = norm(motor_feller(jan))
    m3 = norm(motor_fourier(jan))
    m4 = norm(motor_associacao(jan, ult))
    m5 = norm(mlp.pontuar(jan, ult))
    m6 = norm(motor_multi_janela(cs))
    return {n: (m1[n]+m2[n]+m3[n]+m4[n]+m5[n]+m6[n])/6 for n in range(1, 26)}


def gerar_jogos(scores, ult, hs, n=5):
    jogos = []; vistos = set()
    ranking = sorted(scores.keys(), key=lambda x: -scores[x])

    # Jogo 1: top 15
    base = tuple(sorted(ranking[:15]))
    if passou_filtros(base, ult, hs):
        jogos.append(base); vistos.add(base)

    # Jogos 2..n: amostragem ponderada
    tent = 0
    while len(jogos) < n and tent < 50000:
        tent += 1
        probs = np.array([scores[x] for x in ranking])
        probs = probs / probs.sum()
        cand = tuple(sorted(np.random.choice(ranking, size=15, replace=False, p=probs)))
        if cand in vistos or not passou_filtros(cand, ult, hs): continue
        if any(len(set(cand) & set(j)) > 11 for j in jogos): continue
        jogos.append(cand); vistos.add(cand)

    while len(jogos) < n:
        cand = tuple(sorted(random.sample(range(1, 26), 15)))
        if cand in vistos or not passou_filtros(cand, ult, hs): continue
        jogos.append(cand); vistos.add(cand)

    return jogos


# ============================================================
# PROJECAO
# ============================================================

def projetar(concursos):
    print()
    print("=" * 72)
    print("ASB-10.0 — PROJECAO DE 5 JOGOS")
    print("=" * 72)
    print(f"Total de concursos na base: {len(concursos)}")
    print(f"Ultimo resultado: {' '.join(f'{n:02d}' for n in concursos[-1])}")

    jan = concursos[-JANELA:]
    ult = jan[-1]
    hs = set(tuple(sorted(c)) for c in concursos)

    print()
    print("Analisando com 6 motores...")
    mlp = MotorMLP()
    mlp.treinar(concursos)
    scores = score_final(jan, concursos, ult, mlp)

    print("Ranking dos 10 melhores numeros:")
    rank = sorted(scores.items(), key=lambda x: -x[1])[:10]
    for i, (n, s) in enumerate(rank, 1):
        print(f"  {i:>2}. Numero {n:>2} (score {s:.3f})")

    print()
    print("Gerando 5 jogos diversificados...")
    jogos = gerar_jogos(scores, ult, hs, n=N_JOGOS)

    print()
    print("=" * 72)
    print(f"5 JOGOS SUGERIDOS (R$ {CUSTO_JOGO:.2f} cada | total R$ {CUSTO_JOGO*N_JOGOS:.2f})")
    print("=" * 72)
    for i, jogo in enumerate(jogos, 1):
        nums = " ".join(f"{n:02d}" for n in jogo)
        soma = sum(jogo)
        pares = sum(1 for n in jogo if n % 2 == 0)
        print(f"  Jogo {i}: {nums}")
        print(f"          Soma: {soma} | {pares}P/{15-pares}I")
    print()
    print("=" * 72)
    print("LEMBRETE: este motor nao preve o sorteio.")
    print("A probabilidade de 15 acertos e 1 em 3.268.760.")
    print("Use como ferramenta de organizacao, nunca como promessa de lucro.")
    print("=" * 72)

    return jogos


# ============================================================
# EXECUTAR
# ============================================================

if __name__ == "__main__":
    concursos = baixar_dados()
    jogos = projetar(concursos)