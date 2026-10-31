import numpy as np
import pandas as pd


class Mozg:
    """Mała sieć neuronów LIF z konektomu muszki (parametry: Shiu i wsp., 2024)."""

    # parametry modelu
    V_SPOCZ = -52.0     # napięcie spoczynkowe [mV]
    V_PROG = -45.0      # próg impulsu [mV]
    TAU_M = 20.0        # stała czasowa błony [ms]
    TAU_SYN = 5.0       # stała czasowa synapsy [ms]
    REFRAKCJA = 2.2     # czas "martwy" po impulsie [ms]
    W_SYN = 0.275       # siła jednej synapsy [mV]
    F_POI = 250         # siła sztucznego wejścia z zewnątrz

    def __init__(self, W, nazwy, dt=0.1, seed=0):
        self.W = W
        self.nazwy = np.asarray(nazwy)
        self.N = len(self.nazwy)
        self.dt = dt
        self.rng = np.random.default_rng(seed)
        self.reset()
    
    @classmethod
    def z_plikow(cls, folder, **kwargs):
        """Buduje mózg z plików neurony.csv i podsiec.csv."""
        neurony = pd.read_csv(f"{folder}/neurony.csv")
        polaczenia = pd.read_csv(f"{folder}/podsiec.csv")

        indeks = {b: i for i, b in enumerate(neurony["bodyId"])}
        N = len(neurony)

        # znak neuronu wysyłającego: +1 pobudza, -1 hamuje (zasada Dale'a)
        if "znak" in neurony.columns:
            znaki = neurony["znak"].to_numpy()
        else:
            znaki = np.ones(N)

        W = np.zeros((N, N))
        for pre, post, w in polaczenia[["body_pre", "body_post", "weight"]].itertuples(index=False):
            W[indeks[post], indeks[pre]] = w * znaki[indeks[pre]]

        nazwy = neurony["type"] + "_" + neurony["somaSide"]
        return cls(W, nazwy, **kwargs)
    
    def reset(self):
        """Ustawia wszystkie neurony w stan spoczynku."""
        self.v = np.full(self.N, self.V_SPOCZ)
        self.g = np.zeros(self.N)
        self.blokada = np.zeros(self.N)

    def maska(self, nazwa):
        """Zwraca True/False dla neuronów o danej nazwie, np. 'VS_L'."""
        return self.nazwy == nazwa

    def krok(self, czas_ms, czestotliwosci_hz):
        """Liczy czas_ms symulacji, zaczynając od obecnego stanu.

        czestotliwosci_hz: tablica (N,) z częstotliwością wejścia dla każdego neuronu.
        Zwraca częstotliwość impulsów każdego neuronu w tym oknie [Hz].
        """
        p_wejscia = np.asarray(czestotliwosci_hz) * self.dt / 1000
        impulsy = np.zeros(self.N, dtype=int)

        for _ in range(int(round(czas_ms / self.dt))):
            # 1. wejście z zewnątrz
            losowe = self.rng.random(self.N) < p_wejscia
            self.g += self.W_SYN * self.F_POI * losowe

            # 2. integrate + leaky
            self.v += self.dt * (self.V_SPOCZ - self.v + self.g) / self.TAU_M
            self.g -= self.dt * self.g / self.TAU_SYN
            self.v[self.blokada > 0] = self.V_SPOCZ
            self.blokada -= self.dt

            # 3. fire
            strzal = self.v >= self.V_PROG
            impulsy += strzal
            self.v[strzal] = self.V_SPOCZ
            self.blokada[strzal] = self.REFRAKCJA

            # 4. impulsy do odbiorców
            self.g += self.W_SYN * (self.W @ strzal)

        return impulsy / (czas_ms / 1000)