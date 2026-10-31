import time

import cv2
import numpy as np

from autopilot import FOLDER_DANYCH, IP_XPLANE, PHI, PRZEJECIE_ROLL, WOLANT_ROLL
from mozg import Mozg
from oczy import Oczy
from xplane import XPlane

THETA = "sim/flightmodel/position/theta"
AGL = "sim/flightmodel/position/y_agl"      # wysokość nad terenem - TYLKO do podglądu
WOLANT_PITCH = "sim/joystick/yoke_pitch_ratio"
PRZEJECIE_PITCH = "sim/operation/override/override_joystick_pitch"

# --- wejścia (co muszka widzi) ---
OKNO_MS = 33
BAZA_HZ = 40                # VS, gdy obraz stoi
HZ_NA_RUCH = 40.0           # VS: przechylanie (różnica lewo-prawo)
HZ_NA_RUCH_PION = 60.0      # VS: pochylanie (wspólny ruch w pionie)
MAKS_RUCH = 1.5
CZESC_DO_POCHYLENIA = 0.6   # pochylanie mierzymy tylko w górnych 60% obrazu (horyzont, niebo)
BAZA_PRZYOCZEK = 60.0       # OCG01 (przechylenie)
HZ_NA_ROZNICE = 400.0
BAZA_PION = 50.0            # OCG02b + OCG03 (pochylenie)
HZ_NA_CIEMNOSC_PION = 300.0

# --- narastanie obrazu (LC4 / LPLC2) ---
ROZMIAR_PLAMKI = 4          # co 4 piksele środek pola widzenia jednego neuronu (siatka 8x16)
RAMIE = 3                   # długość "ramienia" LPLC2 w pikselach (w każdą z 4 stron)
PROG_SZUMU = 0.03           # ruch na zewnątrz słabszy niż to [piksel/klatkę] = cisza
ROZCHODZENIE_PELNE = 0.10   # o tyle ponad próg = neuron strzela z pełną siłą
HZ_LOOM_MAKS = 70.0
CZESC_DO_LOOMINGU = 1.0     # neurony patrzą na cały obraz (detektor sam ignoruje przesuwającą się ziemię)
PROG_RUCHU_WLASNEGO = 0.6   # przy takim obrocie całego obrazu narastanie jest całkiem wyciszone

# --- wyjścia (stery) ---
WYGLADZANIE = 0.1
WZMOCNIENIE = 1 / 300        # lotki z różnicy DNp20
KIERUNEK = -1
STERUJ_POCHYLENIEM = True
WZMOCNIENIE_PITCH = 1 / 150  # ster wys.: człon D z sumy DNp20
KIERUNEK_PITCH = -1
WZMOCNIENIE_P19 = 1 / 150    # ster wys.: człon P z sumy DNp19
KIERUNEK_P19 = +1
MAKS_PCHANIE = 0.4           # muszka nigdy nie pcha wolantu mocniej niż -0.4
MAKS_LOTKI = 0.5             # muszka nigdy nie wychyla lotek bardziej niż na pół

# --- odruch ucieczki (Giant Fiber) ---
WYGLADZANIE_GF = 0.15        # kompromis: szum nie odpala, a reakcja nie jest zbyt opóźniona
PROG_GF = 400.0              # suma L+P Giant Fibera, powyżej której muszka ucieka
SILA_UCIECZKI = 0.5          # minimalne "ciągnij" podczas ucieczki
CZAS_UCIECZKI = 1.5          # ile sekund trwa ucieczka (stały czas, bez przedłużania)
MNOZNIK_LOTEK_UCIECZKI = 1.0 # w ucieczce lotki działają normalnie
PROG_NIEBA = -0.10           # obraz o tyle jaśniejszy niż w locie poziomym = widzę niebo
CZAS_PO_UCIECZCE = 3.0       # tyle sekund po ucieczce muszka nie pcha wolantu
REFRAKCJA_UCIECZKI = 1.0     # tyle sekund po powrocie do lotu odruch nie może odpalić ponownie

# --- siatka bezpieczeństwa (NIE biologia - dane z X-Plane, jak GPWS w prawdziwym samolocie) ---
SIATKA_BEZPIECZENSTWA = False
MIN_WYSOKOSC = 60.0          # poniżej tej wysokości nad terenem [m] ucieczka odpala zawsze

# --- podgląd ---
POKAZ_PODGLAD = True         # okno z obrazem muszki i mapą narastania

OCG_ROLL = ["OCG01a", "OCG01b", "OCG01c", "OCG01d", "OCG01e", "OCG01f"]
OCG_PION = ["OCG02b", "OCG03"]
LOOMING = ["LC4", "LPLC2"]


def ruch_obrazu(poprzednia, klatka):
    """Zwraca (obrot, pion, ruch).

    obrot: różnica ruchu pionowego lewa-prawa połowa (przechylanie, w prawo > 0)
    pion:  ruch pionowy tylko w GÓRNEJ części obrazu (pochylanie, nos w górę > 0).
           Daleki horyzont przesuwa się tylko przy obrocie, a bliska ziemia także przy
           zwykłym locie do przodu - dlatego ziemię pomijamy.
    """
    ruch = cv2.calcOpticalFlowFarneback(poprzednia, klatka, None, 0.5, 2, 7, 3, 5, 1.1, 0)
    dy = ruch[..., 1]
    polowa = dy.shape[1] // 2
    gora = int(dy.shape[0] * CZESC_DO_POCHYLENIA)
    obrot = float(dy[:, :polowa].mean() - dy[:, polowa:].mean())
    pion = float(dy[:gora, :].mean())
    return obrot, pion, ruch


def lplc2(ruch):
    """Detektor jak neuron LPLC2: ruch na zewnątrz we WSZYSTKICH 4 kierunkach od środka pola.

    Zwraca siatkę (8 x 16): dla każdego pola widzenia najsłabsze z czterech ramion.
    Jeśli choć jedno ramię nie widzi ruchu na zewnątrz, wynik wynosi 0.
    """
    dx = cv2.blur(ruch[..., 0], (4, 4))     # lokalnie uśredniony ruch poziomy
    dy = cv2.blur(ruch[..., 1], (4, 4))     # lokalnie uśredniony ruch pionowy
    w, s = dx.shape

    ys = np.arange(ROZMIAR_PLAMKI // 2, w, ROZMIAR_PLAMKI)[:, None]   # środki pól: wiersze
    xs = np.arange(ROZMIAR_PLAMKI // 2, s, ROZMIAR_PLAMKI)[None, :]   # środki pól: kolumny

    def probka(mapa, y, x):
        return mapa[np.clip(y, 0, w - 1), np.clip(x, 0, s - 1)]

    prawo = probka(dx, ys, xs + RAMIE)      # na prawo od środka: ruch w prawo
    lewo = -probka(dx, ys, xs - RAMIE)      # na lewo od środka: ruch w lewo
    gora = -probka(dy, ys - RAMIE, xs)      # nad środkiem: ruch w górę
    dol = probka(dy, ys + RAMIE, xs)        # pod środkiem: ruch w dół

    najslabsze = np.minimum(np.minimum(prawo, lewo), np.minimum(gora, dol))
    return np.maximum(najslabsze, 0)


def przypisz_plamki(mozg, ksztalt_plamek, seed=0):
    """Każdemu neuronowi LC4/LPLC2 losuje jedno pole widzenia po jego stronie obrazu."""
    rng = np.random.default_rng(seed)
    wiersze, kolumny = ksztalt_plamek
    wiersze_gora = max(1, int(wiersze * CZESC_DO_LOOMINGU))
    polowa = kolumny // 2
    indeksy, w_idx, k_idx = [], [], []
    for i, nazwa in enumerate(mozg.nazwy):
        typ, _, strona = nazwa.rpartition("_")
        if typ in LOOMING:
            indeksy.append(i)
            w_idx.append(rng.integers(0, wiersze_gora))
            if strona == "L":
                k_idx.append(rng.integers(0, polowa))
            else:
                k_idx.append(rng.integers(polowa, kolumny))
    return np.array(indeksy), np.array(w_idx), np.array(k_idx)


def ciemnosc(klatka):
    """Ciemność lewej połowy, prawej połowy i całego obrazu (0 = biało, 1 = czarno)."""
    polowa = klatka.shape[1] // 2
    lewa = 1 - klatka[:, :polowa].mean() / 255
    prawa = 1 - klatka[:, polowa:].mean() / 255
    cala = 1 - klatka.mean() / 255
    return float(lewa), float(prawa), float(cala)


def wejscie(mozg, maski, obrot, pion, ciemn_l, ciemn_p, zmiana_ciemnosci):
    """Zamienia to, co widzi muszka, na pobudzenie VS i przyoczek."""
    obrot = float(np.clip(obrot, -MAKS_RUCH, MAKS_RUCH))
    pion = float(np.clip(pion, -MAKS_RUCH, MAKS_RUCH))
    roznica = ciemn_l - ciemn_p

    hz = np.zeros(mozg.N)
    hz[maski["vs_l"]] = max(BAZA_HZ + HZ_NA_RUCH * obrot + HZ_NA_RUCH_PION * pion, 0)
    hz[maski["vs_r"]] = max(BAZA_HZ - HZ_NA_RUCH * obrot + HZ_NA_RUCH_PION * pion, 0)
    hz[maski["ocg_l"]] = max(BAZA_PRZYOCZEK + HZ_NA_ROZNICE * roznica, 0)
    hz[maski["ocg_r"]] = max(BAZA_PRZYOCZEK - HZ_NA_ROZNICE * roznica, 0)
    hz[maski["ocg_pion"]] = max(BAZA_PION + HZ_NA_CIEMNOSC_PION * zmiana_ciemnosci, 0)
    return hz


def odczyt(maski, czest):
    """Zwraca (DNp20 L-P, DNp20 L+P, DNp19 L+P, Giant Fiber L+P)."""
    l20, p20 = czest[maski["dn20_l"]].mean(), czest[maski["dn20_r"]].mean()
    l19, p19 = czest[maski["dn19_l"]].mean(), czest[maski["dn19_r"]].mean()
    gf = czest[maski["gf_l"]].mean() + czest[maski["gf_r"]].mean()
    return l20 - p20, l20 + p20, l19 + p19, gf


def kalibruj(mozg, maski, sekundy=2):
    """Stan spoczynku: obraz stoi, strony jednakowo ciemne, nic nie rośnie."""
    hz = wejscie(mozg, maski, 0.0, 0.0, 0.5, 0.5, 0.0)
    okna = int(sekundy * 1000 / OKNO_MS)
    pomiary = [odczyt(maski, mozg.krok(OKNO_MS, hz)) for _ in range(okna)]
    return np.mean(pomiary, axis=0)[:3]     # zera dla DNp20 L-P, DNp20 L+P, DNp19 L+P


def zmierz_jasnosc_poziomu(oczy, sekundy=1.0):
    """Średnia ciemność całego obrazu w locie poziomym (samolot musi być wypoziomowany)."""
    pomiary = []
    koniec = time.time() + sekundy
    while time.time() < koniec:
        try:
            pomiary.append(ciemnosc(oczy.odbierz())[2])
        except TimeoutError:
            pass
    if not pomiary:
        raise RuntimeError("Brak obrazu z PC - czy działa wysylacz_obrazu.py?")
    return float(np.mean(pomiary))


def pokaz_podglad(klatka, pl, stan):
    """Okno: u góry obraz muszki, na dole mapa LPLC2 (gdzie coś leci prosto na muszkę)."""
    oczy_duze = cv2.resize(klatka, (640, 320), interpolation=cv2.INTER_NEAREST)
    oczy_kolor = cv2.cvtColor(oczy_duze, cv2.COLOR_GRAY2BGR)

    mapa = np.clip(pl / 0.05, 0, 1)    # stała skala: widać też aktywność poniżej progu
    mapa_duza = cv2.resize((mapa * 255).astype(np.uint8), (640, 320),
                           interpolation=cv2.INTER_NEAREST)
    mapa_kolor = cv2.applyColorMap(mapa_duza, cv2.COLORMAP_HOT)

    obraz = np.vstack([oczy_kolor, mapa_kolor])

    kolory = {"LOT": (0, 255, 0), "WZNOSZENIE": (0, 200, 255), "UCIECZKA": (0, 0, 255)}
    cv2.putText(obraz, stan, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.1, kolory[stan], 2)
    cv2.imshow("Muszka", obraz)


def pusc_stery(xp):
    """Bezpieczny stan: stery do neutralnego położenia."""
    xp.ustaw(WOLANT_ROLL, 0.0)
    if STERUJ_POCHYLENIEM:
        xp.ustaw(WOLANT_PITCH, 0.0)


def main():
    mozg = Mozg.z_plikow(FOLDER_DANYCH)
    maski = {
        "vs_l": np.isin(mozg.nazwy, ["VS_L", "VSm_L"]),
        "vs_r": np.isin(mozg.nazwy, ["VS_R", "VSm_R"]),
        "ocg_l": np.isin(mozg.nazwy, [f"{t}_L" for t in OCG_ROLL]),
        "ocg_r": np.isin(mozg.nazwy, [f"{t}_R" for t in OCG_ROLL]),
        "ocg_pion": np.isin(mozg.nazwy, [f"{t}_{s}" for t in OCG_PION for s in "LR"]),
        "dn20_l": mozg.maska("DNp20_L"),
        "dn20_r": mozg.maska("DNp20_R"),
        "dn19_l": mozg.maska("DNp19_L"),
        "dn19_r": mozg.maska("DNp19_R"),
        "gf_l": mozg.maska("DNp01_L"),
        "gf_r": mozg.maska("DNp01_R"),
    }

    print("Kalibracja mózgu...")
    zero_roznicy, zero_sumy20, zero_sumy19 = kalibruj(mozg, maski)

    oczy = Oczy()
    ksztalt_plamek = (oczy.ksztalt[0] // ROZMIAR_PLAMKI, oczy.ksztalt[1] // ROZMIAR_PLAMKI)
    loom_idx, loom_w, loom_k = przypisz_plamki(mozg, ksztalt_plamek)
    print(f"Neuronów LC4/LPLC2 z przypisanym polem widzenia: {len(loom_idx)}")

    print("Mierzę jasność w locie poziomym (trzymaj poziom)...")
    ciemnosc_poziomu = zmierz_jasnosc_poziomu(oczy)
    print(f"Ciemność w locie poziomym: {ciemnosc_poziomu:.3f}")

    xp = XPlane(IP_XPLANE, timeout=0.01)
    xp.subskrybuj(PHI, czestotliwosc=30)
    xp.subskrybuj(THETA, czestotliwosc=30)
    xp.subskrybuj(AGL, czestotliwosc=30)
    xp.ustaw(PRZEJECIE_ROLL, 1)
    if STERUJ_POCHYLENIEM:
        xp.ustaw(PRZEJECIE_PITCH, 1)
    print("Muszka steruje. Ctrl+C albo q w oknie podglądu, żeby wyłączyć.")

    poprzednia = None
    wygl_roll, wygl_d, wygl_p, wygl_gf = 0.0, 0.0, 0.0, 0.0
    stan = "LOT"                 # LOT -> UCIECZKA -> WZNOSZENIE -> LOT
    powod = ""                   # kto uruchomił ucieczkę: "muszka" albo "GPWS"
    koniec_stanu = 0.0           # kiedy kończy się UCIECZKA albo WZNOSZENIE
    koniec_refrakcji = 0.0       # do kiedy odruch nie może odpalić ponownie
    phi, theta, agl = float("nan"), float("nan"), float("nan")
    try:
        while True:
            try:
                klatka = oczy.odbierz()
            except TimeoutError:
                print("Brak obrazu - puszczam stery")
                pusc_stery(xp)
                poprzednia = None
                wygl_roll, wygl_d, wygl_p, wygl_gf = 0.0, 0.0, 0.0, 0.0
                stan = "LOT"
                continue

            if poprzednia is None:
                poprzednia = klatka
                continue

            obrot, pion, ruch = ruch_obrazu(poprzednia, klatka)
            poprzednia = klatka
            ciemn_l, ciemn_p, ciemn_cala = ciemnosc(klatka)
            zmiana = ciemn_cala - ciemnosc_poziomu

            hz = wejscie(mozg, maski, obrot, pion, ciemn_l, ciemn_p, zmiana)

            # narastanie obrazu: każdy neuron LC4/LPLC2 patrzy na swoje pole widzenia
            pl = lplc2(ruch)
            sila = np.clip((pl[loom_w, loom_k] - PROG_SZUMU) / ROZCHODZENIE_PELNE, 0, 1)

            # kopia eferentna: gdy cały obraz się OBRACA, wycisz narastanie
            ruch_wlasny = max(abs(obrot), abs(pion))
            tlumienie = float(np.clip(1 - ruch_wlasny / PROG_RUCHU_WLASNEGO, 0, 1))
            hz[loom_idx] = HZ_LOOM_MAKS * sila * tlumienie

            czest = mozg.krok(OKNO_MS, hz)
            roznica, suma20, suma19, gf = odczyt(maski, czest)

            wygl_roll = (1 - WYGLADZANIE) * wygl_roll + WYGLADZANIE * (roznica - zero_roznicy)
            wygl_d = (1 - WYGLADZANIE) * wygl_d + WYGLADZANIE * (suma20 - zero_sumy20)
            wygl_p = (1 - WYGLADZANIE) * wygl_p + WYGLADZANIE * (suma19 - zero_sumy19)
            wygl_gf = (1 - WYGLADZANIE_GF) * wygl_gf + WYGLADZANIE_GF * gf

            lotki = float(np.clip(KIERUNEK * wygl_roll * WZMOCNIENIE, -MAKS_LOTKI, MAKS_LOTKI))
            ster_wys = float(np.clip(KIERUNEK_PITCH * wygl_d * WZMOCNIENIE_PITCH
                                     + KIERUNEK_P19 * wygl_p * WZMOCNIENIE_P19,
                                     -MAKS_PCHANIE, 1))

            # --- maszyna stanów odruchu ucieczki ---
            teraz = time.time()
            if stan == "LOT":
                muszka_ucieka = wygl_gf > PROG_GF and teraz >= koniec_refrakcji
                gpws_ucieka = SIATKA_BEZPIECZENSTWA and agl < MIN_WYSOKOSC
                if muszka_ucieka or gpws_ucieka:
                    stan, koniec_stanu = "UCIECZKA", teraz + CZAS_UCIECZKI
                    powod = "muszka" if muszka_ucieka else "GPWS"
            elif stan == "UCIECZKA":
                if teraz >= koniec_stanu or zmiana < PROG_NIEBA:
                    stan, koniec_stanu = "WZNOSZENIE", teraz + CZAS_PO_UCIECZCE
            elif stan == "WZNOSZENIE":
                if teraz >= koniec_stanu:
                    stan = "LOT"
                    powod = ""
                    koniec_refrakcji = teraz + REFRAKCJA_UCIECZKI
                    wygl_gf = 0.0

            if stan == "UCIECZKA":
                lotki = float(np.clip(lotki * MNOZNIK_LOTEK_UCIECZKI, -MAKS_LOTKI, MAKS_LOTKI))
                ster_wys = max(ster_wys, SILA_UCIECZKI)
            elif stan == "WZNOSZENIE":
                ster_wys = max(ster_wys, 0.0)   # nie pchaj nosa w dół

            xp.ustaw(WOLANT_ROLL, lotki)
            if STERUJ_POCHYLENIEM:
                xp.ustaw(WOLANT_PITCH, ster_wys)

            try:
                w = xp.odbierz()
                phi, theta, agl = w.get(PHI, phi), w.get(THETA, theta), w.get(AGL, agl)
            except TimeoutError:
                pass

            aktywne = (sila > 0).mean() * 100
            print(f"[{stan:10}] {powod:6}  GF: {wygl_gf:5.0f}  LPLC2: {aktywne:4.1f}% (max {pl.max():.3f})  tłum.: {tlumienie:.1f}  "
                  f"|  lotki: {lotki:+.2f}  ster: {ster_wys:+.2f}  |  "
                  f"phi: {phi:+6.1f}°  theta: {theta:+6.1f}°  wys.: {agl:5.0f} m")

            if POKAZ_PODGLAD:
                pokaz_podglad(klatka, pl, stan)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\nWyłączam muszkę.")
    finally:
        pusc_stery(xp)
        xp.ustaw(PRZEJECIE_ROLL, 0)
        xp.ustaw(PRZEJECIE_PITCH, 0)
        xp.zakoncz()
        oczy.zakoncz()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()