import cv2
import numpy as np

from autopilot import FOLDER_DANYCH, IP_XPLANE, PHI, PRZEJECIE_ROLL, WOLANT_ROLL
from mozg import Mozg
from oczy import Oczy
from xplane import XPlane

THETA = "sim/flightmodel/position/theta"
WOLANT_PITCH = "sim/joystick/yoke_pitch_ratio"
PRZEJECIE_PITCH = "sim/operation/override/override_joystick_pitch"

# --- wejścia (co muszka widzi) ---
OKNO_MS = 33
BAZA_HZ = 40              # pobudzenie VS, gdy obraz stoi w miejscu
HZ_NA_RUCH = 40.0         # przechylanie: Hz na 1 piksel/klatkę różnicy lewo-prawo
HZ_NA_RUCH_PION = 60.0    # pochylanie: Hz na 1 piksel/klatkę wspólnego ruchu w pionie
MAKS_RUCH = 1.5
BAZA_PRZYOCZEK = 60.0
HZ_NA_ROZNICE = 400.0

# --- wyjścia (stery) ---
WYGLADZANIE = 0.1
WZMOCNIENIE = 1 / 300          # lotki
KIERUNEK = -1
STERUJ_POCHYLENIEM = True      # False = muszka steruje tylko lotkami
WZMOCNIENIE_PITCH = 1 / 150    # ster wysokości
KIERUNEK_PITCH = -1            # zmień na +1, jeśli muszka rozbuja nos zamiast go uspokoić

OCG = ["OCG01a", "OCG01b", "OCG01c", "OCG01d", "OCG01e", "OCG01f"]


def ruch_obrazu(poprzednia, klatka):
    """Zwraca (obrot, pion): obrót w prawo > 0, obraz jedzie w dół (nos w górę) > 0."""
    ruch = cv2.calcOpticalFlowFarneback(poprzednia, klatka, None, 0.5, 2, 7, 3, 5, 1.1, 0)
    dy = ruch[..., 1]
    polowa = dy.shape[1] // 2
    obrot = float(dy[:, :polowa].mean() - dy[:, polowa:].mean())
    pion = float(dy.mean())
    return obrot, pion


def ciemnosc(klatka):
    """Ciemność lewej i prawej połowy obrazu: 0 = biało, 1 = czarno."""
    polowa = klatka.shape[1] // 2
    lewa = 1 - klatka[:, :polowa].mean() / 255
    prawa = 1 - klatka[:, polowa:].mean() / 255
    return float(lewa), float(prawa)


def wejscie(mozg, maski, obrot, pion, ciemn_l, ciemn_p):
    """Zamienia to, co widzi muszka, na pobudzenie VS i przyoczek."""
    obrot = float(np.clip(obrot, -MAKS_RUCH, MAKS_RUCH))
    pion = float(np.clip(pion, -MAKS_RUCH, MAKS_RUCH))
    roznica_ciemnosci = ciemn_l - ciemn_p

    hz = np.zeros(mozg.N)
    hz[maski["vs_l"]] = max(BAZA_HZ + HZ_NA_RUCH * obrot + HZ_NA_RUCH_PION * pion, 0)
    hz[maski["vs_r"]] = max(BAZA_HZ - HZ_NA_RUCH * obrot + HZ_NA_RUCH_PION * pion, 0)
    hz[maski["ocg_l"]] = max(BAZA_PRZYOCZEK + HZ_NA_ROZNICE * roznica_ciemnosci, 0)
    hz[maski["ocg_r"]] = max(BAZA_PRZYOCZEK - HZ_NA_ROZNICE * roznica_ciemnosci, 0)
    return hz


def odczyt(maski, czest):
    """Zwraca (lewy - prawy, lewy + prawy) dla DNp20."""
    lewy = czest[maski["dn_l"]].mean()
    prawy = czest[maski["dn_r"]].mean()
    return lewy - prawy, lewy + prawy


def kalibruj(mozg, maski, sekundy=2):
    """Stan spoczynku: obraz stoi, obie strony jednakowo ciemne."""
    hz = wejscie(mozg, maski, 0.0, 0.0, 0.5, 0.5)
    okna = int(sekundy * 1000 / OKNO_MS)
    pomiary = [odczyt(maski, mozg.krok(OKNO_MS, hz)) for _ in range(okna)]
    zero_roznicy, zero_sumy = np.mean(pomiary, axis=0)
    return float(zero_roznicy), float(zero_sumy)


def main():
    mozg = Mozg.z_plikow(FOLDER_DANYCH)
    maski = {
        "vs_l": np.isin(mozg.nazwy, ["VS_L", "VSm_L"]),
        "vs_r": np.isin(mozg.nazwy, ["VS_R", "VSm_R"]),
        "ocg_l": np.isin(mozg.nazwy, [f"{t}_L" for t in OCG]),
        "ocg_r": np.isin(mozg.nazwy, [f"{t}_R" for t in OCG]),
        "dn_l": mozg.maska("DNp20_L"),
        "dn_r": mozg.maska("DNp20_R"),
    }

    print("Kalibracja zera...")
    zero_roznicy, zero_sumy = kalibruj(mozg, maski)
    print(f"Zero różnicy: {zero_roznicy:.1f} Hz   zero sumy: {zero_sumy:.1f} Hz")

    oczy = Oczy()
    xp = XPlane(IP_XPLANE, timeout=0.01)
    xp.subskrybuj(PHI, czestotliwosc=30)
    xp.subskrybuj(THETA, czestotliwosc=30)
    xp.ustaw(PRZEJECIE_ROLL, 1)
    if STERUJ_POCHYLENIEM:
        xp.ustaw(PRZEJECIE_PITCH, 1)
    print("Muszka steruje. Ctrl+C, żeby wyłączyć.")

    poprzednia = None
    wygl_roll, wygl_pitch = 0.0, 0.0
    phi, theta = float("nan"), float("nan")
    try:
        while True:
            try:
                klatka = oczy.odbierz()
            except TimeoutError:
                print("Brak obrazu z PC...")
                continue

            if poprzednia is None:
                poprzednia = klatka
                continue

            obrot, pion = ruch_obrazu(poprzednia, klatka)
            poprzednia = klatka
            ciemn_l, ciemn_p = ciemnosc(klatka)

            hz = wejscie(mozg, maski, obrot, pion, ciemn_l, ciemn_p)
            czest = mozg.krok(OKNO_MS, hz)
            roznica, suma = odczyt(maski, czest)
            roznica -= zero_roznicy
            suma -= zero_sumy

            wygl_roll = (1 - WYGLADZANIE) * wygl_roll + WYGLADZANIE * roznica
            wygl_pitch = (1 - WYGLADZANIE) * wygl_pitch + WYGLADZANIE * suma

            lotki = float(np.clip(KIERUNEK * wygl_roll * WZMOCNIENIE, -1, 1))
            ster_wys = float(np.clip(KIERUNEK_PITCH * wygl_pitch * WZMOCNIENIE_PITCH, -1, 1))
            xp.ustaw(WOLANT_ROLL, lotki)
            if STERUJ_POCHYLENIEM:
                xp.ustaw(WOLANT_PITCH, ster_wys)

            try:
                w = xp.odbierz()
                phi, theta = w.get(PHI, phi), w.get(THETA, theta)
            except TimeoutError:
                pass

            print(f"obrót: {obrot:+5.2f}  pion: {pion:+5.2f}  |  "
                  f"lotki: {lotki:+.2f}  ster wys.: {ster_wys:+.2f}  |  "
                  f"phi: {phi:+6.1f}°  theta: {theta:+6.1f}°")
    except KeyboardInterrupt:
        print("\nWyłączam muszkę.")
    finally:
        xp.ustaw(WOLANT_ROLL, 0.0)
        xp.ustaw(WOLANT_PITCH, 0.0)
        xp.ustaw(PRZEJECIE_ROLL, 0)
        xp.ustaw(PRZEJECIE_PITCH, 0)
        xp.zakoncz()
        oczy.zakoncz()


if __name__ == "__main__":
    main()