import time
import numpy as np

from mozg import Mozg
from xplane import XPlane

# --- ustawienia ---
IP_XPLANE = "192.168.1.107"
FOLDER_DANYCH = "/Users/explorjus/Python-projekty/fly_autopilot/data/processed"

OKNO_MS = 50          # co ile ms liczymy mózg i wysyłamy sterowanie
BAZA_HZ = 40          # pobudzenie VS przy locie poziomym
HZ_NA_STOPIEN = 1.0   # ile Hz zmienia się pobudzenie na 1° przechylenia
WZMOCNIENIE = 1 / 300 # przelicznik: Hz różnicy -> wychylenie wolantu
WYGLADZANIE = 0.2     # 0..1, im mniej, tym spokojniej (i wolniej) reaguje
KIERUNEK = -1         # -1 = przeciwdziałaj przechyleniu; zmień na +1, jeśli robi odwrotnie

PHI = "sim/flightmodel/position/phi"
WOLANT_ROLL = "sim/joystick/yoke_roll_ratio"
PRZEJECIE_ROLL = "sim/operation/override/override_joystick_roll"


def czestotliwosci_wejscia(mozg, phi, vs_l, vs_r):
    """Zamienia przechylenie samolotu na pobudzenie lewych i prawych VS."""
    hz = np.zeros(mozg.N)
    hz[vs_l] = max(BAZA_HZ + HZ_NA_STOPIEN * phi, 0)
    hz[vs_r] = max(BAZA_HZ - HZ_NA_STOPIEN * phi, 0)
    return hz


def roznica_dn(mozg, czest, dn_l, dn_r):
    """Różnica aktywności DNp20 lewy - prawy [Hz]."""
    return czest[dn_l].mean() - czest[dn_r].mean()


def kalibruj(mozg, vs_l, vs_r, dn_l, dn_r, sekundy=2):
    """Mierzy przesunięcie zera: różnicę DN przy locie poziomym."""
    hz = czestotliwosci_wejscia(mozg, 0.0, vs_l, vs_r)
    okna = int(sekundy * 1000 / OKNO_MS)
    roznice = [roznica_dn(mozg, mozg.krok(OKNO_MS, hz), dn_l, dn_r) for _ in range(okna)]
    return float(np.mean(roznice))


def main():
    mozg = Mozg.z_plikow(FOLDER_DANYCH)
    vs_l, vs_r = mozg.maska("VS_L"), mozg.maska("VS_R")
    dn_l, dn_r = mozg.maska("DNp20_L"), mozg.maska("DNp20_R")

    print("Kalibracja zera...")
    przesuniecie = kalibruj(mozg, vs_l, vs_r, dn_l, dn_r)
    print(f"Przesunięcie zera: {przesuniecie:.1f} Hz")

    xp = XPlane(IP_XPLANE, timeout=0.5)
    xp.subskrybuj(PHI, czestotliwosc=int(1000 / OKNO_MS))
    xp.ustaw(PRZEJECIE_ROLL, 1)   # od teraz lotkami steruje skrypt, nie joystick
    print("Autopilot włączony. Ctrl+C, żeby wyłączyć.")

    wygladzona = 0.0
    try:
        while True:
            try:
                phi = xp.odbierz().get(PHI, 0.0)
            except TimeoutError:
                print("Brak danych z X-Plane...")
                continue

            hz = czestotliwosci_wejscia(mozg, phi, vs_l, vs_r)
            czest = mozg.krok(OKNO_MS, hz)
            roznica = roznica_dn(mozg, czest, dn_l, dn_r) - przesuniecie

            wygladzona = (1 - WYGLADZANIE) * wygladzona + WYGLADZANIE * roznica
            wolant = float(np.clip(KIERUNEK * wygladzona * WZMOCNIENIE, -1, 1))
            xp.ustaw(WOLANT_ROLL, wolant)

            print(f"phi: {phi:6.1f}°   DN L-P: {roznica:7.1f} Hz   wolant: {wolant:+.2f}")
    except KeyboardInterrupt:
        print("\nWyłączam autopilota.")
    finally:
        xp.ustaw(WOLANT_ROLL, 0.0)
        xp.ustaw(PRZEJECIE_ROLL, 0)   # oddaj lotki joystickowi
        xp.zakoncz()


if __name__ == "__main__":
    main()