import cv2
import numpy as np

from autopilot import FOLDER_DANYCH, IP_XPLANE, PHI, PRZEJECIE_ROLL, WOLANT_ROLL, roznica_dn
from mozg import Mozg
from oczy import Oczy
from xplane import XPlane

# --- ustawienia ---
OKNO_MS = 33            # ~30 klatek/s z PC
BAZA_HZ = 40            # pobudzenie VS, gdy obraz się nie obraca
HZ_NA_RUCH = 40.0       # ile Hz pobudzenia VS na 1 piksel/klatkę obrotu obrazu
MAKS_RUCH = 1.5         # obcięcie skrajnych pomiarów ruchu
BAZA_PRZYOCZEK = 60.0   # pobudzenie przyoczek, gdy obie strony są jednakowo ciemne
HZ_NA_ROZNICE = 400.0   # ile Hz na różnicę ciemności lewo-prawo (0..1)
WZMOCNIENIE = 1 / 300
WYGLADZANIE = 0.1       # mniejsze = spokojniejszy wolant (mniej szumu impulsów)
KIERUNEK = -1

OCG = ["OCG01a", "OCG01c", "OCG01d", "OCG01e"]


def ruch_obrazu(poprzednia, klatka):
    """Obrót obrazu między dwiema klatkami: dodatni = obrót w prawo."""
    ruch = cv2.calcOpticalFlowFarneback(poprzednia, klatka, None, 0.5, 2, 7, 3, 5, 1.1, 0)
    dy = ruch[..., 1]
    polowa = dy.shape[1] // 2
    return float(dy[:, :polowa].mean() - dy[:, polowa:].mean())


def ciemnosc(klatka):
    """Ciemność lewej i prawej połowy obrazu: 0 = biało, 1 = czarno."""
    polowa = klatka.shape[1] // 2
    lewa = 1 - klatka[:, :polowa].mean() / 255
    prawa = 1 - klatka[:, polowa:].mean() / 255
    return float(lewa), float(prawa)


def wejscie(mozg, maski, obrot, ciemn_l, ciemn_p):
    """Zamienia to, co widzi muszka, na pobudzenie VS i przyoczek."""
    obrot = float(np.clip(obrot, -MAKS_RUCH, MAKS_RUCH))
    roznica_ciemnosci = ciemn_l - ciemn_p      # dodatnia = lewa strona ciemniejsza

    hz = np.zeros(mozg.N)
    hz[maski["vs_l"]] = max(BAZA_HZ + HZ_NA_RUCH * obrot, 0)
    hz[maski["vs_r"]] = max(BAZA_HZ - HZ_NA_RUCH * obrot, 0)
    hz[maski["ocg_l"]] = max(BAZA_PRZYOCZEK + HZ_NA_ROZNICE * roznica_ciemnosci, 0)
    hz[maski["ocg_r"]] = max(BAZA_PRZYOCZEK - HZ_NA_ROZNICE * roznica_ciemnosci, 0)
    return hz


def kalibruj(mozg, maski, sekundy=2):
    """Przesunięcie zera: obraz się nie obraca, obie strony jednakowo ciemne."""
    hz = wejscie(mozg, maski, 0.0, 0.5, 0.5)
    okna = int(sekundy * 1000 / OKNO_MS)
    roznice = [roznica_dn(mozg, mozg.krok(OKNO_MS, hz), maski["dn_l"], maski["dn_r"])
               for _ in range(okna)]
    return float(np.mean(roznice))


def main():
    mozg = Mozg.z_plikow(FOLDER_DANYCH)
    maski = {
        "vs_l": mozg.maska("VS_L"),
        "vs_r": mozg.maska("VS_R"),
        "ocg_l": np.isin(mozg.nazwy, [f"{t}_L" for t in OCG]),
        "ocg_r": np.isin(mozg.nazwy, [f"{t}_R" for t in OCG]),
        "dn_l": mozg.maska("DNp20_L"),
        "dn_r": mozg.maska("DNp20_R"),
    }

    print("Kalibracja zera...")
    przesuniecie = kalibruj(mozg, maski)
    print(f"Przesunięcie zera: {przesuniecie:.1f} Hz")

    oczy = Oczy()
    xp = XPlane(IP_XPLANE, timeout=0.01)
    xp.subskrybuj(PHI, czestotliwosc=30)
    xp.ustaw(PRZEJECIE_ROLL, 1)
    print("Muszka steruje lotkami (oczy + przyoczka). Ctrl+C, żeby wyłączyć.")

    poprzednia = None
    wygladzona = 0.0
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

            obrot = ruch_obrazu(poprzednia, klatka)
            poprzednia = klatka
            ciemn_l, ciemn_p = ciemnosc(klatka)

            hz = wejscie(mozg, maski, obrot, ciemn_l, ciemn_p)
            czest = mozg.krok(OKNO_MS, hz)
            roznica = roznica_dn(mozg, czest, maski["dn_l"], maski["dn_r"]) - przesuniecie

            wygladzona = (1 - WYGLADZANIE) * wygladzona + WYGLADZANIE * roznica
            wolant = float(np.clip(KIERUNEK * wygladzona * WZMOCNIENIE, -1, 1))
            xp.ustaw(WOLANT_ROLL, wolant)

            try:
                phi = xp.odbierz().get(PHI, float("nan"))
            except TimeoutError:
                phi = float("nan")

            print(f"ruch: {obrot:+5.2f}   ciemność L/P: {ciemn_l:.2f}/{ciemn_p:.2f}   "
                  f"DN L-P: {roznica:+7.1f}   wolant: {wolant:+.2f}   phi: {phi:+6.1f}°")
    except KeyboardInterrupt:
        print("\nWyłączam muszkę.")
    finally:
        xp.ustaw(WOLANT_ROLL, 0.0)
        xp.ustaw(PRZEJECIE_ROLL, 0)
        xp.zakoncz()
        oczy.zakoncz()


if __name__ == "__main__":
    main()