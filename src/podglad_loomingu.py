import cv2
import numpy as np

from autopilot import IP_XPLANE
from oczy import Oczy
from xplane import XPlane

AGL = "sim/flightmodel/position/y_agl"   # wysokość nad terenem [m], tylko do porównania
ROZMIAR_PLAMKI = 4                       # fragment obrazu 4 x 4 piksele = pole jednego neuronu
PROG_SZUMU = 0.05                        # poniżej tej dywergencji uznajemy, że nic nie rośnie


def dywergencja(ruch):
    """Mapa 'rozciągania się' obrazu: dodatnia = coś się zbliża."""
    dx, dy = ruch[..., 0], ruch[..., 1]
    return np.gradient(dx, axis=1) + np.gradient(dy, axis=0)


def plamki(mapa, rozmiar=ROZMIAR_PLAMKI):
    """Uśrednia mapę w kwadratach rozmiar x rozmiar (32x64 -> 8x16 plamek)."""
    w, s = mapa.shape
    return mapa.reshape(w // rozmiar, rozmiar, s // rozmiar, rozmiar).mean(axis=(1, 3))


oczy = Oczy()
xp = XPlane(IP_XPLANE, timeout=0.01)
xp.subskrybuj(AGL, czestotliwosc=30)

poprzednia = None
agl = float("nan")
print("Podgląd narastania obrazu (q w oknie, żeby zakończyć)")
try:
    while True:
        try:
            klatka = oczy.odbierz()
        except TimeoutError:
            continue
        if poprzednia is None:
            poprzednia = klatka
            continue

        ruch = cv2.calcOpticalFlowFarneback(poprzednia, klatka, None, 0.5, 2, 7, 3, 5, 1.1, 0)
        poprzednia = klatka
        pl = plamki(dywergencja(ruch))

        rosnace = (pl > PROG_SZUMU).mean() * 100     # ile % plamek "rośnie"
        try:
            agl = xp.odbierz().get(AGL, agl)
        except TimeoutError:
            pass
        print(f"max: {pl.max():+.3f}   rośnie: {rosnace:5.1f}% plamek   wysokość: {agl:7.0f} m")

        # podgląd: jasne = rośnie, czarne = nie
        obraz = np.clip(pl / 0.3, 0, 1) * 255
        cv2.imshow("Narastanie obrazu", cv2.resize(obraz.astype(np.uint8), (640, 320),
                                                   interpolation=cv2.INTER_NEAREST))
        cv2.imshow("Oczy muszki", cv2.resize(klatka, (640, 320), interpolation=cv2.INTER_NEAREST))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    xp.zakoncz()
    oczy.zakoncz()
    cv2.destroyAllWindows()