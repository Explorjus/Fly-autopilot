import cv2
import numpy as np

from oczy import Oczy
from xplane import XPlane

P = "sim/flightmodel/position/P"   # prawdziwa prędkość przechylania z X-Plane [°/s]

oczy = Oczy()
xp = XPlane("192.168.1.107", timeout=0.05)
xp.subskrybuj(P, czestotliwosc=30)

poprzednia = None
print("Czekam na obraz... (q w oknie podglądu, żeby zakończyć)")

try:
    while True:
        try:
            klatka = oczy.odbierz()
        except TimeoutError:
            print("Brak obrazu z PC...")
            continue

        if poprzednia is not None:
            # przepływ optyczny: przesunięcie każdego piksela między klatkami
            ruch = cv2.calcOpticalFlowFarneback(
                poprzednia, klatka, None,
                0.5, 2, 7, 3, 5, 1.1, 0
            )
            dy = ruch[..., 1]                  # tylko ruch w pionie
            polowa = dy.shape[1] // 2
            obrot = dy[:, :polowa].mean() - dy[:, polowa:].mean()

            try:
                p_xplane = xp.odbierz().get(P, 0.0)
            except TimeoutError:
                p_xplane = float("nan")

            print(f"oczy muszki (obrót): {obrot:+6.2f}   X-Plane P: {p_xplane:+7.1f} °/s")

        poprzednia = klatka

        duza = cv2.resize(klatka, (640, 320), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Oczy muszki", duza)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    xp.zakoncz()
    oczy.zakoncz()
    cv2.destroyAllWindows()