import cv2
from oczy import Oczy

oczy = Oczy()
print("Czekam na obraz... (q w oknie podglądu, żeby zakończyć)")
try:
    while True:
        klatka = oczy.odbierz()
        duza = cv2.resize(klatka, (640, 320), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Oczy muszki", duza)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    oczy.zakoncz()
    cv2.destroyAllWindows()