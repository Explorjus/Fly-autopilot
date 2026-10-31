import socket
import numpy as np


class Oczy:
    """Odbiera z sieci małe obrazy w odcieniach szarości (np. 64 x 32) z X-Plane."""

    def __init__(self, port=49010, szerokosc=64, wysokosc=32, timeout=1.0):
        self.ksztalt = (wysokosc, szerokosc)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("", port))     # słuchaj na tym porcie na wszystkich kartach sieciowych
        self.sock.settimeout(timeout)
        self.timeout = timeout

    def odbierz(self):
        """Zwraca NAJNOWSZĄ klatkę, starsze zaległe klatki są pomijane."""
        dane, _ = self.sock.recvfrom(65535)        # czekaj na co najmniej jedną klatkę
        self.sock.setblocking(False)               # teraz nie czekaj, tylko zbierz zaległe
        try:
            while True:
                dane, _ = self.sock.recvfrom(65535)
        except BlockingIOError:                    # kolejka pusta, mamy najnowszą
            pass
        finally:
            self.sock.settimeout(self.timeout)     # przywróć normalne czekanie
        return np.frombuffer(dane, dtype=np.uint8).reshape(self.ksztalt)

    def zakoncz(self):
        self.sock.close()