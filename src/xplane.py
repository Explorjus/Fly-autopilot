import socket
import struct


class XPlane:
    """Rozmowa z X-Plane przez UDP: odczyt (RREF) i zapis (DREF) datarefów."""

    def __init__(self, ip, port=49000, timeout=1.0):
        self.adres = (ip, port)
        self.timeout = timeout                  # <- dopisz tę linię
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(timeout)
        self.nazwy = {}      # Twój numer -> nazwa datarefu
        self.wartosci = {}   # nazwa datarefu -> ostatnia wartość

    def subskrybuj(self, dataref, czestotliwosc=20):
        """Prosi X-Plane o regularne przysyłanie wartości datarefu."""
        numer = len(self.nazwy)
        self.nazwy[numer] = dataref
        paczka = struct.pack("<4sxii400s", b"RREF", czestotliwosc, numer, dataref.encode())
        self.sock.sendto(paczka, self.adres)

    def odbierz(self):
        """Odbiera jedną paczkę i aktualizuje słownik wartości."""
        dane, _ = self.sock.recvfrom(2048)
        if dane[:4] == b"RREF":
            for i in range(5, len(dane), 8):
                numer, wartosc = struct.unpack("<if", dane[i:i + 8])
                if numer in self.nazwy:
                    self.wartosci[self.nazwy[numer]] = wartosc
        return self.wartosci

    def ustaw(self, dataref, wartosc):
        """Ustawia wartość datarefu w X-Plane."""
        paczka = struct.pack("<4sxf500s", b"DREF", wartosc, dataref.encode())
        self.sock.sendto(paczka, self.adres)

    def zakoncz(self):
        """Wyłącza subskrypcje (częstotliwość 0) i zamyka gniazdo."""
        for numer, dataref in self.nazwy.items():
            paczka = struct.pack("<4sxii400s", b"RREF", 0, numer, dataref.encode())
            self.sock.sendto(paczka, self.adres)
        self.sock.close()
    
    def odbierz(self):
        """Odbiera wszystkie zaległe paczki; w słowniku zostają najnowsze wartości."""
        pakiety = [self.sock.recvfrom(2048)[0]]    # czekaj na co najmniej jedną
        self.sock.setblocking(False)
        try:
            while True:
                pakiety.append(self.sock.recvfrom(2048)[0])
        except BlockingIOError:
            pass
        finally:
            self.sock.settimeout(self.timeout)

        for dane in pakiety:
            if dane[:4] == b"RREF":
                for i in range(5, len(dane), 8):
                    numer, wartosc = struct.unpack("<if", dane[i:i + 8])
                    if numer in self.nazwy:
                        self.wartosci[self.nazwy[numer]] = wartosc
        return self.wartosci