# Muszka autopilot 🪰✈️

**Mózg muszki owocowej (konektom MaleCNS v1.0) steruje samolotem w X-Plane 12, patrząc wyłącznie na obraz.**

Projekt łączy prawdziwą mapę połączeń układu nerwowego samca *Drosophila melanogaster* z symulatorem lotu. Muszka nie dostaje z symulatora żadnych danych o położeniu samolotu: widzi tylko rozmyty obraz 64 × 32 piksele, tak jak oko złożone, i na tej podstawie steruje lotkami oraz sterem wysokości. Ma też odruch ucieczki przed zbliżającą się ziemią, uruchamiany przez neuron Giant Fiber.

Żadna z zasad sterowania nie jest zaprogramowana ręcznie jako „jeśli przechył w lewo, to wychyl lotki w prawo". Sygnały przepływają przez połączenia odczytane z konektomu, a odpowiedź odczytywana jest z neuronów zstępujących, które w prawdziwej muszce przenoszą polecenia ruchu z mózgu do ciała.

---

## Spis treści

- [Jak to działa](#jak-to-działa)
- [Neurony w modelu](#neurony-w-modelu)
- [Co pokazały dane z konektomu](#co-pokazały-dane-z-konektomu)
- [Model neuronu](#model-neuronu)
- [Architektura](#architektura)
- [Struktura projektu](#struktura-projektu)
- [Uruchomienie](#uruchomienie)
- [Wyniki](#wyniki)
- [Ograniczenia](#ograniczenia)
- [Eksperymenty i ciekawostki](#eksperymenty-i-ciekawostki)
- [Źródła](#źródła)
- [O powstaniu projektu](#o-powstaniu-projektu)

---

## Jak to działa

```
X-Plane 12 (PC)                                   Mac
───────────────                                   ─────────────────────────────────────────
zrzut ekranu ─► 64×32, skala szarości ──UDP──►  "oczy muszki"
                                                   │
                                                   ├─ przepływ optyczny ──────► komórki VS
                                                   ├─ jasność lewo/prawo ─────► przyoczka OCG01
                                                   ├─ jasność całego obrazu ──► przyoczka OCG02b/03
                                                   └─ ruch promienisty ───────► LC4 / LPLC2
                                                                │
                                                     mózg: ~370 neuronów LIF z konektomu
                                                                │
                                                   ├─ DNp20 lewy − prawy ─────► lotki
                                                   ├─ DNp20 lewy + prawy ─────► ster wys. (prędkość)
                                                   ├─ DNp19 lewy + prawy ─────► ster wys. (kąt)
                                                   └─ Giant Fiber (DNp01) ────► odruch ucieczki
                                                                │
stery ◄──────────────────────────────────UDP──── wychylenie wolantu
```

Pętla działa ok. 30 razy na sekundę. Symulacja 33 ms aktywności mózgu zajmuje ok. 10 ms, więc całość mieści się w czasie rzeczywistym z zapasem.

---

## Neurony w modelu

| Grupa | Typy | Liczba | Rola |
|---|---|---|---|
| Komórki wzrokowe ruchu | VS, VSm | 22 | wykrywają obracanie się obrazu (przechylanie, pochylanie) |
| Komórki wzrokowe ruchu poziomego | HSN, HSE, HSS | 6 | wykrywają obrót poziomy (odchylanie, na razie nieużywane do sterowania) |
| Przyoczka – przechylenie | OCG01a–f | 12 | porównują jasność lewej i prawej strony |
| Przyoczka – pochylenie | OCG02b, OCG03 | 4 | reagują na jasność całego obrazu |
| Detektory nadlatujących obiektów | LC4, LPLC2 | 311 | strzelają, gdy coś leci prosto na muszkę |
| Neurony zstępujące | DNp20, DNp22, DNp15, DNp19, DNp102, DNge107 | 12 | polecenia ruchu (wyjścia) |
| Giant Fiber | DNp01 | 2 | odruch ucieczki |

Wszystkie neurony i połączenia między nimi pochodzą z konektomu MaleCNS v1.0. Znak połączenia (pobudzanie/hamowanie) wynika z neuroprzekaźnika neuronu wysyłającego: acetylocholina pobudza, GABA i glutaminian hamują.

---

## Co pokazały dane z konektomu

Wyboru neuronów dokonywałem na podstawie analizy grafu połączeń w pandas, a nie z góry. Kilka wyników okazało się szczególnie ciekawych:

**1. Kanały przechylenia i odchylenia są rozdzielone.**
Komórki VS łączą się z DNp20 i DNp22, a komórki HS z DNp15. Nie ma połączeń „mieszanych", a wszystkie łączą się po tej samej stronie ciała (lewe z lewymi, prawe z prawymi).

**2. DNp20 to przede wszystkim „wskaźnik horyzontu".**
Od przyoczek (OCG01) dostaje ponad 4 razy więcej synaps (3880) niż od komórek VS (936). Zgadza się to z jego dawną nazwą w literaturze: DNOVS, czyli neuron zstępujący układu przyoczkowego i VS.

**3. Przyoczka tworzą układ push-pull.**
- OCG01d i OCG01b (acetylocholina) pobudzają DNp20/DNp22 **po przeciwnej stronie**,
- OCG01a, c, f (glutaminian) hamują je **po tej samej stronie**.

W efekcie lewy DNp20 liczy w praktyce „prawe przyoczko minus lewe przyoczko", czyli różnicę jasności, niezależnie od jej ogólnego poziomu.

**4. OCG02b łączy się z DNp19 symetrycznie.**
Każde przyoczko wysyła sygnał do obu DNp19 w podobnej ilości. Taki sygnał nadaje się do pochylenia: jednakowe ściemnienie obu stron (nos w dół) pobudza oba neurony.

**5. LC4 i LPLC2 tworzą ogromny system ucieczki.**
Do Giant Fibera dochodzi od nich ponad 11 tysięcy synaps. W modelu ujawnił się jednak problem: 96% połączeń w tej części podsieci to wzajemne pobudzenia LC4/LPLC2 (prawie 68 tysięcy synaps). Bez neuronów hamujących, których nie modelujemy, sieć raz pobudzona nie gasła („echo") i Giant Fiber strzelał bez przerwy. Usunięcie połączeń LC4/LPLC2 → LC4/LPLC2 rozwiązało problem.

---

## Model neuronu

Każdy neuron to model **leaky integrate-and-fire (LIF)** z parametrami z pracy Shiu i wsp. (2024):

| Parametr | Wartość |
|---|---|
| napięcie spoczynkowe | −52 mV |
| próg impulsu | −45 mV |
| stała czasowa błony | 20 ms |
| stała czasowa synapsy | 5 ms |
| refrakcja | 2,2 ms |
| siła jednej synapsy | 0,275 mV |
| krok symulacji | 0,1 ms |

Neuron zachowuje się jak kondensator z rezystorem upływowym: sumuje impulsy od innych neuronów, napięcie powoli opada do spoczynku, a po przekroczeniu progu neuron wysyła impuls i wraca do spoczynku. Siła połączenia to liczba synaps × znak neuroprzekaźnika × 0,275 mV.

Symulator jest napisany od zera w NumPy (`src/mozg.py`) i liczy wszystkie neurony naraz, operacjami na macierzach.

---

## Architektura

- **PC z X-Plane 12:** `wysylacz_obrazu.py` robi zrzuty ekranu, zmniejsza je do 64 × 32 w skali szarości i wysyła na Maca przez UDP (2048 bajtów na klatkę, 30 klatek/s).
- **Mac:** odbiera obraz (`oczy.py`), liczy przepływ optyczny (OpenCV), symuluje mózg (`mozg.py`) i wysyła polecenia sterów do X-Plane przez protokół UDP X-Plane (`xplane.py`, pakiety `RREF` i `DREF`).

Z X-Plane odczytywane są tylko kąty i wysokość **do wyświetlania na ekranie**. Muszka z nich nie korzysta.

---

## Struktura projektu

```
fly_autopilot/
├── data/                         # dane z konektomu (nie są w repozytorium, patrz "Uruchomienie")
│   └── processed/                # wybrane neurony i podsieć (tworzone w notatniku)
├── notebooks/
│   └── muszka.ipynb              # analiza konektomu, wybór neuronów, testy "na sucho"
└── src/
    ├── mozg.py                   # klasa Mozg - symulator sieci neuronów LIF
    ├── xplane.py                 # klasa XPlane - komunikacja z X-Plane przez UDP
    ├── oczy.py                   # klasa Oczy - odbiór obrazu z PC
    ├── autopilot.py              # pierwsza wersja: sterowanie na podstawie kąta z X-Plane
    ├── autopilot_oczy.py         # wersja główna: sterowanie wyłącznie wzrokiem
    ├── podglad_ruchu.py          # podgląd przepływu optycznego (diagnostyka)
    ├── podglad_loomingu.py       # podgląd narastania obrazu (diagnostyka)
    └── wysylacz_obrazu.py        # uruchamiany na PC z X-Plane
```

---

## Uruchomienie

### 1. Dane

Pobierz z [male-cns.janelia.org](https://male-cns.janelia.org/download/) do folderu `data/`:

- `connectome-weights-male-cns-v1.0-minconf-0.5.feather` (graf połączeń, ~1 GB)
- `body-annotations-male-cns-v1.0-minconf-0.5.feather` (opisy neuronów, 13 MB)
- `body-neurotransmitters-male-cns-v1.0.feather` (neuroprzekaźniki, 42 MB)

Pliki są dostępne publicznie pod adresem `https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/<nazwa pliku>`.

Następnie uruchom `notebooks/muszka.ipynb`, który tworzy `data/processed/neurony.csv` i `data/processed/podsiec.csv`.

### 2. Komputer z X-Plane (Windows)

```bash
pip install mss numpy opencv-python
python wysylacz_obrazu.py
```

W pliku wpisz adres IP Maca. X-Plane powinien być w widoku **bez kokpitu** (Shift + W): muszka ma widzieć świat, a nie instrumenty.

### 3. Mac

```bash
pip install numpy pandas pyarrow opencv-python matplotlib
cd src
python autopilot_oczy.py
```

W `autopilot.py` ustaw `IP_XPLANE` na adres komputera z X-Plane.

> **Uwaga dla macOS 27:** aplikacje terminala innych firm (iTerm2, Warp, terminal w VS Code) mogą nie mieć dostępu do sieci lokalnej mimo włączonego uprawnienia („No route to host"). Skrypty łączące się z X-Plane uruchamiaj z systemowego **Terminal.app**.

---

## Wyniki

**Przechylenie (lotki)**
- Charakterystyka „przechylenie → różnica aktywności DNp20" jest prawie liniowa (ok. 5 Hz na stopień) i symetryczna.
- Muszka wyprowadza samolot z przechylenia ok. 30° do lotu poziomego, korzystając tylko z obrazu.
- Same komórki VS (bez przyoczek) tylko hamują obracanie się samolotu, ale nie przywracają poziomu. Dopiero przyoczka dają informację o kącie. W języku automatyki: VS to człon D, przyoczka to człon P regulatora PD.

**Pochylenie (ster wysokości)**
- Suma aktywności DNp20 reaguje na ruch obrazu w pionie (prędkość pochylania).
- Suma aktywności DNp19, pobudzanego przez OCG02b/03, reaguje na jasność obrazu (kąt pochylenia).
- Muszka stabilizuje lot w obu osiach naraz. Lekkie falowanie nosa (fugoida) jest tłumione, ale nie znika całkowicie.

**Odruch ucieczki**
- Giant Fiber odpowiada stopniowo na odsetek aktywnych neuronów LC4/LPLC2 (próg ucieczki przy ok. 10%).
- Detektor wzorowany na LPLC2 (ruch na zewnątrz we wszystkich czterech kierunkach naraz) nie reaguje w zwykłym locie, a reaguje, gdy samolot leci w stronę terenu.
- Ucieczka ma stały czas trwania, fazę wznoszenia i okres refrakcji (maszyna stanów).

---

## Ograniczenia

- **Rozdzielczość i prędkość.** Muszka widzi 64 × 32 punkty, a Cessna leci ok. 50 m/s. Teren zaczyna wyraźnie „rosnąć" w obrazie dopiero blisko, więc odruch ucieczki ma mało czasu. Prawdziwa mucha leci kilka metrów na sekundę i ma dużo większy zapas.
- **Duże przechylenia.** Powyżej ok. 90° przyoczka przestają rozróżniać niebo i ziemię, więc muszka traci orientację. Dlatego wychylenie lotek jest ograniczone do połowy.
- **Uproszczony mózg.** ~370 z ok. 166 tysięcy neuronów, bez neuronów hamujących pośredniczących, bez neuromodulatorów (np. dopaminy) i bez synaps elektrycznych, których mikroskop elektronowy nie wykrywa. Komórki HS i VS w prawdziwej muszce są w dużej mierze sprzężone elektrycznie.
- **Kodowanie wejść.** Zamiana obrazu na częstotliwości impulsów (np. „40 Hz + ruch × 40") i odczyt wyjść (np. „różnica DNp20 / 300") to moje założenia inżynierskie, dobrane eksperymentalnie, a nie dane biologiczne.
- **Pola widzenia LC4/LPLC2** są przypisane losowo, bo konektom nie podaje ich bezpośrednio.
- **Brak przezmianek** (żyroskopowych czujników obrotu), które prawdziwe muchy wykorzystują do stabilizacji lotu.

---

## Eksperymenty i ciekawostki

**Dezorientacja przestrzenna.** Po przełączeniu X-Plane na widok z kokpitu muszka wykonała dwie beczki. Nieruchoma tablica przyrządów zajmowała większość obrazu: ruch obrazu zniknął (VS przestały tłumić obrót), asymetria kokpitu wyglądała jak przechylenie (przyoczka), a ogólne ściemnienie jak opuszczony nos. Po powrocie do widoku zewnętrznego muszka sama wyprowadziła samolot. To samo zjawisko u pilotów nazywa się dezorientacją przestrzenną.

**Ruch ziemi brany za pochylenie.** Przy niskim locie tekstura ziemi przesuwa się w obrazie w dół, co muszka początkowo odczytywała jako podnoszenie nosa i pchała wolant od siebie. Rozwiązanie: pochylenie mierzone jest tylko w górnej części obrazu, bo daleki horyzont przesuwa się tylko przy obrocie, a bliska ziemia także przy zwykłym locie do przodu.

**Kopia eferentna.** Podczas gwałtownego manewru cały obraz szybko się przesuwa, a przepływ optyczny daje błędne wyniki, które wyglądały jak nadlatujący obiekt. Na wzór muchy, która wycisza część neuronów wzrokowych podczas własnych manewrów, sygnał LC4/LPLC2 jest tłumiony, gdy obraz jako całość szybko się obraca.

---

## Źródła

- **MaleCNS v1.0** – konektom ośrodkowego układu nerwowego samca *Drosophila melanogaster*. HHMI Janelia (FlyEM), University of Cambridge, MRC LMB, Google Research. [male-cns.janelia.org](https://male-cns.janelia.org). Licencja CC-BY 4.0.
- Shiu P. K. i wsp. (2024). *A Drosophila computational brain model reveals sensorimotor processing.* Nature. Parametry modelu LIF.
- Klapoetke N. C. i wsp. (2017). *Ultra-selective looming detection from radial motion opponency.* Nature. Zasada działania LPLC2.
- [X-Plane datarefs](https://developer.x-plane.com/datarefs/) – dokumentacja zmiennych symulatora.

---

## O powstaniu projektu

Projekt powstał przy pomocy asystenta AI (Claude), który pomagał w projektowaniu, wyjaśnianiu biologii i pisaniu dużej części kodu. Pomysł, decyzje projektowe, przygotowanie środowiska, testy w locie, diagnozowanie problemów na podstawie danych i dostrajanie parametrów były moją pracą.


