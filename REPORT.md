# Detekcja ataków GPS Spoofing przy użyciu uczenia maszynowego

**Kościuszkon 2026 · Honeywell Theme #2**

---

## 1. Zrozumienie problemu

**Atak GPS Spoofing** to klasa ataków cyber-fizycznych, w której atakujący nadaje sfałszowany sygnał GNSS, sprawiając że odbiornik ofiary obliczy błędną pozycję, prędkość lub czas. Dla autonomicznego drona atak może:

- przesunąć raportowaną pozycję poziomą poza prawdziwą trajektorię,
- zafałszować raportowaną wysokość zachowując wiarygodną pozycję poziomą,
- wstrzyknąć syntetyczną trajektorię oderwaną od rzeczywistego ruchu,
- współistnieć z jammingiem lub innymi zakłóceniami RF.

**Cel**: Przewidywanie kolumny `label` (binarna: 0 = czysty lot, 1 = spoofing GPS).

---

## 2. Dataset

**Plik**: `honeywell_gold_dataset.csv` — realne logi PX4 zsynchronizowane z topiców `vehicle_attitude` + `vehicle_global_position` + `vehicle_gps_position` + `vehicle_local_position`.

| | |
|---|---|
| Wiersze | 24 992 |
| Kolumny | 84 (83 cechy + 1 etykieta: `label`) |
| Brakujące wartości | 0 |
| Duplikaty | 0 |
| Punkt referencyjny | `36.2048°N, 138.2529°E, 51.7 m` |

**Rozkład etykiety** `label`:

| label | wiersze | udział |
|---|---|---|
| 0 (czysty lot) | 11 870 | 47,5% |
| 1 (spoofing) | 13 122 | 52,5% |

### 2.1 Struktura segmentowa

Dataset składa się z **6 ciągłych segmentów** (naprzemiennie clean / attack):

| Segment | Label | Wiersze | Indeksy |
|---|---|---|---|
| 1 | 0 (clean) | 8 122 | 0..8 121 |
| 2 | 1 (attack) | 4 374 | 8 122..12 495 |
| 3 | 0 (clean) | 2 499 | 12 496..14 994 |
| 4 | 1 (attack) | 3 749 | 14 995..18 743 |
| 5 | 0 (clean) | 1 249 | 18 744..19 992 |
| 6 | 1 (attack) | 4 999 | 19 993..24 991 |

**UWAGA**: Dataset **nie zawiera kolumny identyfikującej typ scenariusza ataku**. Nie da się zaimplementować Leave-One-Attack-Scenario-Out (LOSO).

### 2.2 Pułapki danych

1. **35 z 84 kolumn jest stałych** — nie niosą sygnału.
2. `timestamp` to **indeks próbki**, nie czas. Użycie go jako cechy powoduje data leakage.
3. **Niezgodność jednostek**: `lat_y`, `lon_y` w stopniach × 1e7; `alt_y` w milimetrach.
4. **Jeden dron, jedno środowisko** — ogranicza transferowalność.
5. **Kolumny zdrowia odbiornika** (`jamming_indicator`, `noise_per_ms`, `satellites_used`) nie separują spoofingu.

---

## 3. Inżynieria cech

### 3.1 Czyszczenie

- Usunięcie 35 stałych kolumn + `timestamp` + kolumn zdrowia odbiornika
- Normalizacja GPS: `lat_y /= 1e7`, `lon_y /= 1e7`, `alt_y /= 1000`

### 3.2 Cechy rozjazdu EKF vs GPS

| cecha | definicja |
|---|---|
| `lat_diff_m`, `lon_diff_m`, `alt_diff_m` | rozjazd pozycyjny EKF vs GPS w metrach |
| `pos_diff_h_m`, `pos_diff_3d_m` | norma pozioma / 3D rozjazdu |
| `vn_diff`, `ve_diff`, `vd_diff` | reszty prędkości NED |
| `vel_diff_h`, `vel_diff_3d` | normy prędkości |
| `speed_diff` | różnica magnitudy prędkości |
| `cog_diff`, `abs_cog_diff` | rozjazd kierunku ruchu |

Po obliczeniu cech rozjazdu surowe kolumny pozycyjne są **usuwane**. Wynikowy zbiór: **48 cech**.

---

## 4. Porównane metody

| # | Metoda | Kategoria | Wymaga treningu? |
|---|---|---|---|
| 1 | **Test innowacji Kalmana (chi-squared)** | klasyczny statystyczny | NIE |
| 2 | **XGBoost** | gradient boosting | TAK |
| 3 | **LightGBM** | gradient boosting | TAK |

### 4.1 Test innowacji Kalmana

Klasyczna metoda oparta na fizyce EKF. Oblicza statystykę χ² porównującą estymaty EKF z surowym GPS, normalizowaną raportowaną niepewnością:

```
χ² = (Δn² + Δe²) / eph² + Δd² / epv²
```

Pod H0 (brak spoofingu) χ² ~ χ²(3). Spoofing łamie zgodność EKF/GPS, więc χ² rośnie. Domyślny próg detekcji: 95% kwantyl χ²₃ = 7.815.

### 4.2 Modele ML

- **XGBoost**: `n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.85`
- **LightGBM**: `n_estimators=600, learning_rate=0.04, num_leaves=63, min_child_samples=20`

Oba trenowane na 48 cechach (po inżynierii) ze StandardScaler.

---

## 5. Protokół ewaluacji

### Block-based 3-Fold CV

Łączymy sąsiednie segmenty clean+attack w 3 foldy:
- **Fold 0**: segmenty 1+2 (8 122 clean + 4 374 attack)
- **Fold 1**: segmenty 3+4 (2 499 clean + 3 749 attack)
- **Fold 2**: segmenty 5+6 (1 249 clean + 4 999 attack)

**Realistyczny test generalizacji**: model trenuje na dwóch sesjach, testowany na trzeciej sesji.

Kalibracja progu: wyznaczana z danych treningowych per fold (bez przecieku z testu).

---

## 6. Wyniki

### 6.1 Porównanie modeli (Block CV)

| Metoda | F1 @domyślny | F1 @kalibrowany | Precision | Recall | AUC |
|---|---|---|---|---|---|
| **Kalman chi²** | **0.876** | **0.840** | **0.862** | **0.859** | **0.900** |
| LightGBM | 0.557 | 0.557 | 0.515 | 0.626 | 0.801 |
| XGBoost | 0.498 | 0.466 | 0.516 | 0.477 | 0.905 |

### 6.2 Kalman chi² — wyniki per fold

| fold | test rows | F1 @7.81 | F1 @cal | precision | recall | AUC |
|---|---|---|---|---|---|---|
| block_0 | 12 496 | 0.883 | 0.709 | 0.587 | 0.893 | 0.942 |
| block_1 | 6 248 | 0.746 | 0.813 | 1.000 | 0.684 | 0.757 |
| block_2 | 6 248 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

### 6.3 XGBoost — wyniki per fold

| fold | test rows | F1 @0.5 | precision | recall | AUC |
|---|---|---|---|---|---|
| block_0 | 12 496 | 0.669 | 0.549 | 0.857 | 0.914 |
| block_1 | 6 248 | **0.000** | 0.000 | 0.000 | 0.800 |
| block_2 | 6 248 | 0.825 | 1.000 | 0.704 | 1.000 |

### 6.4 LightGBM — wyniki per fold

| fold | test rows | F1 @0.5 | precision | recall | AUC |
|---|---|---|---|---|---|
| block_0 | 12 496 | 0.671 | 0.544 | 0.877 | 0.863 |
| block_1 | 6 248 | **0.000** | 0.000 | 0.000 | 0.541 |
| block_2 | 6 248 | 1.000 | 1.000 | 1.000 | 1.000 |

---

## 7. Interpretacja

### 7.1 Kalman wygrywa zdecydowanie

Test innowacji Kalmana (F1 = 0.876) **pokonuje oba modele ML** (F1 ≤ 0.557). Kluczowa przewaga: Kalman wykrywa ataki we **wszystkich foldach**, w tym w block_1 gdzie oba modele ML dają F1 = 0.

### 7.2 Dlaczego ML zawodzi na block_1?

Block_1 to segment ataku (indeksy 14 995..18 743) o innej sygnaturze niż segmenty treningowe. Modele ML nauczone na segmentach 2 i 6 nie rozpoznają wzorca z segmentu 4. AUC > 0.5 sugeruje, że sygnał istnieje w predykcjach, ale rozkład prawdopodobieństw dryfuje — próg 0.5 nie trafia.

### 7.3 Dlaczego Kalman jest lepszy?

Kalman opiera się na **fizyce**, nie na wzorcach statystycznych:
- Pod H0 (brak spoofingu) estymaty EKF i surowy GPS **muszą** się zgadzać w granicach niepewności.
- Spoofing łamie tę zgodność **niezależnie od typu ataku**.
- Nie ma co "przefitwować" — brak parametrów uczonych z danych.

### 7.4 Trade-offs

| Zagadnienie | Implikacja |
|---|---|
| **Kalman vs ML** | Kalman jest lepszy, prostszy i bardziej transferowalny. ML modele nie generalizują na niewidziane typy ataków. |
| **Generalizacja** | Block_1 ujawnia fundamentalny problem modeli ML: nowy typ ataku = porażka. |
| **Brak treningu** | Kalman nie wymaga danych z atakami — daje deployment-ready detektor od razu. |
| **Jeden dron** | Wyniki dla jednego drona. Kalman bardziej transferowalny (fizyka EKF jest ta sama). |

---

## 8. Kanoniczny kod

**Plik źródłowy**: `solution.py`

```
python solution.py
```

**Pipeline**:

```
honeywell_gold_dataset.csv
        │
        ▼
load_and_clean()  →  usunięcie timestamp, 35 stałych, normalizacja GPS
        │
        ├──→ kalman_innovation_chi2()  →  χ² score per próbka
        │                                  ↓
        │                           evaluate_kalman_block()  →  próg + metryki
        │
        ▼
add_divergence_features()  →  +14 cech rozjazdu, usunięcie surowych pozycji
        │
        ▼
evaluate_block_cv()  →  XGBoost / LightGBM
        │
        ▼
metryki + confusion matrix per fold + porównanie
```

---

## 9. Reprodukowalność

| plik | rola |
|---|---|
| `solution.py` | **kanoniczny pipeline** — 3 metody, bez leakage |
| `dataset_description.md` | słownik danych, pułapki |
| `REPORT.md` | ten raport |
| `requirements.txt` | zależności pip |

**UWAGA**: Pliki `.ipynb` zawierają **stary, wadliwy kod** z data leakage i fałszywymi komentarzami. Kanonicznym rozwiązaniem jest `solution.py`.

```
pip install -r requirements.txt
python solution.py
```
