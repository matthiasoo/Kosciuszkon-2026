# Detekcja ataków GPS Spoofing przy użyciu uczenia maszynowego

**Kościuszkon 2026 · Honeywell Theme #2**

---

## 1. Zrozumienie problemu

**Atak GPS Spoofing** to klasa ataków cyber-fizycznych, w której atakujący nadaje sfałszowany sygnał GNSS, sprawiając że odbiornik ofiary obliczy błędną pozycję, prędkość lub czas. Dla autonomicznego drona atak może:

- przesunąć raportowaną pozycję poziomą poza prawdziwą trajektorię,
- zafałszować raportowaną wysokość zachowując wiarygodną pozycję poziomą,
- wstrzyknąć syntetyczną trajektorię oderwaną od rzeczywistego ruchu,
- współistnieć z jammingiem lub innymi zakłóceniami RF.

Atak jest niebezpieczny, bo autopilot ufa GPS dla wyznaczania pozycji absolutnej. Jeśli EKF zaakceptuje sfałszowane pomiary, dron leci po fałszywej trajektorii *myśląc*, że jest na kursie.

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

**UWAGA**: Dataset **nie zawiera kolumny identyfikującej typ scenariusza ataku** (np. "horizontal_drift", "altitude_spoof"). Nie da się więc zaimplementować Leave-One-Attack-Scenario-Out (LOSO). Nazwy scenariuszy, jeśli występują w literaturze, nie są odwzorowane w danych.

### 2.2 Pułapki danych

1. **35 z 84 kolumn jest stałych** — liczniki resetów EKF, flagi walidacji, punkt referencyjny. Nie niosą żadnego sygnału.
2. `timestamp` to **indeks próbki**, nie czas rzeczywisty. `time_utc_usec` jest stały (zero). Użycie `timestamp` jako cechy powoduje data leakage (identyfikuje segment → label).
3. **Niezgodność jednostek**: `lat_y`, `lon_y` są w stopniach × 1e7; `alt_y` w milimetrach. Trzeba przeskalować przed porównaniem z kolumnami EKF.
4. **Jeden dron, jedno środowisko** — ogranicza transferowalność.
5. **Kolumny zdrowia odbiornika** (`jamming_indicator`, `noise_per_ms`, `satellites_used`) nie separują spoofingu od czystego lotu.

---

## 3. Inżynieria cech

Trzy warstwy przetwarzania:

### 3.1 Czyszczenie

- Usunięcie 35 stałych kolumn
- Usunięcie `timestamp` (zapobieganie leakage)
- Usunięcie kolumn zdrowia odbiornika (brak sygnału)
- Normalizacja GPS: `lat_y /= 1e7`, `lon_y /= 1e7`, `alt_y /= 1000`

### 3.2 Cechy rozjazdu EKF vs GPS

Główny sygnał detekcji: rozjazd między estymatą EKF (kolumny `*_x`) a surowym GPS (`*_y`). Pod hipotezą zerową (brak spoofingu) rozjazd powinien być bliski zeru.

| cecha | definicja | dlaczego użyteczna |
|---|---|---|
| `lat_diff_m` | `(lat_x - lat_y) × 111 000` | rozjazd latitude w metrach |
| `lon_diff_m` | `(lon_x - lon_y) × 111 000 × cos(lat)` | rozjazd longitude w metrach |
| `alt_diff_m` | `alt_x - alt_y` | rozjazd wysokości w metrach |
| `pos_diff_h_m` | norma pozioma | magnituda rozjazdu w płaszczyźnie |
| `pos_diff_3d_m` | norma 3D | magnituda rozjazdu w przestrzeni |
| `vn_diff`, `ve_diff`, `vd_diff` | reszty prędkości NED | niefizyczne zmiany prędkości GPS |
| `vel_diff_h`, `vel_diff_3d` | normy prędkości | magnituda rozjazdu prędkości |
| `speed_diff` | `‖(vx,vy,vz)‖ - vel_m_s` | spójność magnitudy prędkości |
| `cog_diff` | różnica kątowa kursu | rozjazd kierunku ruchu |

**Po obliczeniu cech rozjazdu, surowe kolumny pozycyjne (`lat_x`, `lon_x`, `alt_x`, `lat_y`, `lon_y`, `alt_y`, `x`, `y`, `z`) są usuwane** — ich bezpośrednie użycie w modelu powodowałoby leakage.

### 3.3 Wynikowy zbiór cech

Po przetworzeniu: **48 cech** (34 oczyszczone surowe + 14 rozjazdu).

---

## 4. Protokoły ewaluacji

Stosujemy **dwa protokoły** ewaluacji, aby uczciwie raportować zarówno górny limit jak i realistyczną generalizację:

### 4.1 Protokół A — Stratified 4-Fold CV (losowy podział)

Losowy podział stratyfikowany. Daje **zawyżone wyniki** (F1 ≈ 1.0) ponieważ sąsiednie próbki w tym samym segmencie są prawie identyczne — losowy podział rozrzuca je między train i test, co jest de facto data leakage z sąsiedztwa próbek. Raportujemy jako **górny limit**, nie realistyczny wynik.

### 4.2 Protokół B — Block-based 3-Fold CV (podział po segmentach)

Łączymy sąsiednie segmenty clean+attack w 3 foldy:
- **Fold 0**: segmenty 1+2 (8 122 clean + 4 374 attack)
- **Fold 1**: segmenty 3+4 (2 499 clean + 3 749 attack)
- **Fold 2**: segmenty 5+6 (1 249 clean + 4 999 attack)

To jest **realistyczny test generalizacji**: model trenuje na danych z dwóch sesji i jest testowany na trzeciej sesji, której nigdy nie widział. Nie ma leakage z sąsiedztwa próbek.

---

## 5. Model

**XGBoost** (gradient boosted trees):

```
n_estimators=400, max_depth=6, learning_rate=0.05,
subsample=0.85, colsample_bytree=0.85
```

Standardowe skalowanie cech (`StandardScaler`) przed treningiem.

---

## 6. Wyniki

### 6.1 Protokół A — Stratified CV (górny limit, zawyżony)

| fold | F1 | precision | recall | AUC |
|---|---|---|---|---|
| fold_0 | 1.000 | 1.000 | 1.000 | 1.000 |
| fold_1 | 0.999 | 0.999 | 0.999 | 1.000 |
| fold_2 | 1.000 | 1.000 | 0.999 | 1.000 |
| fold_3 | 1.000 | 1.000 | 0.999 | 1.000 |
| **Średnia** | **1.000** | **1.000** | **0.999** | **1.000** |

**UWAGA**: Te wyniki są zawyżone (data leakage z sąsiedztwa próbek). Nie są miarodajne.

### 6.2 Protokół B — Block CV (realistyczny)

| fold | test rows | pos_rate | F1 | precision | recall | AUC |
|---|---|---|---|---|---|---|
| block_0 | 12 496 | 0.35 | 0.669 | 0.549 | 0.857 | 0.914 |
| block_1 | 6 248 | 0.60 | 0.000 | 0.000 | 0.000 | 0.800 |
| block_2 | 6 248 | 0.80 | 0.825 | 1.000 | 0.704 | 1.000 |
| **Średnia** | | | **0.498** | **0.516** | **0.521** | **0.905** |

### 6.3 Interpretacja wyników

**Block fold 0** (F1 = 0.67): Model wykrywa atak z recall = 0.86 ale precision = 0.55, co daje dużo fałszywych alarmów. AUC = 0.91 sugeruje że ranking jest dobry.

**Block fold 1** (F1 = 0.00): Kompletna porażka — model **nie potrafi** wykryć żadnego ataku w segmencie 4. AUC = 0.80 pokazuje że ranking jest ponadlosowy (model "widzi" sygnał w predykcjach), ale próg 0.5 nie trafia w rozkład tego foldu.

**Block fold 2** (F1 = 0.83): Dobra detekcja z precision = 1.00 (zero fałszywych alarmów), ale recall = 0.70 (30% ataków pominięte).

**Kluczowy wniosek**: Model generalizuje się **bardzo nierówno** między segmentami. AUC jest stabilnie > 0.80, co sugeruje że cechy niosą sygnał, ale rozkład prawdopodobieństw dryfuje drastycznie — stąd ogromna wrażliwość na próg.

---

## 7. Analiza

### 7.1 Dlaczego stratified CV daje F1 = 1.0?

Dane telemetryczne w obrębie jednego segmentu są **silnie autokorelowane** — sąsiednie próbki różnią się minimalnie. Losowy podział rozrzuca te prawie identyczne próbki między train i test. Model po prostu "zapamiętuje" wartości i odnajduje niemal identyczne próbki w teście. To nie jest generalizacja.

### 7.2 Dlaczego block fold 1 to F1 = 0?

Segment 4 (attack, indeksy 14 995..18 743) ma **inną sygnaturę** ataku niż segmenty 2 i 6. Model wytrenowany na segmentach 2 i 6 (z segmentami clean 1, 5 i 6) nie potrafi rozpoznać wzorca ataku z segmentu 4.

AUC = 0.80 sugeruje, że model widzi pewien sygnał (ranking jest lepszy niż losowy), ale próg 0.5 nie trafia — wszystkie predykcje dla segment 4 attack są poniżej 0.5.

### 7.3 Trade-offs

| zagadnienie | implikacja |
|---|---|
| **Generalizacja** | Model nie generalizuje się równomiernie na wszystkie typy ataków. Nowy, niewidziany typ ataku może nie być wykryty. |
| **AUC vs F1** | AUC jest stabilnie > 0.80 — sygnał istnieje, ale potrzebna jest lepsza strategia ustawiania progu. |
| **Jeden dron** | Wyniki dotyczą jednego drona w jednym środowisku. Transferowalność jest ograniczona. |
| **Brak identyfikacji scenariuszy** | Bez kolumny scenariusza nie możemy przeprowadzić prawdziwego LOSO. |

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
load_and_clean()  →  usunięcie timestamp, 35 stałych kolumn,
                      normalizacja GPS, usunięcie kolumn zdrowia odbiornika
        │
        ▼
add_divergence_features()  →  +14 cech rozjazdu EKF vs GPS,
                               usunięcie surowych kolumn pozycyjnych
        │
        ▼
evaluate_stratified_cv()  →  losowy 4-fold (górny limit)
evaluate_block_cv()       →  podział po segmentach (realistyczny)
        │
        ▼
XGBoost  →  StandardScaler  →  predict_proba
        │
        ▼
metryki + confusion matrix per fold
```

---

## 9. Reprodukowalność

| plik | rola |
|---|---|
| `solution.py` | **kanoniczny pipeline** — czysty, zweryfikowany, bez leakage |
| `dataset_description.md` | słownik danych, pułapki, sugerowane cechy |
| `REPORT.md` | ten raport |
| `requirements.txt` | zależności pip |

**UWAGA**: Pliki `.ipynb` (notebooki) w tym repo zawierają **stary, wadliwy kod** z data leakage, nieprawidłowymi komentarzami o LOSO, i niewywoływanymi funkcjami. Nie są miarodajne. Kanonicznym rozwiązaniem jest `solution.py`.

Kod jest deterministyczny (`random_state=42` wszędzie). Re-egzekucja na świeżym środowisku odtwarza wyniki.

```
pip install -r requirements.txt
python solution.py
```
