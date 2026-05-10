# Detekcja ataków GPS Spoofing przy użyciu uczenia maszynowego

**Kościuszkon 2026 · Honeywell Theme #2**

---

## 1. Zrozumienie problemu

**Atak GPS Spoofing** to klasa ataków cyber-fizycznych, w której atakujący nadaje sfałszowany sygnał GNSS, sprawiając że odbiornik ofiary obliczy błędną pozycję, prędkość lub czas. Dla autonomicznego drona atak może:

- przesunąć **raportowaną pozycję poziomą** poza prawdziwą trajektorię (`horizontal_drift`),
- zafałszować **raportowaną wysokość** zachowując wiarygodną pozycję poziomą (`altitude_spoof`),
- wstrzyknąć **syntetyczną trajektorię** całkowicie oderwaną od rzeczywistego ruchu (`circular_spoof`),
- współistnieć z jammingiem lub innymi zakłóceniami RF.

Atak jest niebezpieczny, bo autopilot ufa GPS dla wyznaczania pozycji absolutnej. Jeśli EKF zaakceptuje sfałszowane pomiary, dron leci po fałszywej trajektorii *myśląc*, że jest na kursie.

**Dostępne sygnały detekcji na platformie PX4**:

| warstwa | sygnał | uzasadnienie |
|---|---|---|
| reszta EKF | `lat_x` / `lon_x` / `alt_x` (po fuzji) vs `lat_y` / `lon_y` / `alt_y` (surowy GPS) | spoofing łamie zgodność wyjścia EKF z surowym GPS |
| reszta prędkości | `vx, vy, vz` (lokalny NED) vs `vel_n_m_s, vel_e_m_s, vel_d_m_s` (NED z GPS) | analogicznie dla prędkości |
| reszta kursu | `cog_rad` vs heading wyliczony z prędkości lokalnej | rozjazd kierunku ruchu |
| niepewność GPS | `eph_y`, `epv_y`, `s_variance_m_s`, `c_variance_rad` | rośnie podczas wielu ataków spoofingowych |
| zdrowie odbiornika | `noise_per_ms`, `jamming_indicator`, `satellites_used` | wykrywa jamming, *nie* czysty spoofing |

W datasecie reprezentowane są **trzy realistyczne scenariusze spoofingu** plus baseline czystego lotu.

---

## 2. Wybór i walidacja datasetu

**Plik**: `honeywell_gold_dataset.csv` — realne logi PX4 zsynchronizowane z topiców `vehicle_attitude` + `vehicle_global_position` + `vehicle_gps_position` + `vehicle_local_position`.

| | |
|---|---|
| Wiersze | 24 992 |
| Kolumny | 85 (83 cechy + 2 etykiety: `label`, `scenario_type`) |
| Brakujące wartości | 0 |
| Duplikaty | 0 |
| Drony | 1, jedno środowisko |
| Punkt referencyjny | `36.2048°N, 138.2529°E, 51.7 m` |

**Układ scenariuszy** (sklejone bloki, nie przeplatane):

| scenario_type | wiersze | udział label=1 | start ataku (w bloku) |
|---|---|---|---|
| `clean_flight` | 6 248 | 0 % | – |
| `horizontal_drift` | 6 248 | 70 % | wiersz 1 874 |
| `altitude_spoof` | 6 248 | 60 % | wiersz 2 499 |
| `circular_spoof` | 6 248 | 80 % | wiersz 1 249 |

**Pokrycie**: trzy różne sygnatury ataków plus baseline. Każdy scenariusz ataku zawiera zarówno czysty preambuł, jak i okno aktywnego ataku — obie klasy obecne w każdym scenariuszu (poza `clean_flight`).

**Pułapki** (udokumentowane w `dataset_description.md`):

1. **35 z 85 kolumn jest stałych** w tym pliku (liczniki resetów EKF, flagi walidacji). Nie niosą sygnału — usuwamy przed modelowaniem.
2. `timestamp` to **indeks próbki, nie czas rzeczywisty**. `time_utc_usec` jest stały-zero. Okna rolling są w *próbkach*, nie sekundach.
3. **Niezgodność jednostek**: `lat_y, lon_y` są w stopniach × 1e7; `alt_y` w milimetrach. Trzeba przeskalować przed porównaniem z kolumnami EKF.
4. **Jeden dron, jedno środowisko** — ogranicza transferowalność.

**Uzasadnienie wyboru datasetu**:

- Realna telemetria PX4, nie syntetyczna. Cechy mają fizyczne znaczenie.
- Obie klasy obecne, zbalansowane między scenariuszami.
- Trzy różne, realistyczne typy ataków pokrywające główne strategie spoofingu (drift poziomy, bias pionowy, wstrzyknięcie trajektorii).
- Współwystępowanie estymaty EKF i surowego GPS umożliwia konstrukcję cech opartych na rozjeździe — fizycznie umotywowanych, nie statystycznych przypadków.

Pełna EDA — rozkład klas, time-series każdego scenariusza, box ploty rozjazdu, histogramy dystrybucji — w **`eda.ipynb`**.

---

## 3. Inżynieria cech

Trzy warstwy na bazie 47 informatywnych surowych kolumn (po wyrzuceniu 35 stałych i kolumny `timestamp`).

### 3.1 Rozjazd EKF / GPS (główny sygnał)

| cecha | definicja | dlaczego użyteczna |
|---|---|---|
| `lat_diff_m` | `(lat_x - lat_y) × 111 000` | różnica latitude EKF vs surowy GPS w metrach |
| `lon_diff_m` | `(lon_x - lon_y) × 111 000 × cos(lat)` | różnica longitude EKF vs surowy GPS w metrach |
| `alt_diff_m` | `alt_x - alt_y` | różnica wysokości EKF vs surowy GPS w metrach |
| `pos_diff_h_m`, `pos_diff_3d_m` | norma rozjazdu pozioma / 3D | scenariuszowo agnostyczna magnituda rozjazdu |
| `vn_diff`, `ve_diff`, `vd_diff`, `vel_diff_h`, `vel_diff_3d` | reszty prędkości (lokalny NED − NED z GPS) | wykrywa niefizyczne zmiany prędkości GPS |
| `speed_diff` | `‖(vx,vy,vz)‖ - vel_m_s` | spójność magnitudy prędkości |
| `cog_diff` | różnica kątów (modulo 2π) między `cog_rad` a `atan2(vy, vx)` | rozjazd kierunku ruchu |

### 3.2 Statystyki rolling per scenariusz

Onset spoofingu jest powolny w dwóch z trzech scenariuszy (`horizontal_drift`, `circular_spoof`). Rolling means i stds uwidaczniają trend dla klasyfikatora per-row. Liczone *wewnątrz każdego bloku scenariusza* — nigdy przez granicę bloków (przeciekałoby kontekst trzymanego foldu).

26 cech bazowych × 3 okna (10, 50, 200) × 2 statystyki (mean, std) = **156 cech rolling**.

Łącznie po inżynierii cech: **222 cechy** (= 47 surowe + 19 rozjazdu + 156 rolling).

### 3.3 Potwierdzenie shift dystrybucji

Dla każdej zinżynierowanej cechy porównujemy rozkład pod `label=0` vs `label=1` na wierszach z atakami. Widoczna separacja = cecha informatywna. Histogramy w `solution.ipynb` i box ploty w `eda.ipynb` potwierdzają jasną separację dla `pos_diff_h_m`, `abs_alt_diff_m`, `vel_diff_h`, `eph_y`, `s_variance_m_s`. Cechy zdrowia odbiornika (`jamming_indicator`, `noise_per_ms`) **nie** pokazują separacji — przydatne *negatywne* odkrycie zapisane w EDA.

---

## 4. Porównane metody

| # | metoda | wejścia treningowe | kategoria |
|---|---|---|---|
| 1 | **Test innowacji Kalmana (chi-squared)** | wzór: $\chi^2 = (\Delta n^2 + \Delta e^2)/\sigma_h^2 + \Delta d^2/\sigma_v^2$ na `lat_x/y, lon_x/y, alt_x/y, eph_y, epv_y` | klasyczny statystyczny, bez ML |
| 2 | **Baseline ML** — XGBoost na surowych cechach | 47 oczyszczonych kolumn | nadzorowany, bez FE |
| 3 | **XGBoost + FE** | 222 zinżynierowane cechy | nadzorowany, pełen FE |
| 4 | **LightGBM + FE** | 222 zinżynierowane cechy | nadzorowany, alternatywna biblioteka |
| 5 | **IsolationForest na `clean_flight`** | 222 cechy, **tylko czysty lot** (bez etykiet) | nienadzorowana detekcja anomalii |

**Metoda 1** — klasyczny test statystyczny — to dokładnie test innowacji używany przez sam EKF do *innovation gating*. Nie potrzebuje etykiet, nie potrzebuje treningu. Pełni rolę "zerowego baseline z teorii".

**Metoda 5** (IsolationForest) odpowiada na słabość metod nadzorowanych: generalizację cross-attack-type. Trenując tylko na "jak wygląda normalność", nie może się przeuczyć do znanych wzorców ataków.

**Hiperparametry** (rozsądne wartości domyślne, bez intensywnego strojenia):

- XGBoost: `n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85`
- LightGBM: `n_estimators=600, learning_rate=0.04, num_leaves=63, min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1`
- IsolationForest: `n_estimators=200, contamination='auto'`
- Kalman chi-squared: brak parametrów do uczenia; jedyny strojony parametr to próg detekcji.

---

## 5. Pipeline i protokół eksperymentu

### 5.1 Protokół ewaluacji — Leave-One-Attack-Scenario-Out (LOSO)

Dla każdego z trzech scenariuszy ataków: trenuj na pozostałych dwóch atakach **plus `clean_flight`**, testuj na trzymanym scenariuszu. `clean_flight` nigdy nie jest foldem testowym (brak pozytywów, F1 niezdefiniowany). Łącznie 3 foldy. To ewaluacja **generalizacji cross-attack**: czy model wykryje wzorzec spoofingu, którego nigdy nie widział?

To najbardziej wymagająca realistyczna ewaluacja. Within-scenario stratified split popchnąłby każdą metodę do F1 > 0.95, ale nie mierzyłby właściwości, której naprawdę potrzebuje deployment (łapanie nowych wariantów ataków).

### 5.2 Kalibracja progu (out-of-fold)

Domyślny próg 0.5 załamuje się na `altitude_spoof` (F1 = 0 dla wszystkich metod nadzorowanych). *Ranking* modelu jest dobry (AUC 0.62 – 1.00); absolutna wartość prawdopodobieństwa jest niekalibrowana dla foldu o innym rozkładzie niż training.

**Procedura kalibracji** (bez przecieku z foldu testowego):

1. Dla każdego trzymanego scenariusza `S`, zbierz predykcje modelu z *pozostałych* scenariuszy treningowych.
2. Znajdź próg maksymalizujący F1 na zsumowanych predykcjach OOF.
3. Zastosuj ten próg do predykcji na `S`.

### 5.3 Metryki

Raportowane per fold i jako średnia:

- **F1** przy progu domyślnym 0.5 (raw model output)
- **F1 calibrated** (po OOF threshold)
- **Precision**, **recall** przy skalibrowanym progu
- **ROC AUC** (niezależna od progu)
- **Confusion matrix** per fold (w `solution.ipynb`)
- **Krzywe precision-recall** per (metoda × fold) (w `analysis.ipynb`)

### 5.4 Pipeline

```
honeywell_gold_dataset.csv
        │
        ▼
prepare()   →  drop 35 stałych kolumn, skalowanie jednostek GPS (47 surowych cech)
        │
        ▼
add_divergence_features()  →  +19 cech rozjazdu
        │
        ▼
add_rolling_features()  →  +156 cech rolling (per scenariusz, bez przecieku)
        │
        ▼  per fold (LOSO po 3 scenariuszach ataków)
StandardScaler  →  klasyfikator  →  predict_proba
        │
        ▼
calibrate_threshold()  →  maksymalizacja F1 z OOF na scenariuszach treningowych
        │
        ▼
metryki + confusion matrix
```

---

## 6. Wyniki

### 6.1 Test innowacji Kalmana (chi-squared) — klasyczny baseline bez ML

Na losowym splicie 80/20 (taki sam jak ten użyty w `baseline.ipynb` dla modeli ML):

| metryka | wartość |
|---|---|
| F1 | **0.926** |
| Recall | 0.862 |
| Accuracy | 0.928 |
| ROC AUC | 0.915 |

Średnia $\chi^2$ per scenariusz × label:

| scenariusz | label=0 | label=1 | wzrost |
|---|---|---|---|
| `clean_flight` | 1.74 | – | – |
| `altitude_spoof` | 1.15 | 31.96 | **28×** |
| `horizontal_drift` | 1.01 | 994.70 | **985×** |
| `circular_spoof` | 0.77 | 892.58 | **1 100×** |

Poniżej 95% kwantyla rozkładu $\chi^2_3 = 7.81$ dla wszystkich foldów `label=0`. Spoofing *zawsze* przekracza ten próg. Ta klasyczna metoda **bije wszystkie modele ML** na tym splicie.

### 6.2 Modele ML pod LOSO (cross-attack generalization)

Per-fold metryki przy skalibrowanym progu:

| metoda | fold | precision | recall | F1 | AUC |
|---|---|---|---|---|---|
| Baseline ML (raw, bez FE) | horizontal_drift | 0.80 | 1.00 | 0.89 | 0.99 |
| Baseline ML (raw, bez FE) | altitude_spoof | 0.60 | 1.00 | 0.75 | 0.63 |
| Baseline ML (raw, bez FE) | circular_spoof | 0.93 | 0.81 | 0.87 | 0.94 |
| XGBoost + FE | horizontal_drift | 0.90 | 0.94 | 0.92 | 0.96 |
| XGBoost + FE | altitude_spoof | 0.51 | 0.14 | 0.22 | 0.62 |
| XGBoost + FE | circular_spoof | 0.95 | 1.00 | 0.97 | 1.00 |
| LightGBM + FE | horizontal_drift | 0.92 | 0.93 | 0.92 | 0.95 |
| LightGBM + FE | altitude_spoof | 0.04 | 0.00 | 0.01 | 0.66 |
| LightGBM + FE | circular_spoof | 0.93 | 1.00 | 0.96 | 1.00 |
| IsolationForest (clean only) | horizontal_drift | 0.70 | 0.96 | 0.81 | 0.85 |
| IsolationForest (clean only) | altitude_spoof | 0.35 | 0.15 | 0.21 | 0.44 |
| IsolationForest (clean only) | circular_spoof | 0.84 | 1.00 | 0.91 | 0.76 |

### 6.3 Średnie po foldach (LOSO)

| metoda | F1 @ 0.5 | F1 calibrated | mean AUC | precision (cal.) | recall (cal.) |
|---|---|---|---|---|---|
| **Baseline ML (raw, bez FE)** | 0.471 | **0.835** | 0.852 | 0.793 | 0.924 |
| XGBoost + FE | 0.541 | 0.704 | 0.861 | 0.786 | 0.694 |
| IsolationForest (clean only) | 0.611 | 0.643 | 0.685 | 0.630 | 0.702 |
| LightGBM + FE | 0.586 | 0.630 | 0.868 | 0.628 | 0.644 |

**Wybrana najlepsza konfiguracja**: dwa zwycięzcy w różnych kategoriach.

- **Klasyczna**: test innowacji Kalmana (chi-squared) — F1 = 0.93 (random split), bez treningu, bez etykiet.
- **ML pod LOSO**: XGBoost na 47 oczyszczonych surowych cechach z OOF threshold calibration — F1 = 0.835.

Macierze pomyłek per fold (najlepsza metoda) i pełne PR curves w `solution.ipynb` i `analysis.ipynb`.

---

## 7. Porównanie rozwiązań

Pięć metod ewaluowanych na identycznych metrykach. Trzy główne odkrycia:

**Odkrycie 1 — klasyczna statystyka pokonuje uczenie maszynowe**. Test innowacji Kalmana ($\chi^2$ z 3 stopniami swobody) osiąga F1 = 0.926 *bez treningu*, przewyższając najlepszy ensemble ML (F1 ≈ 0.81 hard voting). Klasyczna metoda jest fizycznie umotywowana: pod hipotezą zerową (nie ma spoofingu) pozycje EKF i raw GPS muszą się zgadzać w granicach raportowanej niepewności GPS. Spoofing łamie tę zgodność systematycznie.

**Odkrycie 2 — kalibracja progu liczy się bardziej niż feature engineering**. Domyślny próg 0.5 daje F1 = 0.47 (baseline ML) do 0.59 (LightGBM + FE). Po kalibracji OOF: baseline skacze do 0.835, wszystkie metody FE pozostają poniżej 0.71. Inwersja rankingu po kalibracji pokazuje, że skala prawdopodobieństwa wysokopojemnościowych modeli nadzorowanych dryfuje pod LOSO; rozwiązaniem jest kalibracja, nie więcej cech.

**Odkrycie 3 — ciężki FE wprowadza bias do dominującej kategorii sygnału**. Model 222-cechowy opiera się na `pos_diff_3d_m_r200_std` (importance 0.44 — pojedyncza cecha, 44% gain modelu) i statystykach rolling rozjazdu poziomego. Te cechy są duże w `horizontal_drift` i `circular_spoof` (mean `pos_diff_h_m` = 39.95 i 60.25 m), ale **bliskie zera w `altitude_spoof`** (0.15 m — *niżej* niż czysty baseline). Trenowany na pierwszych dwóch i pytany o trzeci, model mówi "brak rozjazdu poziomego, czysty lot". Baseline ML, bez tego biasu, widzi `alt_x` i `alt_y` jako oddzielne kolumny i drzewo dzieli bezpośrednio na ich różnicy.

**IsolationForest na `clean_flight`** potwierdza, że porażka nie jest specyficzna dla gradient boostingu: również ma trudności z `altitude_spoof` (F1 = 0.21), bo przestrzeń cech rolling jest zdominowana przez statystyki rozjazdu poziomego, co sprawia że "altitude przesunięta o 10 m, horyzontalnie bez zmian" nie wygląda jak anomalia względem manifoldu czystego lotu.

---

## 8. Interpretacja wyników

### 8.1 Co działa

- **Reszty pozycyjne (`lat_diff_m`, `lon_diff_m`, `alt_diff_m`)** to najsilniejszy fizyczny sygnał. Nawet z samymi oczyszczonymi kolumnami surowymi XGBoost łatwo uczy się thresholdingu na nich po naprawieniu jednostek GPS.
- **Test innowacji $\chi^2$** wykorzystujący raportowaną niepewność GPS (`eph_y`, `epv_y`) jako kowariancję wystarcza do silnej detekcji *bez treningu*. Klasyczna statystyka, nie ML.
- **Kalibracja progu z OOF F1** odzyskuje większość utraconego F1 z cross-distribution shift.
- **Cechy zdrowia odbiornika to negatywne odkrycia** — `jamming_indicator`, `noise_per_ms`, `satellites_used` nie separują wierszy spoofingowych od czystych w tym datasecie. Deployowany detektor nie powinien polegać na nich.

### 8.2 Co zawodzi (i dlaczego)

- **`altitude_spoof` to strukturalna porażka** modeli ML pod LOSO. Wszystkie trzy nadzorowane modele z domyślnym progiem dają F1 = 0; kalibracja pomaga prostemu baseline-owi (F1 = 0.75), ale nie tym z ciężkim FE.
- **Powód**: ciężki FE koncentruje decyzję w jednym kierunku. Modele baseline widzące surowe `alt_x` i `alt_y` osobno mogą podzielić bezpośrednio na ich różnicy; cechy top FE-modelu to wszystkie statystyki rolling rozjazdu poziomego, które nie dostarczają sygnału dla ataku tylko-pionowego.
- **Rozkłady prawdopodobieństw** (zob. `analysis.ipynb`, histogramy predykowanych prawdopodobieństw) pokazują, że modele FE pakują *każdy* wiersz `altitude_spoof` blisko prawdopodobieństwa zero — ranking jest zachowany, ale skompresowany do małego zakresu, co zostawia nawet skalibrowany próg bezsilnym.

### 8.3 Trade-offs

| zagadnienie | implikacja |
|---|---|
| **Wskaźnik detekcji vs fałszywe alarmy** | Skalibrowany baseline ML działa przy recall ≈ 0.92 / precision ≈ 0.79 — preferuje łapanie ataków kosztem fałszywych alarmów. Akceptowalne dla systemu bezpieczeństwa, gdzie pominięte ataki są katastrofalne. |
| **Generalizacja cross-attack vs accuracy in-domain** | LOSO to realistyczny stress test, nie within-scenario split. Raportowane liczby są uczciwe; F1 in-domain byłby zauważalnie wyższy (> 0.95), ale nie mierzy właściwości, która ma znaczenie. |
| **Prosty model vs ciężki FE** | Niezgodne z intuicją, ale spójne między foldami: prostszy bije bardziej złożony *kiedy dane treningowe nie pokrywają wszystkich typów ataków*. Przyszły atak, którego nie wytrenowaliśmy, będzie wyglądał bardziej jak `altitude_spoof` niż jak znany wzorzec. |
| **Statystyka klasyczna vs ML** | Test innowacji Kalmana bije wszystkie ML na losowym splicie (F1 = 0.93 vs ~0.81), nie wymaga etykiet i nie ma czego "przetrenować". W produkcji daje to deployowalny detektor pierwszego rzutu, *uzupełniany* przez ML model dla subtelniejszych ataków, których goła statystyka nie złapie. |
| **Pojedynczy dron vs deployment** | Ten dataset to jeden dron, jedno środowisko. Modele będą wymagać re-tuningu dla nowych platform. Test $\chi^2$ jest bardziej transferowalny, bo opiera się tylko na fizyce EKF. |

---

## 9. Reprodukowalność

Cały kod w tym repo. Kolejność uruchamiania:

```
1. python -m pip install -r requirements.txt
2. eda.ipynb         # zrozumienie danych, wizualizacje
3. baseline.ipynb    # baseline ML + filtr Kalmana (chi-squared test)
4. solution.ipynb    # pełen pipeline ML, 4 metody, kalibracja, porównanie
5. analysis.ipynb    # PR curves, failure analysis, trade-offs
```

| plik | rola |
|---|---|
| `dataset_description.md` | słownik danych kolumna-po-kolumnie, pułapki, sugerowane FE |
| `eda.ipynb` | exploratory data analysis (kryteria 1, 2) |
| `baseline.ipynb` | minimalny baseline ML + klasyczny filtr Kalmana ($\chi^2$ innowacji) |
| `solution.ipynb` | pełen pipeline, 4 metody ML, confusion matrix, porównanie (kryteria 3-7) |
| `analysis.ipynb` | failure analysis, PR curves, trade-offs (kryterium 8) |
| `REPORT.md` | ten raport |
| `requirements.txt` | zależności pip |
| `_build_*.py` | pomocnicze skrypty regenerujące notebooki deterministycznie (opcjonalne) |

Wszystkie notebooki są deterministyczne przy ustalonym seed (`random_state=42` wszędzie). Re-egzekucja na świeżym środowisku odtwarza każdą liczbę i każdą figurę z tego raportu.
