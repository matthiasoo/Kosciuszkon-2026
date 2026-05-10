# Analiza słabych stron i trade-offów modeli

## 1. Filtr Kalmana (Test innowacji χ²) — model główny

### Zalety
- **Najwyższy F1 = 0.876** w realistycznym block-based CV
- **Nie wymaga treningu** — zero parametrów uczonych z danych
- **Fizycznie umotywowany** — opiera się na fundamentalnej właściwości EKF
- **Transferowalny** — działa na dowolnym dronie z EKF (PX4, ArduPilot) bez rekalibracji
- **Odporny na nowe typy ataków** — wykrywa block_1 (F1=0.746), gdzie ML daje F1=0
- **Zero ryzyka overfittingu** — brak fazy treningowej
- **Interpretowalny** — wartość χ² ma jasną statystyczną interpretację

### Słabe strony

| # | Słabość | Opis | Wpływ |
|---|---|---|---|
| 1 | **Zależność od jakości EKF** | Jeśli EKF sam zaakceptuje sfałszowane pomiary GPS (tzw. „slow-drift attack"), rozjazd EKF vs GPS jest bliski zeru → atak niewidoczny | **Krytyczny** — ataki z wolnym dryftem (<0.1 m/s) mogą unikać detekcji |
| 2 | **Sztywny próg** | Domyślny próg χ²₃ @95% = 7.815 jest stały — nie adaptuje się do warunków lotu (turbulencje, GPS multipath) | **Umiarkowany** — w realistycznych warunkach (las, miasto) FP rate może wzrosnąć |
| 3 | **Brak kontekstu czasowego** | Każda próbka analizowana niezależnie — brak pamięci o historii (np. narastający trend rozjazdu) | **Umiarkowany** — ataki typu „ramp" mogą być wykryte później |
| 4 | **Zależność od eph/epv** | Normalizacja χ² wymaga rzetelnych wartości `eph_y`, `epv_y` z odbiornika. Spoofer może sfałszować te pola | **Istotny** — zaawansowany spoofer raportujący niskie eph "maskuje" atak |
| 5 | **Nie rozróżnia typu ataku** | Zwraca binarne: "spoofing" / "nie spoofing" — nie identyfikuje wektora ataku | **Niski** — wystarczy do alertu; identyfikacja typu wymaga dodatkowej analizy |
| 6 | **Wrażliwość na fałszywe alarmy** | Block_0: precision = 0.587 (41% FP) — wiele czystych próbek klasyfikowane jako atak | **Istotny** — w produkcji FP rate musi być niższy |
| 7 | **Wymaga surowych danych GPS** | Potrzebuje kolumn `lat_x/y`, `lon_x/y`, `alt_x/y`, `eph_y`, `epv_y` — nie działa z przetworzonym featuresetem | **Niski** — te kolumny są standardowe w logach PX4 |

### Trade-offy projektowe

| Trade-off | Wybrany kompromis |
|---|---|
| Czułość vs. specyficzność | Domyślny próg 7.81 faworyzuje recall kosztem precision (więcej FP) |
| Prostota vs. kontekst | Brak pamięci ⟹ szybkość i prostotę kosztem zdolności detekcji wolnych ataków |
| Transferowalność vs. dopasowanie | Jeden próg dla wszystkich → transferowalny ale nie optymalny per-platform |

---

## 2. XGBoost — model ML #1

### Zalety
- **Silna zdolność modelowania** złożonych interakcji między cechami
- Doskonały na block_0 (F1 = 0.789) i block_2 (F1 = 1.000)
- Dobrze zoptymalizowany (GPU support, regularyzacja)
- Dojrzały ekosystem (SHAP, feature importance)

### Słabe strony
- **Kompletna porażka na block_1** (F1 = 0.000) — nie generalizuje na niewidziany typ ataku
- Wymaga etykietowanych danych treningowych
- Ryzyko overfittingu do specyficznych wzorców jednego datasetu
- AUC = 0.702 na block_1 sugeruje słaby sygnał nawet na poziomie rankingu

### Kiedy wybrać XGBoost?
Gdy mamy **duży, zróżnicowany dataset** z wieloma typami ataków i chcemy **wycisnąć maksymalny F1** na znanej dystrybucji. Nie nadaje się jako standalone detektor w deployment bez Kalmana.

---

## 3. LightGBM — model ML #2

### Zalety
- **Najszybszy z trzech** — szybszy trening niż XGBoost
- Lepszy F1 na block_0 (0.807 vs 0.789 XGBoost)
- Mniejsze zużycie pamięci (histogram-based splitting)
- Block_2: perfekcyjna detekcja (F1 = 1.000)

### Słabe strony
- **Identyczny problem jak XGBoost**: block_1 F1 = 0.000
- AUC na block_1 = 0.579 (jeszcze gorszy ranking niż XGBoost)
- Bardziej wrażliwy na szum w małych zbiorach danych
- Wymaga strojenia `num_leaves` pod konkretny problem

### Kiedy wybrać LightGBM?
Gdy potrzebny jest **szybki benchmark** na dużych danych lub jako **drugi model w ensemble** obok XGBoost. Samodzielnie nie radzi sobie lepiej niż XGBoost.

---

## 4. Wnioski porównawcze

```
                 F1@def   Precision   Recall    AUC     block_1 F1
Kalman chi²      0.876    0.862      0.859     0.900    0.746 ✓
LightGBM         0.602    0.582      0.626     0.819    0.000 ✗
XGBoost          0.596    0.577      0.619     0.861    0.000 ✗
```

**Kluczowy wniosek**: W zadaniu detekcji GPS spoofing, gdzie typ ataku w produkcji jest **nieznany a priori**, metoda klasyczna oparta na fizyce (Kalman) **zdecydowanie pokonuje** modele ML, które są wrażliwe na distribution shift.

**Rekomendacja architektury produkcyjnej**: Kalman chi² jako **gatekeeper** (pierwszy detektor) + model ML jako **drugi etap** (redukcja FP) na próbkach flagowanych przez Kalmana.
