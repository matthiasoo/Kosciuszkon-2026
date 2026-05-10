# Detekcja ataków GPS Spoofing przy użyciu uczenia maszynowego

**Kościuszkon 2026 · Honeywell Theme #2**

---

## 1. Zrozumienie problemu

**Atak GPS Spoofing** to klasa ataków cyber-fizycznych, w której atakujący nadaje sfałszowany sygnał GNSS, sprawiając że odbiornik ofiary obliczy błędną pozycję, prędkość lub czas.

**Cel**: Przewidywanie kolumny `label` (binarna: 0 = czysty lot, 1 = spoofing GPS).

---

## 2. Dataset

**Plik**: `honeywell_gold_dataset.csv` — realne logi PX4.

| | |
|---|---|
| Wiersze | 24 992 |
| Kolumny | 84 (83 cechy + 1 etykieta: `label`) |
| Brakujące wartości | 0 |

**Rozkład etykiety**: 0 (clean) = 11 870 (47,5%) · 1 (spoofing) = 13 122 (52,5%)

### 2.1 Struktura segmentowa

6 ciągłych segmentów (naprzemiennie clean/attack), łączonych w 3 foldy do Block CV.

---

## 3. Wyselekcjonowane cechy (15 cech)

### Cechy innowacji (rozjazd EKF vs GPS)
| cecha | opis |
|---|---|
| `pos_diff_h_m` | Rozjazd pozycji w poziomie |
| `pos_diff_3d_m` | Całkowity rozjazd pozycji 3D |
| `vel_diff_h` | Rozjazd wektora prędkości w poziomie |
| `vel_diff_3d` | Całkowity rozjazd prędkości 3D |
| `speed_diff` | Różnica skalarna prędkości |
| `abs_cog_diff` | Bezwzględna różnica kursu (COG) |

### Cechy jakości sygnału GPS
| cecha | opis |
|---|---|
| `eph_y` | Estymowany błąd horyzontalny GPS |
| `epv_y` | Estymowany błąd wertykalny GPS |
| `hdop` | Horizontal Dilution of Precision |
| `vdop` | Vertical Dilution of Precision |
| `s_variance_m_s` | Wariancja prędkości GPS |
| `c_variance_rad` | Wariancja kursu GPS |

### Cechy dynamiki lotu (IMU)
| cecha | opis |
|---|---|
| `ax`, `ay`, `az` | Akceleracja w 3 osiach |

Uzasadnienie selekcji: patrz `SELECTED_FEATURES` w `solution.py`.

---

## 4. Porównane metody

| # | Metoda | Kategoria | Wymaga treningu? |
|---|---|---|---|
| 1 | **Test innowacji Kalmana (chi-squared)** | klasyczny statystyczny | NIE |
| 2 | **XGBoost** | gradient boosting | TAK |
| 3 | **LightGBM** | gradient boosting | TAK |

---

## 5. Wyniki (Block-based 3-Fold CV)

### 5.1 Porównanie modeli

| Metoda | F1 @domyślny | Precision | Recall | AUC |
|---|---|---|---|---|
| **Kalman chi²** | **0.876** | **0.862** | **0.859** | **0.900** |
| LightGBM | 0.602 | 0.582 | 0.626 | 0.819 |
| XGBoost | 0.596 | 0.577 | 0.619 | 0.861 |

### 5.2 Kalman chi² — per fold

| fold | F1 @7.81 | F1 @cal | precision | recall | AUC |
|---|---|---|---|---|---|
| block_0 | 0.883 | 0.709 | 0.587 | 0.893 | 0.942 |
| block_1 | 0.746 | 0.813 | 1.000 | 0.684 | 0.757 |
| block_2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

### 5.3 XGBoost — per fold

| fold | F1 @0.5 | precision | recall | AUC |
|---|---|---|---|---|
| block_0 | 0.789 | 0.730 | 0.857 | 0.881 |
| block_1 | **0.000** | 0.000 | 0.000 | 0.702 |
| block_2 | 1.000 | 1.000 | 1.000 | 1.000 |

### 5.4 LightGBM — per fold

| fold | F1 @0.5 | precision | recall | AUC |
|---|---|---|---|---|
| block_0 | 0.807 | 0.747 | 0.877 | 0.878 |
| block_1 | **0.000** | 0.000 | 0.000 | 0.579 |
| block_2 | 1.000 | 1.000 | 1.000 | 1.000 |

---

## 6. Interpretacja

### Kalman wygrywa zdecydowanie
Kalman (F1 = 0.876) pokonuje oba modele ML (F1 ≤ 0.602). Kluczowa przewaga: wykrywa ataki we **wszystkich foldach**, w tym block_1 (F1 = 0.746) gdzie ML daje F1 = 0.

### Dlaczego ML zawodzi na block_1?
Block_1 to segment ataku o innej sygnaturze. ML modele nie generalizują na niewidziany typ ataku — uczą się wzorców specyficznych dla datasetu, nie fizyki.

### Dlaczego Kalman jest lepszy?
Opiera się na fizyce: pod H0 estymaty EKF i GPS **muszą** się zgadzać. Spoofing łamie tę zgodność **niezależnie od typu ataku**. Brak parametrów uczonych → zero ryzyka overfittingu.

---

## 7. Wygenerowane wizualizacje

| Plik | Zawartość |
|---|---|
| `plots_distributions.png` | Histogramy dystrybucji 15 cech (clean vs spoofing) |
| `plots_confusion_matrices.png` | Macierze pomyłek 3 modeli side-by-side |

---

## 8. Dokumentacja uzupełniająca

| Plik | Opis |
|---|---|
| `TRADE_OFFS.md` | Szczegółowa analiza słabych stron i trade-offów modeli |
| `DOCUMENTATION.md` | Dokumentacja techniczna programu |
| `dataset_description.md` | Opis datasetu |
