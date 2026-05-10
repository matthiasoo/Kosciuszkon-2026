# Dokumentacja programu — `solution.py`

## Wymagania systemowe

| Wymaganie | Wersja |
|---|---|
| Python | ≥ 3.9 |
| pandas | ≥ 1.5 |
| numpy | ≥ 1.23 |
| scikit-learn | ≥ 1.2 |
| xgboost | ≥ 1.7 |
| lightgbm | ≥ 3.3 |
| scipy | ≥ 1.10 |
| matplotlib | ≥ 3.6 |

## Instalacja

```bash
pip install -r requirements.txt
```

## Uruchomienie

```bash
python solution.py
```

Skrypt automatycznie:
1. Wczytuje dane z `honeywell_gold_dataset.csv`
2. Generuje wykresy dystrybucji cech → `plots_distributions.png`
3. Trenuje i ewaluuje XGBoost + LightGBM (Block CV)
4. Ewaluuje filtr Kalmana (Block CV)
5. Generuje macierze pomyłek → `plots_confusion_matrices.png`
6. Drukuje porównanie modeli

## Struktura pliku `solution.py`

```
Sekcja 1: load_and_clean()
  └─ Wczytanie CSV, usunięcie stałych kolumn, normalizacja GPS

Sekcja 2: add_divergence_features()
  └─ 14 cech rozjazdu EKF vs GPS

Sekcja 3: assign_segments() / assign_block_folds()
  └─ Podział na 6 segmentów → 3 foldy (clean+attack)

Sekcja 4: SELECTED_FEATURES / get_feature_cols()
  └─ Whitelist 15 cech z uzasadnieniem

Sekcja 4b: plot_feature_distributions()
  └─ Histogram per cecha: clean vs spoofing

Sekcja 4c: plot_confusion_matrices()
  └─ Macierze pomyłek side-by-side

Sekcja 5: kalman_innovation_chi2() / evaluate_kalman_block()
  └─ Test innowacji χ² (klasyczny, bez ML)

Sekcja 5b: evaluate_stratified_cv() / evaluate_block_cv()
  └─ Block-based 3-Fold CV z kalibracją progu

Sekcja 6: make_xgb() / make_lgbm()
  └─ Fabryki modeli ML

MAIN: Pipeline uruchamiany przy `python solution.py`
```

## Parametry modeli

### XGBoost
| Parametr | Wartość | Uzasadnienie |
|---|---|---|
| `n_estimators` | 400 | Wystarczająco duży ensemble |
| `max_depth` | 6 | Ogranicza przeuczenie |
| `learning_rate` | 0.05 | Wolniejsze uczenie = lepszy tuning |
| `subsample` | 0.85 | Stochastic boosting |
| `colsample_bytree` | 0.85 | Feature bagging |

### LightGBM
| Parametr | Wartość | Uzasadnienie |
|---|---|---|
| `n_estimators` | 600 | Więcej drzew, mniejszy LR |
| `learning_rate` | 0.04 | Wolniejsze uczenie |
| `num_leaves` | 63 | ~2^6, porównywalny z max_depth=6 |
| `min_child_samples` | 20 | Zapobiega przeuczeniu |
| `reg_alpha` / `reg_lambda` | 0.1 | L1/L2 regularyzacja |

### Filtr Kalmana
| Parametr | Wartość | Uzasadnienie |
|---|---|---|
| Próg domyślny | χ²₃ @95% = 7.815 | Standardowy próg statystyczny |
| `eph` floor | 0.1 m | Zabezpieczenie przed dzieleniem przez 0 |

## Pliki wyjściowe

| Plik | Zawartość |
|---|---|
| `plots_distributions.png` | Histogramy dystrybucji cech |
| `plots_confusion_matrices.png` | Macierze pomyłek (Kalman, XGBoost, LightGBM) |
| stdout | Metryki per fold + porównanie modeli |

## Pliki projektu

| Plik | Rola |
|---|---|
| `solution.py` | Kanoniczny pipeline (jedyny plik do uruchomienia) |
| `REPORT.md` | Raport z wynikami |
| `TRADE_OFFS.md` | Analiza słabych stron modeli |
| `DOCUMENTATION.md` | Ten dokument |
| `dataset_description.md` | Opis datasetu |
| `requirements.txt` | Zależności pip |
| `honeywell_gold_dataset.csv` | Dataset wejściowy |
