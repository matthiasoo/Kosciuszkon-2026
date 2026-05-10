# Guardian of the Skies
## Detekcja ataków GPS Spoofing przy użyciu uczenia maszynowego

**Projekt stworzony na hackathon Kościuszkon 2026 · Ścieżka Honeywell #2**

## Zespół deweloperski - *If_else*:
- Emil Pierzchała (team leader)
- Adam Brodziak
- Maciej Miazek
- Klim Hudzenko

---

Projekt skupia się na budowie pipeline'u do wykrywania ataków **GPS Spoofing** w bezzałogowych statkach powietrznych (UAV). Wykorzystuje realne logi z systemu oblatywanych dronów PX4 (`honeywell_gold_dataset.csv`) i porównuje zaawansowane algorytmy uczenia maszynowego z analitycznym podejściem bazującym na teście innowacji filtru Kalmana.

## O projekcie

Atak GPS Spoofing polega na nadawaniu fałszywego sygnału GNSS, co prowadzi do błędnego obliczenia pozycji przez odbiornik statku powietrznego. Projekt ma na celu stworzenie skutecznego detektora rozróżniającego czysty lot od lotu, w którym sygnał GPS został sfałszowany. 

Głównym problemem, który ujawnił się podczas eksperymentów na zebranych danych, jest to, że modele uczenia maszynowego uczą się wzorców specyficznych dla danych uczących i mają problem z generalizacją na niespotykane wcześniej typy ataków i sygnatury fałszujące. W związku z tym, wykazano wysoką skuteczność weryfikacji wskazań GPS z odczytami rozszerzonego filtra Kalmana (EKF), bazując na statystycznych właściwościach innowacji (test chi-kwadrat).

## Najważniejsze funkcjonalności

- **Klasyfikacja anomalii**: Identyfikacja ataków GPS na podstawie logów lotu.
- **Inżynieria cech (Feature Engineering)**: Ekstrakcja 15 kluczowych cech (m.in. odchylenia innowacji EKF vs GPS, jakości sygnału GPS i dynamiki lotu IMU).
- **Klasyczny detektor fizyczny**: Implementacja i weryfikacja testu innowacji Kalmana ($\chi^2$), który osiąga znacznie lepsze rezultaty w generalizacji.
- **Weryfikacja metod ML**: Budowa modeli XGBoost i LightGBM z systemem walidacji `Block CV`.

## Wymagania i instalacja

Projekt wymaga języka **Python w wersji 3.9 lub wyższej**.

```bash
# Klonowanie repozytorium (lub wejście do katalogu projektu)
# cd Kosciuszkon-2026

# Instalacja wymaganych pakietów
pip install -r requirements.txt
```

Wymagane biblioteki obejmują m.in. `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `scipy` oraz `matplotlib`.

## Uruchomienie

Aby uruchomić pełny pipeline (przetworzenie danych, wygenerowanie wykresów, trening i ewaluację modeli), należy wywołać skrypt `solution.py`:

```bash
python solution.py
```

Skrypt uruchamia cały proces:
1. Wczytuje i oczyszcza dane wejściowe.
2. Dodaje cechy wynikające z różnic w odczytach GPS oraz EKF.
3. Generuje wykresy dystrybucji cech (`plots_distributions.png`).
4. Uruchamia i ewaluuje klasyczny detektor $\chi^2$ (Kalman) oraz trenuje modele ML (XGBoost, LightGBM) przy użyciu techniki Block CV.
5. Generuje macierze pomyłek (`plots_confusion_matrices.png`).
6. Wyświetla końcowe metryki i porównanie użytych mechanizmów.

## Struktura projektu

- `/data`:
  - `honeywell_gold_dataset.csv` - Wejściowy zbiór danych (logi PX4) wykorzystany w ewaluacji.
- `/docs`:
  - `trade_offs.md` - Analiza mocnych i słabych stron wykorzystywanych detektorów.
  - `documentation.md` - Dokumentacja z parametrami poszczególnych modeli oraz implementacji skryptu.
  - `dataset_description.md` - Opis datasetu wejściowego z wymiarowością.
- `/src`:
  - `solution.py` - Główny plik wykonawczy zawierający cały pipeline przetwarzania danych, inżynierii cech oraz detekcji.
  - `visualize.py` - Kod generujący szczegółowe wykresy.
- `/visualization` - Katalog z wykresami.
- `requirements.txt` - Zależności Pythona potrzebe do instalacji.

## Podsumowanie wyników

Z przeprowadzonych eksperymentów wynika bezspornie, że **klasyczny test innowacji Kalmana ($\chi^2$)** sprawdza się w tym ujęciu najlepiej. Zachowuje on fizyczny kontekst działania sensorów drona. Osiągnął wynik F1 Score na poziomie **0.876**, skutecznie wykazując uniwersalność w wykrywaniu nowych typów ataku (co zademonstrowano np. na bloku testowym nr 1), na których zaawansowane modele ML (LightGBM, XGBoost) całkowicie zawiodły (F1 = 0.000). Szczegółowe zestawienia znajdują się w pliku `REPORT.md`.
