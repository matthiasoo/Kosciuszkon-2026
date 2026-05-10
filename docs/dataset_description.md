# Zestaw danych Honeywell Gold — telemetria PX4 do wykrywania podszywania się pod GPS

Rzeczywista telemetria autopilota scalona z tematów PX4 uORB (`vehicle_attitude`, `vehicle_global_position`, `vehicle_gps_position`, `vehicle_local_position`). Każdy wiersz to jedna zsynchronizowana próbka pojedynczego drona w locie; wiersze z czterech scenariuszy lotu są połączone od końca do końca.

## Przegląd zestawu danych

| | |
|---|---|
| Plik | `honeywell_gold_dataset.csv` |
| Wiersze | 24 992 |
| Kolumny | 84 (83 obiekty + 1 cel: `label`) |
| Wartości null | 0 |
| Drony | 1 |
| Źródło odniesienia | `ref_lat=36.2048055, ref_lon=138.2529121, ref_alt=51.6988 m` |

### Dystrybucja etykiet

| etykieta | liczba | ułamek |
|---|---|---|
| 0 | 11 870 | 47,5% |
| 1 | 13 122 | 52,5% |

Zestaw danych zawiera wiersze z czterech sąsiadujących bloków lotu (każdy po 6248 wierszy). Niektóre bloki są całkowicie czyste (`etykieta=0`), podczas gdy inne zawierają okna spoofingowe z `etykietą=1`.

## Ważne zastrzeżenia

1. **Podział na treningi/testy** — dane są przechowywane jako cztery sąsiadujące bloki, *nie* przetasowane. Losowy `train_test_split` może powodować wyciek wierszy z tego samego bloku lotu do obu foldów i zawyżać wyniki. Rozważ zastosowanie strategii podziału uwzględniających bloki.
2. **`timestamp` to indeks próbki, a nie czas rzeczywisty.** Wykonuje `0..24991` z krokiem 1 wewnątrz każdego bloku (nie zegar ścienny). `time_utc_usec` i `timestamp_time_relative` są stałymi zerami w tym pliku, więc nie ma odniesienia do czasu absolutnego. Jeśli częstotliwość próbkowania ma znaczenie dla Twoich funkcji, potraktuj indeks jako znaczniki i załóż, że typowa dla PX4 wartość to 50 Hz / 100 Hz.
3. **35 z 85 kolumn w tym pliku jest stałych** (głównie liczniki resetowania EKF i flagi ważności). Nie niosą one żadnych informacji; należy je usunąć z początku. Pełna lista znajduje się na dole tego dokumentu.
4. **EKF (`*_x`) i surowy GPS (`*_y`) używają różnych jednostek**:
- `lat_x`, `lon_x` — stopnie (zmiennoprzecinkowe)
- `lat_y`, `lon_y` — stopnie × 1e7 (format czujnika zakodowany jako liczba całkowita)
- `alt_x`, `alt_ellipsoid_x` — metry
- `alt_y`, `alt_ellipsoid_y` — milimetry

Aby porównać rozbieżność EKF z GPS (silny sygnał spoofingowy), znormalizuj:
```python
df['lat_diff_m'] = (df['lat_x'] - df['lat_y'] / 1e7) * 111_000
df['lon_diff_m'] = (df['lon_x'] - df['lon_y'] / 1e7) * 111_000 * np.cos(np.radians(df['lat_x']))
df['alt_diff_m'] = df['alt_x'] - df['alt_y'] / 1000.0
```

## Odwołanie do kolumny

### Próbka / oś czasu
- **`timestamp`** — indeks próbki w pliku (0..24991, krok 1). Nie sekundy.
- **`time_utc_usec`** — mikrosekundy GPS UTC. Tutaj stała `0`.
- **`timestamp_time_relative`** — pole czasu względnego z `sensor_gps`. Tutaj stała `0`.

### Położenie (`vehicle_attitude`)
- **`q[0]..q[3]`** — kwaternion położenia pojazdu (`[w, x, y, z]`, znormalizowany).
- **`delta_q_reset[0]..delta_q_reset[3]`** — kwaternion ostatniego resetu estymatora położenia (stała w tym pliku).
- **`quat_reset_counter`** — inkrementacja przy każdym resecie położenia (tu stała `3`).

## Pozycja globalna EKF (`vehicle_global_position`, sufiks `_x`)
- **`lat_x`** — łączna wartość szacunkowa szerokości geograficznej, w stopniach.
- **`lon_x`** — łączna wartość szacunkowa długości geograficznej, w stopniach.
- **`alt_x`** — łączna wysokość nad poziomem morza, w metrach.
- **`alt_ellipsoid_x`** — suma wysokości nad elipsoidą WGS-84, metry.
- **`delta_alt`** — różnica wysokości przy ostatnim resecie (tu stała `0`).
- **`eph_x`** — odchylenie standardowe oszacowania położenia poziomego, metry.
- **`epv_x`** — odchylenie standardowe oszacowania położenia pionowego, metry.
- **`terrain_alt`** — oszacowanie wysokości terenu (AMSL), metry.
- **`terrain_alt_valid`** — oszacowanie terenu prawidłowe (1) / nieprawidłowe (0).
- **`lat_lon_reset_counter`** — przyrosty przy każdym resecie szerokości/długości geograficznej (stała `2`).
- **`alt_reset_counter`** — przyrosty przy każdym resecie wysokości (stała `0`).
- **`dead_reckoning`** — `1`, jeśli EKF nie ma GPS i działa w trybie nawigacji zliczeniowej (stała `0`).

## Surowy czujnik GPS (`vehicle_gps_position`, przyrostek `_y`)
- **`lat_y`** — szerokość geograficzna GPS, stopnie × 1e7.
- **`lon_y`** — długość geograficzna GPS, stopnie × 1e7.
- **`alt_y`** — wysokość GPS nad poziomem morza (AMSL), milimetry.
- **`alt_ellipsoid_y`** — wysokość GPS powyżej elipsoidy WGS-84, milimetry.
- **`s_variance_m_s`** — wariancja oszacowania prędkości GPS, (m/s)².
- **`c_variance_rad`** — wariancja oszacowania kursu GPS, rad².
- **`eph_y`** — niepewność pozycji poziomej zgłoszona przez GPS (1σ), metry.
- **`epv_y`** — niepewność pozycji pionowej zgłoszona przez GPS (1σ), metry.
- **`hdop`** — poziome rozmycie precyzji (bezjednostkowe; im niższa wartość, tym lepiej).
- **`vdop`** — pionowe rozmycie precyzji.
- **`noise_per_ms`** — średni szum tła odbiornika na milisekundę. Wyższy poziom oznacza bardziej zaszumione środowisko radiowe; zagłuszanie zwiększa ten poziom.
- **`jamming_indicator`** — prawdopodobieństwo zagłuszania po stronie odbiornika, 0–255 (wyższy poziom = większe zakłócenia).
- **`vel_m_s`** — całkowita prędkość GPS nad ziemią, m/s.
- **`vel_n_m_s`** — prędkość GPS na północ, m/s.
- **`vel_e_m_s`** — prędkość GPS na wschód, m/s.
- **`vel_d_m_s`** — prędkość GPS w dół, m/s.
- **`cog_rad`** — kurs GPS nad ziemią, radiany.
- **`heading_offset`** — przesunięcie kursu magnetycznego (rad).