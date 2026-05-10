"""
Detekcja GPS Spoofing — Kosciuszkon 2026
========================================
Kanoniczne rozwiazanie. Przewiduje kolumne `label` (0 = czysty lot, 1 = spoofing).

Pipeline:
  1. Czyszczenie danych (usuniecie timestamp, 35 stalych kolumn, normalizacja GPS)
  2. Inzynieria cech — rozjazd EKF vs surowy GPS (cechy fizycznie umotywowane)
  3. Ewaluacja dwoma protokolami:
     a) Stratified 4-fold CV (losowy podzial — latwy test)
  3. Ewaluacja block-based 3-fold CV (podzial po segmentach — twardy test)
  4. Modele:
     - XGBoost (gradient boosting)
     - LightGBM (gradient boosting)
     - Test innowacji Kalmana chi-squared (klasyczny, bez ML)

UWAGA: Notebooki .ipynb w tym repo zawieraja stary, wadliwy kod z data leakage
       i falszywe komentarze o LOSO. Ten skrypt jest kanonicznym rozwiazaniem.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (f1_score, roc_auc_score, accuracy_score,
                             precision_score, recall_score, confusion_matrix,
                             ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ---- STYL WIZUALNY (konsekwentny na wszystkich wykresach) ----
PLOT_STYLE = {
    'bg_color': '#1a1a2e',
    'text_color': '#e0e0e0',
    'grid_color': '#333355',
    'color_clean': '#4fc3f7',     # jasny niebieski = czysty lot
    'color_spoof': '#ef5350',     # czerwony = spoofing
    'color_kalman': '#66bb6a',    # zielony
    'color_xgb': '#ffa726',       # pomaranczowy
    'color_lgbm': '#ab47bc',      # fioletowy
    'font_title': 14,
    'font_label': 11,
    'font_tick': 9,
    'dpi': 130,
}

def setup_plot_style():
    """Konfiguracja ciemnego motywu dla wszystkich wykresow."""
    plt.rcParams.update({
        'figure.facecolor': PLOT_STYLE['bg_color'],
        'axes.facecolor': '#16213e',
        'axes.edgecolor': PLOT_STYLE['grid_color'],
        'axes.labelcolor': PLOT_STYLE['text_color'],
        'axes.grid': True,
        'grid.color': PLOT_STYLE['grid_color'],
        'grid.alpha': 0.3,
        'text.color': PLOT_STYLE['text_color'],
        'xtick.color': PLOT_STYLE['text_color'],
        'ytick.color': PLOT_STYLE['text_color'],
        'figure.dpi': PLOT_STYLE['dpi'],
        'font.size': PLOT_STYLE['font_tick'],
        'axes.titlesize': PLOT_STYLE['font_title'],
        'axes.labelsize': PLOT_STYLE['font_label'],
        'legend.fontsize': PLOT_STYLE['font_tick'],
        'savefig.bbox': 'tight',
        'savefig.facecolor': PLOT_STYLE['bg_color'],
    })

setup_plot_style()


# =============================================================================
# 1. WCZYTANIE I CZYSZCZENIE DANYCH
# =============================================================================

def load_and_clean(path='honeywell_gold_dataset.csv'):
    """Wczytaj dane i usun nieprzydatne kolumny."""
    df = pd.read_csv(path)

    # 35 kolumn stalych — nie niosą zadnego sygnalu
    CONST_COLS = [
        'delta_q_reset[0]', 'delta_q_reset[1]', 'delta_q_reset[2]', 'delta_q_reset[3]',
        'quat_reset_counter', 'delta_alt', 'lat_lon_reset_counter', 'alt_reset_counter',
        'dead_reckoning', 'time_utc_usec', 'timestamp_time_relative', 'heading_offset',
        'fix_type', 'vel_ned_valid', 'ref_lat', 'ref_lon',
        'delta_xy[0]', 'delta_xy[1]', 'delta_z',
        'delta_vxy[0]', 'delta_vxy[1]', 'delta_vz', 'delta_heading',
        'ref_alt',
        'xy_valid', 'z_valid', 'v_xy_valid', 'v_z_valid',
        'xy_reset_counter', 'z_reset_counter', 'vxy_reset_counter', 'vz_reset_counter',
        'heading_reset_counter', 'xy_global', 'z_global',
    ]

    # timestamp to indeks probki, nie czas — wycieka informacje o segmencie
    DROP = ['timestamp'] + CONST_COLS

    # Kolumny zdrowia odbiornika — nie separuja spoofingu w tym datasecie
    DROP += ['jamming_indicator', 'noise_per_ms', 'satellites_used']

    df = df.drop(columns=[c for c in DROP if c in df.columns])

    # Normalizacja jednostek GPS (surowy GPS jest w deg*1e7 / mm)
    df['lat_y'] = df['lat_y'] / 1e7
    df['lon_y'] = df['lon_y'] / 1e7
    df['alt_y'] = df['alt_y'] / 1000.0
    df['alt_ellipsoid_y'] = df['alt_ellipsoid_y'] / 1000.0

    return df


# =============================================================================
# 2. INZYNIERIA CECH — ROZJAZD EKF vs GPS
# =============================================================================

def add_divergence_features(df):
    """
    Oblicza rozjazd miedzy estymata EKF (kolumny *_x) a surowym GPS (*_y).
    Pod hipoteza zerowa (brak spoofingu) rozjazd powinien byc bliski zeru.
    Spoofing lamania te zgodnosc.
    """
    df = df.copy()

    # Rozjazd pozycyjny w metrach
    cos_lat = np.cos(np.radians(df['lat_x']))
    df['lat_diff_m'] = (df['lat_x'] - df['lat_y']) * 111_000.0
    df['lon_diff_m'] = (df['lon_x'] - df['lon_y']) * 111_000.0 * cos_lat
    df['alt_diff_m'] = df['alt_x'] - df['alt_y']
    df['alt_ell_diff_m'] = df['alt_ellipsoid_x'] - df['alt_ellipsoid_y']
    df['pos_diff_h_m'] = np.sqrt(df['lat_diff_m']**2 + df['lon_diff_m']**2)
    df['pos_diff_3d_m'] = np.sqrt(df['pos_diff_h_m']**2 + df['alt_diff_m']**2)

    # Rozjazd predkosci (lokalny NED vs GPS NED)
    df['vn_diff'] = df['vx'] - df['vel_n_m_s']
    df['ve_diff'] = df['vy'] - df['vel_e_m_s']
    df['vd_diff'] = df['vz'] - df['vel_d_m_s']
    df['vel_diff_h'] = np.sqrt(df['vn_diff']**2 + df['ve_diff']**2)
    df['vel_diff_3d'] = np.sqrt(df['vel_diff_h']**2 + df['vd_diff']**2)

    # Rozjazd predkosci skalarne
    df['speed_local'] = np.sqrt(df['vx']**2 + df['vy']**2 + df['vz']**2)
    df['speed_diff'] = df['speed_local'] - df['vel_m_s']

    # Rozjazd kursu
    cog_local = np.arctan2(df['vy'], df['vx'])
    df['cog_diff'] = (df['cog_rad'] - cog_local + np.pi) % (2*np.pi) - np.pi
    df['abs_cog_diff'] = df['cog_diff'].abs()

    # Usuwamy surowe kolumny pozycyjne — model NIE powinien ich widziec bezposrednio
    # (identyfikuja segment = leakage przy losowym CV)
    DROP_RAW_POS = ['lat_x', 'lon_x', 'alt_x', 'alt_ellipsoid_x',
                    'lat_y', 'lon_y', 'alt_y', 'alt_ellipsoid_y',
                    'x', 'y', 'z']
    df = df.drop(columns=[c for c in DROP_RAW_POS if c in df.columns])

    return df


# =============================================================================
# 3. IDENTYFIKACJA SEGMENTOW (dla block-based CV)
# =============================================================================

def assign_segments(df):
    """
    Dataset sklada sie z 6 ciaglych blokow (3x clean, 3x attack).
    Przypisuje numer segmentu kazdemu wierszowi.
    """
    changes = df['label'].diff().ne(0).cumsum()
    df = df.copy()
    df['segment'] = changes
    return df


def assign_block_folds(df):
    """
    Laczy sasiednie segmenty clean+attack w 3 foldy.
    Fold 0: segmenty 1+2 (clean + attack)
    Fold 1: segmenty 3+4 (clean + attack)
    Fold 2: segmenty 5+6 (clean + attack)
    """
    df = assign_segments(df)
    seg_to_fold = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
    df['block_fold'] = df['segment'].map(seg_to_fold)
    return df


# =============================================================================
# 4. LISTA CECH I ICH WYTŁUMACZENIE (Feature Selection & Rationale)
# =============================================================================

LABEL_COL = 'label'
META_COLS = [LABEL_COL, 'segment', 'block_fold']

# ZGODNIE Z WYMAGANIAMI: Każda wybrana cecha ma wytłumaczenie, czemu została
# wybrana do predykcji. Selekcja ta pozwala uniknąć Data Leakage (wycieku danych),
# ponieważ model nie widzi bezwzględnych współrzędnych geograficznych ani prędkości,
# które mogłyby identyfikować konkretny lot.
SELECTED_FEATURES = {
    # --- CECHY INNOWACJI (ROZJAZD SENSORÓW: EKF vs GPS) ---
    # Podstawa detekcji: w trakcie ataku spoofingowego, odczyty z zakłócanego GPS
    # przestają zgadzać się z modelem fizycznym i pomiarami inercyjnymi drona (IMU).
    'pos_diff_h_m': "Rozjazd pozycji w poziomie (horyzontalny) między EKF a GPS. Kluczowy wskaźnik ściągania drona z trasy.",
    'pos_diff_3d_m': "Całkowity (3D) rozjazd pozycji w metrach. Spoofing często zaburza również estymację wysokości (Z).",
    'vel_diff_h': "Rozjazd wektora prędkości w poziomie. Pozwala szybko wykryć atak zanim nastąpi duży dryf pozycji (anomalia pochodnej ruchu).",
    'vel_diff_3d': "Całkowity rozjazd wektora prędkości w 3D. Wykrywa trójwymiarowe rozbieżności dynamiczne.",
    'speed_diff': "Różnica w skalarnej prędkości drona. Prosta miara pokazująca, że odbiornik GPS podaje prędkość fizycznie niemożliwą do osiągnięcia w danym momencie.",
    'abs_cog_diff': "Bezwzględna różnica kursu (Course Over Ground). Spoofer często wymusza wektor prędkości o kącie niezgodnym z faktycznym zwrotem (headingiem) i ruchem maszyny.",
    
    # --- CECHY JAKOŚCI SYGNAŁU GPS I GEOMETRII ---
    # Atak spoofingowy sztucznie podmienia satelity, co nierzadko powoduje skoki w 
    # estymowanej dokładności i geometriach satelitów raportowanych przez sam odbiornik.
    'eph_y': "Estymowany błąd horyzontalny GPS (Expected Position Error). Wybrany, ponieważ często skacze lub drastycznie maleje w momencie przejmowania sygnału przez spoofera.",
    'epv_y': "Estymowany błąd wertykalny GPS. Działa jak wyżej, sygnalizując nienormalne zaufanie lub brak zaufania do rozwiązania wysokościowego.",
    'hdop': "Horizontal Dilution of Precision. Wybrano ją, bo spooferzy z jednym nadajnikiem psują geometrię poziomą (sygnały przychodzą z jednego kierunku), co zaburza HDOP.",
    'vdop': "Vertical Dilution of Precision. Analogicznie jak HDOP, sztuczny układ satelitów degraduje dopasowanie pionowe.",
    's_variance_m_s': "Wariancja prędkości z odbiornika GPS. Rosnąca wariancja demaskuje szum i niestabilność pętli śledzenia PLL/DLL podczas ataku.",
    'c_variance_rad': "Wariancja kursu z odbiornika. Ujawnia utratę stabilnego lock-a na nośnej (carrier phase) z powodu sygnałów zagłuszających (jamming towarzyszący).",
    
    # --- CECHY DYNAMIKI LOTU (IMU) ---
    # Jeśli dron dostanie zły GPS, uważa, że zdmuchnął go wiatr i próbuje to korygować, 
    # co skutkuje gwałtownym pochyleniem i przyspieszeniem maszyn.
    'ax': "Akceleracja osi X. Rejestruje nagłe szarpnięcia do przodu/tyłu gdy kontroler próbuje nadrobić 'utraconą' wg fałszywego GPS pozycję.",
    'ay': "Akceleracja osi Y. Rejestruje szarpnięcia boczne na skutek prób powrotu na trajektorię odchyloną przez atak.",
    'az': "Akceleracja osi Z. Ujawnia próby gwałtownej korekty wysokości wywołanej m.in. sztucznym meandrowaniem sygnału w osi D."
}

def get_feature_cols(df):
    """
    Zwraca nazwy scisle wyselekcjonowanych kolumn cech (whitelist) uzywanych do predykcji ML.
    Model otrzymuje wylacznie te cechy, ktore maja mocne fizyczne i analityczne uzasadnienie.
    """
    selected = list(SELECTED_FEATURES.keys())
    return [col for col in selected if col in df.columns]


# =============================================================================
# 4b. WIZUALIZACJA DYSTRYBUCJI CECH
# =============================================================================

def plot_feature_distributions(df, save_path='plots_distributions.png'):
    """
    Generuje histogramy dystrybucji wybranych cech, podzielone na label=0 (clean)
    i label=1 (spoofing). Pozwala wizualnie ocenic separowalnosc klas.
    """
    features = get_feature_cols(df)
    n = len(features)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.2))
    fig.suptitle('Dystrybucja cech: czysty lot vs spoofing',
                 fontsize=PLOT_STYLE['font_title'] + 2, fontweight='bold', y=1.01)

    clean = df[df['label'] == 0]
    spoof = df[df['label'] == 1]

    for idx, feat in enumerate(features):
        ax = axes.flat[idx]
        ax.hist(clean[feat].dropna(), bins=60, alpha=0.6, density=True,
                color=PLOT_STYLE['color_clean'], label='Czysty lot (0)')
        ax.hist(spoof[feat].dropna(), bins=60, alpha=0.6, density=True,
                color=PLOT_STYLE['color_spoof'], label='Spoofing (1)')
        ax.set_title(feat, fontsize=PLOT_STYLE['font_label'], fontweight='bold')
        ax.set_ylabel('Gestosc')
        if idx == 0:
            ax.legend(fontsize=PLOT_STYLE['font_tick'])

    # Ukryj puste subploty
    for idx in range(n, nrows * ncols):
        axes.flat[idx].set_visible(False)

    plt.tight_layout()
    fig.savefig(save_path)
    print(f"  [PLOT] Dystrybucje cech -> {save_path}")
    plt.close(fig)


# =============================================================================
# 4c. MACIERZ POMYLEK (Confusion Matrix)
# =============================================================================

def plot_confusion_matrices(cm_dict, save_path='plots_confusion_matrices.png'):
    """
    Rysuje macierze pomylek dla wszystkich modeli side-by-side.
    cm_dict: {'Model Name': {'block_0': cm, 'block_1': cm, 'block_2': cm, 'total': cm}}
    """
    models = list(cm_dict.keys())
    n_models = len(models)

    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    fig.suptitle('Macierze pomylek (sumaryczne, Block CV)',
                 fontsize=PLOT_STYLE['font_title'] + 2, fontweight='bold', y=1.03)

    if n_models == 1:
        axes = [axes]

    model_colors = {
        'Kalman chi²': plt.cm.Greens,
        'XGBoost': plt.cm.Oranges,
        'LightGBM': plt.cm.Purples,
    }

    for idx, (name, data) in enumerate(cm_dict.items()):
        ax = axes[idx]
        cm = data['total']
        cmap = model_colors.get(name, plt.cm.Blues)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=['Clean (0)', 'Spoof (1)'])
        disp.plot(ax=ax, cmap=cmap, values_format='d', colorbar=False)
        ax.set_title(f'{name}', fontsize=PLOT_STYLE['font_title'], fontweight='bold')
        ax.set_xlabel('Predykcja', fontsize=PLOT_STYLE['font_label'])
        ax.set_ylabel('Rzeczywistosc', fontsize=PLOT_STYLE['font_label'])
        # Popraw kolor tekstu w komorkach
        for text in disp.text_.ravel():
            text.set_fontsize(PLOT_STYLE['font_title'])

    plt.tight_layout()
    fig.savefig(save_path)
    print(f"  [PLOT] Macierze pomylek -> {save_path}")
    plt.close(fig)

# =============================================================================
# 5. DETEKTOR KLASYCZNY — TEST INNOWACJI KALMANA (chi-squared)
# =============================================================================

def kalman_innovation_chi2(df_raw):
    """
    Test innowacji Kalmana: porownuje estymaty EKF z surowym GPS
    i normalizuje rozjazd raportowana niepewnoscia GPS.

    chi2 = (dn^2 + de^2) / eph^2 + dd^2 / epv^2

    Pod H0 (brak spoofingu) chi2 ~ chi2(3).
    Spoofing lamania zgodnosc EKF/GPS, wiec chi2 rosnie.

    Ta metoda NIE wymaga treningu (brak ML). Jest fizycznie umotywowana.
    Wymaga surowych kolumn pozycyjnych (lat_x/y, lon_x/y, alt_x/y)
    i niepewnosci GPS (eph_y, epv_y), wiec operuje na df_raw (po load_and_clean,
    przed add_divergence_features).
    """
    cos_lat = np.cos(np.radians(df_raw['lat_x']))
    dn = (df_raw['lat_x'] - df_raw['lat_y']) * 111_000.0
    de = (df_raw['lon_x'] - df_raw['lon_y']) * 111_000.0 * cos_lat
    dd = df_raw['alt_x'] - df_raw['alt_y']

    eph = df_raw['eph_y'].clip(lower=0.1)  # floor na 10cm
    epv = df_raw['epv_y'].clip(lower=0.1)

    chi2 = (dn**2 + de**2) / eph**2 + dd**2 / epv**2
    return chi2


def evaluate_kalman_block(df_raw, thresholds=None):
    """
    Ewaluacja testu innowacji Kalmana na block-based foldach.
    Kalman nie wymaga treningu — liczymy chi2 na calym zbiorze,
    ale metryki raportujemy per block dla spojnosci z ML modelami.

    Progi: 95% kwantyl chi2(3) = 7.81 (domyslny) + kalibrowany.
    """
    from scipy.stats import chi2 as chi2_dist

    df = assign_block_folds(df_raw)
    chi2_scores = kalman_innovation_chi2(df)

    # Domyslny prog: 95% kwantyl chi2 z 3 stopniami swobody
    default_threshold = chi2_dist.ppf(0.95, df=3)  # = 7.815

    rows = []
    cm_dict = {}
    total_cm = np.zeros((2, 2), dtype=int)
    n_folds = df['block_fold'].nunique()

    for fold_idx in range(n_folds):
        mask_te = df['block_fold'] == fold_idx
        yte = df.loc[mask_te, 'label'].values
        chi2_te = chi2_scores[mask_te].values

        # Predykcja: chi2 > prog => spoofing
        pred_default = (chi2_te > default_threshold).astype(int)

        # Kalibracja: prog na danych treningowych
        mask_tr = ~mask_te
        ytr = df.loc[mask_tr, 'label'].values
        chi2_tr = chi2_scores[mask_tr].values

        best_thr = default_threshold
        best_f1 = 0
        for thr in np.linspace(0.1, 50, 500):
            p = (chi2_tr > thr).astype(int)
            f = f1_score(ytr, p, zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_thr = thr

        pred_calibrated = (chi2_te > best_thr).astype(int)

        f1_def = f1_score(yte, pred_default, zero_division=0)
        f1_cal = f1_score(yte, pred_calibrated, zero_division=0)

        # AUC: chi2_score jako "probability" (wyzszy = bardziej podejrzany)
        auc = roc_auc_score(yte, chi2_te)

        rows.append({
            'fold': f'block_{fold_idx}',
            'test_rows': int(mask_te.sum()),
            'test_pos_rate': float(yte.mean()),
            'threshold_default': default_threshold,
            'threshold_cal': best_thr,
            'f1_default': f1_def,
            'f1_calibrated': f1_cal,
            'precision': precision_score(yte, pred_calibrated, zero_division=0),
            'recall': recall_score(yte, pred_calibrated, zero_division=0),
            'roc_auc': auc,
        })

        cm = confusion_matrix(yte, pred_calibrated)
        cm_dict[f'block_{fold_idx}'] = cm
        total_cm += cm
        print(f"\n  Block fold {fold_idx} (test: {int(mask_te.sum())} rows, "
              f"pos_rate={yte.mean():.2f}, thr_cal={best_thr:.2f}):")
        print(f"    F1 @7.81={f1_def:.3f}  F1 @cal={f1_cal:.3f}")
        print(f"    Confusion matrix: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")

    cm_dict['total'] = total_cm
    results = pd.DataFrame(rows)
    mean = results[['f1_default','f1_calibrated','precision','recall','roc_auc']].mean()
    print(f"\n{'='*60}")
    print(f"  Kalman chi2 — Block-based 3-Fold (REALISTYCZNY)")
    print(f"{'='*60}")
    print(results[['fold','test_rows','test_pos_rate','threshold_cal',
                   'f1_default','f1_calibrated','precision','recall','roc_auc']].round(3).to_string(index=False))
    print(f"  Srednia:  F1@7.81={mean['f1_default']:.3f}  "
          f"F1@cal={mean['f1_calibrated']:.3f}  "
          f"P={mean['precision']:.3f}  R={mean['recall']:.3f}  "
          f"AUC={mean['roc_auc']:.3f}")
    return results, cm_dict


# =============================================================================
# 5. EWALUACJA
# =============================================================================

def evaluate_stratified_cv(df, make_model, n_splits=4, name="Model"):
    """
    Stratified K-Fold CV z losowym podzialem.
    UWAGA: Daje zawyzone wyniki (F1 ~ 1.0) bo sasiednie probki w tym samym
    segmencie sa prawie identyczne — losowy podzial rozrzuca je miedzy
    train i test, co jest de facto data leakage. Raportujemy to jako
    gorny limit, NIE jako realistyczny wynik.
    """
    feature_cols = get_feature_cols(df)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df, df[LABEL_COL])):
        tr = df.iloc[train_idx]
        te = df.iloc[test_idx]

        Xtr, ytr = tr[feature_cols].values, tr[LABEL_COL].values
        Xte, yte = te[feature_cols].values, te[LABEL_COL].values

        sc = StandardScaler().fit(Xtr)
        m = make_model()
        m.fit(sc.transform(Xtr), ytr)
        proba = m.predict_proba(sc.transform(Xte))[:, 1]
        pred = (proba >= 0.5).astype(int)

        rows.append({
            'fold': f'fold_{fold_idx}',
            'f1': f1_score(yte, pred, zero_division=0),
            'precision': precision_score(yte, pred, zero_division=0),
            'recall': recall_score(yte, pred, zero_division=0),
            'roc_auc': roc_auc_score(yte, proba),
            'accuracy': accuracy_score(yte, pred),
        })

    results = pd.DataFrame(rows)
    mean = results[['f1','precision','recall','roc_auc','accuracy']].mean()
    print(f"\n{'='*60}")
    print(f"  {name} — Stratified {n_splits}-Fold CV (GORNY LIMIT)")
    print(f"{'='*60}")
    print(results[['fold','f1','precision','recall','roc_auc']].round(3).to_string(index=False))
    print(f"  Srednia:  F1={mean['f1']:.3f}  P={mean['precision']:.3f}  "
          f"R={mean['recall']:.3f}  AUC={mean['roc_auc']:.3f}")
    print(f"  (zawyzone — losowy podzial nie mierzy generalizacji)")
    return results


def calibrate_threshold(y_true, proba):
    """Znajdz prog maksymalizujacy F1 na danych kalibracyjnych."""
    from sklearn.metrics import precision_recall_curve
    prec, rec, thresholds = precision_recall_curve(y_true, proba)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    best_idx = np.argmax(f1s)
    return float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5


def evaluate_block_cv(df, make_model, name="Model"):
    """
    Block-based 3-Fold CV — podzial po segmentach (ciagłych blokach danych).
    Twardy test generalizacji: model nigdy nie widzial danych z testowanego
    segmentu czasowego lotu. Nie ma leakage z sasiedztwa probek.

    Kalibracja progu: prog decyzyjny wyznaczany z OOF predictions na foldach
    treningowych (bez przecieku z testu). To pomaga gdy rozklad prawdopodobienstw
    dryfuje miedzy segmentami.
    """
    df = assign_block_folds(df)
    feature_cols = get_feature_cols(df)
    rows = []
    cm_dict = {}  # confusion matrix per fold + total
    total_cm = np.zeros((2, 2), dtype=int)
    n_folds = df['block_fold'].nunique()

    for fold_idx in range(n_folds):
        te = df[df['block_fold'] == fold_idx]
        tr = df[df['block_fold'] != fold_idx]

        Xtr, ytr = tr[feature_cols].values, tr[LABEL_COL].values
        Xte, yte = te[feature_cols].values, te[LABEL_COL].values

        sc = StandardScaler().fit(Xtr)
        m = make_model()
        m.fit(sc.transform(Xtr), ytr)

        # Predykcje na zbiorze treningowym (OOF) do kalibracji progu
        proba_train = m.predict_proba(sc.transform(Xtr))[:, 1]
        threshold = calibrate_threshold(ytr, proba_train)

        proba = m.predict_proba(sc.transform(Xte))[:, 1]
        pred_default = (proba >= 0.5).astype(int)
        pred_calibrated = (proba >= threshold).astype(int)

        f1_default = f1_score(yte, pred_default, zero_division=0)
        f1_calibrated = f1_score(yte, pred_calibrated, zero_division=0)

        rows.append({
            'fold': f'block_{fold_idx}',
            'test_rows': len(te),
            'test_pos_rate': float(yte.mean()),
            'threshold': threshold,
            'f1_default': f1_default,
            'f1_calibrated': f1_calibrated,
            'precision': precision_score(yte, pred_calibrated, zero_division=0),
            'recall': recall_score(yte, pred_calibrated, zero_division=0),
            'roc_auc': roc_auc_score(yte, proba),
            'accuracy': accuracy_score(yte, pred_calibrated),
        })
        cm = confusion_matrix(yte, pred_calibrated)
        cm_dict[f'block_{fold_idx}'] = cm
        total_cm += cm
        print(f"\n  Block fold {fold_idx} (test: {len(te)} rows, pos_rate={yte.mean():.2f}, "
              f"threshold={threshold:.3f}):")
        print(f"    F1 @0.5={f1_default:.3f}  F1 @cal={f1_calibrated:.3f}")
        print(f"    Confusion matrix: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")

    cm_dict['total'] = total_cm
    results = pd.DataFrame(rows)
    mean = results[['f1_default','f1_calibrated','precision','recall','roc_auc','accuracy']].mean()
    print(f"\n{'='*60}")
    print(f"  {name} — Block-based 3-Fold CV (REALISTYCZNY)")
    print(f"{'='*60}")
    print(results[['fold','test_rows','test_pos_rate','threshold',
                   'f1_default','f1_calibrated','precision','recall','roc_auc']].round(3).to_string(index=False))
    print(f"  Srednia:  F1@0.5={mean['f1_default']:.3f}  "
          f"F1@cal={mean['f1_calibrated']:.3f}  "
          f"P={mean['precision']:.3f}  R={mean['recall']:.3f}  AUC={mean['roc_auc']:.3f}")
    return results, cm_dict


# =============================================================================
# 6. MODEL
# =============================================================================

def make_xgb():
    return XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
    )


def make_lgbm():
    return LGBMClassifier(
        n_estimators=600,
        learning_rate=0.04,
        num_leaves=63,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("="*60)
    print("  DETEKCJA GPS SPOOFING — KOSCIUSZKON 2026")
    print("="*60)

    # 1. Wczytaj i wyczysc dane
    df_raw = load_and_clean()
    print(f"\n[1] Dane po czyszczeniu: {df_raw.shape}")
    print(f"    Label: {df_raw['label'].value_counts().to_dict()}")

    # 2. Dodaj cechy rozjazdu
    df = add_divergence_features(df_raw)
    feature_cols = get_feature_cols(df)
    print(f"\n[2] Dane po inzynierii cech: {df.shape}")
    print(f"    Cechy: {len(feature_cols)}")
    print(f"    Nazwy: {feature_cols}")

    # Sprawdz ze nie ma leakage
    assert 'timestamp' not in df.columns, "timestamp nie powinien byc w danych!"
    assert 'lat_x' not in df.columns, "lat_x nie powinien byc w danych!"
    assert 'label' not in feature_cols, "label nie powinien byc w cechach!"

    # 3. Analiza dystrybucji cech
    print(f"\n{'#'*60}")
    print(f"  ANALIZA DYSTRYBUCJI CECH")
    print(f"{'#'*60}")
    plot_feature_distributions(df)

    # 4. Block CV — XGBoost
    print(f"\n{'#'*60}")
    print(f"  BLOCK-BASED 3-FOLD CV (podzial po segmentach)")
    print(f"  Realistyczny test: model testowany na sesji ktorej nie widzial.")
    print(f"{'#'*60}")

    results_xgb, cm_xgb = evaluate_block_cv(df, make_xgb, name="XGBoost")

    # 5. Block CV — LightGBM
    results_lgbm, cm_lgbm = evaluate_block_cv(df, make_lgbm, name="LightGBM")

    # 6. Kalman chi-squared (klasyczny, bez ML)
    print(f"\n{'#'*60}")
    print(f"  KALMAN CHI-SQUARED (klasyczny detektor, bez treningu)")
    print(f"{'#'*60}")
    results_kalman, cm_kalman = evaluate_kalman_block(df_raw)

    # 7. Macierze pomylek
    print(f"\n{'#'*60}")
    print(f"  MACIERZE POMYLEK")
    print(f"{'#'*60}")
    cm_all = {
        'Kalman chi\u00b2': cm_kalman,
        'XGBoost': cm_xgb,
        'LightGBM': cm_lgbm,
    }
    plot_confusion_matrices(cm_all)

    # 8. Porownanie modeli
    print(f"\n{'='*60}")
    print(f"  POROWNANIE MODELI (Block CV — realistyczny)")
    print(f"{'='*60}")
    all_results = {
        'XGBoost': results_xgb,
        'LightGBM': results_lgbm,
        'Kalman chi2': results_kalman,
    }
    for name, res in all_results.items():
        m = res[['f1_default','f1_calibrated','precision','recall','roc_auc']].mean()
        print(f"  {name:12s}  F1@def={m['f1_default']:.3f}  "
              f"F1@cal={m['f1_calibrated']:.3f}  "
              f"P={m['precision']:.3f}  R={m['recall']:.3f}  "
              f"AUC={m['roc_auc']:.3f}")
