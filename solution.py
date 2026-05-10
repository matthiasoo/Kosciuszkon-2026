"""
Detekcja GPS Spoofing — Kosciuszkon 2026
========================================
Kanoniczne rozwiazanie. Przewiduje kolumne `label` (0 = czysty lot, 1 = spoofing).

Pipeline:
  1. Czyszczenie danych (usuniecie timestamp, 35 stalych kolumn, normalizacja GPS)
  2. Inzynieria cech — rozjazd EKF vs surowy GPS (cechy fizycznie umotywowane)
  3. Ewaluacja dwoma protokolami:
     a) Stratified 4-fold CV (losowy podzial — latwy test)
     b) Block-based 3-fold CV (podzial po segmentach — twardy test generalizacji)
  4. Model: XGBoost (gradient boosting)

UWAGA: Notebooki .ipynb w tym repo zawieraja stary, wadliwy kod z data leakage
       i falszywe komentarze o LOSO. Ten skrypt jest kanonicznym rozwiazaniem.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (f1_score, roc_auc_score, accuracy_score,
                             precision_score, recall_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


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
# 4. LISTA CECH
# =============================================================================

LABEL_COL = 'label'
META_COLS = [LABEL_COL, 'segment', 'block_fold']

def get_feature_cols(df):
    """Zwraca nazwy kolumn cech (wszystko co nie jest metadana/label)."""
    return [c for c in df.columns if c not in META_COLS]


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
        print(f"\n  Block fold {fold_idx} (test: {len(te)} rows, pos_rate={yte.mean():.2f}, "
              f"threshold={threshold:.3f}):")
        print(f"    F1 @0.5={f1_default:.3f}  F1 @cal={f1_calibrated:.3f}")
        print(f"    Confusion matrix: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")

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
    return results


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

    # 3. Ewaluacja — Stratified CV (latwy test)
    print(f"\n{'#'*60}")
    print(f"  PROTOKOL A: STRATIFIED 4-FOLD CV (losowy podzial)")
    print(f"{'#'*60}")
    results_strat = evaluate_stratified_cv(df, make_xgb, name="XGBoost + divergence")

    # 4. Ewaluacja — Block CV (twardy test)
    print(f"\n{'#'*60}")
    print(f"  PROTOKOL B: BLOCK-BASED 3-FOLD CV (podzial po segmentach)")
    print(f"  Twardy test: model nigdy nie widzial danych z testowanego")
    print(f"  segmentu czasowego lotu.")
    print(f"{'#'*60}")
    results_block = evaluate_block_cv(df, make_xgb, name="XGBoost + divergence")

    # 5. Podsumowanie
    print(f"\n{'='*60}")
    print(f"  POROWNANIE")
    print(f"{'='*60}")
    strat_mean = results_strat[['f1','precision','recall','roc_auc']].mean()
    block_mean = results_block[['f1_default','f1_calibrated','precision','recall','roc_auc']].mean()
    print(f"  Stratified CV:    F1={strat_mean['f1']:.3f}  AUC={strat_mean['roc_auc']:.3f}  (zawyzone)")
    print(f"  Block CV @0.5:    F1={block_mean['f1_default']:.3f}  AUC={block_mean['roc_auc']:.3f}")
    print(f"  Block CV @cal:    F1={block_mean['f1_calibrated']:.3f}  AUC={block_mean['roc_auc']:.3f}")
    print(f"\n  Block CV jest realistycznym testem — mierzy generalizacje")
    print(f"  miedzy roznymi sesjami lotowymi.")
