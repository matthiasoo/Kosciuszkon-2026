import numpy as np
import matplotlib.pyplot as plt
from math import pi

# ---- STYL WIZUALNY Z SOLUTION.PY ----
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
    """Konfiguracja ciemnego motywu dla wykresów."""
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

# =============================================================================
# 1. TIME-SERIES PLOT: SIGNAL DRIFT
# =============================================================================
def plot_signal_drift(save_path='plot_1_signal_drift.png'):
    setup_plot_style()
    
    t = np.linspace(0, 100, 500)
    # Prawdziwa pozycja (np. prosta trajektoria + lekki szum)
    true_pos = np.sin(t * 0.1) * 20 + t * 0.5
    
    # Pozycja spoofowana (zaczyna powoli dryfować po t=40)
    spoofed_pos = true_pos.copy()
    drift_start = 40
    drift_mask = t > drift_start
    # Paraboliczny dryf przypominający powolne ściąganie drona
    spoofed_pos[drift_mask] += 0.05 * (t[drift_mask] - drift_start)**2
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(t, true_pos, label='Prawdziwa Pozycja (EKF/IMU)', color=PLOT_STYLE['color_clean'], linewidth=2.5)
    ax.plot(t, spoofed_pos, label='Pozycja z zakłóconego GPS', color=PLOT_STYLE['color_spoof'], linewidth=2.5, linestyle='--')
    
    # Detekcja Kalmana (zaczyna alarmować tuż po rozpoczęciu dryfu, np. t=45)
    detection_time = 43
    ax.axvspan(detection_time, 100, color=PLOT_STYLE['color_spoof'], alpha=0.15, label='Aura detekcji: Alarm Filtr Kalmana (χ²)')
    
    # Linia pionowa oznaczająca moment detekcji
    ax.axvline(x=detection_time, color=PLOT_STYLE['color_kalman'], linestyle=':', linewidth=2)
    ax.text(detection_time + 1.5, true_pos.max(), 'Wykrycie rozjazdu\nprzez Innowację Kalmana', 
            color=PLOT_STYLE['color_kalman'], fontsize=PLOT_STYLE['font_label'], fontweight='bold',
            verticalalignment='top')
    
    ax.set_title('Demonstracja Ataku Spoofingowego i Detekcji', fontweight='bold')
    ax.set_xlabel('Czas [s]')
    ax.set_ylabel('Pozycja [m]')
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    fig.savefig(save_path)
    print(f"Wygenerowano: {save_path}")
    plt.close()

# =============================================================================
# 2. BAR CHART: PORÓWNANIE OPÓŹNIENIA (LATENCJI)
# =============================================================================
def plot_latency_comparison(save_path='plot_2_latency.png'):
    setup_plot_style()
    
    models = ['Filtr Kalmana (χ²)', 'LightGBM', 'XGBoost']
    # Latencja w mikrosekundach (symulowana, pokazująca rząd wielkości na korzyść prostych obliczeń macierzowych)
    latencies_us = [2.5, 450.0, 1200.0] 
    colors = [PLOT_STYLE['color_kalman'], PLOT_STYLE['color_lgbm'], PLOT_STYLE['color_xgb']]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    bars = ax.bar(models, latencies_us, color=colors, alpha=0.85, width=0.5)
    
    ax.set_title('Zapotrzebowanie Obliczeniowe (Latencja Infrerencji per próbka)', fontweight='bold')
    ax.set_ylabel('Czas wykonania [µs] (Skala logarytmiczna)')
    
    # Skala logarytmiczna żeby pokazać jak ogromna jest przewaga Kalmana
    ax.set_yscale('log')
    
    # Dodanie wartości nad słupkami
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height} µs',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 punkty w górę
                    textcoords="offset points",
                    ha='center', va='bottom', color=PLOT_STYLE['text_color'], fontweight='bold')

    plt.tight_layout()
    fig.savefig(save_path)
    print(f"Wygenerowano: {save_path}")
    plt.close()

# =============================================================================
# 3. RADAR CHART: PORÓWNANIE MODELI 
# =============================================================================
def plot_radar_comparison(save_path='plot_3_radar.png'):
    setup_plot_style()
    
    categories = ['Dokładność (F1 Score)', 'Szybkość Obliczeń', 'Brak Zapotrzebowania\nna Dane (No-Training)', 'Wyjaśnialność\n(Explainability)']
    N = len(categories)
    
    # Skala 1-10
    # Kalman: Dokładność bardzo wysoka, szybkość max, nie potrzebuje danych, pełna wyjaśnialność wzorami fizycznymi
    kalman_stats = [9.0, 10.0, 10.0, 10.0]
    # XGBoost: Dokładność gorsza na out-of-distribution (Block CV), szybkość ok ale gorsza, potrzebuje mega dużo danych, wyjaśnialność (SHAP - umiarkowana)
    xgb_stats = [6.0, 6.0, 3.0, 5.0]
    # LightGBM: Podobnie do XGB, odrobinę szybszy
    lgbm_stats = [6.5, 7.5, 3.0, 5.0]
    
    # Dodajemy pierwszy element na koniec, żeby zamknąć wielokąt
    kalman_stats += kalman_stats[:1]
    xgb_stats += xgb_stats[:1]
    lgbm_stats += lgbm_stats[:1]
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Konfiguracja osi
    plt.xticks(angles[:-1], categories, color=PLOT_STYLE['text_color'], size=10, fontweight='bold')
    ax.set_rlabel_position(0)
    plt.yticks([2, 4, 6, 8, 10], ["2", "4", "6", "8", "10"], color="grey", size=8)
    plt.ylim(0, 10)
    
    # Plot Kalman
    ax.plot(angles, kalman_stats, linewidth=2, linestyle='solid', color=PLOT_STYLE['color_kalman'], label='Filtr Kalmana (χ²)')
    ax.fill(angles, kalman_stats, color=PLOT_STYLE['color_kalman'], alpha=0.25)
    
    # Plot XGBoost
    ax.plot(angles, xgb_stats, linewidth=2, linestyle='solid', color=PLOT_STYLE['color_xgb'], label='XGBoost')
    ax.fill(angles, xgb_stats, color=PLOT_STYLE['color_xgb'], alpha=0.1)
    
    # Plot LightGBM
    ax.plot(angles, lgbm_stats, linewidth=2, linestyle='solid', color=PLOT_STYLE['color_lgbm'], label='LightGBM')
    ax.fill(angles, lgbm_stats, color=PLOT_STYLE['color_lgbm'], alpha=0.1)
    
    ax.set_title('Kompleksowe Porównanie Algorytmów', size=PLOT_STYLE['font_title']+2, fontweight='bold', y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    fig.savefig(save_path)
    print(f"Wygenerowano: {save_path}")
    plt.close()

if __name__ == '__main__':
    plot_signal_drift()
    plot_latency_comparison()
    plot_radar_comparison()
    print("Zakończono tworzenie wizualizacji.")
