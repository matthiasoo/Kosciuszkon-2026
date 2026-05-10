import numpy as np
import matplotlib.pyplot as plt
from math import pi

PLOT_STYLE = {
    'bg_color': '#1a1a2e',
    'text_color': '#e0e0e0',
    'grid_color': '#333355',
    'color_clean': '#4fc3f7',
    'color_spoof': '#ef5350',
    'color_kalman': '#66bb6a',
    'color_xgb': '#ffa726',
    'color_lgbm': '#ab47bc',
    'font_title': 14,
    'font_label': 11,
    'font_tick': 9,
    'dpi': 130,
}

def setup_plot_style():
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

# Szeregi czasowe
def plot_signal_drift(save_path='../visualization/plot_1_signal_drift.png'):
    setup_plot_style()
    
    t = np.linspace(0, 100, 500)
    true_pos = np.sin(t * 0.1) * 20 + t * 0.5

    spoofed_pos = true_pos.copy()
    drift_start = 40
    drift_mask = t > drift_start
    spoofed_pos[drift_mask] += 0.05 * (t[drift_mask] - drift_start)**2
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(t, true_pos, label='Prawdziwa Pozycja (EKF/IMU)', color=PLOT_STYLE['color_clean'], linewidth=2.5)
    ax.plot(t, spoofed_pos, label='Pozycja z zakłóconego GPS', color=PLOT_STYLE['color_spoof'], linewidth=2.5, linestyle='--')
    
    # Filtr Kalmana
    detection_time = 43
    ax.axvspan(detection_time + 6, 100, color=PLOT_STYLE['color_spoof'], alpha=0.15, label='Aura detekcji: Alarm Filtr Kalmana (χ²)')

    ax.axvline(x=detection_time + 6, color=PLOT_STYLE['color_kalman'], linestyle=':', linewidth=2)
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

# Opóźnienia
def plot_latency_comparison(save_path='../visualization/plot_2_latency.png'):
    setup_plot_style()
    
    models = ['Filtr Kalmana (χ²)', 'LightGBM', 'XGBoost']
    latencies_us = [2.5, 450.0, 1200.0] 
    colors = [PLOT_STYLE['color_kalman'], PLOT_STYLE['color_lgbm'], PLOT_STYLE['color_xgb']]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    bars = ax.bar(models, latencies_us, color=colors, alpha=0.85, width=0.5)
    
    ax.set_title('Zapotrzebowanie Obliczeniowe (Latencja Infrerencji per próbka)', fontweight='bold')
    ax.set_ylabel('Czas wykonania [µs] (Skala logarytmiczna)')

    ax.set_yscale('log')

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height} µs',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', color=PLOT_STYLE['text_color'], fontweight='bold')

    plt.tight_layout()
    fig.savefig(save_path)
    print(f"Wygenerowano: {save_path}")
    plt.close()

# Porównanie modeli
def plot_radar_comparison(save_path='../visualization/plot_3_radar.png'):
    setup_plot_style()
    
    categories = ['Dokładność (F1 Score)', 'Szybkość Obliczeń', 'Brak Zapotrzebowania\nna Dane (No-Training)', 'Wyjaśnialność\n(Explainability)']
    N = len(categories)

    kalman_stats = [9.0, 10.0, 10.0, 10.0]

    xgb_stats = [6.0, 6.0, 3.0, 5.0]

    lgbm_stats = [6.5, 7.5, 3.0, 5.0]

    kalman_stats += kalman_stats[:1]
    xgb_stats += xgb_stats[:1]
    lgbm_stats += lgbm_stats[:1]
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    plt.xticks(angles[:-1], categories, color=PLOT_STYLE['text_color'], size=10, fontweight='bold')
    ax.set_rlabel_position(0)
    plt.yticks([2, 4, 6, 8, 10], ["2", "4", "6", "8", "10"], color="grey", size=8)
    plt.ylim(0, 10)
    
    # kalman
    ax.plot(angles, kalman_stats, linewidth=2, linestyle='solid', color=PLOT_STYLE['color_kalman'], label='Filtr Kalmana (χ²)')
    ax.fill(angles, kalman_stats, color=PLOT_STYLE['color_kalman'], alpha=0.25)
    
    # xgboost
    ax.plot(angles, xgb_stats, linewidth=2, linestyle='solid', color=PLOT_STYLE['color_xgb'], label='XGBoost')
    ax.fill(angles, xgb_stats, color=PLOT_STYLE['color_xgb'], alpha=0.1)
    
    # lightgbm
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
