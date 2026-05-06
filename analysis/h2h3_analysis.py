"""
Анализ гипотез H2 и H3
-----------------------
H2: Существует пороговый взнос в ПДС, выше которого предельная выгода от
    со-финансирования перестаёт перекрывать комиссии → стимул перейти в ИИС-3.
H3: Со-финансирование создаёт убывающую доходность при росте взносов внутри ПДС.

Логика проверки
  H2 — advantage(r) = E[ROI_ПДС(r)] − E[ROI_ИИС3(r)]
       Должна иметь перевёрнутую U-форму: растёт до cap-порога, затем убывает.
       Точка перегиба сравнивается с теоретическим порогом.
  H3 — d(E[ROI_ПДС])/d(r) должна стать отрицательной после порога насыщения.
       Проверяется полиномиальной регрессией 2-й степени: знак коэфф. при r² < 0.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.lines as mlines
import seaborn as sns
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

DATA_RAW   = os.path.join(os.path.dirname(__file__), '..', 'scenarios', 'temp_data', 'h2h3_raw.csv')
DATA_INF   = os.path.join(os.path.dirname(__file__), '..', 'scenarios', 'temp_data', 'h2h3_inflection_points.csv')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'scenarios', 'temp_data', 'figures')

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.05)

SALARY_LABELS = {
    50_000:  '50 тыс. ₽\n(ниже 80 тыс.)',
    100_000: '100 тыс. ₽\n(80–150 тыс.)',
    150_000: '150 тыс. ₽\n(150+ тыс.)',
    200_000: '200 тыс. ₽\n(150+ тыс.)',
}

COLOR_PDS  = '#2166ac'
COLOR_IIS  = '#d73027'
COLOR_DIFF = '#4dac26'


def load_data():
    for p in [DATA_RAW, DATA_INF]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Файл не найден: {p}\n"
                "Сначала запустите: python -m scenarios.h2h3_cofin_threshold"
            )
    df  = pd.read_csv(DATA_RAW)
    inf = pd.read_csv(DATA_INF)
    return df, inf


# ─────────────────────────── Figure 1 ────────────────────────────────────────
def plot_roi_curves(df: pd.DataFrame, inf: pd.DataFrame):
    """
    E[ROI_ПДС] и E[ROI_ИИС3] vs payment_rate — для каждой зарплаты.
    Вертикальная линия = аналитический порог со-финансирования.
    """
    salary_vals = sorted(df['salary'].unique())
    n_sal = len(salary_vals)
    fig, axes = plt.subplots(1, n_sal, figsize=(4.5 * n_sal, 4.5), sharey=False)
    if n_sal == 1:
        axes = [axes]
    fig.suptitle('H2/H3 — E[ROI] ПДС и ИИС-3 в зависимости от ставки взноса',
                 fontsize=13, y=1.02)

    for ax, salary in zip(axes, salary_vals):
        sub  = df[df['salary'] == salary]
        agg  = sub.groupby('payment_rate')[['roi_pds', 'roi_iis']].mean().reset_index()
        thresh = inf.loc[inf['salary'] == salary, 'analytical_threshold_rate'].values[0]

        ax.plot(agg['payment_rate'] * 100, agg['roi_pds'],
                color=COLOR_PDS, lw=2, label='ПДС')
        ax.plot(agg['payment_rate'] * 100, agg['roi_iis'],
                color=COLOR_IIS, lw=2, linestyle='--', label='ИИС-3 (то же доходн.)')
        ax.axvline(thresh * 100, color='#555', linestyle=':', lw=1.4,
                   label=f'Аналит. порог {thresh:.0%}')

        ax.set_title(SALARY_LABELS.get(salary, f'{salary//1000} тыс.'), fontsize=10)
        ax.set_xlabel('Ставка взноса (%)')
        if ax is axes[0]:
            ax.set_ylabel('E[ROI]')
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'h2h3_roi_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


# ─────────────────────────── Figure 2 ────────────────────────────────────────
def plot_advantage_curve(df: pd.DataFrame, inf: pd.DataFrame):
    """
    advantage = E[ROI_ПДС] − E[ROI_ИИС3] с отметками точек перегиба.
    H2 проверяется: кривая имеет перевёрнутую U-форму и пересекает 0.
    """
    salary_vals = sorted(df['salary'].unique())
    n_sal = len(salary_vals)
    fig, axes = plt.subplots(1, n_sal, figsize=(4.5 * n_sal, 4.5), sharey=False)
    if n_sal == 1:
        axes = [axes]
    fig.suptitle('H2 — Преимущество ПДС над ИИС-3: advantage = E[ROI_ПДС] − E[ROI_ИИС-3]',
                 fontsize=12, y=1.02)

    for ax, salary in zip(axes, salary_vals):
        sub  = df[df['salary'] == salary]
        agg  = sub.groupby('payment_rate')[['roi_pds', 'roi_iis']].mean().reset_index()
        agg['advantage'] = agg['roi_pds'] - agg['roi_iis']

        inf_row = inf[inf['salary'] == salary].iloc[0]
        thresh  = inf_row['analytical_threshold_rate']
        peak_r  = inf_row['h2_advantage_peak_rate']
        zero_r  = inf_row['h2_zero_cross_rate']

        ax.fill_between(agg['payment_rate'] * 100, agg['advantage'], 0,
                        where=agg['advantage'] >= 0, alpha=0.20, color=COLOR_PDS, label='ПДС лучше')
        ax.fill_between(agg['payment_rate'] * 100, agg['advantage'], 0,
                        where=agg['advantage'] < 0, alpha=0.20, color=COLOR_IIS, label='ИИС-3 лучше')
        ax.plot(agg['payment_rate'] * 100, agg['advantage'],
                color=COLOR_DIFF, lw=2)
        ax.axhline(0, color='#333', lw=0.8)

        # Точки перегиба
        ax.axvline(thresh * 100, color='#555', linestyle=':', lw=1.4,
                   label=f'Аналит. порог {thresh:.0%}')
        peak_adv = agg.loc[(agg['payment_rate'] - peak_r).abs().idxmin(), 'advantage']
        ax.scatter([peak_r * 100], [peak_adv], color='#333', zorder=5, s=60,
                   label=f'Пик {peak_r:.0%}')
        if zero_r is not None:
            ax.axvline(zero_r * 100, color=COLOR_IIS, linestyle='--', lw=1.2,
                       label=f'Пересечение 0: {zero_r:.0%}')

        ax.set_title(SALARY_LABELS.get(salary, f'{salary//1000} тыс.'), fontsize=10)
        ax.set_xlabel('Ставка взноса (%)')
        if ax is axes[0]:
            ax.set_ylabel('advantage (ROI_ПДС − ROI_ИИС3)')
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
        ax.legend(fontsize=7.5)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'h2h3_advantage.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


# ─────────────────────────── Figure 3 ────────────────────────────────────────
def plot_diminishing_returns(df: pd.DataFrame):
    """
    H3 — E[ROI_ПДС] с квадратичной аппроксимацией.
    Убывающая доходность = отрицательный коэффициент при r².
    """
    salary_vals = sorted(df['salary'].unique())
    n_sal = len(salary_vals)
    fig, axes = plt.subplots(1, n_sal, figsize=(4.5 * n_sal, 4.5), sharey=False)
    if n_sal == 1:
        axes = [axes]
    fig.suptitle('H3 — Убывающая доходность ПДС: E[ROI_ПДС] и квадратичный тренд',
                 fontsize=12, y=1.02)

    poly_results = {}
    for ax, salary in zip(axes, salary_vals):
        sub = df[df['salary'] == salary]
        agg = sub.groupby('payment_rate')['roi_pds'].mean().reset_index()
        x   = agg['payment_rate'].values * 100   # в %
        y   = agg['roi_pds'].values

        # Квадратичная регрессия
        coeffs = np.polyfit(x, y, deg=2)
        poly_results[salary] = coeffs
        x_fit  = np.linspace(x.min(), x.max(), 200)
        y_fit  = np.polyval(coeffs, x_fit)

        ax.plot(x, y, color=COLOR_PDS, lw=1.8, label='E[ROI_ПДС]')
        ax.plot(x_fit, y_fit, color='#e6550d', linestyle='--', lw=1.4,
                label=f'Квадр. тренд\na₂={coeffs[0]:.4f}')
        ax.fill_between(x, y, y.min(), alpha=0.10, color=COLOR_PDS)

        concave = coeffs[0] < 0
        mark = '✓ вогнутая' if concave else '✗ не вогнутая'
        ax.set_title(f'{SALARY_LABELS.get(salary, salary)} | {mark}', fontsize=9.5)
        ax.set_xlabel('Ставка взноса (%)')
        if ax is axes[0]:
            ax.set_ylabel('E[ROI_ПДС]')
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'h2h3_diminishing.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")
    return poly_results


# ─────────────────────────── Figure 4 ────────────────────────────────────────
def plot_inflection_table(inf: pd.DataFrame):
    """Сводная таблица точек перегиба + визуализация."""
    fig, ax = plt.subplots(figsize=(11, 2.8))
    ax.axis('off')
    disp = inf[[
        'salary', 'analytical_threshold_rate',
        'h2_advantage_peak_rate', 'h2_zero_cross_rate',
        'h3_diminishing_return_rate',
    ]].copy()
    disp.columns = [
        'Зарплата (₽)', 'Аналит. порог', 'H2: пик пр-ства',
        'H2: пересеч. 0', 'H3: убыв. дох-ть',
    ]
    for c in disp.columns[1:]:
        disp[c] = disp[c].apply(
            lambda x: f'{float(x):.1%}' if pd.notna(x) and x != 'None' and str(x) != 'None'
            else '—'
        )
    disp['Зарплата (₽)'] = disp['Зарплата (₽)'].apply(lambda x: f'{int(x):,}')

    tbl = ax.table(
        cellText=disp.values,
        colLabels=disp.columns,
        cellLoc='center', loc='center',
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#2166ac')
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#f7f7f7')

    ax.set_title('H2/H3 — Сводная таблица численных точек перегиба',
                 fontsize=11, pad=8)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'h2h3_inflection_table.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


# ─────────────────────────── Verdict ─────────────────────────────────────────
def print_verdict(df: pd.DataFrame, inf: pd.DataFrame, poly_results: dict):
    print('\n' + '=' * 65)
    print('  ВЕРДИКТ: ГИПОТЕЗЫ H2 и H3')
    print('=' * 65)

    print('\nH2 — Пороговый взнос:')
    print(f'  {"Зарплата":>10} | {"Аналит.порог":>13} | {"Эмпир.пик":>10} | {"Пересеч.0":>10} | {"Подтв."}')
    print('  ' + '-' * 58)
    h2_scores = []
    h2_peak_scores = []
    for _, row in inf.iterrows():
        peak_r = row['h2_advantage_peak_rate']
        zero_r = row['h2_zero_cross_rate']
        # zero_r is NaN (stored as CSV null) when ИИС-3 never catches up
        zero_valid = pd.notna(zero_r)
        peak_valid = pd.notna(peak_r)
        confirmed = peak_valid and zero_valid
        h2_scores.append(confirmed)
        h2_peak_scores.append(peak_valid)
        z_str = f'{float(zero_r):.1%}' if zero_valid else '—'
        print(f"  {int(row['salary']):>10,} | {row['analytical_threshold_rate']:>13.1%} | "
              f"{peak_r:>10.1%} | {z_str:>10} | {'✓' if confirmed else ('△' if peak_valid else '✗')}")

    n_full = sum(h2_scores)
    n_peak = sum(h2_peak_scores)
    if n_full > len(h2_scores) / 2:
        verdict = 'ПОДТВЕРЖДАЕТСЯ (полностью)'
    elif n_peak >= len(h2_peak_scores) / 2:
        verdict = 'ЧАСТИЧНО ПОДТВЕРЖДАЕТСЯ (пик найден, нет пересечения нуля у высоких зарплат)'
    else:
        verdict = 'НЕ ПОДТВЕРЖДАЕТСЯ'
    print(f'\n  → H2 {verdict}')
    print(f'     пик преимущества: {n_peak}/{len(h2_peak_scores)} групп | '
          f'пересечение нуля: {n_full}/{len(h2_scores)} групп')

    print('\nH3 — Убывающая доходность ПДС (знак коэфф. a₂ квадр. регрессии):')
    print(f'  {"Зарплата":>10} | {"a₂":>8} | {"Вогнутость":>12}')
    print('  ' + '-' * 37)
    h3_scores = []
    for salary, coeffs in poly_results.items():
        a2 = coeffs[0]
        concave = a2 < 0
        h3_scores.append(concave)
        print(f'  {salary:>10,} | {a2:>8.5f} | {"✓ вогнутая" if concave else "✗ выпуклая"}')

    print(f'\n  → H3 {"ПОДТВЕРЖДАЕТСЯ" if sum(h3_scores) >= len(h3_scores) / 2 else "НЕ ПОДТВЕРЖДАЕТСЯ"}: '
          f'убывающая доходность в {sum(h3_scores)}/{len(h3_scores)} группах зарплат.')
    print('=' * 65)


if __name__ == '__main__':
    print("Загружаю данные H2/H3...")
    df, inf = load_data()
    print(f"  Строк: {len(df)}, payment_rate: {df['payment_rate'].nunique()} уровней")

    print("Строю графики...")
    plot_roi_curves(df, inf)
    plot_advantage_curve(df, inf)
    poly_results = plot_diminishing_returns(df)
    plot_inflection_table(inf)

    print_verdict(df, inf, poly_results)
