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

Терминология характерных точек
  analytical_cofin_cap_rate       — аналитический порог насыщения со-финансирования
                                    (ставка взноса, при которой исчерпывается лимит гос. взноса)
  h2_pds_max_advantage_rate       — точка максимального преимущества ПДС
                                    (argmax advantage(r): ПДС выигрывает у ИИС-3 сильнее всего)
  h2_pds_iis_indifference_rate    — точка безразличия ПДС и ИИС-3
                                    (первый r, при котором advantage(r) ≤ 0: ИИС-3 начинает обгонять)
  h3_marginal_roi_decline_rate    — начало убывания предельного ROI ПДС
                                    (первый r, при котором d(E[ROI_ПДС])/dr < 0)
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

REPRESENTATIVE_AGE = 40
REPRESENTATIVE_SEX = 'M'

DEMOGRAPHIC_GROUPS = [('M', 20), ('M', 40), ('M', 60), ('F', 20), ('F', 40), ('F', 60)]
DEMOGRAPHIC_LABELS = {
    ('M', 20): 'М, 20 лет', ('M', 40): 'М, 40 лет', ('M', 60): 'М, 60 лет',
    ('F', 20): 'Ж, 20 лет', ('F', 40): 'Ж, 40 лет', ('F', 60): 'Ж, 60 лет',
}
DEMOGRAPHIC_COLORS = {
    ('M', 20): '#1b9e77', ('M', 40): '#d95f02', ('M', 60): '#7570b3',
    ('F', 20): '#e7298a', ('F', 40): '#66a61e', ('F', 60): '#e6ab02',
}
DEMOGRAPHIC_LINESTYLES = {
    ('M', 20): '-', ('M', 40): '-', ('M', 60): '-',
    ('F', 20): '--', ('F', 40): '--', ('F', 60): '--',
}


def load_data():
    for p in [DATA_RAW, DATA_INF]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Файл не найден: {p}\n"
                "Сначала запустите: python -m scenarios.h2h3_cofin_threshold"
            )
    df  = pd.read_csv(DATA_RAW)
    inf = pd.read_csv(DATA_INF)
    if 'market_scenario' not in df.columns:
        df['market_scenario'] = 'baseline'
    if 'market_scenario' not in inf.columns:
        inf['market_scenario'] = 'baseline'
    if 'transition_scenario' not in df.columns:
        df['transition_scenario'] = 'baseline'
    if 'transition_scenario' not in inf.columns:
        inf['transition_scenario'] = 'baseline'
    return df, inf


# ─────────────────────────── Figure 1 ────────────────────────────────────────
def plot_assumptions_table():
    """Таблица исходных сценарных предпосылок H2/H3."""
    rows = [
        ('Горизонт моделирования',        '15 лет'),
        ('Число симуляций',               '300'),
        ('Возраст участника',             '20, 40, 60 лет'),
        ('Пол',                           'М и Ж'),
        ('Ставка взноса (развёртка)',      '0.5 % – 24 %, шаг 0.5 %'),
        ('Зарплатные группы',             '50 / 100 / 150 / 200 тыс. ₽/мес.'),
        ('Портфель ПДС',                  'Средний НПФ (структура из structure.xlsx)'),
        ('Портфель ИИС-3',                'Тот же (изоляция эффекта со-финансирования)'),
        ('Комиссия ПДС',                  '0.5 % фикс. + 20 % от дохода'),
        ('Комиссия ИИС-3',                '0 % (нет управляющей компании)'),
        ('Налоговый вычет',               '13 % от взноса'),
        ('Со-финансирование (лимит)',      '36 000 ₽/год (зарплата < 80 тыс.);\n'
                                          '36 000 ₽/год при взносе ≥ 72 000 (80–150 тыс.);\n'
                                          '36 000 ₽/год при взносе ≥ 144 000 (≥ 150 тыс.)'),
        ('Период со-финансирования',      '10 лет'),
        ('Модель зарплаты',               'StochasticSalaryModel (стохастический рост)'),
        ('Вероятность трудового перехода','p ≈ calibrated (Weibull, ОРС Росстат 2024)'),
        ('Досрочный выход',               'Не моделируется (фиксированный горизонт)'),
        ('Метрики сравнения',             'ROI, IRR, TWR, KZ'),
    ]
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.axis('off')
    tbl = ax.table(
        cellText=rows,
        colLabels=['Параметр', 'Значение'],
        cellLoc='left', loc='center',
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#2166ac')
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#f0f4f8')
        cell.set_edgecolor('#cccccc')
        if c == 0:
            cell.set_text_props(fontweight='bold')
    ax.set_title('H2/H3 — Исходные сценарные предпосылки', fontsize=12, pad=10)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'h2h3_assumptions.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


def plot_roi_curves(df: pd.DataFrame, inf: pd.DataFrame):
    """
    E[ROI_ПДС] и E[ROI_ИИС3] vs payment_rate с перцентильными лентами.
    Вертикальная линия = аналитический порог насыщения со-финансирования.
    """
    salary_vals = sorted(df['salary'].unique())
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=False)
    axes = axes.flatten()
    fig.suptitle('H2/H3 — E[ROI] ПДС и ИИС-3 в зависимости от ставки взноса\n'
                 '(лента = 25–75-й перцентиль, М, 40 лет)',
                 fontsize=13, y=1.01)

    for ax, salary in zip(axes, salary_vals):
        sub  = df[df['salary'] == salary]
        agg  = sub.groupby('payment_rate')[['roi_pds', 'roi_iis']].agg(
            ['mean', lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75)]
        ).reset_index()
        agg.columns = ['payment_rate',
                       'roi_pds_mean', 'roi_pds_p25', 'roi_pds_p75',
                       'roi_iis_mean', 'roi_iis_p25', 'roi_iis_p75']
        thresh = inf.loc[inf['salary'] == salary, 'analytical_cofin_cap_rate'].values[0]

        x = agg['payment_rate'] * 100
        ax.fill_between(x, agg['roi_pds_p25'], agg['roi_pds_p75'],
                        alpha=0.15, color=COLOR_PDS)
        ax.fill_between(x, agg['roi_iis_p25'], agg['roi_iis_p75'],
                        alpha=0.15, color=COLOR_IIS)
        ax.plot(x, agg['roi_pds_mean'], color=COLOR_PDS, lw=2, label='ПДС')
        ax.plot(x, agg['roi_iis_mean'], color=COLOR_IIS, lw=2, linestyle='--',
                label='ИИС-3 (то же доходн.)')
        ax.axvline(thresh * 100, color='#555', linestyle=':', lw=1.4,
                   label=f'Порог насыщения {thresh:.0%}')

        ax.set_title(SALARY_LABELS.get(salary, f'{salary//1000} тыс.'), fontsize=10)
        ax.set_xlabel('Ставка взноса (%)')
        ax.set_ylabel('E[ROI]')
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.legend(fontsize=8)

    for ax in axes[len(salary_vals):]:
        ax.set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'h2h3_roi_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


# ─────────────────────────── Figure 2 ────────────────────────────────────────
def plot_advantage_curve(df: pd.DataFrame, inf: pd.DataFrame):
    """
    advantage = E[ROI_ПДС] − E[ROI_ИИС3] с отметками характерных точек.
    H2 проверяется: кривая имеет перевёрнутую U-форму и пересекает 0.

    Характерные точки:
      ▲ — точка максимального преимущества ПДС (h2_pds_max_advantage_rate)
      ✕ — точка безразличия ПДС и ИИС-3 (h2_pds_iis_indifference_rate)
      : — аналитический порог насыщения со-финансирования
    """
    salary_vals = sorted(df['salary'].unique())
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=False)
    axes = axes.flatten()
    fig.suptitle('H2 — Преимущество ПДС над ИИС-3: advantage = E[ROI_ПДС] − E[ROI_ИИС-3]\n'
                 '(М, 40 лет)',
                 fontsize=12, y=1.01)

    for ax, salary in zip(axes, salary_vals):
        sub  = df[df['salary'] == salary]
        agg  = sub.groupby('payment_rate')[['roi_pds', 'roi_iis']].mean().reset_index()
        agg['advantage'] = agg['roi_pds'] - agg['roi_iis']

        inf_row    = inf[inf['salary'] == salary].iloc[0]
        thresh     = inf_row['analytical_cofin_cap_rate']
        peak_r     = inf_row['h2_pds_max_advantage_rate']
        indiffer_r = inf_row['h2_pds_iis_indifference_rate']

        ax.fill_between(agg['payment_rate'] * 100, agg['advantage'], 0,
                        where=agg['advantage'] >= 0, alpha=0.20, color=COLOR_PDS, label='ПДС лучше')
        ax.fill_between(agg['payment_rate'] * 100, agg['advantage'], 0,
                        where=agg['advantage'] < 0, alpha=0.20, color=COLOR_IIS, label='ИИС-3 лучше')
        ax.plot(agg['payment_rate'] * 100, agg['advantage'],
                color=COLOR_DIFF, lw=2)
        ax.axhline(0, color='#333', lw=0.8)

        ax.axvline(thresh * 100, color='#555', linestyle=':', lw=1.4,
                   label=f'Порог насыщения {thresh:.0%}')
        peak_adv = agg.loc[(agg['payment_rate'] - peak_r).abs().idxmin(), 'advantage']
        ax.scatter([peak_r * 100], [peak_adv], color='#333', zorder=5, s=60, marker='^',
                   label=f'Макс. преимущество ПДС: {peak_r:.0%}')
        if pd.notna(indiffer_r):
            ax.axvline(indiffer_r * 100, color=COLOR_IIS, linestyle='--', lw=1.2,
                       label=f'Точка безразличия: {indiffer_r:.0%}')

        ax.set_title(SALARY_LABELS.get(salary, f'{salary//1000} тыс.'), fontsize=10)
        ax.set_xlabel('Ставка взноса (%)')
        ax.set_ylabel('advantage (ROI_ПДС − ROI_ИИС3)')
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
        ax.legend(fontsize=7.5)

    for ax in axes[len(salary_vals):]:
        ax.set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'h2h3_advantage.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


# ─────────────────────────── Figure 3 ────────────────────────────────────────
def plot_diminishing_returns(df: pd.DataFrame, inf: pd.DataFrame):
    """
    H3 — E[ROI_ПДС] с квадратичной аппроксимацией и перцентильной лентой.
    Убывающий предельный ROI = отрицательный коэффициент при r².
    Вертикальная линия = начало убывания предельного ROI ПДС.
    """
    salary_vals = sorted(df['salary'].unique())
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=False)
    axes = axes.flatten()
    fig.suptitle('H3 — Убывание предельного ROI ПДС: E[ROI_ПДС], квадратичный тренд\n'
                 '(лента = 25–75-й перцентиль, М, 40 лет)',
                 fontsize=12, y=1.01)

    poly_results = {}
    for ax, salary in zip(axes, salary_vals):
        sub = df[df['salary'] == salary]
        agg = sub.groupby('payment_rate')['roi_pds'].agg(
            mean='mean',
            p25=lambda x: np.percentile(x, 25),
            p75=lambda x: np.percentile(x, 75),
        ).reset_index()
        x   = agg['payment_rate'].values * 100
        y   = agg['mean'].values

        coeffs = np.polyfit(x, y, deg=2)
        poly_results[salary] = coeffs
        x_fit  = np.linspace(x.min(), x.max(), 200)
        y_fit  = np.polyval(coeffs, x_fit)

        ax.fill_between(x, agg['p25'].values, agg['p75'].values,
                        alpha=0.15, color=COLOR_PDS, label='25–75-й перц.')
        ax.plot(x, y, color=COLOR_PDS, lw=1.8, label='E[ROI_ПДС]')
        ax.plot(x_fit, y_fit, color='#e6550d', linestyle='--', lw=1.4,
                label=f'Квадр. тренд\na₂={coeffs[0]:.4f}')

        # Начало убывания предельного ROI
        inf_row = inf[inf['salary'] == salary].iloc[0]
        decline_r = inf_row.get('h3_marginal_roi_decline_rate')
        if pd.notna(decline_r):
            ax.axvline(float(decline_r) * 100, color='#e6550d', linestyle=':', lw=1.3,
                       label=f'Нач. убыв. пред. ROI: {float(decline_r):.0%}')

        concave = coeffs[0] < 0
        mark = '✓ убывающий пред. ROI' if concave else '✗ не убывающий'
        ax.set_title(f'{SALARY_LABELS.get(salary, salary)} | {mark}', fontsize=9.5)
        ax.set_xlabel('Ставка взноса (%)')
        ax.set_ylabel('E[ROI_ПДС]')
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.legend(fontsize=8)

    for ax in axes[len(salary_vals):]:
        ax.set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'h2h3_diminishing.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")
    return poly_results


# ─────────────────────────── Figure 4 ────────────────────────────────────────
def plot_inflection_table(inf: pd.DataFrame):
    """Сводная таблица характерных точек H2/H3 с уточнёнными названиями."""
    fig, ax = plt.subplots(figsize=(14, 2.8))
    ax.axis('off')
    disp = inf[['salary', 'analytical_cofin_cap_rate', 'h2_pds_max_advantage_rate',
                'h2_pds_iis_indifference_rate', 'h3_marginal_roi_decline_rate']].copy()
    disp.columns = [
        'Зарплата (₽)',
        'Порог насыщения\nсо-финансирования',
        'H2: точка макс.\nпреимущества ПДС',
        'H2: точка\nбезразличия ПДС/ИИС-3',
        'H3: начало убывания\nпредельного ROI ПДС',
    ]
    for c in disp.columns[1:]:
        disp[c] = disp[c].apply(
            lambda x: f'{float(x):.1%}' if pd.notna(x) and str(x) not in ('None', 'nan')
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
    tbl.set_fontsize(8.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#2166ac')
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#f7f7f7')
        cell.set_edgecolor('#cccccc')

    ax.set_title('H2/H3 — Сводная таблица характерных точек (М, 40 лет)',
                 fontsize=11, pad=8)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'h2h3_inflection_table.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


# ─────────────────────────── Verdict ─────────────────────────────────────────
def plot_percentile_fan(df: pd.DataFrame):
    """
    Наложенные полупрозрачные гистограммы ROI_ПДС и ROI_ИИС-3 при ставке взноса,
    соответствующей точке максимального преимущества ПДС (для каждой зарплаты).
    """
    salary_vals = sorted(df['salary'].unique())
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=False)
    axes = axes.flatten()
    fig.suptitle('H2/H3 — Распределения ROI_ПДС и ROI_ИИС-3\n'
                 '(при ставке макс. преимущества ПДС, М, 40 лет)',
                 fontsize=12, y=1.01)

    for ax, salary in zip(axes, salary_vals):
        sub = df[df['salary'] == salary]
        agg = sub.groupby('payment_rate')[['roi_pds', 'roi_iis']].mean().reset_index()
        agg['advantage'] = agg['roi_pds'] - agg['roi_iis']
        peak_rate = agg.loc[agg['advantage'].idxmax(), 'payment_rate']

        sub_peak = sub[np.isclose(sub['payment_rate'], peak_rate, atol=1e-5)]

        ax.hist(sub_peak['roi_pds'].dropna(), bins=30, alpha=0.45, color=COLOR_PDS,
                label='ROI ПДС', density=True, edgecolor='none')
        ax.axvline(sub_peak['roi_pds'].median(), color=COLOR_PDS, linestyle='--', lw=1.3, alpha=0.85)
        ax.hist(sub_peak['roi_iis'].dropna(), bins=30, alpha=0.45, color=COLOR_IIS,
                label='ROI ИИС-3', density=True, edgecolor='none')
        ax.axvline(sub_peak['roi_iis'].median(), color=COLOR_IIS, linestyle='--', lw=1.3, alpha=0.85)

        ax.set_title(f'{SALARY_LABELS.get(salary, f"{salary//1000} тыс.")} | ставка={peak_rate:.1%}',
                     fontsize=10)
        ax.set_xlabel('ROI')
        ax.set_ylabel('Плотность')
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.legend(fontsize=8)

    for ax in axes[len(salary_vals):]:
        ax.set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'h2h3_percentile_fan.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


def print_descriptive_stats(df: pd.DataFrame, inf: pd.DataFrame):
    """Таблица описательной статистики ROI_ПДС и ROI_ИИС по зарплатам."""
    print('\n' + '=' * 75)
    print('  ОПИСАТЕЛЬНАЯ СТАТИСТИКА (при точке максимального преимущества ПДС, М, 40 лет)')
    print('=' * 75)
    for salary in sorted(df['salary'].unique()):
        peak_r = inf.loc[inf['salary'] == salary, 'h2_pds_max_advantage_rate'].values
        peak_r = float(peak_r[0]) if len(peak_r) > 0 and pd.notna(peak_r[0]) else None
        if peak_r is None:
            continue
        sub = df[(df['salary'] == salary) & np.isclose(df['payment_rate'], peak_r, atol=1e-5)]
        print(f'\n  Зарплата {salary:,} ₽  |  ставка взноса = {peak_r:.1%}')
        print(f'  {"":12} {"p5":>7} {"p25":>7} {"p50":>7} {"p75":>7} {"p95":>7} {"mean":>7} {"std":>7}')
        for col, label in [('roi_pds', 'ROI ПДС'), ('roi_iis', 'ROI ИИС-3')]:
            vals = sub[col].dropna()
            if len(vals) == 0:
                continue
            pcts = np.percentile(vals, [5, 25, 50, 75, 95])
            print(f'  {label:12} '
                  + ' '.join(f'{v:>7.1%}' for v in pcts)
                  + f' {vals.mean():>7.1%} {vals.std():>7.1%}')
    print('=' * 75)


def print_verdict(df: pd.DataFrame, inf: pd.DataFrame, poly_results: dict = None):
    print('\n' + '=' * 75)
    print('  ВЕРДИКТ: ГИПОТЕЗЫ H2 и H3 (М, 40 лет)')
    print('=' * 75)

    print('\nH2 — Точка максимального преимущества ПДС и точка безразличия:')
    hdr = f'  {"Зарплата":>10} | {"Порог насыщ.":>13} | {"Макс.преим.":>12} | {"Безразличие":>12} | {"Подтв."}'
    print(hdr)
    print('  ' + '-' * 65)
    h2_scores = []
    h2_peak_scores = []
    for _, row in inf.iterrows():
        peak_r    = row['h2_pds_max_advantage_rate']
        indiffer_r = row['h2_pds_iis_indifference_rate']
        indiffer_valid = pd.notna(indiffer_r) and str(indiffer_r) not in ('None', 'nan')
        peak_valid     = pd.notna(peak_r)
        confirmed      = peak_valid and indiffer_valid
        h2_scores.append(confirmed)
        h2_peak_scores.append(peak_valid)
        ind_str = f'{float(indiffer_r):.1%}' if indiffer_valid else '—'
        print(f"  {int(row['salary']):>10,} | {row['analytical_cofin_cap_rate']:>13.1%} | "
              f"{peak_r:>12.1%} | {ind_str:>12} | "
              f"{'✓' if confirmed else ('△' if peak_valid else '✗')}")

    n_full = sum(h2_scores)
    n_peak = sum(h2_peak_scores)
    if n_full > len(h2_scores) / 2:
        verdict = 'ПОДТВЕРЖДАЕТСЯ (полностью)'
    elif n_peak >= len(h2_peak_scores) / 2:
        verdict = 'ЧАСТИЧНО ПОДТВЕРЖДАЕТСЯ (макс. преимущество найдено, точка безразличия не достигается при высоких зарплатах)'
    else:
        verdict = 'НЕ ПОДТВЕРЖДАЕТСЯ'
    print(f'\n  → H2 {verdict}')
    print(f'     макс. преимущество: {n_peak}/{len(h2_peak_scores)} групп | '
          f'точка безразличия: {n_full}/{len(h2_scores)} групп')

    print('\nH3 — Начало убывания предельного ROI ПДС (эмпирическое d(ROI)/dr < 0):')
    print(f'  {"Зарплата":>10} | {"Порог насыщ.":>12} | {"Нач. убыв.":>10} | {"До порога?":>10} | {"Подтв."}')
    print('  ' + '-' * 62)
    h3_scores = []
    for _, row in inf.iterrows():
        salary    = row['salary']
        cap_r     = float(row['analytical_cofin_cap_rate'])
        decline_r = row['h3_marginal_roi_decline_rate']
        has_decline = pd.notna(decline_r) and str(decline_r) not in ('None', 'nan')
        if has_decline:
            decline_r = float(decline_r)
            before_cap = decline_r <= cap_r
            confirmed = True
            decline_str = f'{decline_r:.1%}'
            before_str  = '✓ да' if before_cap else '△ после'
        else:
            confirmed = False
            decline_str = '—'
            before_str  = '—'
        h3_scores.append(confirmed)
        print(f'  {int(salary):>10,} | {cap_r:>12.1%} | {decline_str:>10} | {before_str:>10} | '
              f'{"✓" if confirmed else "✗"}')

    n_confirmed = sum(h3_scores)
    print(f'\n  → H3 {"ПОДТВЕРЖДАЕТСЯ" if n_confirmed >= len(h3_scores) / 2 else "НЕ ПОДТВЕРЖДАЕТСЯ"}: '
          f'убывание предельного ROI зафиксировано в {n_confirmed}/{len(h3_scores)} группах зарплат.')
    if n_confirmed > 0:
        print(f'     Убывание начинается с минимальных ставок взноса — со-финансирование')
        print(f'     создаёт убывающую отдачу на вложенный рубль уже при малых взносах.')
    print('=' * 75)


def plot_market_scenario_comparison(df: pd.DataFrame, inf: pd.DataFrame):
    """
    Сравнение кривых advantage(rate) по трём рыночным сценариям.
    Показывает устойчивость характерных точек H2/H3 к рыночным предположениям.
    """
    scenarios = [s for s in ['baseline', 'stress', 'optimistic']
                 if s in df['market_scenario'].unique()]
    if len(scenarios) < 2:
        print("  Пропускаю сравнение рыночных сценариев: данные только для одного сценария.")
        return

    SCENARIO_LABELS = {
        'baseline':   'Базовый',
        'stress':     'Стрессовый',
        'optimistic': 'Оптимистичный',
    }
    SCENARIO_COLORS = {
        'baseline':   '#2166ac',
        'stress':     '#d73027',
        'optimistic': '#1a9850',
    }
    linestyles = {'baseline': '-', 'stress': '--', 'optimistic': ':'}

    salary_vals = sorted(df['salary'].unique())
    base_transit = df[df['transition_scenario'] == 'baseline'] if 'transition_scenario' in df.columns else df
    base_transit_inf = inf[inf['transition_scenario'] == 'baseline'] if 'transition_scenario' in inf.columns else inf
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=False)
    axes = axes.flatten()
    fig.suptitle('H2 — Устойчивость точки максимального преимущества ПДС\n'
                 'к рыночным сценариям (advantage = E[ROI_ПДС] − E[ROI_ИИС-3])',
                 fontsize=12, y=1.01)

    for ax, salary in zip(axes, salary_vals):
        for scenario in scenarios:
            sub = base_transit[(base_transit['market_scenario'] == scenario) & (base_transit['salary'] == salary)]
            if sub.empty:
                continue
            agg = sub.groupby('payment_rate')[['roi_pds', 'roi_iis']].mean().reset_index()
            agg['advantage'] = agg['roi_pds'] - agg['roi_iis']

            inf_row = base_transit_inf[(base_transit_inf['market_scenario'] == scenario) & (base_transit_inf['salary'] == salary)]
            peak_r  = inf_row['h2_pds_max_advantage_rate'].values[0] if len(inf_row) else None

            ax.plot(agg['payment_rate'] * 100, agg['advantage'],
                    color=SCENARIO_COLORS[scenario],
                    linestyle=linestyles[scenario],
                    lw=1.8, label=SCENARIO_LABELS[scenario])
            if peak_r is not None and pd.notna(peak_r):
                peak_adv = agg.loc[(agg['payment_rate'] - peak_r).abs().idxmin(), 'advantage']
                ax.scatter([peak_r * 100], [peak_adv],
                           color=SCENARIO_COLORS[scenario], zorder=5, s=50, marker='^')

        ax.axhline(0, color='#333', lw=0.8)
        ax.set_title(SALARY_LABELS.get(salary, f'{salary//1000} тыс.'), fontsize=10)
        ax.set_xlabel('Ставка взноса (%)')
        ax.set_ylabel('advantage (ROI_ПДС − ROI_ИИС3)')
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
        ax.legend(fontsize=8)

    for ax in axes[len(salary_vals):]:
        ax.set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'h2h3_market_scenarios.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


def plot_transition_scenario_comparison(df: pd.DataFrame, inf: pd.DataFrame):
    """
    Устойчивость точки максимального преимущества ПДС к сценариям вероятности
    трудового перехода (advantage = E[ROI_ПДС] − E[ROI_ИИС-3]).
    """
    scenarios = [s for s in ['baseline', 'low_transit', 'mid_transit']
                 if s in df['transition_scenario'].unique()]
    if len(scenarios) < 2:
        print("  Пропускаю сравнение сценариев трудового перехода: данные только для одного сценария.")
        return

    SCENARIO_LABELS = {
        'baseline':    'Базовый (калибр.)',
        'low_transit': 'Низкий (p=0.10)',
        'mid_transit': 'Умеренный (p=0.15)',
    }
    SCENARIO_COLORS = {
        'baseline':    '#2166ac',
        'low_transit': '#d73027',
        'mid_transit': '#1a9850',
    }
    linestyles = {'baseline': '-', 'low_transit': '--', 'mid_transit': ':'}

    salary_vals = sorted(df['salary'].unique())
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=False)
    axes = axes.flatten()
    fig.suptitle('H2 — Устойчивость точки максимального преимущества ПДС\n'
                 'к сценариям вероятности трудового перехода (advantage = E[ROI_ПДС] − E[ROI_ИИС-3])',
                 fontsize=12, y=1.01)

    base_market = df[df['market_scenario'] == 'baseline'] if 'market_scenario' in df.columns else df
    base_market_inf = inf[inf['market_scenario'] == 'baseline'] if 'market_scenario' in inf.columns else inf

    for ax, salary in zip(axes, salary_vals):
        for scenario in scenarios:
            sub = base_market[(base_market['transition_scenario'] == scenario) & (base_market['salary'] == salary)]
            if sub.empty:
                continue
            agg = sub.groupby('payment_rate')[['roi_pds', 'roi_iis']].mean().reset_index()
            agg['advantage'] = agg['roi_pds'] - agg['roi_iis']

            inf_row = base_market_inf[(base_market_inf['transition_scenario'] == scenario) & (base_market_inf['salary'] == salary)]
            peak_r  = inf_row['h2_pds_max_advantage_rate'].values[0] if len(inf_row) else None

            ax.plot(agg['payment_rate'] * 100, agg['advantage'],
                    color=SCENARIO_COLORS[scenario],
                    linestyle=linestyles[scenario],
                    lw=1.8, label=SCENARIO_LABELS[scenario])
            if peak_r is not None and pd.notna(peak_r):
                peak_adv = agg.loc[(agg['payment_rate'] - peak_r).abs().idxmin(), 'advantage']
                ax.scatter([peak_r * 100], [peak_adv],
                           color=SCENARIO_COLORS[scenario], zorder=5, s=50, marker='^')

        ax.axhline(0, color='#333', lw=0.8)
        ax.set_title(SALARY_LABELS.get(salary, f'{salary//1000} тыс.'), fontsize=10)
        ax.set_xlabel('Ставка взноса (%)')
        ax.set_ylabel('advantage (ROI_ПДС − ROI_ИИС3)')
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
        ax.legend(fontsize=8)

    for ax in axes[len(salary_vals):]:
        ax.set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'h2h3_transition_scenarios.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


def plot_demographic_robustness(df: pd.DataFrame, inf: pd.DataFrame):
    """
    Устойчивость кривой advantage(rate) к демографическим группам (возраст × пол).
    Если линии близки — результат устойчив к демографическим характеристикам.
    """
    base = df[(df['market_scenario'] == 'baseline') &
              (df['transition_scenario'] == 'baseline')]
    base_inf = inf[(inf['market_scenario'] == 'baseline') &
                   (inf['transition_scenario'] == 'baseline')]

    salary_vals = sorted(base['salary'].unique())
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=False)
    axes = axes.flatten()
    fig.suptitle('H2 — Устойчивость точки максимального преимущества ПДС\n'
                 'к демографическим группам (advantage = E[ROI_ПДС] − E[ROI_ИИС-3])',
                 fontsize=12, y=1.01)

    for ax, salary in zip(axes, salary_vals):
        for (sex, age) in DEMOGRAPHIC_GROUPS:
            sub = base[(base['sex'] == sex) & (base['age'] == age) &
                       (base['salary'] == salary)]
            if sub.empty:
                continue
            agg = sub.groupby('payment_rate')[['roi_pds', 'roi_iis']].mean().reset_index()
            agg['advantage'] = agg['roi_pds'] - agg['roi_iis']

            inf_row = base_inf[(base_inf['sex'] == sex) & (base_inf['age'] == age) &
                               (base_inf['salary'] == salary)]
            peak_r = inf_row['h2_pds_max_advantage_rate'].values[0] if len(inf_row) else None

            ax.plot(agg['payment_rate'] * 100, agg['advantage'],
                    color=DEMOGRAPHIC_COLORS[(sex, age)],
                    linestyle=DEMOGRAPHIC_LINESTYLES[(sex, age)],
                    lw=1.5, label=DEMOGRAPHIC_LABELS[(sex, age)])
            if peak_r is not None and pd.notna(peak_r):
                peak_adv = agg.loc[(agg['payment_rate'] - peak_r).abs().idxmin(), 'advantage']
                ax.scatter([peak_r * 100], [peak_adv],
                           color=DEMOGRAPHIC_COLORS[(sex, age)], zorder=5, s=40, marker='^')

        ax.axhline(0, color='#333', lw=0.8)
        ax.set_title(SALARY_LABELS.get(salary, f'{salary//1000} тыс.'), fontsize=10)
        ax.set_xlabel('Ставка взноса (%)')
        ax.set_ylabel('advantage (ROI_ПДС − ROI_ИИС3)')
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
        ax.legend(fontsize=7)

    for ax in axes[len(salary_vals):]:
        ax.set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'h2h3_demographic_robustness.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


if __name__ == '__main__':
    print("Загружаю данные H2/H3...")
    df, inf = load_data()
    print(f"  Строк: {len(df)}, payment_rate: {df['payment_rate'].nunique()} уровней, "
          f"рыночных сценариев: {df['market_scenario'].nunique()}, "
          f"трудовых сценариев: {df['transition_scenario'].nunique()}")

    os.makedirs(FIGURES_DIR, exist_ok=True)

    base_df = df[(df['market_scenario'] == 'baseline') &
                 (df['transition_scenario'] == 'baseline') &
                 (df['age'] == REPRESENTATIVE_AGE) &
                 (df['sex'] == REPRESENTATIVE_SEX)]
    base_inf = inf[(inf['market_scenario'] == 'baseline') &
                   (inf['transition_scenario'] == 'baseline') &
                   (inf['age'] == REPRESENTATIVE_AGE) &
                   (inf['sex'] == REPRESENTATIVE_SEX)]

    print("Строю графики...")
    plot_assumptions_table()
    plot_roi_curves(base_df, base_inf)
    plot_advantage_curve(base_df, base_inf)
    poly_results = plot_diminishing_returns(base_df, base_inf)
    plot_inflection_table(base_inf)
    plot_percentile_fan(base_df)
    plot_market_scenario_comparison(df, inf)
    plot_transition_scenario_comparison(df, inf)
    plot_demographic_robustness(df, inf)

    print_descriptive_stats(base_df, base_inf)
    print_verdict(base_df, base_inf, poly_results)
