"""
Анализ гипотезы H1
------------------
H1: Программы с самостоятельным управлением (ИИС-3) имеют более высокую ожидаемую
доходность, чем ПДС, за счёт менее ограниченных требований к портфелю.

Логика проверки
  1. TWR (time-weighted return) — изолированная портфельная доходность.
     Если E[TWR_ИИС3] > E[TWR_ПДС], свобода выбора портфеля даёт преимущество.
  2. IRR — полная доходность программы (портфель + комиссии + со-финансирование).
     Если IRR_ПДС > IRR_ИИС3, несмотря на более высокий TWR у ИИС-3 —
     со-финансирование перекрывает портфельное преимущество.
  Критерий: критерий Манна-Уитни (непараметрический), α = 0.05.
  Размер эффекта: rank-biserial correlation r = 2U/(n1·n2) - 1.

Сценарии вероятности трудового перехода
  baseline    — откалиброванное значение (ОРС Росстат 2024)
  low_transit — p = 0.10 (низкая вероятность трудового перехода)
  mid_transit — p = 0.15 (умеренная вероятность трудового перехода)
  Если результаты радикально меняются между сценариями — это важный вывод
  об устойчивости гипотезы к предположениям о занятости.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# ── пути ──────────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join(os.path.dirname(__file__), '..', 'scenarios', 'temp_data', 'h1_portfolio_freedom.csv')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'scenarios', 'temp_data', 'figures')

# ── стиль ─────────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
COLORS = {
    'pds_avg':    '#2166ac',
    'iis3_20/80': '#d9ef8b',
    'iis3_50/50': '#fee08b',
    'iis3_80/20': '#d73027',
}
LABELS = {
    'pds_avg':    'ПДС (НПФ ср.)',
    'iis3_20/80': 'ИИС-3 (20/80)',
    'iis3_50/50': 'ИИС-3 (50/50)',
    'iis3_80/20': 'ИИС-3 (80/20)',
}
ALPHA = 0.05

TRANSITION_LABELS = {
    'baseline':    'Базовый (калибр.)',
    'low_transit': 'Низкий (p=0.10)',
    'mid_transit': 'Умеренный (p=0.15)',
}


def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Файл не найден: {DATA_PATH}\n"
            "Сначала запустите: python -m scenarios.h1_portfolio_freedom"
        )
    df = pd.read_csv(DATA_PATH)
    df['program_label'] = df['portfolio'].map(LABELS)
    if 'transition_scenario' not in df.columns:
        df['transition_scenario'] = 'baseline'
    return df


def plot_assumptions_table():
    """Таблица исходных сценарных предпосылок H1."""
    rows = [
        ('Горизонт моделирования',              '15 лет'),
        ('Число симуляций',                     '500'),
        ('Ставка взноса',                       '6 % от зарплаты'),
        ('Зарплатные группы',                   '50 / 100 / 150 / 200 тыс. ₽/мес.'),
        ('Возрастные группы',                   '30 и 45 лет'),
        ('Пол',                                 'М и Ж'),
        ('Портфель ПДС',                        'Средний НПФ (structure.xlsx)'),
        ('Портфели ИИС-3',                      '20/80, 50/50, 80/20 (акции/облигации)'),
        ('Облигационная часть ИИС-3',           'Пропорции среднего НПФ: ОФЗ≈63.5%, корп.≈33.4%, мун.≈3.1%'),
        ('Комиссия ПДС',                        '0.5 % фикс. + 20 % от дохода'),
        ('Комиссия ИИС-3',                      '0 % (нет управляющей компании)'),
        ('Налоговый вычет',                     '13 % от взноса'),
        ('Со-финансирование ПДС',               'До 36 000 ₽/год, первые 10 лет'),
        ('Модель зарплаты',                     'StochasticSalaryModel'),
        ('Вероятность трудового перехода',      'Базовый: калибр. (ОРС 2024); альт.: p=0.10, p=0.15'),
        ('Метрики сравнения',                   'TWR, IRR, ROI, KZ'),
        ('Статистический критерий',             'Манн-Уитни U, α=0.05; размер эффекта: rank-biserial r'),
    ]
    fig, ax = plt.subplots(figsize=(14, 5.5))
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
    ax.set_title('H1 — Исходные сценарные предпосылки', fontsize=12, pad=10)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'h1_assumptions.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


def plot_percentile_distributions(df: pd.DataFrame):
    """
    Веерный график перцентилей TWR и IRR по портфелям (базовый сценарий).
    Показывает риск: 5-й, 25-й, 50-й, 75-й, 95-й перцентили.
    """
    base = df[df['transition_scenario'] == 'baseline']
    portfolio_order = list(LABELS.keys())
    pctiles = [5, 25, 50, 75, 95]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('H1 — Перцентили TWR и IRR по типу программы/портфеля\n'
                 '(базовый сценарий трудового перехода)',
                 fontsize=13, y=1.01)

    for ax, metric, title in zip(
        axes,
        ['twr', 'irr'],
        ['TWR — взвешенная по времени доходность', 'IRR — полная доходность программы'],
    ):
        x = np.arange(len(portfolio_order))
        width = 0.15
        offsets = np.linspace(-2 * width, 2 * width, len(pctiles))

        PCTILE_COLORS = {5: '#2166ac', 25: '#74add1', 50: '#4dac26', 75: '#f46d43', 95: '#d73027'}
        for offset, p in zip(offsets, pctiles):
            vals = [base.loc[base['portfolio'] == pf, metric].dropna()
                    .quantile(p / 100) for pf in portfolio_order]
            ax.bar(x + offset, vals, width, color=PCTILE_COLORS[p], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[p] for p in portfolio_order], rotation=15, ha='right')
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(metric.upper())
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

        # Легенда перцентилей
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=PCTILE_COLORS[p], label=f'p{p}') for p in pctiles]
        ax.legend(handles=legend_elements, fontsize=8, title='Перцентиль')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'h1_percentiles.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


def plot_transition_scenario_comparison(df: pd.DataFrame):
    """
    Сравнение медианных TWR и IRR по сценариям вероятности трудового перехода.
    Если результаты радикально меняются — это важный вывод об устойчивости.
    """
    scenarios = [s for s in ['baseline', 'low_transit', 'mid_transit']
                 if s in df['transition_scenario'].unique()]
    if len(scenarios) < 2:
        print("  Пропускаю сравнение сценариев: данные только для одного сценария.")
        return

    portfolio_order = list(LABELS.keys())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('H1 — Устойчивость результатов к сценариям вероятности трудового перехода\n'
                 '(медианный TWR и IRR)',
                 fontsize=12, y=1.02)

    linestyles = {'baseline': '-', 'low_transit': '--', 'mid_transit': ':'}

    for ax, metric, title in zip(
        axes,
        ['twr', 'irr'],
        ['TWR — взвешенная по времени доходность', 'IRR — полная доходность программы'],
    ):
        for scenario in scenarios:
            sub = df[df['transition_scenario'] == scenario]
            medians = [sub.loc[sub['portfolio'] == pf, metric].median()
                       for pf in portfolio_order]
            ax.plot(range(len(portfolio_order)), medians,
                    marker='o', lw=1.8,
                    linestyle=linestyles.get(scenario, '-'),
                    label=TRANSITION_LABELS.get(scenario, scenario))

        ax.set_xticks(range(len(portfolio_order)))
        ax.set_xticklabels([LABELS[p] for p in portfolio_order], rotation=15, ha='right')
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(f'Медиана {metric.upper()}')
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'h1_transition_scenarios.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


def print_descriptive_stats(df: pd.DataFrame):
    """Таблица описательной статистики TWR, IRR, KZ по портфелям (базовый сценарий)."""
    base = df[df['transition_scenario'] == 'baseline']
    pctiles = [5, 25, 50, 75, 95]
    print('\n' + '=' * 80)
    print('  ОПИСАТЕЛЬНАЯ СТАТИСТИКА (базовый сценарий трудового перехода)')
    print('=' * 80)
    for metric in ['twr', 'irr', 'kz']:
        print(f"\n  {metric.upper()}:")
        print(f"  {'Портфель':15} " + " ".join(f"p{p:>3}" for p in pctiles) + "   mean    std")
        for pf in list(LABELS.keys()):
            vals = base.loc[base['portfolio'] == pf, metric].dropna()
            if len(vals) == 0:
                continue
            pcts = np.percentile(vals, pctiles)
            print(f"  {pf:15} " + " ".join(f"{v:>6.1%}" for v in pcts)
                  + f"  {vals.mean():>6.1%}  {vals.std():>6.1%}")
    print('=' * 80)


def mann_whitney(a, b):
    """Манн-Уитни U + rank-biserial r."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    r = 2 * u / (len(a) * len(b)) - 1          # rank-biserial correlation
    return float(p), float(r)


def sig_label(p: float) -> str:
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'


# ─────────────────────────── Figure 1 ────────────────────────────────────────
def plot_distributions(df: pd.DataFrame):
    """Ящики с усами: TWR и IRR по всем портфелям."""
    portfolio_order = list(LABELS.keys())
    palette = [COLORS[p] for p in portfolio_order]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('H1 — Распределение доходностей по типу программы/портфеля\n'
                 '(все группы зарплат и возрастов объединены)', fontsize=13, y=1.01)

    for ax, metric, title, fmt in zip(
        axes,
        ['twr', 'irr'],
        ['TWR (взвешенная по времени доходность)\nбез учёта размера взносов',
         'IRR (внутренняя норма доходности)\nполная доходность программы'],
        [':.1%', ':.1%'],
    ):
        plot_df = df[['portfolio', metric]].copy()
        plot_df[metric] = plot_df[metric].clip(
            *np.nanpercentile(df[metric].dropna(), [1, 99])   # обрезаем выбросы
        )
        sns.boxplot(
            data=plot_df,
            x='portfolio', y=metric,
            order=portfolio_order,
            palette=palette,
            width=0.55, fliersize=2,
            ax=ax,
        )
        ax.set_xticklabels([LABELS[p] for p in portfolio_order], rotation=15, ha='right')
        ax.set_xlabel('')
        ax.set_ylabel(metric.upper())
        ax.set_title(title, fontsize=11)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

        # медианы как аннотация
        for i, pf in enumerate(portfolio_order):
            med = df.loc[df['portfolio'] == pf, metric].median()
            ax.annotate(f'{med:.1%}', xy=(i, med), ha='center', va='bottom',
                        fontsize=8.5, color='#333333', fontweight='bold')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'h1_distributions.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


# ─────────────────────────── Figure 2 ────────────────────────────────────────
def plot_by_salary(df: pd.DataFrame):
    """Средний TWR и IRR по уровням зарплаты — показывает устойчивость результата."""
    salary_vals = sorted(df['salary'].unique())
    portfolio_order = list(LABELS.keys())
    palette = [COLORS[p] for p in portfolio_order]

    fig, axes = plt.subplots(2, len(salary_vals), figsize=(14, 7), sharey='row')
    fig.suptitle('H1 — Средние TWR и IRR по уровням зарплаты', fontsize=13, y=1.02)

    for col, salary in enumerate(salary_vals):
        sub = df[df['salary'] == salary]
        for row, metric in enumerate(['twr', 'irr']):
            ax = axes[row, col]
            means = [sub.loc[sub['portfolio'] == p, metric].median() for p in portfolio_order]
            bars = ax.bar(range(len(portfolio_order)), means, color=palette, width=0.6, edgecolor='white')
            ax.set_xticks(range(len(portfolio_order)))
            ax.set_xticklabels([LABELS[p].replace(' ', '\n') for p in portfolio_order],
                                fontsize=7.5)
            if col == 0:
                ax.set_ylabel(f'Медиана {metric.upper()}')
                ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
            if row == 0:
                ax.set_title(f'Зарплата {salary//1000} тыс. ₽', fontsize=10)
            ax.set_xlabel('')
            for bar, val in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.1%}', ha='center', va='bottom', fontsize=7.5)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'h1_by_salary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


# ─────────────────────────── Figure 3 ────────────────────────────────────────
def plot_twr_vs_irr(df: pd.DataFrame):
    """Scatter: медианный TWR vs медианный IRR — карта «портфель против программы»."""
    agg = (
        df.groupby('portfolio')[['twr', 'irr']]
        .median()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, row in agg.iterrows():
        ax.scatter(row['twr'], row['irr'],
                   color=COLORS[row['portfolio']], s=130, zorder=3,
                   edgecolors='white', linewidths=1.2)
        ax.annotate(LABELS[row['portfolio']],
                    (row['twr'], row['irr']),
                    textcoords='offset points', xytext=(7, 4), fontsize=9.5)

    ax.axhline(agg.loc[agg['portfolio'] == 'pds_avg', 'irr'].values[0],
               color=COLORS['pds_avg'], linestyle='--', alpha=0.4, lw=1)
    ax.axvline(agg.loc[agg['portfolio'] == 'pds_avg', 'twr'].values[0],
               color=COLORS['pds_avg'], linestyle='--', alpha=0.4, lw=1)

    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax.set_xlabel('Медианный TWR (портфельная доходность)', fontsize=10)
    ax.set_ylabel('Медианный IRR (полная доходность программы)', fontsize=10)
    ax.set_title('H1 — Портфельная vs полная доходность\n'
                 'Пунктир = уровень ПДС', fontsize=11)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'h1_twr_vs_irr.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


# ─────────────────────────── Figure: market scenario robustness ──────────────
def plot_market_scenario_comparison(df: pd.DataFrame):
    """Устойчивость медианных TWR и IRR к рыночным сценариям."""
    scenarios = [s for s in ['baseline', 'stress', 'optimistic']
                 if s in df['market_scenario'].unique()]
    if len(scenarios) < 2:
        print("  Пропускаю сравнение рыночных сценариев: данные только для одного сценария.")
        return

    SCENARIO_LABELS = {'baseline': 'Базовый', 'stress': 'Стрессовый', 'optimistic': 'Оптимистичный'}
    SCENARIO_COLORS = {'baseline': '#2166ac', 'stress': '#d73027', 'optimistic': '#1a9850'}
    linestyles = {'baseline': '-', 'stress': '--', 'optimistic': ':'}
    portfolio_order = list(LABELS.keys())

    base_transit = df[df['transition_scenario'] == 'baseline'] if 'transition_scenario' in df.columns else df

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('H1 — Устойчивость результатов к рыночным сценариям\n'
                 '(медианный TWR и IRR, базовый сценарий трудового перехода)',
                 fontsize=12, y=1.02)

    for ax, metric, title in zip(
        axes,
        ['twr', 'irr'],
        ['TWR — взвешенная по времени доходность', 'IRR — полная доходность программы'],
    ):
        for scenario in scenarios:
            sub = base_transit[base_transit['market_scenario'] == scenario]
            medians = [sub.loc[sub['portfolio'] == pf, metric].median()
                       for pf in portfolio_order]
            ax.plot(range(len(portfolio_order)), medians,
                    marker='o', lw=1.8,
                    color=SCENARIO_COLORS[scenario],
                    linestyle=linestyles[scenario],
                    label=SCENARIO_LABELS.get(scenario, scenario))

        ax.set_xticks(range(len(portfolio_order)))
        ax.set_xticklabels([LABELS[p] for p in portfolio_order], rotation=15, ha='right')
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(f'Медиана {metric.upper()}')
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'h1_market_scenarios.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


# ─────────────────────────── Statistical tests ───────────────────────────────
def run_tests(df: pd.DataFrame) -> dict:
    """Манна-Уитни: ПДС vs каждый ИИС-3 по TWR и IRR."""
    pds = df[df['portfolio'] == 'pds_avg']
    results = {}
    for metric in ['twr', 'irr']:
        a_pds = pds[metric].dropna().values
        for pf in ['iis3_20/80', 'iis3_50/50', 'iis3_80/20']:
            b    = df.loc[df['portfolio'] == pf, metric].dropna().values
            p, r = mann_whitney(a_pds, b)
            med_pds = float(np.median(a_pds))
            med_iis = float(np.median(b))
            results[f'{metric}|{pf}'] = {
                'metric':     metric.upper(),
                'portfolio':  LABELS[pf],
                'med_pds':    med_pds,
                'med_iis3':   med_iis,
                'delta':      med_iis - med_pds,
                'direction':  'ИИС-3 >' if med_iis > med_pds else 'ПДС >',
                'p_value':    p,
                'sig':        sig_label(p),
                'effect_r':   r,
            }
    return results


def print_verdict(results: dict, df: pd.DataFrame):
    print('\n' + '=' * 65)
    print('  ВЕРДИКТ: ГИПОТЕЗА H1')
    print('=' * 65)

    # Сводная таблица
    rows = list(results.values())
    tbl = pd.DataFrame(rows)[['metric', 'portfolio', 'med_pds', 'med_iis3',
                               'delta', 'direction', 'p_value', 'sig', 'effect_r']]
    tbl[['med_pds', 'med_iis3', 'delta']] = tbl[['med_pds', 'med_iis3', 'delta']].applymap(
        lambda x: f'{x:.2%}'
    )
    tbl['p_value'] = tbl['p_value'].apply(lambda x: f'{x:.4f}')
    tbl['effect_r'] = tbl['effect_r'].apply(lambda x: f'{x:.3f}')
    print(tbl.to_string(index=False))

    # TWR: портфельная свобода
    twr_wins = sum(1 for k, v in results.items() if 'twr' in k and v['delta'] > 0 and v['p_value'] < ALPHA
                   for k2 in [k] if float(v['p_value'].replace(',', '.')) < ALPHA
                   if True)
    twr_wins = sum(1 for k, v in results.items() if 'twr' in k
                   and v['delta'] > 0 and v['p_value'] < ALPHA)

    irr_wins = sum(1 for k, v in results.items() if 'irr' in k
                   and v['delta'] > 0 and v['p_value'] < ALPHA)

    print()
    if twr_wins >= 2:
        print('✓ H1 по TWR: ПОДТВЕРЖДАЕТСЯ — ИИС-3 обеспечивает более высокую')
        print('  взвешенную по времени доходность (свобода выбора портфеля даёт преимущество).')
    else:
        print('✗ H1 по TWR: НЕ ПОДТВЕРЖДАЕТСЯ — взвешенная по времени доходность ИИС-3')
        print('  не превышает ПДС значимо.')

    if irr_wins >= 2:
        print('✓ H1 по IRR: ИИС-3 выгоднее и по полной доходности программы.')
    else:
        print('  H1 по IRR: ПДС конкурентоспособен за счёт со-финансирования —')
        print('  нет оснований считать ИИС-3 безусловно лучше по полной доходности.')

    print('=' * 65)


# ─────────────────────────── main ────────────────────────────────────────────
if __name__ == '__main__':
    print("Загружаю данные H1...")
    df = load_data()
    print(f"  Строк: {len(df)}, портфелей: {df['portfolio'].nunique()}, "
          f"сценариев: {df['transition_scenario'].nunique()}")

    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Строю графики...")
    plot_assumptions_table()
    plot_distributions(df[df['transition_scenario'] == 'baseline'])
    plot_by_salary(df[df['transition_scenario'] == 'baseline'])
    plot_twr_vs_irr(df[df['transition_scenario'] == 'baseline'])
    plot_percentile_distributions(df)
    plot_transition_scenario_comparison(df)
    plot_market_scenario_comparison(df)

    print_descriptive_stats(df)

    results = run_tests(df[df['transition_scenario'] == 'baseline'])
    print_verdict(results, df[df['transition_scenario'] == 'baseline'])
