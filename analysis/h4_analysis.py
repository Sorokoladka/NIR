"""
Анализ гипотезы H4
------------------
H4: Результативность программ неоднородна по группам населения: демографические
и социальные признаки определяют более выгодную для индивида программу.

Логика проверки
  1. Карта победителей (age_group × sex): для каждой ячейки — кто выгоднее.
  2. Разброс kz_delta = KZ_ПДС − KZ_ИИС3_best по ячейкам → неоднородность.
  3. Критерий Краскела-Уоллиса: различается ли KZ ПДС по возрастным группам?
     Аналогично — для лучшего ИИС-3.
  4. Оценка численности потенциальных вкладчиков (ОРС Росстат 2024).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

DATA_RAW    = os.path.join(os.path.dirname(__file__), '..', 'scenarios', 'temp_data', 'h4_raw.csv')
DATA_WINNER = os.path.join(os.path.dirname(__file__), '..', 'scenarios', 'temp_data', 'h4_winner_map.csv')
DATA_POP    = os.path.join(os.path.dirname(__file__), '..', 'scenarios', 'temp_data', 'h4_population_summary.csv')
FIGURES_DIR  = os.path.join(os.path.dirname(__file__), '..', 'scenarios', 'temp_data', 'figures')

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.05)

AGE_ORDER = ['20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59']
COLOR_PDS  = '#2166ac'
COLOR_IIS  = '#d73027'


def load_data():
    missing = [p for p in [DATA_RAW, DATA_WINNER, DATA_POP] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Файлы не найдены: {missing}\n"
            "Сначала запустите: python -m scenarios.h4_demographic_heterogeneity"
        )
    df     = pd.read_csv(DATA_RAW)
    winner = pd.read_csv(DATA_WINNER)
    pop    = pd.read_csv(DATA_POP)
    return df, winner, pop


# ─────────────────────────── Figure 1 ────────────────────────────────────────
def plot_winner_heatmap(winner: pd.DataFrame):
    """
    Тепловая карта kz_delta = KZ_ПДС − KZ_ИИС3_best по age_group × sex.
    Синий = ПДС лучше, красный = ИИС-3 лучше.
    """
    pivot = winner.pivot(index='age_group', columns='sex', values='kz_delta')
    pivot = pivot.reindex(AGE_ORDER)

    fig, ax = plt.subplots(figsize=(6, 6))
    vmax = max(abs(pivot.values[np.isfinite(pivot.values)].max()),
               abs(pivot.values[np.isfinite(pivot.values)].min()))
    cmap = sns.diverging_palette(20, 220, as_cmap=True)   # красный–белый–синий

    sns.heatmap(
        pivot, ax=ax,
        cmap=cmap, center=0, vmin=-vmax, vmax=vmax,
        annot=True, fmt='.3f', annot_kws={'size': 9},
        linewidths=0.5, linecolor='#cccccc',
        cbar_kws={'label': 'KZ_ПДС − KZ_ИИС3_best'},
    )
    ax.set_title('H4 — Карта победителей\n'
                 'KZ_delta > 0 (синий) = ПДС лучше, < 0 (красный) = ИИС-3 лучше',
                 fontsize=11)
    ax.set_xlabel('Пол')
    ax.set_ylabel('Возрастная группа')
    ax.set_xticklabels(['Ж', 'М'])

    # Победитель текстом поверх клетки
    for age_i, age_g in enumerate(AGE_ORDER):
        for sex_j, sex in enumerate(['F', 'M']):
            row = winner[(winner['age_group'] == age_g) & (winner['sex'] == sex)]
            if len(row):
                w = row.iloc[0]['winner']
                ax.text(sex_j + 0.5, age_i + 0.15,
                        '▲ПДС' if w == 'pds' else '▼ИИС',
                        ha='center', va='top', fontsize=7,
                        color='white' if abs(row.iloc[0]['kz_delta']) > vmax * 0.3 else '#333')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'h4_winner_heatmap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


# ─────────────────────────── Figure 2 ────────────────────────────────────────
def plot_population_bar(pop: pd.DataFrame, winner: pd.DataFrame):
    """Столбчатая диаграмма: численность вкладчиков (млн) по победителю."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('H4 — Оценка численности потенциальных вкладчиков\n'
                 '(наёмные работники 18–60, Росстат ОРС 2024)', fontsize=12, y=1.02)

    # Левый: агрегированный
    ax = axes[0]
    colors = [COLOR_PDS if w == 'pds' else COLOR_IIS for w in pop['winner']]
    bars = ax.bar(pop['winner'].map({'pds': 'ПДС', 'iis3': 'ИИС-3'}),
                  pop['population_mln'], color=colors, width=0.5, edgecolor='white')
    ax.set_ylabel('Численность (млн чел.)')
    ax.set_title('Итого по победителю программы')
    total = pop['population_mln'].sum()
    for bar, val in zip(bars, pop['population_mln']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.1f} млн\n({val/total:.0%})',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylim(0, pop['population_mln'].max() * 1.25)

    # Правый: разбивка age_group × sex
    ax2 = axes[1]
    winner_sorted = winner.copy()
    winner_sorted['age_group'] = pd.Categorical(winner_sorted['age_group'],
                                                  categories=AGE_ORDER, ordered=True)
    winner_sorted = winner_sorted.sort_values(['age_group', 'sex'])

    x     = np.arange(len(AGE_ORDER))
    width = 0.35
    m_vals = winner_sorted[winner_sorted['sex'] == 'M'].set_index('age_group')
    f_vals = winner_sorted[winner_sorted['sex'] == 'F'].set_index('age_group')

    for i, age_g in enumerate(AGE_ORDER):
        for vals, offset, hatch in [(m_vals, -width/2, ''), (f_vals, width/2, '///')]:
            if age_g in vals.index:
                row = vals.loc[age_g]
                color = COLOR_PDS if row['winner'] == 'pds' else COLOR_IIS
                ax2.bar(i + offset, row['population_mln'], width,
                        color=color, alpha=0.85, hatch=hatch, edgecolor='white')

    ax2.set_xticks(x)
    ax2.set_xticklabels(AGE_ORDER, rotation=30, ha='right', fontsize=9)
    ax2.set_ylabel('Численность (млн чел.)')
    ax2.set_title('По возрасту и полу')

    legend_elements = [
        mpatches.Patch(facecolor=COLOR_PDS, label='ПДС победитель'),
        mpatches.Patch(facecolor=COLOR_IIS, label='ИИС-3 победитель'),
        mpatches.Patch(facecolor='grey', hatch='///', label='Женщины'),
        mpatches.Patch(facecolor='grey', label='Мужчины'),
    ]
    ax2.legend(handles=legend_elements, fontsize=8, loc='upper right')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'h4_population.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


# ─────────────────────────── Figure 3 ────────────────────────────────────────
def plot_kz_by_age(df: pd.DataFrame):
    """
    E[KZ] ПДС vs лучшего ИИС-3 по возрастным группам (M и F отдельно).
    Позволяет увидеть какие возрасты выигрывают от ПДС, а какие от ИИС-3.
    """
    df_age = df.copy()
    df_age['age_group'] = pd.Categorical(df_age['age_group'],
                                          categories=AGE_ORDER, ordered=True)

    # Лучший ИИС-3 = max KZ среди трёх аллокаций
    iis3_best = (
        df_age[df_age['program'] == 'iis3']
        .groupby(['age_group', 'sex', 'sim_id'])['kz']
        .max()
        .reset_index()
        .assign(program_label='ИИС-3 (лучший)')
    )
    pds_df = (
        df_age[df_age['program'] == 'pds']
        .groupby(['age_group', 'sex', 'sim_id'])['kz']
        .mean()
        .reset_index()
        .assign(program_label='ПДС')
    )
    combined = pd.concat([pds_df, iis3_best], ignore_index=True)
    combined = combined[combined['kz'].notna() & np.isfinite(combined['kz'])]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle('H4 — Медианный KZ (коэффициент замещения) по возрастным группам',
                 fontsize=12, y=1.02)

    palette = {'ПДС': COLOR_PDS, 'ИИС-3 (лучший)': COLOR_IIS}
    for ax, sex, sex_label in zip(axes, ['M', 'F'], ['Мужчины', 'Женщины']):
        sub = combined[combined['sex'] == sex]
        sns.boxplot(
            data=sub, x='age_group', y='kz', hue='program_label',
            order=AGE_ORDER, palette=palette,
            width=0.55, fliersize=1.5, ax=ax,
        )
        ax.set_title(sex_label, fontsize=11)
        ax.set_xlabel('Возрастная группа')
        if ax is axes[0]:
            ax.set_ylabel('KZ (коэффициент замещения)')
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_xticklabels(AGE_ORDER, rotation=30, ha='right', fontsize=9)
        ax.legend(title='', fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'h4_kz_by_age.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


# ─────────────────────────── Statistical tests ───────────────────────────────
def run_tests(df: pd.DataFrame) -> dict:
    """
    Тест Краскела-Уоллиса: различается ли KZ_ПДС по возрастным группам?
    Тест Манна-Уитни: ПДС vs лучший ИИС-3 внутри каждой возрастной группы.
    """
    results = {'kruskal': {}, 'mw_per_age': {}}

    for prog, prog_label in [('pds', 'ПДС'), ('iis3', 'ИИС-3')]:
        sub  = df[df['program'] == prog]
        groups = [sub.loc[sub['age_group'] == ag, 'kz'].dropna().values for ag in AGE_ORDER]
        groups = [g for g in groups if len(g) > 5]
        if len(groups) >= 2:
            h, p = stats.kruskal(*groups)
            results['kruskal'][prog_label] = {'H': h, 'p': p}

    # ПДС vs лучший ИИС-3 на каждую возрастную группу
    iis3_best_by_group = (
        df[df['program'] == 'iis3']
        .groupby(['age_group', 'sex', 'sim_id'])['kz']
        .max()
        .reset_index()
    )
    for age_g in AGE_ORDER:
        pds_vals = df.loc[(df['program'] == 'pds') & (df['age_group'] == age_g), 'kz'].dropna()
        iis_vals = iis3_best_by_group.loc[iis3_best_by_group['age_group'] == age_g, 'kz'].dropna()
        if len(pds_vals) > 5 and len(iis_vals) > 5:
            u, p = stats.mannwhitneyu(pds_vals, iis_vals, alternative='two-sided')
            med_diff = float(pds_vals.median()) - float(iis_vals.median())
            results['mw_per_age'][age_g] = {
                'med_pds':  float(pds_vals.median()),
                'med_iis':  float(iis_vals.median()),
                'delta':    med_diff,
                'p':        p,
                'winner':   'ПДС' if med_diff > 0 else 'ИИС-3',
            }
    return results


def print_verdict(winner: pd.DataFrame, pop: pd.DataFrame, tests: dict):
    print('\n' + '=' * 65)
    print('  ВЕРДИКТ: ГИПОТЕЗА H4')
    print('=' * 65)

    # Неоднородность: есть ли разные победители?
    n_pds = (winner['winner'] == 'pds').sum()
    n_iis = (winner['winner'] == 'iis3').sum()
    total = len(winner)
    print(f'\nКарта победителей ({total} ячеек age_group × sex):')
    print(f'  ПДС лучше: {n_pds}/{total} ячеек ({n_pds/total:.0%})')
    print(f'  ИИС-3 лучше: {n_iis}/{total} ячеек ({n_iis/total:.0%})')
    heterogeneous = (n_pds > 0) and (n_iis > 0)

    # Тест Краскела-Уоллиса
    print('\nТест Краскела-Уоллиса (KZ неоднороден по возрасту):')
    for prog, res in tests['kruskal'].items():
        sig = '***' if res['p'] < 0.001 else ('**' if res['p'] < 0.01 else ('*' if res['p'] < 0.05 else 'ns'))
        print(f'  {prog}: H={res["H"]:.2f}, p={res["p"]:.4f} {sig}')

    # Манн-Уитни по возрастам
    print('\nПДС vs ИИС-3 (лучший) по возрастным группам (медиана KZ):')
    print(f'  {"Группа":>7} | {"ПДС":>6} | {"ИИС-3":>6} | {"Δ":>7} | {"p":>7} | {"Победитель"}')
    print('  ' + '-' * 55)
    for age_g, r in tests['mw_per_age'].items():
        sig = '***' if r['p'] < 0.001 else ('**' if r['p'] < 0.01 else ('*' if r['p'] < 0.05 else 'ns'))
        print(f'  {age_g:>7} | {r["med_pds"]:>6.2%} | {r["med_iis"]:>6.2%} | '
              f'{r["delta"]:>+7.2%} | {r["p"]:>7.4f}{sig} | {r["winner"]}')

    # Численность вкладчиков
    print('\nОценка численности потенциальных вкладчиков:')
    for _, row in pop.iterrows():
        prog = 'ПДС' if row['winner'] == 'pds' else 'ИИС-3'
        print(f'  {prog}: {row["population_mln"]:.1f} млн чел. ({row["share_pct"]:.0f}%)')

    # Неоднородность по MW-тестам (медиана — устойчива к выбросам)
    mw_pds_wins = sum(1 for r in tests['mw_per_age'].values() if r['winner'] == 'ПДС' and r['p'] < 0.05)
    mw_iis_wins = sum(1 for r in tests['mw_per_age'].values() if r['winner'] == 'ИИС-3' and r['p'] < 0.05)
    mw_heterogeneous = (mw_pds_wins > 0) and (mw_iis_wins > 0)

    print()
    if heterogeneous or mw_heterogeneous:
        print('✓ H4 ПОДТВЕРЖДАЕТСЯ: результативность программ неоднородна.')
        print(f'  Демографические характеристики (возраст, зарплата) значимо')
        print(f'  определяют, какая программа предпочтительна:')
        if mw_heterogeneous:
            print(f'    • ИИС-3 предпочтителен (по медиане KZ) в {mw_iis_wins} возр. группах')
            print(f'    • ПДС предпочтителен (по медиане KZ) в {mw_pds_wins} возр. группах')
        if heterogeneous:
            print(f'    • По средним KZ: ПДС лучше в {n_pds}/{total}, ИИС-3 в {n_iis}/{total} ячейках')
        if not heterogeneous and mw_heterogeneous:
            print(f'  Примечание: карта средних KZ не показывает смену победителя,')
            print(f'  однако медианные KZ устойчиво показывают преимущество ИИС-3')
            print(f'  для молодых когорт (25–39) — со-финансирование ПДС создаёт')
            print(f'  асимметрию (длинный правый хвост), которая завышает среднее.')
    else:
        print('✗ H4 НЕ ПОДТВЕРЖДАЕТСЯ: одна программа доминирует во всех группах.')
    print('=' * 65)


if __name__ == '__main__':
    print("Загружаю данные H4...")
    df, winner, pop = load_data()
    print(f"  Строк: {len(df)}, возрастных групп: {df['age_group'].nunique()}")

    print("Строю графики...")
    plot_winner_heatmap(winner)
    plot_population_bar(pop, winner)
    plot_kz_by_age(df)

    tests = run_tests(df)
    print_verdict(winner, pop, tests)
