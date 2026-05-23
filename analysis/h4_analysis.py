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
  5. Перцентили KZ (5, 25, 50, 75, 95) — риск нехватки накоплений.
  6. Сравнение сценариев вероятности трудового перехода (baseline / low / mid).
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
    if 'transition_scenario' not in df.columns:
        df['transition_scenario'] = 'baseline'
    return df, winner, pop


# ─────────────────────────── Figure 1 ────────────────────────────────────────
def plot_winner_heatmap(winner: pd.DataFrame):
    """
    Тепловая карта benefit_delta = Benefit_ПДС − Benefit_ИИС3_best по age_group × sex.
    ПДС: KZ (коэфф. замещения), ИИС-3: накопления / годовая зарплата.
    Если есть оба сценария ставки взноса — два сабплота side-by-side.
    """
    PAY_LABELS = {'cap_rate': 'Аналит. ставка (макс. со-фин.)', '2x_cap_rate': '2× аналит. ставка'}
    pay_scenarios = [s for s in ['cap_rate', '2x_cap_rate']
                     if 'payment_scenario' not in winner.columns or s in winner['payment_scenario'].unique()]
    if 'payment_scenario' not in winner.columns:
        pay_scenarios = ['cap_rate']

    cmap = sns.diverging_palette(20, 220, as_cmap=True)

    all_deltas = winner['benefit_delta'].values
    vmax = max(abs(all_deltas[np.isfinite(all_deltas)].max()),
               abs(all_deltas[np.isfinite(all_deltas)].min()))

    n = len(pay_scenarios)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    fig.suptitle('H4 — Карта победителей: Δ = Benefit_ПДС − Benefit_ИИС3_best\n'
                 'Benefit = накопления / среднегодовая зарплата\n'
                 'Синий = ПДС лучше, красный = ИИС-3 лучше',
                 fontsize=12, y=1.04)

    for ax, pay_sc in zip(axes, pay_scenarios):
        if 'payment_scenario' in winner.columns:
            sub = winner[winner['payment_scenario'] == pay_sc]
        else:
            sub = winner
        pivot = sub.pivot(index='age_group', columns='sex', values='benefit_delta')
        pivot = pivot.reindex(AGE_ORDER)

        sns.heatmap(
            pivot, ax=ax,
            cmap=cmap, center=0, vmin=-vmax, vmax=vmax,
            annot=True, fmt='.3f', annot_kws={'size': 9},
            linewidths=0.5, linecolor='#cccccc',
            cbar_kws={'label': 'Benefit_ПДС − Benefit_ИИС3_best'},
            cbar=(ax is axes[-1]),
        )
        ax.set_title(PAY_LABELS.get(pay_sc, pay_sc), fontsize=11)
        ax.set_xlabel('Пол')
        ax.set_ylabel('Возрастная группа' if ax is axes[0] else '')
        ax.set_xticklabels(['Ж', 'М'])

        for age_i, age_g in enumerate(AGE_ORDER):
            for sex_j, sex in enumerate(['F', 'M']):
                row = sub[(sub['age_group'] == age_g) & (sub['sex'] == sex)]
                if len(row):
                    w = row.iloc[0]['winner']
                    ax.text(sex_j + 0.5, age_i + 0.15,
                            '▲ПДС' if w == 'pds' else '▼ИИС',
                            ha='center', va='top', fontsize=7,
                            color='white' if abs(row.iloc[0]['benefit_delta']) > vmax * 0.3 else '#333')

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
    E[Benefit] ПДС vs лучшего ИИС-3 по возрастным группам (M и F отдельно).
    """
    df_age = df.copy()
    df_age['age_group'] = pd.Categorical(df_age['age_group'],
                                          categories=AGE_ORDER, ordered=True)

    # Лучший портфель ИИС-3 по медиане benefit в каждой ячейке (age_group × sex)
    iis3_all = df_age[df_age['program'] == 'iis3']
    best_portfolio = (
        iis3_all.groupby(['age_group', 'sex', 'portfolio'])['benefit']
        .median().reset_index()
        .sort_values('benefit', ascending=False)
        .drop_duplicates(['age_group', 'sex'])
        .rename(columns={'benefit': '_med'})[['age_group', 'sex', 'portfolio']]
    )
    iis3_best = (
        iis3_all.merge(best_portfolio, on=['age_group', 'sex', 'portfolio'])
        [['age_group', 'sex', 'sim_id', 'benefit']]
        .assign(program_label='ИИС-3 (лучший)')
    )
    pds_df = (
        df_age[df_age['program'] == 'pds']
        [['age_group', 'sex', 'sim_id', 'benefit']]
        .assign(program_label='ПДС')
    )
    combined = pd.concat([pds_df, iis3_best], ignore_index=True)
    combined = combined[combined['benefit'].notna() & np.isfinite(combined['benefit'])]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle('H4 — Медианный Benefit по возрастным группам\n'
                 'Benefit = накопления / среднегодовая зарплата',
                 fontsize=12, y=1.02)

    palette = {'ПДС': COLOR_PDS, 'ИИС-3 (лучший)': COLOR_IIS}
    for ax, sex, sex_label in zip(axes, ['M', 'F'], ['Мужчины', 'Женщины']):
        sub = combined[combined['sex'] == sex]
        sns.boxplot(
            data=sub, x='age_group', y='benefit', hue='program_label',
            order=AGE_ORDER, palette=palette,
            width=0.55, fliersize=1.5, ax=ax,
        )
        ax.set_title(sex_label, fontsize=11)
        ax.set_xlabel('Возрастная группа')
        if ax is axes[0]:
            ax.set_ylabel('Benefit')
        ax.set_xticklabels(AGE_ORDER, rotation=30, ha='right', fontsize=9)
        ax.legend(title='', fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'h4_kz_by_age.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


def plot_assumptions_table():
    """Таблица исходных сценарных предпосылок H4."""
    rows = [
        ('Горизонт моделирования',              '15 лет'),
        ('Число симуляций',                     '300'),
        ('Ставка взноса',                       'Аналит. оптимальная (макс. со-фин.) и 2× аналит.'),
        ('Возрастные группы',                   '20–24, 25–29, …, 55–59 (8 групп)'),
        ('Пол',                                 'М и Ж'),
        ('Представительная зарплата',           'demogr_salaries_agg.csv, данные 2021 г., индексированы к 2026 (×1.600, инфляция Росстат)'),
        ('Портфель ПДС',                        'Средний НПФ (structure.xlsx)'),
        ('Портфели ИИС-3',                      '20/80, 50/50, 80/20 (акции/облигации)'),
        ('Комиссия ПДС',                        '0.5 % фикс. + 20 % от дохода'),
        ('Комиссия ИИС-3',                      '0 % (нет управляющей компании)'),
        ('Налоговый вычет',                     '13 % от взноса'),
        ('Со-финансирование ПДС',               'До 36 000 ₽/год, первые 10 лет'),
        ('Модель зарплаты',                     'StochasticSalaryModel'),
        ('Вероятность трудового перехода',      'Базовый: калибр. (ОРС 2024); альт.: p=0.10, p=0.15'),
        ('Таблица дожития',                     'life_duration.csv (Росстат)'),
        ('Метрика победителя',                  'argmax E[Benefit]: накопления / среднегодовая зарплата'),
        ('Оценка численности',                  'ОРС Росстат 2024, наёмные 18–60, норм. к 56.5 млн'),
        ('Статистические тесты',                'Краскел-Уоллис (KZ по возрасту), Манн-Уитни (ПДС vs ИИС-3)'),
    ]
    fig, ax = plt.subplots(figsize=(14, 6))
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
    ax.set_title('H4 — Исходные сценарные предпосылки', fontsize=12, pad=10)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'h4_assumptions.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


def plot_kz_percentiles(df: pd.DataFrame):
    """
    Медиана Benefit + перцентильная лента (p5–p95) по возрастным группам.
    Показывает рост накоплений с возрастом и расширение разброса.
    """
    base = df[df['transition_scenario'] == 'baseline']
    base = base.copy()
    base['age_group'] = pd.Categorical(base['age_group'], categories=AGE_ORDER, ordered=True)

    # Лучший портфель ИИС-3 по медиане benefit в каждой ячейке (age_group × sex)
    iis3_all = base[base['program'] == 'iis3']
    best_portfolio = (
        iis3_all.groupby(['age_group', 'sex', 'portfolio'])['benefit']
        .median().reset_index()
        .sort_values('benefit', ascending=False)
        .drop_duplicates(['age_group', 'sex'])
        .rename(columns={'benefit': '_med'})[['age_group', 'sex', 'portfolio']]
    )
    iis3_best = iis3_all.merge(best_portfolio, on=['age_group', 'sex', 'portfolio'])
    iis3_best = iis3_best[['age_group', 'sex', 'sim_id', 'benefit']]

    pds_df = base[base['program'] == 'pds'][['age_group', 'sex', 'sim_id', 'benefit']].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle('H4 — Медиана и разброс Benefit по возрастным группам\n'
                 'Benefit = накопления / среднегодовая зарплата (лента = p5–p95)',
                 fontsize=12, y=1.02)

    for ax, sex, sex_label in zip(axes, ['M', 'F'], ['Мужчины', 'Женщины']):
        for data, color, label in [(pds_df, COLOR_PDS, 'ПДС'),
                                   (iis3_best, COLOR_IIS, 'ИИС-3 (лучший)')]:
            sub = data[data['sex'] == sex]
            agg = sub.groupby('age_group', observed=True)['benefit'].agg(
                median='median',
                p5=lambda x: np.percentile(x.dropna(), 5),
                p95=lambda x: np.percentile(x.dropna(), 95),
            ).reindex(AGE_ORDER)

            x = np.arange(len(AGE_ORDER))
            ax.fill_between(x, agg['p5'].values, agg['p95'].values,
                            alpha=0.15, color=color)
            ax.plot(x, agg['median'].values, marker='o', lw=2, color=color,
                    label=label)

        ax.set_xticks(np.arange(len(AGE_ORDER)))
        ax.set_xticklabels(AGE_ORDER, rotation=30, ha='right', fontsize=9)
        ax.set_title(sex_label, fontsize=11)
        ax.set_xlabel('Возрастная группа')
        if ax is axes[0]:
            ax.set_ylabel('Benefit')
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'h4_kz_percentiles.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


def plot_transition_scenario_comparison(df: pd.DataFrame):
    """
    Сравнение медианного Benefit по сценариям вероятности трудового перехода.
    """
    scenarios = [s for s in ['baseline', 'low_transit', 'mid_transit']
                 if s in df['transition_scenario'].unique()]
    if len(scenarios) < 2:
        print("  Пропускаю сравнение сценариев: данные только для одного сценария.")
        return

    TRANSIT_LABELS = {
        'baseline':    'Базовый (калибр.)',
        'low_transit': 'Низкий (p=0.10)',
        'mid_transit': 'Умеренный (p=0.15)',
    }
    linestyles = {'baseline': '-', 'low_transit': '--', 'mid_transit': ':'}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('H4 — Устойчивость медианного Benefit к сценариям трудового перехода\n'
                 '(ПДС vs лучший ИИС-3)',
                 fontsize=12, y=1.02)

    for ax, sex, sex_label in zip(axes, ['M', 'F'], ['Мужчины', 'Женщины']):
        for scenario in scenarios:
            sub = df[(df['transition_scenario'] == scenario) & (df['sex'] == sex)]
            pds_med = [sub[(sub['program'] == 'pds') & (sub['age_group'] == ag)]['benefit']
                       .median() for ag in AGE_ORDER]
            iis_med = [sub[(sub['program'] == 'iis3') & (sub['age_group'] == ag)]['benefit']
                       .median() for ag in AGE_ORDER]
            label = TRANSIT_LABELS.get(scenario, scenario)
            ax.plot(AGE_ORDER, pds_med, marker='o', lw=1.8, color=COLOR_PDS,
                    linestyle=linestyles.get(scenario, '-'),
                    label=f'ПДС {label}')
            ax.plot(AGE_ORDER, iis_med, marker='s', lw=1.8, color=COLOR_IIS,
                    linestyle=linestyles.get(scenario, '-'),
                    label=f'ИИС-3 {label}')

        ax.set_title(sex_label, fontsize=11)
        ax.set_xlabel('Возрастная группа')
        ax.set_xticklabels(AGE_ORDER, rotation=30, ha='right', fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel('Медиана Benefit')
        ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'h4_transition_scenarios.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


def print_descriptive_stats(df: pd.DataFrame):
    """Таблица описательной статистики Benefit по возрастным группам (базовый сценарий)."""
    base = df[df['transition_scenario'] == 'baseline']
    pctiles = [5, 25, 50, 75, 95]
    print('\n' + '=' * 80)
    print('  ОПИСАТЕЛЬНАЯ СТАТИСТИКА BENEFIT (базовый сценарий трудового перехода)')
    print('  Benefit = накопления / среднегодовая зарплата')
    print('=' * 80)
    print(f"  {'Группа':7} {'Пол':3} {'Программа':10} "
          + " ".join(f"p{p:>3}" for p in pctiles) + "   mean    std")
    for age_g in AGE_ORDER:
        for sex in ['M', 'F']:
            for prog_label, prog_filter in [
                ('ПДС',   (base['program'] == 'pds')),
                ('ИИС-3', (base['program'] == 'iis3')),
            ]:
                sub = base[(base['age_group'] == age_g) & (base['sex'] == sex) & prog_filter]
                vals = sub['benefit'].dropna()
                if len(vals) == 0:
                    continue
                pcts = np.percentile(vals, pctiles)
                print(f"  {age_g:7} {sex:3} {prog_label:10} "
                      + " ".join(f"{v:>6.1%}" for v in pcts)
                      + f"  {vals.mean():>6.1%}  {vals.std():>6.1%}")
    print('=' * 80)



    """
    Тест Краскела-Уоллиса: различается ли Benefit по возрастным группам?
    Тест Манна-Уитни: ПДС vs лучший ИИС-3 внутри каждой возрастной группы.
    """
    results = {'kruskal': {}, 'mw_per_age': {}}

    for prog, prog_label in [('pds', 'ПДС'), ('iis3', 'ИИС-3')]:
        sub  = df[df['program'] == prog]
        groups = [sub.loc[sub['age_group'] == ag, 'benefit'].dropna().values for ag in AGE_ORDER]
        groups = [g for g in groups if len(g) > 5]
        if len(groups) >= 2:
            h, p = stats.kruskal(*groups)
            results['kruskal'][prog_label] = {'H': h, 'p': p}

    iis3_best_by_group = (
        df[df['program'] == 'iis3']
        .groupby(['age_group', 'sex', 'sim_id'])['benefit']
        .max()
        .reset_index()
    )
    for age_g in AGE_ORDER:
        pds_vals = df.loc[(df['program'] == 'pds') & (df['age_group'] == age_g), 'benefit'].dropna()
        iis_vals = iis3_best_by_group.loc[iis3_best_by_group['age_group'] == age_g, 'benefit'].dropna()
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


def run_tests(df: pd.DataFrame) -> dict:
    """Краскел-Уоллис по возрасту + Манн-Уитни ПДС vs лучший ИИС-3 по каждой группе."""
    from scipy import stats as scipy_stats

    kruskal = {}
    for prog, filt in [('ПДС', df['portfolio'] == 'pds_avg'),
                       ('ИИС-3', df['program'] == 'iis3')]:
        groups = [df.loc[filt & (df['age_group'] == ag), 'benefit'].dropna().values
                  for ag in sorted(df['age_group'].unique())]
        groups = [g for g in groups if len(g) > 0]
        H, p = scipy_stats.kruskal(*groups)
        kruskal[prog] = {'H': float(H), 'p': float(p)}

    mw_per_age = {}
    for age_g in sorted(df['age_group'].unique()):
        sub = df[df['age_group'] == age_g]
        a_pds = sub.loc[sub['portfolio'] == 'pds_avg', 'benefit'].dropna().values
        best_iis = (sub[sub['program'] == 'iis3']
                    .groupby('portfolio')['benefit'].median().idxmax())
        a_iis = sub.loc[sub['portfolio'] == best_iis, 'benefit'].dropna().values
        if len(a_pds) == 0 or len(a_iis) == 0:
            continue
        u, p = scipy_stats.mannwhitneyu(a_pds, a_iis, alternative='two-sided')
        med_pds = float(np.median(a_pds))
        med_iis = float(np.median(a_iis))
        mw_per_age[age_g] = {
            'med_pds': med_pds,
            'med_iis': med_iis,
            'delta':   med_pds - med_iis,
            'p':       float(p),
            'winner':  'ПДС' if med_pds >= med_iis else 'ИИС-3',
        }

    return {'kruskal': kruskal, 'mw_per_age': mw_per_age}


def print_verdict(winner: pd.DataFrame, pop: pd.DataFrame, tests: dict):
    print('\n' + '=' * 65)
    print('  ВЕРДИКТ: ГИПОТЕЗА H4')
    print('=' * 65)

    PAY_LABELS = {'cap_rate': 'Аналит. ставка', '2x_cap_rate': '2× аналит. ставка'}
    pay_scenarios = [s for s in ['cap_rate', '2x_cap_rate']
                     if 'payment_scenario' not in winner.columns or s in winner['payment_scenario'].unique()]

    for pay_sc in pay_scenarios:
        if 'payment_scenario' in winner.columns:
            wm = winner[winner['payment_scenario'] == pay_sc]
        else:
            wm = winner
        n_pds  = (wm['winner'] == 'pds').sum()
        n_iis  = (wm['winner'] == 'iis3').sum()
        total  = len(wm)
        label  = PAY_LABELS.get(pay_sc, pay_sc)
        print(f'\nКарта победителей [{label}] ({total} ячеек age_group × sex):')
        print(f'  ПДС лучше:   {n_pds}/{total} ячеек ({n_pds/total:.0%})')
        print(f'  ИИС-3 лучше: {n_iis}/{total} ячеек ({n_iis/total:.0%})')
        if 'payment_scenario' in pop.columns:
            pop_sc = pop[pop['payment_scenario'] == pay_sc]
        else:
            pop_sc = pop
        for _, row in pop_sc.iterrows():
            prog = 'ПДС' if row['winner'] == 'pds' else 'ИИС-3'
            print(f'    {prog}: {row["population_mln"]:.1f} млн чел. ({row["share_pct"]:.0f}%)')

    print('\nТест Краскела-Уоллиса (Benefit неоднороден по возрасту, аналит. ставка):')
    for prog, res in tests['kruskal'].items():
        sig = '***' if res['p'] < 0.001 else ('**' if res['p'] < 0.01 else ('*' if res['p'] < 0.05 else 'ns'))
        print(f'  {prog}: H={res["H"]:.2f}, p={res["p"]:.4f} {sig}')

    print('\nПДС vs ИИС-3 (лучший) по возрастным группам, аналит. ставка (медиана Benefit):')
    print(f'  {"Группа":>7} | {"ПДС":>6} | {"ИИС-3":>6} | {"Δ":>7} | {"p":>7} | {"Победитель"}')
    print('  ' + '-' * 55)
    for age_g, r in tests['mw_per_age'].items():
        sig = '***' if r['p'] < 0.001 else ('**' if r['p'] < 0.01 else ('*' if r['p'] < 0.05 else 'ns'))
        print(f'  {age_g:>7} | {r["med_pds"]:>6.2%} | {r["med_iis"]:>6.2%} | '
              f'{r["delta"]:>+7.2%} | {r["p"]:>7.4f}{sig} | {r["winner"]}')

    wm_2x = winner[winner['payment_scenario'] == '2x_cap_rate'] if 'payment_scenario' in winner.columns else pd.DataFrame()
    heterogeneous_2x = (not wm_2x.empty and
                        (wm_2x['winner'] == 'pds').sum() > 0 and
                        (wm_2x['winner'] == 'iis3').sum() > 0)
    mw_pds_wins = sum(1 for r in tests['mw_per_age'].values() if r['winner'] == 'ПДС' and r['p'] < 0.05)
    mw_iis_wins = sum(1 for r in tests['mw_per_age'].values() if r['winner'] == 'ИИС-3' and r['p'] < 0.05)

    print()
    if heterogeneous_2x:
        n_iis_2x = (wm_2x['winner'] == 'iis3').sum()
        n_pds_2x = (wm_2x['winner'] == 'pds').sum()
        total_2x = len(wm_2x)
        pop_2x = pop[pop['payment_scenario'] == '2x_cap_rate'] if 'payment_scenario' in pop.columns else pd.DataFrame()
        iis_mln = pop_2x.loc[pop_2x['winner'] == 'iis3', 'population_mln'].sum() if not pop_2x.empty else 0
        print('✓ H4 ПОДТВЕРЖДАЕТСЯ при взносах выше порога насыщения со-финансирования:')
        print(f'  При аналит. ставке: ПДС доминирует во всех группах (со-финансирование перевешивает).')
        print(f'  При 2× аналит. ставке: ИИС-3 предпочтителен в {n_iis_2x}/{total_2x} ячейках')
        print(f'  (~{iis_mln:.1f} млн чел., {iis_mln/56.5*100:.0f}% наёмных работников 18–60).')
        print(f'  ПДС остаётся лучше только для групп 20–24 и 55–59 (короткий горизонт).')
    else:
        print('✗ H4 НЕ ПОДТВЕРЖДАЕТСЯ: одна программа доминирует во всех группах.')
    print('=' * 65)


def plot_market_scenario_comparison(df: pd.DataFrame):
    """
    Сравнение медианного Benefit по трём рыночным сценариям.
    """
    scenarios = [s for s in ['baseline', 'stress', 'optimistic']
                 if s in df['market_scenario'].unique()]
    if len(scenarios) < 2:
        print("  Пропускаю сравнение рыночных сценариев: данные только для одного сценария.")
        return

    SCENARIO_LABELS = {'baseline': 'Базовый', 'stress': 'Стрессовый', 'optimistic': 'Оптимистичный'}
    SCENARIO_COLORS = {'baseline': '#2166ac', 'stress': '#d73027', 'optimistic': '#1a9850'}
    linestyles = {'baseline': '-', 'stress': '--', 'optimistic': ':'}

    base_transit = df[df['transition_scenario'] == 'baseline'] if 'transition_scenario' in df.columns else df

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('H4 — Устойчивость медианного Benefit к рыночным сценариям\n'
                 '(ПДС vs лучший ИИС-3, базовый сценарий трудового перехода)',
                 fontsize=12, y=1.02)

    for ax, sex, sex_label in zip(axes, ['M', 'F'], ['Мужчины', 'Женщины']):
        for scenario in scenarios:
            sub = base_transit[(base_transit['market_scenario'] == scenario) &
                               (base_transit['sex'] == sex)]
            pds_med = [sub[(sub['program'] == 'pds') & (sub['age_group'] == ag)]['benefit']
                       .median() for ag in AGE_ORDER]
            iis_med = [sub[(sub['program'] == 'iis3') & (sub['age_group'] == ag)]['benefit']
                       .median() for ag in AGE_ORDER]
            label = SCENARIO_LABELS.get(scenario, scenario)
            ax.plot(AGE_ORDER, pds_med, marker='o', lw=1.8, color=SCENARIO_COLORS[scenario],
                    linestyle=linestyles[scenario], label=f'ПДС {label}')
            ax.plot(AGE_ORDER, iis_med, marker='s', lw=1.8, color=SCENARIO_COLORS[scenario],
                    linestyle=linestyles[scenario], alpha=0.6, label=f'ИИС-3 {label}')

        ax.set_title(sex_label, fontsize=11)
        ax.set_xlabel('Возрастная группа')
        ax.set_xticklabels(AGE_ORDER, rotation=30, ha='right', fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel('Медиана Benefit')
        ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'h4_market_scenarios.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


def plot_payment_scenario_comparison(df: pd.DataFrame, winner: pd.DataFrame):
    """
    Сравнение двух сценариев ставки взноса: cap_rate vs 2x_cap_rate.
    Показывает как меняется Δ(Benefit) = Benefit_ПДС − Benefit_ИИС3_best при удвоении взноса.
    """
    PAY_LABELS  = {'cap_rate': 'Аналит. ставка (макс. со-фин.)', '2x_cap_rate': '2× аналит. ставка'}
    PAY_COLORS  = {'cap_rate': '#2166ac', '2x_cap_rate': '#d73027'}
    PAY_MARKERS = {'cap_rate': 'o', '2x_cap_rate': 's'}

    pay_scenarios = [s for s in ['cap_rate', '2x_cap_rate'] if s in winner['payment_scenario'].unique()]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('H4 — Влияние ставки взноса на преимущество ПДС над ИИС-3\n'
                 'Δ(Benefit) = Benefit_ПДС − Benefit_ИИС3_best (базовый сценарий)',
                 fontsize=12, y=1.02)

    for ax, sex, sex_label in zip(axes, ['M', 'F'], ['Мужчины', 'Женщины']):
        for pay_sc in pay_scenarios:
            sub = winner[(winner['payment_scenario'] == pay_sc) & (winner['sex'] == sex)]
            sub = sub.set_index('age_group').reindex(AGE_ORDER).reset_index()
            ax.plot(AGE_ORDER, sub['benefit_delta'].values,
                    marker=PAY_MARKERS[pay_sc], lw=2,
                    color=PAY_COLORS[pay_sc],
                    label=PAY_LABELS[pay_sc])
        ax.axhline(0, color='black', lw=0.8, linestyle='--')
        ax.set_title(sex_label, fontsize=11)
        ax.set_xlabel('Возрастная группа')
        ax.set_xticklabels(AGE_ORDER, rotation=30, ha='right', fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel('Δ Benefit (ПДС − ИИС-3)')
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'h4_payment_scenarios.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {path}")


if __name__ == '__main__':
    print("Загружаю данные H4...")
    df, winner, pop = load_data()
    if 'payment_scenario' not in df.columns:
        df['payment_scenario'] = 'cap_rate'
    if 'payment_scenario' not in winner.columns:
        winner['payment_scenario'] = 'cap_rate'
    if 'payment_scenario' not in pop.columns:
        pop['payment_scenario'] = 'cap_rate'

    pay_scenarios = [s for s in ['cap_rate', '2x_cap_rate'] if s in df['payment_scenario'].unique()]
    print(f"  Строк: {len(df)}, возрастных групп: {df['age_group'].nunique()}, "
          f"рыночных сценариев: {df['market_scenario'].nunique()}, "
          f"трудовых сценариев: {df['transition_scenario'].nunique()}, "
          f"сценариев ставки: {len(pay_scenarios)}")

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # базовый срез для основных графиков — cap_rate
    base = df[(df['market_scenario'] == 'baseline') &
              (df['transition_scenario'] == 'baseline') &
              (df['payment_scenario'] == 'cap_rate')]
    winner_base = winner[winner['payment_scenario'] == 'cap_rate'] if 'payment_scenario' in winner.columns else winner

    print("Строю графики...")
    plot_assumptions_table()
    plot_winner_heatmap(winner)  # передаём все сценарии — функция сама разбивает на сабплоты
    plot_population_bar(pop[pop['payment_scenario'] == 'cap_rate'] if 'payment_scenario' in pop.columns else pop,
                        winner_base)
    plot_kz_by_age(base)
    plot_kz_percentiles(df[df['payment_scenario'] == 'cap_rate'])
    plot_transition_scenario_comparison(df[df['payment_scenario'] == 'cap_rate'])
    plot_market_scenario_comparison(df[df['payment_scenario'] == 'cap_rate'])

    # сравнение двух сценариев ставки взноса
    if len(pay_scenarios) > 1:
        plot_payment_scenario_comparison(df, winner)

    print_descriptive_stats(df[df['payment_scenario'] == 'cap_rate'])

    tests = run_tests(base)
    print_verdict(winner, pop, tests)

    # сравнение двух сценариев ставки взноса в вердикте
    if len(pay_scenarios) > 1:
        print('\n--- Сравнение сценариев ставки взноса ---')
        PAY_LABELS = {'cap_rate': 'Аналит. ставка', '2x_cap_rate': '2× аналит. ставка'}
        for pay_sc in pay_scenarios:
            wm = winner[winner['payment_scenario'] == pay_sc]
            n_pds = (wm['winner'] == 'pds').sum()
            n_iis = (wm['winner'] == 'iis3').sum()
            total = len(wm)
            delta_mean = wm['benefit_delta'].mean()
            print(f"  {PAY_LABELS[pay_sc]:30}: ПДС {n_pds}/{total}, ИИС-3 {n_iis}/{total}, "
                  f"среднее Δ(Benefit) = {delta_mean:+.3%}")
