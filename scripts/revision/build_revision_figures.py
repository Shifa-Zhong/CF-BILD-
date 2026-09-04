'''Build publication-quality main and SI figures for the major revision.'''

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from rdkit import Chem
from rdkit.Chem import Draw


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cf_bild.predictive import zero_truncated_normal_quantile  # noqa: E402


RESULTS = ROOT / 'output' / 'revision_2026'
FIGURE_DIR = ROOT / 'figures'
BLUE = '#2F6690'
TEAL = '#3A7D8C'
ORANGE = '#E07A5F'
GREEN = '#5B8E7D'
GOLD = '#D8A23A'
GRAY = '#7A7A7A'
LIGHT = '#EEF3F6'


mpl.rcParams.update({
    'font.family': 'Arial',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8,
    'axes.linewidth': 0.8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


def save_figure(figure, name):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        FIGURE_DIR / f'{name}.png',
        dpi=600,
        bbox_inches='tight',
        facecolor='white',
    )
    figure.savefig(
        FIGURE_DIR / f'{name}.pdf',
        bbox_inches='tight',
        facecolor='white',
    )
    plt.close(figure)


def panel_label(axis, label):
    axis.text(
        -0.13, 1.05, label,
        transform=axis.transAxes,
        fontsize=11,
        fontweight='bold',
        va='top',
    )


def figure_1_workflow():
    figure, axis = plt.subplots(figsize=(12.0, 3.8))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis('off')
    boxes = [
        (0.02, 0.22, 0.20, 0.60, BLUE, '1  Ion vocabulary',
         '505 cations x 173 anions\n87,365 formal, charge-balanced\ncandidate ion pairs'),
        (0.275, 0.22, 0.20, 0.60, TEAL, '2  Probabilistic surrogate',
         'Collision-free Morgan features\nCompositional GP kernel\nFull non-test refit + uncertainty'),
        (0.53, 0.22, 0.20, 0.60, ORANGE, '3  Multi-objective ranking',
         'FW-AEI primary screen\nAnalytical q = 1 EHVI benchmark\nTraining-derived operating points'),
        (0.785, 0.22, 0.20, 0.60, GREEN, '4  Chemistry triage',
         'Tier I proton-transfer flag\nTier II liquid-likeness score\nTier III pair-reactivity flag'),
    ]
    for x, y, width, height, color, title, body in boxes:
        patch = FancyBboxPatch(
            (x, y), width, height,
            boxstyle='round,pad=0.012,rounding_size=0.018',
            linewidth=1.5,
            edgecolor=color,
            facecolor='white',
        )
        axis.add_patch(patch)
        axis.add_patch(FancyBboxPatch(
            (x, y + height - 0.14), width, 0.14,
            boxstyle='round,pad=0.012,rounding_size=0.018',
            linewidth=0,
            facecolor=color,
        ))
        axis.text(
            x + 0.014, y + height - 0.07, title,
            color='white', fontweight='bold', fontsize=10.5,
            va='center',
        )
        axis.text(
            x + width / 2, y + 0.27, body,
            ha='center', va='center', fontsize=9.2, linespacing=1.45,
        )
    for start, end in ((0.225, 0.27), (0.48, 0.525), (0.735, 0.78)):
        axis.add_patch(FancyArrowPatch(
            (start, 0.52), (end, 0.52),
            arrowstyle='-|>', mutation_scale=13,
            linewidth=1.4, color=GRAY,
        ))
    axis.text(
        0.5, 0.08,
        'Held-out species-level test sets are used only for final evaluation; '
        'ranked candidates remain hypotheses for experimental validation.',
        ha='center', va='center', fontsize=9.2, color='#333333',
    )
    save_figure(figure, 'Figure_1_revision')


def figure_2_vocabulary():
    cations = pd.read_csv(RESULTS / 'vocabulary_cation_families.csv')
    anions = pd.read_csv(RESULTS / 'vocabulary_anion_families.csv')
    tsne = pd.read_csv(RESULTS / 'vocabulary_tsne_coordinates.csv')
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    for axis, table, color, title in (
        (axes[0], cations, BLUE, 'Cation families (n = 505)'),
        (axes[1], anions, TEAL, 'Anion families (n = 173)'),
    ):
        plot = table.sort_values('count')
        axis.barh(plot['family'], plot['count'], color=color, alpha=0.9)
        for y, value in enumerate(plot['count']):
            axis.text(value + max(plot['count']) * 0.02, y, str(value), va='center')
        axis.set_xlabel('Number of unique ions')
        axis.set_title(title)
        axis.spines[['top', 'right']].set_visible(False)
    family_codes = pd.Categorical(tsne['family'])
    axes[2].scatter(
        tsne['tsne_1'], tsne['tsne_2'],
        c=family_codes.codes,
        cmap='tab20',
        s=11,
        alpha=0.75,
        linewidths=0,
    )
    axes[2].set_xlabel('t-SNE coordinate 1')
    axes[2].set_ylabel('t-SNE coordinate 2')
    axes[2].set_title('CF-MF structural map')
    axes[2].spines[['top', 'right']].set_visible(False)
    for label, axis in zip(('a', 'b', 'c'), axes):
        panel_label(axis, label)
    figure.tight_layout(w_pad=2.3)
    save_figure(figure, 'Figure_2_revision')


def empirical_coverage(table, property_name, levels):
    mu = table['latent_mu'].to_numpy()
    variance = table['latent_std'].to_numpy() ** 2
    y_true = table['y_true'].to_numpy()
    observed = []
    for level in levels:
        tail = (1.0 - level) / 2.0
        if property_name == 'co2':
            lower = zero_truncated_normal_quantile(mu, variance, tail)
            upper = zero_truncated_normal_quantile(
                mu, variance, 1.0 - tail
            )
        else:
            sigma = np.sqrt(variance)
            from scipy.stats import norm
            lower = mu + sigma * norm.ppf(tail)
            upper = mu + sigma * norm.ppf(1.0 - tail)
        observed.append(np.mean((y_true >= lower) & (y_true <= upper)))
    return np.asarray(observed)


def figure_3_surrogates():
    with (RESULTS / 'revision_model_summary.json').open(encoding='utf-8') as h:
        summary = json.load(h)
    labels = {
        'co2': ('CO2 capacity (mol mol$^{-1}$)', BLUE),
        'vis': ('ln viscosity (Pa s)', ORANGE),
        'tox': ('ln EC50', GREEN),
    }
    figure, axes = plt.subplots(2, 3, figsize=(11.4, 7.0))
    levels = np.linspace(0.1, 0.95, 18)
    for column, property_name in enumerate(('co2', 'vis', 'tox')):
        table = pd.read_csv(
            RESULTS / f'test_predictions_{property_name}.csv'
        )
        label, color = labels[property_name]
        axis = axes[0, column]
        axis.scatter(
            table['y_true'], table['pred_mean'],
            s=11, alpha=0.45, color=color, edgecolors='none',
        )
        low = min(table['y_true'].min(), table['pred_mean'].min())
        high = max(table['y_true'].max(), table['pred_mean'].max())
        axis.plot([low, high], [low, high], '--', color=GRAY, linewidth=1)
        axis.set_xlabel(f'Experimental {label}')
        axis.set_ylabel(f'Predicted {label}')
        metric = summary['properties'][property_name]
        axis.text(
            0.04, 0.94,
            f'R$^2$ = {metric["r2"]:.3f}\nRMSE = {metric["rmse"]:.3f}',
            transform=axis.transAxes,
            va='top',
            bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.85,
                  'edgecolor': LIGHT},
        )
        axis.spines[['top', 'right']].set_visible(False)
        panel_label(axis, chr(ord('a') + column))

        coverage_axis = axes[1, column]
        observed = empirical_coverage(table, property_name, levels)
        coverage_axis.plot(
            levels, observed, marker='o', markersize=3.5,
            linewidth=1.4, color=color,
        )
        coverage_axis.plot([0, 1], [0, 1], '--', color=GRAY, linewidth=1)
        coverage_axis.set_xlim(0.05, 1.0)
        coverage_axis.set_ylim(0.05, 1.02)
        coverage_axis.set_xlabel('Nominal interval coverage')
        coverage_axis.set_ylabel('Empirical coverage')
        coverage_axis.spines[['top', 'right']].set_visible(False)
        panel_label(coverage_axis, chr(ord('d') + column))
    figure.tight_layout(h_pad=2.0, w_pad=2.0)
    save_figure(figure, 'Figure_3_revision')


def figure_4_screening():
    comparison = pd.read_csv(RESULTS / 'acquisition_comparison_revision.csv')
    top = pd.read_csv(RESULTS / 'top_candidates_revision.csv')
    stability = pd.read_csv(RESULTS / 'stability_screening_revision.csv')
    post_filter = pd.read_csv(RESULTS / 'post_filter_top10_revision.csv')
    figure, axes = plt.subplots(2, 3, figsize=(12.0, 7.4))

    method_colors = [ORANGE, GRAY, BLUE, '#B9BDC1']
    names = ['FW-AEI', 'Additive\nEI', 'Analytical\nq=1 EHVI', 'Random\n(5 seeds)']
    x = np.arange(len(comparison))
    errors = comparison['hypervolume_se'].fillna(0.0)
    axes[0, 0].bar(
        x, comparison['hypervolume'], yerr=errors,
        color=method_colors, capsize=3,
    )
    axes[0, 0].set_xticks(x, names)
    axes[0, 0].set_ylabel('Top-100 dominated hypervolume')
    axes[0, 0].spines[['top', 'right']].set_visible(False)

    axes[0, 1].bar(
        x, comparison['mean_sigma'], color=method_colors
    )
    axes[0, 1].set_xticks(x, names)
    axes[0, 1].set_ylabel('Mean predictive standard deviation')
    axes[0, 1].spines[['top', 'right']].set_visible(False)

    score_color = np.log10(np.maximum(top['fw_aei_score'], 1e-12))
    for axis, x_column, x_label in (
        (axes[0, 2], 'vis_pred', 'Predicted ln viscosity (Pa s)'),
        (axes[1, 0], 'tox_pred', 'Predicted ln EC50'),
    ):
        scatter = axis.scatter(
            top[x_column], top['co2_pred'],
            c=score_color, cmap='viridis',
            s=22, alpha=0.78, edgecolors='none',
        )
        pareto = top['pareto_within_selected_set'].astype(bool)
        axis.scatter(
            top.loc[pareto, x_column],
            top.loc[pareto, 'co2_pred'],
            marker='*', s=70, facecolors='none',
            edgecolors=ORANGE, linewidths=1.0,
            label='Pareto within top 100',
        )
        axis.set_xlabel(x_label)
        axis.set_ylabel('Predicted CO2 capacity (mol mol$^{-1}$)')
        axis.spines[['top', 'right']].set_visible(False)
        axis.legend(loc='best', frameon=False)
    color_axis = axes[0, 2].inset_axes([0.50, 0.78, 0.45, 0.045])
    colorbar = figure.colorbar(
        scatter, cax=color_axis, orientation='horizontal'
    )
    colorbar.set_label('log10(FW-AEI)', fontsize=7)
    color_axis.tick_params(labelsize=6, length=2)

    status_order = ['Pass', 'Caution', 'Fail']
    counts = stability['overall_status'].value_counts().reindex(
        status_order, fill_value=0
    )
    axes[1, 1].bar(
        status_order, counts,
        color=[GREEN, GOLD, ORANGE],
    )
    for index, value in enumerate(counts):
        axes[1, 1].text(index, value + 1, str(value), ha='center')
    axes[1, 1].set_ylabel('Number of FW-AEI top-100 candidates')
    axes[1, 1].spines[['top', 'right']].set_visible(False)

    raw = np.column_stack([
        post_filter['co2_pred'],
        -post_filter['vis_pred'],
        post_filter['tox_pred'],
    ])
    minimum = raw.min(axis=0)
    span = np.maximum(raw.max(axis=0) - minimum, 1e-12)
    normalized = (raw - minimum) / span
    image = axes[1, 2].imshow(
        normalized, aspect='auto', cmap='Blues', vmin=0, vmax=1
    )
    axes[1, 2].set_xticks(
        range(3), ['CO2', '-ln(viscosity)', 'ln(EC50)']
    )
    axes[1, 2].set_yticks(
        range(len(post_filter)),
        [f'Rank {int(value)}' for value in post_filter['rank']],
    )
    for row in range(raw.shape[0]):
        for column in range(raw.shape[1]):
            axes[1, 2].text(
                column, row, f'{raw[row, column]:.2f}',
                ha='center', va='center',
                color='white' if normalized[row, column] > 0.55 else '#222222',
                fontsize=7.5,
            )
    axes[1, 2].set_title('Top 10 post-filter candidates')
    axes[1, 2].tick_params(length=0)

    for label, axis in zip(('a', 'b', 'c', 'd', 'e', 'f'), axes.flat):
        panel_label(axis, label)
    figure.subplots_adjust(
        left=0.07, right=0.94, bottom=0.08, top=0.97,
        wspace=0.34, hspace=0.38,
    )
    save_figure(figure, 'Figure_4_revision')


def figure_s1_vocabulary_mw():
    table = pd.read_csv(RESULTS / 'vocabulary_ions_with_mw.csv')
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for axis, ion_type, color in (
        (axes[0], 'cation', BLUE),
        (axes[1], 'anion', TEAL),
    ):
        subset = table[table['type'] == ion_type]
        order = (
            subset.groupby('family')['mw'].median()
            .sort_values().index.tolist()
        )
        values = [
            subset.loc[subset['family'] == family, 'mw'].to_numpy()
            for family in order
        ]
        boxes = axis.boxplot(
            values, tick_labels=order, vert=False,
            patch_artist=True, showfliers=False,
        )
        for box in boxes['boxes']:
            box.set_facecolor(color)
            box.set_alpha(0.75)
        axis.set_xlabel('Molecular weight (Da)')
        axis.set_title(f'{ion_type.capitalize()} vocabulary')
        axis.spines[['top', 'right']].set_visible(False)
    panel_label(axes[0], 'a')
    panel_label(axes[1], 'b')
    figure.tight_layout(w_pad=2.2)
    save_figure(figure, 'Figure_S1_revision')


def figure_s2_model_comparison():
    with (RESULTS / 'revision_model_summary.json').open(encoding='utf-8') as h:
        tuned = json.load(h)
    with (RESULTS / 'median_heuristic_baseline.json').open(encoding='utf-8') as h:
        baseline = json.load(h)
    properties = ['CO2', 'Viscosity', 'Toxicity']
    keys = ['co2', 'vis', 'tox']
    tuned_r2 = [tuned['properties'][key]['r2'] for key in keys]
    baseline_r2 = [
        baseline['properties'][key]['metrics']['r2'] for key in keys
    ]
    coverage = [
        tuned['properties'][key]['coverage_95'] for key in keys
    ]
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    positions = np.arange(3)
    width = 0.36
    axes[0].bar(
        positions - width / 2, tuned_r2, width,
        label='CV-selected GP', color=BLUE,
    )
    axes[0].bar(
        positions + width / 2, baseline_r2, width,
        label='Median-heuristic GP', color=GRAY,
    )
    axes[0].set_xticks(positions, properties)
    axes[0].set_ylabel('Held-out test R$^2$')
    axes[0].legend(frameon=False)
    axes[0].spines[['top', 'right']].set_visible(False)

    axes[1].bar(positions, coverage, color=[BLUE, ORANGE, GREEN])
    axes[1].axhline(0.95, color=GRAY, linestyle='--', linewidth=1)
    axes[1].set_xticks(positions, properties)
    axes[1].set_ylim(0.85, 1.01)
    axes[1].set_ylabel('Empirical coverage of nominal 95% interval')
    axes[1].spines[['top', 'right']].set_visible(False)
    panel_label(axes[0], 'a')
    panel_label(axes[1], 'b')
    figure.tight_layout(w_pad=2.5)
    save_figure(figure, 'Figure_S2_revision')


def figure_s3_pareto_projection():
    top = pd.read_csv(RESULTS / 'top_candidates_revision.csv')
    figure, axis = plt.subplots(figsize=(5.4, 4.4))
    score = np.log10(np.maximum(top['fw_aei_score'], 1e-12))
    scatter = axis.scatter(
        top['vis_pred'], top['tox_pred'],
        c=score, cmap='viridis', s=28, alpha=0.8, edgecolors='none',
    )
    pareto = top['pareto_within_selected_set'].astype(bool)
    axis.scatter(
        top.loc[pareto, 'vis_pred'],
        top.loc[pareto, 'tox_pred'],
        marker='*', s=85, facecolors='none',
        edgecolors=ORANGE, linewidths=1.0,
        label='Pareto within top 100',
    )
    axis.set_xlabel('Predicted ln viscosity (Pa s)')
    axis.set_ylabel('Predicted ln EC50')
    axis.legend(frameon=False)
    axis.spines[['top', 'right']].set_visible(False)
    colorbar = figure.colorbar(scatter, ax=axis)
    colorbar.set_label('log10(FW-AEI score)')
    figure.tight_layout()
    save_figure(figure, 'Figure_S3_revision')


def figure_s4_threshold_sensitivity():
    table = pd.read_csv(RESULTS / 'threshold_sensitivity_revision.csv')
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.7))
    axes[0].plot(
        table['threshold_percentile'],
        table['pareto_count_top50'],
        marker='o', color=BLUE, linewidth=1.6,
    )
    axes[0].set_xlabel('Feasibility threshold percentile')
    axes[0].set_ylabel('Pareto candidates in top 50')
    axes[0].spines[['top', 'right']].set_visible(False)
    axes[1].semilogy(
        table['threshold_percentile'],
        table['mean_fw_aei_top50'],
        marker='o', color=ORANGE, linewidth=1.6,
    )
    axes[1].set_xlabel('Feasibility threshold percentile')
    axes[1].set_ylabel('Mean FW-AEI score in top 50')
    axes[1].spines[['top', 'right']].set_visible(False)
    axes[2].plot(
        table['threshold_percentile'],
        table['hypervolume_top50'],
        marker='o', color=GREEN, linewidth=1.6,
    )
    axes[2].set_xlabel('Feasibility threshold percentile')
    axes[2].set_ylabel('Top-50 dominated hypervolume')
    axes[2].spines[['top', 'right']].set_visible(False)
    panel_label(axes[0], 'a')
    panel_label(axes[1], 'b')
    panel_label(axes[2], 'c')
    figure.tight_layout(w_pad=2.2)
    save_figure(figure, 'Figure_S4_revision')


def figure_s5_structures():
    table = pd.read_csv(
        RESULTS / 'post_filter_top10_revision.csv'
    ).query('overall_status == "Pass"').head(5)
    molecules = []
    legends = []
    for _, row in table.iterrows():
        rank = int(row['rank'])
        molecules.extend([
            Chem.MolFromSmiles(row['cation']),
            Chem.MolFromSmiles(row['anion']),
        ])
        legends.extend([
            f'Rank {rank} cation',
            f'Rank {rank} anion',
        ])
    draw_options = Draw.rdMolDraw2D.MolDrawOptions()
    draw_options.legendFontSize = 32
    draw_options.minFontSize = 20
    draw_options.maxFontSize = 32
    image = Draw.MolsToGridImage(
        molecules,
        molsPerRow=2,
        subImgSize=(900, 420),
        legends=legends,
        drawOptions=draw_options,
        useSVG=False,
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    image.save(
        FIGURE_DIR / 'Figure_S5_revision.png',
        dpi=(600, 600),
    )
    figure, axis = plt.subplots(figsize=(7.5, 8.8))
    axis.imshow(image)
    axis.axis('off')
    figure.tight_layout(pad=0.1)
    figure.savefig(
        FIGURE_DIR / 'Figure_S5_revision.pdf',
        bbox_inches='tight',
        facecolor='white',
    )
    plt.close(figure)


def toc_graphic():
    figure, axis = plt.subplots(figsize=(3.15, 1.57))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis('off')
    items = [
        (0.02, BLUE, '505 x 173\nion pairs'),
        (0.36, ORANGE, 'GP uncertainty\n+ FW-AEI'),
        (0.70, GREEN, 'transparent\nstability triage'),
    ]
    for x, color, label in items:
        axis.add_patch(FancyBboxPatch(
            (x, 0.25), 0.27, 0.50,
            boxstyle='round,pad=0.012,rounding_size=0.025',
            facecolor=color, edgecolor='none',
        ))
        axis.text(
            x + 0.135, 0.50, label,
            color='white', ha='center', va='center',
            fontsize=8, fontweight='bold',
        )
    for start, end in ((0.29, 0.35), (0.63, 0.69)):
        axis.add_patch(FancyArrowPatch(
            (start, 0.50), (end, 0.50),
            arrowstyle='-|>', mutation_scale=10,
            linewidth=1.0, color=GRAY,
        ))
    save_figure(figure, 'TOC_graphic_revision')
    text = (
        'CF-BILD combines fragment-constrained ion-pair enumeration, '
        'compositional Gaussian-process uncertainty, feasibility-weighted '
        'multiobjective ranking and transparent stability triage for '
        'reproducible ionic-liquid screening.'
    )
    (FIGURE_DIR.parent / 'TOC_text_revision.txt').write_text(
        text, encoding='utf-8'
    )


def main():
    figure_1_workflow()
    figure_2_vocabulary()
    restricted_predictions = [
        RESULTS / f'test_predictions_{name}.csv'
        for name in ('co2', 'vis', 'tox')
    ]
    if all(path.exists() for path in restricted_predictions):
        figure_3_surrogates()
    else:
        print(
            'Skipping Figure 3: regeneration requires the three restricted '
            'held-out target/prediction tables. The submitted 600 dpi PNG '
            'and vector PDF remain in figures/.'
        )
    figure_4_screening()
    figure_s1_vocabulary_mw()
    figure_s2_model_comparison()
    figure_s3_pareto_projection()
    figure_s4_threshold_sensitivity()
    figure_s5_structures()
    toc_graphic()
    print(f'Figures written to {FIGURE_DIR}')


if __name__ == '__main__':
    main()
