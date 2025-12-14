#!/usr/bin/env python3
import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
plt.rcParams.update({'figure.max_open_warning': 0})

def load_and_clean(csv_path, solver='CBS'):
    df = pd.read_csv(csv_path, dtype=str, sep=None, engine='python')
    df.columns = [c.strip() for c in df.columns]
    if 'solver' in df.columns:
        df = df[df['solver'] == solver].copy()
    # numeric conversions
    for col in ['cost', 'high_expanded', 'high_generated', 'low_expanded', 'low_generated']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'cpu_time_or_status' in df.columns:
        df['cpu_time_raw'] = df['cpu_time_or_status']
        df['cpu_time'] = pd.to_numeric(df['cpu_time_or_status'], errors='coerce')
    else:
        df['cpu_time_raw'] = np.nan
        df['cpu_time'] = np.nan
    if 'mode' in df.columns:
        df['mode'] = df['mode'].astype(str).str.strip().str.lower()
    df['instance_short'] = df['instance'].apply(lambda s: os.path.basename(str(s)))
    return df

def pivot_wide(df, metrics=('cost','low_expanded','low_generated','cpu_time')):
    instances = sorted(df['instance'].unique())
    modes = sorted(df['mode'].unique())
    rows = []
    for inst in instances:
        row = {'instance': inst, 'instance_short': os.path.basename(inst)}
        sub = df[df['instance'] == inst]
        for m in modes:
            subm = sub[sub['mode'] == m]
            if len(subm) == 0:
                for metric in metrics:
                    row[f'{metric}_{m}'] = np.nan
            else:
                # prefer a numeric cpu_time row if possible
                if 'cpu_time' in subm.columns:
                    numeric = subm[~subm['cpu_time'].isna()]
                    chosen = numeric.iloc[0] if len(numeric) > 0 else subm.iloc[0]
                else:
                    chosen = subm.iloc[0]
                for metric in metrics:
                    row[f'{metric}_{m}'] = chosen.get(metric, np.nan)
        rows.append(row)
    return pd.DataFrame(rows), modes

def plot_line_by_instance(wide, modes, metric, outpath, sort_by_mode='astar'):
    base_col = f'{metric}_{sort_by_mode}'
    wide_sorted = wide.sort_values(by=base_col, ascending=True, na_position='last').reset_index(drop=True)
    insts = wide_sorted['instance_short']
    x = np.arange(len(insts))
    plt.figure(figsize=(max(8, len(insts)*0.12), 4.5))
    for m in modes:
        col = f'{metric}_{m}'
        if col in wide_sorted.columns:
            plt.plot(x, wide_sorted[col], marker='o', label=m)
    plt.xticks(x, insts, rotation=90, fontsize=6)
    plt.xlabel('instance')
    plt.ylabel(metric)
    plt.title(f'Per-instance {metric} by mode (sorted by {sort_by_mode})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_boxplots(df, modes, out_low, out_cpu, showfliers=False):
    data = [df[df['mode']==m]['low_expanded'].dropna().astype(float).values for m in modes]
    plt.figure(figsize=(6,4))
    # matplotlib will accept labels param; if it warns upgrade later it's harmless
    plt.boxplot(data, labels=modes, showmeans=True, showfliers=showfliers)
    plt.ylabel('low_expanded (nodes)')
    plt.title('Low-level expanded nodes distributions')
    plt.tight_layout(); plt.savefig(out_low, dpi=200); plt.close()

    data = [df[df['mode']==m]['cpu_time'].dropna().astype(float).values for m in modes]
    plt.figure(figsize=(6,4))
    plt.boxplot(data, labels=modes, showmeans=True, showfliers=showfliers)
    plt.ylabel('cpu_time (s)')
    plt.title('CPU time distribution by mode')
    plt.tight_layout(); plt.savefig(out_cpu, dpi=200); plt.close()

def plot_ecdf(wide, modes, metric, outpath):
    plt.figure(figsize=(6,4))
    for m in modes:
        col = f'{metric}_{m}'
        if col not in wide.columns:
            continue
        vals = wide[col].dropna().astype(float)
        if len(vals)==0:
            continue
        vals_sorted = np.sort(vals)
        y = np.arange(1, len(vals_sorted)+1) / float(len(vals_sorted))
        plt.step(vals_sorted, y, where='post', label=m)
    plt.xlabel(metric); plt.ylabel('ECDF')
    plt.title(f'Empirical CDF of {metric} by mode'); plt.legend(); plt.tight_layout()
    plt.savefig(outpath, dpi=200); plt.close()

def plot_scatter_low_vs_cpu(df, modes, outpath):
    plt.figure(figsize=(7,5))
    for m in modes:
        sub = df[df['mode']==m]
        x = sub['low_expanded'].astype(float)
        y = sub['cpu_time'].astype(float)
        mask = (~x.isna()) & (~y.isna())
        if mask.sum() > 0:
            plt.scatter(x[mask], y[mask], alpha=0.6, label=m, s=20)
    plt.xscale('symlog'); plt.yscale('symlog')
    plt.xlabel('low_expanded'); plt.ylabel('cpu_time (s)')
    plt.title('Low-level expanded vs CPU time'); plt.legend(); plt.tight_layout()
    plt.savefig(outpath, dpi=200); plt.close()

def plot_scatter_cost_nodes_time(wide, modes, outdir):
    # cost vs nodes
    plt.figure(figsize=(6,4))
    for m in modes:
        cn = f'low_expanded_{m}'; cc = f'cost_{m}'
        if cn in wide.columns and cc in wide.columns:
            x = wide[cn].astype(float); y = wide[cc].astype(float)
            mask = (~x.isna()) & (~y.isna())
            if mask.sum()>0:
                plt.scatter(x[mask], y[mask], alpha=0.6, label=m, s=20)
    plt.xscale('symlog'); plt.xlabel('low_expanded'); plt.ylabel('cost')
    plt.title('Cost vs low_expanded'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir,'scatter_cost_vs_nodes.png'), dpi=200); plt.close()

    # cpu_time vs nodes
    plt.figure(figsize=(6,4))
    for m in modes:
        cn = f'low_expanded_{m}'; ct = f'cpu_time_{m}'
        if cn in wide.columns and ct in wide.columns:
            x = wide[cn].astype(float); y = wide[ct].astype(float)
            mask = (~x.isna()) & (~y.isna())
            if mask.sum()>0:
                plt.scatter(x[mask], y[mask], alpha=0.6, label=m, s=20)
    plt.xscale('symlog'); plt.yscale('symlog')
    plt.xlabel('low_expanded'); plt.ylabel('cpu_time (s)')
    plt.title('CPU time vs low_expanded'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir,'scatter_cost_vs_time.png'), dpi=200); plt.close()

def plot_ratio_over_baseline(wide, modes, baseline='astar', outdir='analysis3'):
    for m in modes:
        if m == baseline: continue
        col_base = f'low_expanded_{baseline}'; col_other = f'low_expanded_{m}'
        if col_base in wide.columns and col_other in wide.columns:
            base = wide[col_base].astype(float)
            other = wide[col_other].astype(float)
            ratio = np.where((base == 0) | np.isnan(base), np.nan, other/base)
            plt.figure(figsize=(8,3))
            plt.plot(ratio, marker='o', linestyle='none')
            plt.title(f'Per-instance ratio low_expanded ({m} / {baseline})')
            plt.xlabel('instance index (sorted by astar)'); plt.ylabel('ratio')
            plt.tight_layout(); plt.savefig(os.path.join(outdir,f'ratio_{m}_over_{baseline}.png'), dpi=200); plt.close()

def heatmap_instances_modes(wide, modes, metric, outpath):
    insts = list(wide['instance_short'])
    mat = []
    for m in modes:
        col = f'{metric}_{m}'
        if col in wide.columns:
            vals = wide[col].astype(float).values
        else:
            vals = np.array([np.nan]*len(insts))
        mat.append(vals)
    mat = np.array(mat)
    plt.figure(figsize=(max(6, len(insts)*0.12), max(3, len(modes)*0.4)))
    im = plt.imshow(mat, aspect='auto', interpolation='nearest', cmap='viridis')
    plt.colorbar(im, label=metric)
    plt.yticks(np.arange(len(modes)), modes)
    plt.xticks(np.arange(len(insts)), insts, rotation=90, fontsize=6)
    plt.title(f'Heatmap: {metric} (modes x instances)'); plt.tight_layout()
    plt.savefig(outpath, dpi=200); plt.close()


def make_outputs(csv_path='results.csv', outdir='analysis3', solver='CBS'):
    print('Loading:', csv_path)
    df = load_and_clean(csv_path, solver)
    if df.empty:
        print('No rows after filtering. Exiting.')
        return
    os.makedirs(outdir, exist_ok=True)

    # ensure numeric columns exist
    for col in ['low_expanded','cpu_time','cost']:
        if col not in df.columns:
            df[col] = np.nan

    wide, modes = pivot_wide(df)

    # produce plots (no extra CSV files)
    plot_line_by_instance(wide, modes, 'low_expanded', os.path.join(outdir,'line_low_expanded_sorted_by_astar.png'), sort_by_mode='astar')
    plot_line_by_instance(wide, modes, 'cpu_time', os.path.join(outdir,'line_cpu_time_sorted_by_astar.png'), sort_by_mode='astar')
    plot_boxplots(df, modes, os.path.join(outdir,'boxplot_low_expanded.png'), os.path.join(outdir,'boxplot_cpu_time.png'))
    plot_ecdf(wide, modes, 'low_expanded', os.path.join(outdir,'ecdf_low_expanded.png'))
    plot_scatter_low_vs_cpu(df, modes, os.path.join(outdir,'scatter_nodes_vs_time.png'))
    plot_scatter_cost_nodes_time(wide, modes, outdir)
    plot_ratio_over_baseline(wide, modes, baseline='astar', outdir=outdir)
    heatmap_instances_modes(wide, modes, 'low_expanded', os.path.join(outdir,'heatmap_instance_vs_mode_low_expanded.png'))

    # histograms
    plt.figure(figsize=(6,3))
    df['low_expanded'].dropna().astype(float).clip(0,1e6).hist(bins=60)
    plt.title('Histogram low_expanded'); plt.tight_layout()
    plt.savefig(os.path.join(outdir,'hist_low_expanded.png'), dpi=200); plt.close()

    plt.figure(figsize=(6,3))
    df['cpu_time'].dropna().astype(float).clip(0,1e3).hist(bins=60)
    plt.title('Histogram cpu_time (s)'); plt.tight_layout()
    plt.savefig(os.path.join(outdir,'hist_cpu_time.png'), dpi=200); plt.close()

    # consolidated single CSV to summarize all
    wide_to_merge = wide.copy()
    merged = df.merge(wide_to_merge, on='instance', how='left', suffixes=('','_wide'))
    combined_csv = os.path.join(outdir, 'summary_all.csv')
    merged.to_csv(combined_csv, index=False)
    print('Wrote combined CSV:', combined_csv)


def main():
    make_outputs(csv_path='results.csv', outdir='analysis3', solver='CBS')

if __name__ == '__main__':
    main()
