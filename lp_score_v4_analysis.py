
import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from pathlib import Path

def main(/content/dex-temp-db.score_v4.csv, out_dir="lp_score_v4_outputs"):
    OUT_DIR = Path(out_dir)
    FIG_DIR = OUT_DIR / "figures"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(/content/dex-temp-db.score_v4.csv, low_memory=False)

    # Identify pool indices
    import re
    pool_indices = sorted(set(int(re.search(r'lp_scores\[(\d+)\]', c).group(1))
                              for c in df.columns if re.search(r'lp_scores\[(\d+)\]', c)))

    # Aggregated and per-pool totals
    total_score_cols = [f'lp_scores[{i}].total_score'.format(i=i) for i in pool_indices if f'lp_scores[{i}].total_score'.format(i=i) in df.columns]
    br_total_cols    = [f'lp_scores[{i}].score_breakdown.total_score'.format(i=i) for i in pool_indices if f'lp_scores[{i}].score_breakdown.total_score'.format(i=i) in df.columns]

    df['_sum_pool_total_score'] = df[total_score_cols].fillna(0).sum(axis=1)
    df['_sum_pool_breakdown_total'] = df[br_total_cols].fillna(0).sum(axis=1)
    df['_agg'] = df['aggregated_lp_score']
    df['_diff_sum_pool'] = df['_agg'] - df['_sum_pool_total_score']

    # Category breakdown
    cat_cols = ['lp_category_breakdown.stable-stable','lp_category_breakdown.stable-volatile','lp_category_breakdown.volatile-volatile']
    df['_cat_sum'] = df[cat_cols].fillna(0).sum(axis=1)

    # Aggregate retention
    pool_fields = {
        'dep': 'total_deposit_all_time',
        'wd': 'total_withdraw_all_time',
        'ret': 'retained_liquidity',
        'rem': 'liquidity_percent_remaining'
    }
    dep_cols = [f'lp_scores[{i}].'+pool_fields['dep'] for i in pool_indices if f'lp_scores[{i}].'+pool_fields['dep'] in df.columns]
    wd_cols  = [f'lp_scores[{i}].'+pool_fields['wd']  for i in pool_indices if f'lp_scores[{i}].'+pool_fields['wd']  in df.columns]
    ret_cols = [f'lp_scores[{i}].'+pool_fields['ret'] for i in pool_indices if f'lp_scores[{i}].'+pool_fields['ret'] in df.columns]
    rem_cols = [f'lp_scores[{i}].'+pool_fields['rem'] for i in pool_indices if f'lp_scores[{i}].'+pool_fields['rem'] in df.columns]

    df['_sum_dep'] = df[dep_cols].fillna(0).sum(axis=1)
    df['_sum_wd']  = df[wd_cols].fillna(0).sum(axis=1)
    df['_sum_retained'] = df[ret_cols].fillna(0).sum(axis=1)
    df['_mean_pct_remaining'] = df[rem_cols].replace(0, np.nan).mean(axis=1, skipna=True)
    df['_max_pct_remaining']  = df[rem_cols].max(axis=1, skipna=True)

    # Long pool-level frame
    fields = [
        'pool_id','pool_name','num_deposits','num_withdrawals','avg_holding_days',
        'liquidity_percent_remaining','retained_liquidity','lp_volatility_stddev',
        'dust_deposit_count','dust_deposit_volume','total_deposit_all_time','total_withdraw_all_time',
        'score_breakdown.deposit_volume_score','score_breakdown.withdraw_volume_score','score_breakdown.deposit_frequency_score',
        'score_breakdown.avg_holding_time_score','score_breakdown.liquidity_retention_score','score_breakdown.lp_volatility_score',
        'score_breakdown.time_score','score_breakdown.total_score','total_score','last_tx_timestamp','timestamp','tvl'
    ]
    long_rows = []
    for i in pool_indices:
        prefix = f'lp_scores[{i}]'
        cols_map = {f: f'{prefix}.{f}' for f in fields if f'{prefix}.{f}' in df.columns}
        sub = df[['_id','wallet_id'] + list(cols_map.values())].copy()
        sub = sub.rename(columns={v:k for k,v in cols_map.items()})
        sub['slot'] = i
        long_rows.append(sub)
    pools = pd.concat(long_rows, ignore_index=True)
    pools['last_tx_dt'] = pd.to_datetime(pools['last_tx_timestamp'], errors='coerce', utc=True)
    max_seen = pools['last_tx_dt'].max()
    pools['days_since_last_tx'] = (max_seen - pools['last_tx_dt']).dt.days

    # Figures and graphs
    plt.figure(); df['_agg'].plot(kind='hist', bins=50); plt.title('Distribution of aggregated_lp_score'); plt.xlabel('aggregated_lp_score'); plt.ylabel('count'); plt.savefig(FIG_DIR/'fig1_agg_hist.png', bbox_inches='tight'); plt.close()
    plt.figure(); plt.scatter(df['_sum_pool_total_score'], df['_agg'], s=1); plt.title('Aggregated vs Sum of Per-Pool total_score'); plt.xlabel('Sum of per-pool total_score'); plt.ylabel('aggregated_lp_score'); plt.savefig(FIG_DIR/'fig2_agg_vs_sum.png', bbox_inches='tight'); plt.close()

    # Deposit deciles
    def cohort_table(series, score, bins=10):
        s = series.copy(); valid = s.notna() & score.notna(); s = s[valid]; sc = score[valid]
        q = pd.qcut(s.rank(method='first')/len(s), bins, labels=False, duplicates='drop')
        dfc = pd.DataFrame({'cohort': q, 'metric': s, 'score': sc})
        return dfc.groupby('cohort').agg(metric_median=('metric','median'), score_median=('score','median')).reset_index()

    cohort_dep = cohort_table(df['_sum_dep'], df['_agg'], 10)
    plt.figure(); plt.plot(cohort_dep['cohort'], cohort_dep['score_median']); plt.title('Median aggregated score by deposit decile'); plt.xlabel('Deposit decile (0=low, 9=high)'); plt.ylabel('Median aggregated score'); plt.savefig(FIG_DIR/'fig3_cohort_deposit.png', bbox_inches='tight'); plt.close()

    # Retention vs score
    pools_lr = pools[['liquidity_percent_remaining','score_breakdown.liquidity_retention_score']].dropna()
    pools_lr['bin'] = pd.qcut(pools_lr['liquidity_percent_remaining'].rank(method='first')/len(pools_lr), 20, labels=False, duplicates='drop')
    bin_lr = pools_lr.groupby('bin')['score_breakdown.liquidity_retention_score'].median().reset_index()
    plt.figure(); plt.plot(bin_lr['bin'], bin_lr['score_breakdown.liquidity_retention_score']); plt.title('Liquidity percent remaining vs retention score (median by bin)'); plt.xlabel('Percentile bin (0..19)'); plt.ylabel('Median liquidity_retention_score'); plt.savefig(FIG_DIR/'fig4_retention_vs_score.png', bbox_inches='tight'); plt.close()

    # Time vs score
    pools_ts = pools[['days_since_last_tx','score_breakdown.time_score']].dropna()
    pools_ts['bin'] = pd.qcut(pools_ts['days_since_last_tx'].rank(method='first')/len(pools_ts), 20, labels=False, duplicates='drop')
    bin_ts = pools_ts.groupby('bin')['score_breakdown.time_score'].median().reset_index()
    plt.figure(); plt.plot(bin_ts['bin'], bin_ts['score_breakdown.time_score']); plt.title('Days since last tx vs time_score (median by bin)'); plt.xlabel('Percentile bin (0..19) - more recent to older'); plt.ylabel('Median time_score'); plt.savefig(FIG_DIR/'fig5_time_vs_score.png', bbox_inches='tight'); plt.close()

    # Volatility vs score
    pools_vs = pools[['lp_volatility_stddev','score_breakdown.lp_volatility_score']].dropna()
    pools_vs['bin'] = pd.qcut(pools_vs['lp_volatility_stddev'].rank(method='first')/len(pools_vs), 20, labels=False, duplicates='drop')
    bin_vs = pools_vs.groupby('bin')['score_breakdown.lp_volatility_score'].median().reset_index()
    plt.figure(); plt.plot(bin_vs['bin'], bin_vs['score_breakdown.lp_volatility_score']); plt.title('Volatility stddev vs lp_volatility_score (median by bin)'); plt.xlabel('Percentile bin (0..19)'); plt.ylabel('Median lp_volatility_score'); plt.savefig(FIG_DIR/'fig6_volatility_vs_score.png', bbox_inches='tight'); plt.close()

    # Component attribution - top pool per wallet checking
    pools['total_score_filled'] = pools['total_score'].fillna(0)
    top_pool_idx = pools.groupby('wallet_id')['total_score_filled'].idxmax()
    top_pools = pools.loc[top_pool_idx]

    comp_cols = ['score_breakdown.deposit_volume_score','score_breakdown.withdraw_volume_score','score_breakdown.deposit_frequency_score','score_breakdown.avg_holding_time_score','score_breakdown.liquidity_retention_score','score_breakdown.lp_volatility_score','score_breakdown.time_score']
    comp_means = top_pools[comp_cols].fillna(0).mean()
    totals = top_pools['total_score'].replace(0, np.nan)
    comp_props = (top_pools[comp_cols].div(totals, axis=0)).mean()

    plt.figure(); plt.bar(range(len(comp_means)), comp_means.values); plt.xticks(range(len(comp_means)), [c.split('.')[-1] for c in comp_means.index], rotation=45, ha='right'); plt.title('Mean component scores for top pool per wallet'); plt.ylabel('Mean component score'); plt.tight_layout(); plt.savefig(FIG_DIR/'fig7_component_means.png', bbox_inches='tight'); plt.close()
    plt.figure(); plt.bar(range(len(comp_props)), comp_props.values); plt.xticks(range(len(comp_props)), [c.split('.')[-1] for c in comp_props.index], rotation=45, ha='right'); plt.title('Mean proportional contribution to total_score (top pool per wallet)'); plt.ylabel('Mean proportion'); plt.tight_layout(); plt.savefig(FIG_DIR/'fig8_component_props.png', bbox_inches='tight'); plt.close()

    # Anomalies 
    anomalies = []
    agg_median = df['_agg'].median()
    agg_p90 = df['_agg'].quantile(0.9)
    dep_p90 = df['_sum_dep'].quantile(0.9)
    dep_p10 = df['_sum_dep'].quantile(0.1)

    cand1 = df.loc[(df['_sum_dep'] >= dep_p90) & (df['_max_pct_remaining'] >= 0.8) & (df['_agg'] < agg_median),
                   ['wallet_id','_sum_dep','_max_pct_remaining','_agg']]
    for _, r in cand1.iterrows():
        anomalies.append({'wallet_id': r['wallet_id'], 'pool_id': '', 'reason': 'High deposits + high retention but below-median agg score','metric': 'agg_vs_dep_ret', 'value': float(r['_agg']), 'threshold': float(agg_median)})

    cand2 = df.loc[(df['_sum_dep'] <= dep_p10) & (df['_agg'] >= agg_p90), ['wallet_id','_sum_dep','_agg']]
    for _, r in cand2.iterrows():
        anomalies.append({'wallet_id': r['wallet_id'], 'pool_id': '', 'reason': 'Low deposits but high agg score (top decile)','metric': 'agg_vs_dep', 'value': float(r['_agg']), 'threshold': float(agg_p90)})

    time_score_p90 = pools['score_breakdown.time_score'].dropna().quantile(0.9)
    cand3 = pools.loc[(pools['days_since_last_tx'] >= 365) & (pools['score_breakdown.time_score'] >= time_score_p90),
                      ['wallet_id','pool_id','days_since_last_tx','score_breakdown.time_score']]
    for _, r in cand3.iterrows():
        anomalies.append({'wallet_id': r['wallet_id'], 'pool_id': str(r['pool_id']), 'reason': 'Old last_tx but high time_score','metric': 'days_since_last_tx', 'value': float(r['days_since_last_tx']), 'threshold': 365.0})

    anom_df = pd.DataFrame(anomalies)
    anom_df.to_csv(OUT_DIR/'anomalies.csv', index=False)

   
    summary = {
        'rows': int(len(df)),
        'cols': int(df.shape[1]),
        'exact_reconstruct_ratio': float((df['_agg'].round(6) == df['_sum_pool_total_score'].round(6)).mean()),
        'diff_mean': float(df['_diff_sum_pool'].mean()),
        'diff_min': float(df['_diff_sum_pool'].min()),
        'diff_max': float(df['_diff_sum_pool'].max())
    }
    with open(OUT_DIR/'summary.json','w') as f: f.write(pd.Series(summary).to_json())

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to dex-temp-db.score_v4.csv")
    ap.add_argument("--out", default="lp_score_v4_outputs", help="Output directory")
    args = ap.parse_args()
    main(args.csv, args.out)
