
import matplotlib.pyplot as plt
from common.style import setup_matplotlib, FIG_W, FIG_H, COLOR_BLUE, COLOR_RED, COLOR_GREY
from common.io_utils import read_csv, read_json
from common.save_utils import save_pair

def main():
    setup_matplotlib(); df=read_csv('application_primary_raw.csv'); meta=read_json('application_primary_summary.json'); sample=df.iloc[:80].copy()
    fig, axes = plt.subplots(1,2, figsize=(FIG_W, FIG_H))
    ax=axes[0]
    for _, row in sample.iterrows(): ax.plot([0,1],[row['candidate_stopping_count'], row['full_stopping_count']], color=COLOR_GREY, alpha=0.25, linewidth=1)
    ax.scatter([0]*len(sample), sample['candidate_stopping_count'], color=COLOR_BLUE, s=12, label='Candidate-only')
    ax.scatter([1]*len(sample), sample['full_stopping_count'], color=COLOR_RED, s=12, label='Full-set')
    ax.set_xticks([0,1], ['Candidate-only','Full-set']); ax.set_ylabel('Cap-restricted stopping count'); ax.set_title('Paired stopping counts (sample of 80 replications)'); ax.axhline(2000, linestyle='--', linewidth=1, color=COLOR_RED); ax.legend(frameon=False, loc='upper left')
    ax2=axes[1]; ax2.axis('off')
    txt=(f"Runs: {meta['runs']}\nCandidate completions: {meta['candidate_completion_count']} / {meta['runs']}\nFull-set completions: {meta['full_completion_count']} / {meta['runs']}\nCandidate median stopping count: {meta['candidate_restricted_median']}\nFull-set median stopping count: {meta['full_restricted_median']}\nMean paired difference: {meta['mean_paired_restricted_difference']:.2f}\n95% interval: [{meta['mean_difference_ci_low']:.2f}, {meta['mean_difference_ci_high']:.2f}]\nPermitted interpretation:\n{meta['permitted_interpretation']}")
    ax2.text(0.02,0.98,txt,va='top',ha='left'); fig.suptitle('F04 — Primary paired stopping counts'); fig.tight_layout(); save_pair(fig,'F04_primary_paired_counts_demo')
if __name__=='__main__': main()
