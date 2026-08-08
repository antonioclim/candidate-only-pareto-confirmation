
import matplotlib.pyplot as plt, pandas as pd
from common.style import setup_matplotlib, FIG_W, FIG_H, COLOR_BLUE, COLOR_RED, COLOR_GREY
from common.io_utils import read_json
from common.save_utils import save_pair

def main():
    setup_matplotlib(); df=pd.DataFrame(read_json('application_sensitivity_summary.json')).sort_values('mean_paired_restricted_difference')
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H)); y=range(len(df))
    cols=[COLOR_RED if r['mean_difference_ci_low'] <= 0 <= r['mean_difference_ci_high'] else COLOR_BLUE for _,r in df.iterrows()]
    for yi, ((_,r), c) in enumerate(zip(df.iterrows(), cols)):
        ax.plot([r['mean_difference_ci_low'], r['mean_difference_ci_high']], [yi,yi], color=c, linewidth=2); ax.scatter(r['mean_paired_restricted_difference'], yi, color=c, s=28)
    ax.axvline(0, color=COLOR_GREY, linestyle='--', linewidth=1); ax.set_yticks(list(y), df['cell_id']); ax.set_xlabel('Mean paired restricted difference with 99% interval'); ax.set_title('F05 — Sensitivity intervals'); ax.text(0.99,0.01,'Red = interval crosses zero', transform=ax.transAxes, ha='right', va='bottom'); fig.tight_layout(); save_pair(fig,'F05_sensitivity_intervals_demo')
if __name__=='__main__': main()
