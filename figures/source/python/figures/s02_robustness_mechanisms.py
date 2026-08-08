
import matplotlib.pyplot as plt, pandas as pd
from common.style import setup_matplotlib, FIG_W, FIG_H, COLOR_BLUE, COLOR_GREEN, COLOR_GOLD, COLOR_RED
from common.io_utils import read_json
from common.save_utils import save_pair
COLORS={'coordinate':COLOR_BLUE,'hotelling':COLOR_GREEN,'hybrid':COLOR_GOLD}

def main():
    setup_matplotlib(); df=pd.DataFrame(read_json('robustness_summary.json')).copy(); cell_order=sorted(df['cell_id'].unique()); methods=['coordinate','hotelling','hybrid']; width=0.22; x=list(range(len(cell_order))); fig, ax = plt.subplots(figsize=(FIG_W+2, FIG_H)); offsets=[-width,0,width]
    for m,off in zip(methods, offsets):
        vals=[float(df[(df['cell_id']==c)&(df['method']==m)]['false_certification_rate'].iloc[0]) for c in cell_order]
        ax.bar([i+off for i in x], vals, width=width, label=m, color=COLORS[m])
    ax.axhline(0.05, color=COLOR_RED, linestyle='--', linewidth=2); ax.set_xticks(x, cell_order); ax.set_ylabel('False-certification rate'); ax.set_xlabel('Robustness cell'); ax.set_title('S02 — Robustness across mechanisms'); ax.legend(frameon=False, ncol=3); fig.tight_layout(); save_pair(fig,'S02_robustness_mechanisms_demo')
if __name__=='__main__': main()
