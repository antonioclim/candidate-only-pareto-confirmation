
import matplotlib.pyplot as plt, pandas as pd
from common.style import setup_matplotlib, FIG_W, FIG_H, COLOR_BLUE, COLOR_GREEN, COLOR_GOLD, COLOR_RED
from common.io_utils import read_json
from common.save_utils import save_pair
COLORS={'coordinate':COLOR_BLUE,'hotelling':COLOR_GREEN,'hybrid':COLOR_GOLD}

def main():
    setup_matplotlib(); df=pd.DataFrame(read_json('robustness_summary.json')); df=df[df['stress_class'].isin(['serial dependence','strong serial dependence'])].copy(); pivot=df.pivot(index='cell_id', columns='method', values='false_certification_rate').loc[['ROB-05','ROB-06']]
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H)); x=range(len(pivot.index)); width=0.22; methods=['coordinate','hotelling','hybrid']; offsets=[-width,0,width]
    for m,off in zip(methods, offsets): ax.bar([i+off for i in x], pivot[m].values, width=width, label=m, color=COLORS[m])
    ax.axhline(0.05, color=COLOR_RED, linestyle='--', linewidth=2); ax.set_xticks(list(x), ['AR(1) rho=0.3','AR(1) rho=0.6']); ax.set_ylabel('False-certification rate'); ax.set_title('F07 — Serial dependence'); ax.legend(frameon=False); fig.tight_layout(); save_pair(fig,'F07_serial_dependence_demo')
if __name__=='__main__': main()
