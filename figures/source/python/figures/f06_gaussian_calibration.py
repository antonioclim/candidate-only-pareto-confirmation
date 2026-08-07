
import matplotlib.pyplot as plt, pandas as pd
from common.style import setup_matplotlib, FIG_W, FIG_H, COLOR_BLUE, COLOR_RED
from common.io_utils import read_json
from common.save_utils import save_pair

def main():
    setup_matplotlib(); df=pd.DataFrame(read_json('null_calibration_summary.json')).sort_values('cell_id')
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H)); ax.bar(df['cell_id'], df['bonferroni_one_sided_upper_familywise_95'], color=COLOR_BLUE); ax.axhline(df['gate_threshold'].iloc[0], color=COLOR_RED, linestyle='--', linewidth=2); ax.set_ylabel('Family-wise adjusted one-sided upper 95% bound'); ax.set_xlabel('Calibration cell'); ax.set_title('F06 — Exact Gaussian calibration'); ax.tick_params(axis='x', rotation=45); fig.tight_layout(); save_pair(fig,'F06_gaussian_calibration_demo')
if __name__=='__main__': main()
