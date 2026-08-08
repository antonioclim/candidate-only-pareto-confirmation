
import matplotlib.pyplot as plt
from common.style import setup_matplotlib, FIG_W, FIG_H, COLOR_BLUE
from common.io_utils import read_csv
from common.save_utils import save_pair

def main():
    setup_matplotlib(); df=read_csv('application_primary_candidate_distribution.csv'); fig, ax = plt.subplots(figsize=(FIG_W, FIG_H)); ax.bar(df['candidate_policy_id'], df['count'], color=COLOR_BLUE); ax.set_ylabel('Count across 400 primary replications'); ax.set_xlabel('Candidate identity'); ax.set_title('S01 — Candidate identity distribution'); ax.tick_params(axis='x', rotation=25); fig.tight_layout(); save_pair(fig,'S01_candidate_identity_demo')
if __name__=='__main__': main()
