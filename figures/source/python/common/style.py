
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "evidence_extracts"
OUT_PNG = ROOT / "outputs" / "demo_png"
OUT_SVG = ROOT / "outputs" / "demo_svg"
DEFAULT_DPI = 300
FIG_W = 12
FIG_H = 8
FONT_FAMILY = "DejaVu Sans"
COLOR_BLUE = "#1F3A5F"
COLOR_GREEN = "#B9D9B0"
COLOR_GOLD = "#F1D37A"
COLOR_RED = "#D96B6B"
COLOR_GREY = "#6B7280"

def setup_matplotlib():
    import matplotlib
    matplotlib.rcParams['font.family'] = FONT_FAMILY
    matplotlib.rcParams['axes.titlesize'] = 14
    matplotlib.rcParams['axes.labelsize'] = 12
    matplotlib.rcParams['xtick.labelsize'] = 10
    matplotlib.rcParams['ytick.labelsize'] = 10
    matplotlib.rcParams['figure.titlesize'] = 16
