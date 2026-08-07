
from common.style import OUT_PNG, OUT_SVG, DEFAULT_DPI

def save_pair(fig, stem):
    OUT_PNG.mkdir(parents=True, exist_ok=True)
    OUT_SVG.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG / f"{stem}.png", dpi=DEFAULT_DPI, bbox_inches='tight')
    fig.savefig(OUT_SVG / f"{stem}.svg", bbox_inches='tight')
