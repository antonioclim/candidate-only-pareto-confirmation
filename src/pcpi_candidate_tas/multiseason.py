"""Multi-season, mirror-derived battery-policy application helpers.

The module consumes the pinned DK1 mirror subset distributed with the research
package.  The subset is not the complete official OPSD package; its provenance
and limits are part of the data contract.
"""
from __future__ import annotations
from pathlib import Path
import csv
import numpy as np


def load_multiseason_subset(path: Path | str) -> dict:
    rows=[]
    with open(path,newline="",encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    required={"utc_timestamp","season","split",
              "DK_1_price_day_ahead_EUR_MWh",
              "DK_1_load_actual_entsoe_transparency_MWh"}
    if not rows or set(rows[0]) != required:
        raise ValueError("unexpected multi-season columns")
    price=np.asarray([float(r["DK_1_price_day_ahead_EUR_MWh"]) for r in rows])
    load=np.asarray([float(r["DK_1_load_actual_entsoe_transparency_MWh"]) for r in rows])
    season=np.asarray([r["season"] for r in rows])
    split=np.asarray([r["split"] for r in rows])
    timestamps=np.asarray([r["utc_timestamp"] for r in rows])
    if not np.all(np.isfinite(price)) or not np.all(np.isfinite(load)):
        raise ValueError("finite price and load are required")
    if np.any(load <= 0):
        raise ValueError("strictly positive load is required")
    return {"price":price,"load":load,"season":season,"split":split,
            "timestamps":timestamps}


def seasonal_windows(data: dict, split: str, window: int = 24) -> tuple[tuple[str,np.ndarray,np.ndarray], ...]:
    if window < 2:
        raise ValueError("window >= 2 required")
    blocks=[]
    for season in sorted(set(data["season"])):
        mask=(data["season"]==season)&(data["split"]==split)
        prices=np.asarray(data["price"])[mask]
        loads=np.asarray(data["load"])[mask]
        if len(prices) < window:
            continue
        # Use non-redundant starts but retain a final tail window when needed.
        starts=list(range(0,len(prices)-window+1,window))
        final=len(prices)-window
        if final not in starts:
            starts.append(final)
        for start in sorted(set(starts)):
            blocks.append((str(season),prices[start:start+window].copy(),loads[start:start+window].copy()))
    if not blocks:
        raise ValueError(f"no {window}-hour windows for split {split!r}")
    return tuple(blocks)


def generate_multiseason_scenarios(
    data: dict,
    split: str,
    count: int,
    seed: int,
    *,
    window: int = 24,
    price_noise: float = 0.045,
    load_noise: float = 0.020,
    additive_price_sd: float = 1.25,
    additive_load_sd: float = 12.0,
) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    if count < 1:
        raise ValueError("positive scenario count required")
    blocks=seasonal_windows(data,split,window)
    rng=np.random.default_rng(seed)
    prices=[];loads=[];seasons=[]
    for _ in range(count):
        season,p,l=blocks[int(rng.integers(len(blocks)))]
        innovation=rng.normal(size=window)
        smooth=np.convolve(innovation,np.ones(5)/5,mode="same")
        seasonal_scale=rng.lognormal(mean=-0.5*0.02**2,sigma=0.02)
        ps=p*seasonal_scale*(1+price_noise*smooth)+rng.normal(0,additive_price_sd,size=window)
        ls=l*seasonal_scale*(1+load_noise*smooth)+rng.normal(0,additive_load_sd,size=window)
        prices.append(ps);loads.append(np.clip(ls,1,None));seasons.append(season)
    return np.asarray(prices),np.asarray(loads),np.asarray(seasons)
