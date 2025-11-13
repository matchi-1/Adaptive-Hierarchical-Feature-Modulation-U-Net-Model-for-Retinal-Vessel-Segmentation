import numpy as np, pandas as pd, matplotlib.pyplot as plt

def flatten_metrics(out: dict) -> dict:
    bio = out["biomarkers"]; g = bio["global"]; t = bio["topology"]
    row = {
        "area_density": g["area_density"],
        "length_density_PD_inv": g.get("length_density_PD_inv", np.nan),
        "median_width_PD": g.get("median_width_PD", np.nan),
        "iqr_width_PD": g.get("iqr_width_PD", np.nan),
        "tortuosity_mean_PD2_x1e3": g.get("tortuosity_mean_PD2_x1e3", np.nan),
        "tortuosity_mean": g.get("tortuosity_mean", np.nan),
        "junction_density": t.get("junction_density", np.nan),
        "endpoint_density": t.get("endpoint_density", np.nan),
        "angle_mean_2PD": (t.get("angles_2PD") or {}).get("angle_mean", np.nan),
    }
    rings = bio.get("rings") or {}
    ordered = []
    for k in rings.keys():
        try: ordered.append((float(k.split("-")[0]), k))
        except: pass
    for _, k in sorted(ordered):
        r = rings[k]
        row[f"{k}|area_density"]     = r.get("area_density", np.nan)
        row[f"{k}|median_width"]     = r.get("median_width", np.nan)
        row[f"{k}|tortuosity_x1e3"]  = r.get("tortuosity_mean_PD2_x1e3", np.nan)
    return row

def agg_compare(df_norm, df_dr, metrics):
    nm, dm = df_norm[metrics].mean(), df_dr[metrics].mean()
    ns, ds = df_norm[metrics].std(),  df_dr[metrics].std()
    comp = pd.concat([nm.rename("Normal_mean"),
                      ns.rename("Normal_sd"),
                      dm.rename("DR_mean"),
                      ds.rename("DR_sd")], axis=1)
    comp["Δ (DR-Norm)"] = comp["DR_mean"] - comp["Normal_mean"]
    comp["%Δ vs Norm"]  = 100.0 * comp["Δ (DR-Norm)"] / np.where(comp["Normal_mean"]!=0, comp["Normal_mean"], np.nan)
    return comp

def bar_compare(dfN, dfD, metrics, title, ylabel="Value"):
    meansN, sdsN = dfN[metrics].mean(), dfN[metrics].std()
    meansD, sdsD = dfD[metrics].mean(), dfD[metrics].std()
    ind = np.arange(len(metrics)); w = 0.38
    plt.figure(figsize=(12, 4.5))
    plt.bar(ind - w/2, meansN.values, yerr=sdsN.values, capsize=3, label="Normal")
    plt.bar(ind + w/2, meansD.values, yerr=sdsD.values, capsize=3, label="DR")
    plt.xticks(ind, metrics, rotation=25, ha="right")
    plt.ylabel(ylabel); plt.title(title); plt.legend(); plt.tight_layout(); plt.show()

def ring_profile(df, metric_suffix):
    cols = [c for c in df.columns if c.endswith("|"+metric_suffix)]
    keyed=[]
    for c in cols:
        try: keyed.append((float(c.split("|")[0].split("-")[0]), c))
        except: pass
    cols = [c for _, c in sorted(keyed)]
    xs = [float(c.split("|")[0].split("-")[0]) for c in cols]
    return np.array(xs), df[cols].to_numpy(dtype=float)

def draw_overlay_ax(ax, out, title=None, alpha=0.35):
    rgb_iso    = out["rgb_iso"]
    disc_mask  = out["disc_mask"]
    center_yx  = tuple(out["od"]["center_yx"])
    PD_px      = out["od"]["PD_px"]
    mask_thr05 = out["pred_mask_thr05"]

    overlay = np.zeros_like(rgb_iso, dtype=np.uint8)
    overlay[..., 1] = (mask_thr05.astype(np.uint8) * 255)
    ax.imshow(rgb_iso); ax.imshow(overlay, alpha=alpha); ax.axis("off")
    try: ax.contour(disc_mask.astype(bool), levels=[0.5], colors='r', linewidths=1.0)
    except: pass

    from src.retina_biomarkers.notebook_utils.viz.overlays import _draw_pd_rings
    _draw_pd_rings(ax, center_yx, PD_px, color='white')
    if title: ax.set_title(title, fontsize=10)
