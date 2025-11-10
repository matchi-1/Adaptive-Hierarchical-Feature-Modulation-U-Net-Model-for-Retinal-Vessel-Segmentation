
import numpy as np
from typing import Dict, List, Tuple, Optional
from .geometry import to_bool_mask

def junction_metrics(graph: Dict, image_shape: Tuple[int, int]) -> Dict[str, float]:
    H, W = image_shape
    n_junc = sum(1 for n in graph["nodes"].values() if n["type"] == "junction")
    n_endp = sum(1 for n in graph["nodes"].values() if n["type"] == "endpoint")
    area = float(H * W)
    return {
        "junction_count": float(n_junc),
        "endpoint_count": float(n_endp),
        "junction_density": float(n_junc) / area,
        "endpoint_density": float(n_endp) / area,
    }


def _unit_vec(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-8:
        return v
    return v / n


def _angle_between(u: np.ndarray, v: np.ndarray) -> float:
    uu, vv = _unit_vec(u), _unit_vec(v)
    dot = float(np.clip(np.dot(uu, vv), -1.0, 1.0))
    ang = float(np.degrees(np.arccos(dot)))
    return ang


def branching_and_bifurcation_angles(graph: Dict, k_ahead: int = 3) -> Dict[str, float]:
    """
    For each junction, compute direction vectors of incident edges (from junction into the edge),
    then compute all pairwise angles. Report summary statistics.
    k_ahead: how many pixels ahead from the junction to define the edge direction (robustness).
    """
    angles = []
    for eid, e in enumerate(graph["edges"]):
        # build full path with endpoints
        u = e["u"]; v = e["v"]
        path = [(graph["nodes"][u]["y"], graph["nodes"][u]["x"])] + e["pixels"] + [(graph["nodes"][v]["y"], graph["nodes"][v]["x"])]
        # directions at both ends
        if len(path) >= 2:
            # at u side
            j_yx = path[0]; nxt = path[min(k_ahead, len(path)-1)]
            ju = np.array([nxt[1]-j_yx[1], nxt[0]-j_yx[0]], dtype=np.float32)  # (dx,dy)
            # at v side
            j_yx2 = path[-1]; prv = path[max(0, len(path)-1-k_ahead)]
            jv = np.array([prv[1]-j_yx2[1], prv[0]-j_yx2[0]], dtype=np.float32)
        else:
            continue

        # store directions keyed by node id
        for nid, vec in [(u, ju), (v, jv)]:
            if "dirs" not in graph["nodes"][nid]:
                graph["nodes"][nid]["dirs"] = []
            graph["nodes"][nid]["dirs"].append(vec)

    # now compute angles at nodes with 3+ dirs
    for nid, n in graph["nodes"].items():
        if "dirs" not in n or len(n["dirs"]) < 2:
            continue
        dirs = n["dirs"]
        L = len(dirs)
        for i in range(L):
            for j in range(i+1, L):
                ang = _angle_between(dirs[i], dirs[j])
                angles.append(ang)

    # cleanup temp
    for n in graph["nodes"].values():
        if "dirs" in n:
            del n["dirs"]

    if not angles:
        return {"angle_mean": 0.0, "angle_std": 0.0, "angle_p10": 0.0, "angle_p50": 0.0, "angle_p90": 0.0}
    arr = np.asarray(angles, dtype=np.float32)
    return {
        "angle_mean": float(arr.mean()),
        "angle_std": float(arr.std(ddof=0)),
        "angle_p10": float(np.percentile(arr, 10)),
        "angle_p50": float(np.percentile(arr, 50)),
        "angle_p90": float(np.percentile(arr, 90)),
    }


def _bresenham_line(y0, x0, y1, x1):
    """
    Bresenham's line algorithm; returns list of (y,x) coordinates from (y0,x0) to (y1,x1).
    """
    points = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        points.append((y, x))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
    return points


def gap_metrics(mask: np.ndarray, graph: Dict, max_gap_px: int = 10, align_cos_thresh: float = 0.5) -> Dict[str, float]:
    """
    Identify candidate 'gaps' as pairs of endpoints within max_gap_px whose straight-line
    connection passes through background only and whose local directions are roughly aligned.
    align_cos_thresh: cosine of angle between endpoint directions must be <= -align_cos_thresh
                      (i.e., pointing towards each other). Use 0.5 ~ 60 degrees tolerance.
    """
    mask_bin = to_bool_mask(mask)
    H, W = mask_bin.shape
    # collect endpoints and local direction
    endpoints = []
    for nid, n in graph["nodes"].items():
        if n["type"] != "endpoint":
            continue
        y, x = n["y"], n["x"]
        # estimate direction by looking into its single incident edge
        # find an edge that uses this node
        vec = None
        for e in graph["edges"]:
            if e["u"] == nid or e["v"] == nid:
                path = [(graph["nodes"][e["u"]]["y"], graph["nodes"][e["u"]]["x"])] + e["pixels"] + [(graph["nodes"][e["v"]]["y"], graph["nodes"][e["v"]]["x"])]
                if path[0] == (y, x) and len(path) > 1:
                    nxt = path[min(3, len(path)-1)]
                    vec = np.array([nxt[1]-x, nxt[0]-y], dtype=np.float32)
                    break
                if path[-1] == (y, x) and len(path) > 1:
                    prv = path[max(0, len(path)-2)]
                    vec = np.array([prv[1]-x, prv[0]-y], dtype=np.float32)
                    break
        if vec is None:
            vec = np.array([0.0, 0.0], dtype=np.float32)
        endpoints.append((nid, y, x, vec))

    # pair endpoints
    gaps = []
    for i in range(len(endpoints)):
        nid1, y1, x1, v1 = endpoints[i]
        for j in range(i+1, len(endpoints)):
            nid2, y2, x2, v2 = endpoints[j]
            dy = y2 - y1
            dx = x2 - x1
            dist = float(np.hypot(dy, dx))
            if dist <= 1.0 or dist > max_gap_px:
                continue
            # check direction roughly facing each other
            v1u = v1 / (np.linalg.norm(v1) + 1e-8)
            v2u = v2 / (np.linalg.norm(v2) + 1e-8)
            # vectors from endpoints towards each other
            to2 = np.array([dx, dy], dtype=np.float32) / (dist + 1e-8)
            to1 = -to2
            c1 = float(np.dot(v1u, to2))
            c2 = float(np.dot(v2u, to1))
            if c1 < align_cos_thresh or c2 < align_cos_thresh:
                continue

            # line-of-sight test: must be background-only except at endpoints
            line = _bresenham_line(y1, x1, y2, x2)
            interior = line[1:-1] if len(line) > 2 else []
            if all((not mask_bin[yy, xx]) for (yy, xx) in interior):
                gaps.append(dist)

    if not gaps:
        return {"gap_count": 0.0, "gap_mean_len": 0.0, "gap_total_len": 0.0}
    gaps = np.asarray(gaps, dtype=np.float32)
    return {"gap_count": float(len(gaps)), "gap_mean_len": float(gaps.mean()), "gap_total_len": float(gaps.sum())}
