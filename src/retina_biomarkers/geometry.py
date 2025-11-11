
import numpy as np
from typing import Dict, List, Tuple, Optional

import numpy as np

def _edge_full_path(graph, e):
    return [(graph["nodes"][e["u"]]["y"], graph["nodes"][e["u"]]["x"])] + e["pixels"] + [(graph["nodes"][e["v"]]["y"], graph["nodes"][e["v"]]["x"])]

def _tangent_at_index(path, i, k=3):
    # robust finite-diff tangent using k-ahead/back sampling
    j0 = max(0, i-k)
    j1 = min(len(path)-1, i+k)
    (y0,x0),(y1,x1) = path[j0], path[j1]
    v = np.array([x1-x0, y1-y0], dtype=np.float32)
    n = np.linalg.norm(v)
    return v / (n + 1e-8)

def _ray_width_from_point(mask, yx, normal, step=0.5, max_radius=20.0):
    """
    March along +n and -n from center until crossing mask boundary.
    Subpixel DDA-style stepping on a binary mask.
    Returns chord length in pixels (float).
    """
    H, W = mask.shape
    def march(sign):
        t = 0.0
        y, x = float(yx[0]), float(yx[1])
        ny, nx = float(normal[1])*sign, float(normal[0])*sign
        prev_inside = True
        while t < max_radius:
            t += step
            yy = int(round(y + ny*t))
            xx = int(round(x + nx*t))
            if yy < 0 or yy >= H or xx < 0 or xx >= W:
                break
            inside = bool(mask[yy, xx])
            if prev_inside and not inside:
                return t  # crossed boundary between last and this step
            prev_inside = inside
        return t
    t_plus  = march(+1.0)
    t_minus = march(-1.0)
    return t_plus + t_minus  # total chord length (px)

def sample_widths_orthogonal(mask, graph, k_tangent=3, step=0.5, max_radius=20.0, stride=1):
    """
    For each edge, compute orthogonal-chord width at (optionally strided) pixels.
    Returns list[np.ndarray] aligned with graph['edges'] (like your EDT widths).
    """
    widths = []
    for e in graph["edges"]:
        path = _edge_full_path(graph, e)
        if len(path) == 0:
            widths.append(np.zeros(0, dtype=np.float32))
            continue
        w = []
        for idx in range(0, len(path), max(1, stride)):
            tan = _tangent_at_index(path, idx, k=k_tangent)
            # normal = rotate tangent by +90°
            normal = np.array([-tan[1], tan[0]], dtype=np.float32)
            yx = path[idx]
            w.append(_ray_width_from_point(mask, yx, normal, step=step, max_radius=max_radius))
        widths.append(np.asarray(w, dtype=np.float32))
    return widths


# Attempt to use skimage skeletonize; fall back to Zhang-Suen if unavailable
def _skeletonize_fallback_zhang_suen(img: np.ndarray) -> np.ndarray:
    """
    Zhang-Suen thinning producing a 1-pixel skeleton.
    Input: img as boolean array (True for foreground).
    """
    img = img.copy().astype(np.uint8)
    changed = True

    def neighbors(y, x):
        # 8-neighbors clockwise starting from P2 (north)
        return [
            img[y-1, x],     # P2
            img[y-1, x+1],   # P3
            img[y,   x+1],   # P4
            img[y+1, x+1],   # P5
            img[y+1, x],     # P6
            img[y+1, x-1],   # P7
            img[y,   x-1],   # P8
            img[y-1, x-1]    # P9
        ]

    def transitions(neigh):
        # number of 0->1 transitions in circular sequence
        n = 0
        for i in range(8):
            if neigh[i] == 0 and neigh[(i+1) % 8] == 1:
                n += 1
        return n

    h, w = img.shape
    # Pad to avoid boundary checks
    pad = 1
    padded = np.zeros((h+2*pad, w+2*pad), dtype=np.uint8)
    padded[pad:pad+h, pad:pad+w] = img
    img = padded
    h, w = img.shape

    while changed:
        changed = False
        # Step 1
        to_delete = []
        for y in range(1, h-1):
            for x in range(1, w-1):
                P1 = img[y, x]
                if P1 == 0:
                    continue
                neigh = neighbors(y, x)
                B = sum(neigh)
                if B < 2 or B > 6:
                    continue
                A = transitions(neigh)
                if A != 1:
                    continue
                if neigh[0]*neigh[2]*neigh[4] != 0:
                    continue
                if neigh[2]*neigh[4]*neigh[6] != 0:
                    continue
                to_delete.append((y, x))
        if to_delete:
            changed = True
            for (y, x) in to_delete:
                img[y, x] = 0

        # Step 2
        to_delete = []
        for y in range(1, h-1):
            for x in range(1, w-1):
                P1 = img[y, x]
                if P1 == 0:
                    continue
                neigh = neighbors(y, x)
                B = sum(neigh)
                if B < 2 or B > 6:
                    continue
                A = transitions(neigh)
                if A != 1:
                    continue
                if neigh[0]*neigh[2]*neigh[6] != 0:
                    continue
                if neigh[0]*neigh[4]*neigh[6] != 0:
                    continue
                to_delete.append((y, x))
        if to_delete:
            changed = True
            for (y, x) in to_delete:
                img[y, x] = 0

    # Unpad
    result = img[pad:h-pad, pad:w-pad]
    return result.astype(bool)


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Return a 1-pixel-thin skeleton (boolean) of a binary mask.
    Tries scikit-image's skeletonize if available; otherwise uses Zhang-Suen fallback.
    """
    mask = to_bool_mask(mask)
    try:
        from skimage.morphology import skeletonize as _skel
        return _skel(mask)
    except Exception:
        return _skeletonize_fallback_zhang_suen(mask)


def distance_transform(mask: np.ndarray) -> np.ndarray:
    """
    Euclidean distance transform inside the vessel mask.
    Returns float32 array where dist[y,x] is distance (px) to nearest background.
    """
    mask = to_bool_mask(mask)
    try:
        from scipy.ndimage import distance_transform_edt
    except Exception as e:
        raise ImportError("scipy is required for distance transform (scipy.ndimage.distance_transform_edt).") from e
    dist = distance_transform_edt(mask).astype(np.float32)
    return dist


def to_bool_mask(mask: np.ndarray) -> np.ndarray:
    """
    Normalize a 2D mask (0/1, 0/255, uint8, etc.) to boolean.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be HxW, got shape={mask.shape}")
    return mask.astype(np.uint8) > 0


def _neighbors8_coords(y: int, x: int, H: int, W: int) -> List[Tuple[int, int]]:
    coords = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                coords.append((ny, nx))
    return coords


def _degree_map(skel: np.ndarray) -> np.ndarray:
    """
    Degree at each skeleton pixel = number of 8-neighbor skeleton pixels.
    """
    H, W = skel.shape
    deg = np.zeros_like(skel, dtype=np.uint8)
    ys, xs = np.where(skel)
    for y, x in zip(ys, xs):
        cnt = 0
        for ny, nx in _neighbors8_coords(y, x, H, W):
            if skel[ny, nx]:
                cnt += 1
        deg[y, x] = cnt
    return deg


def build_skeleton_graph(skel: np.ndarray) -> Dict:
    """
    Convert a skeleton (boolean) to a simple graph:
      nodes: dict[id] = {"y": int, "x": int, "deg": int, "type": "endpoint"/"junction"/"interior"}
      edges: list of {"u": node_id, "v": node_id, "pixels": [(y,x), ...]} excluding the endpoints (u,v)
    Only nodes with degree != 2 are exported as graph nodes; interior degree-2 pixels are absorbed into edges.
    """
    skel = to_bool_mask(skel)
    H, W = skel.shape
    deg = _degree_map(skel)
    # Identify node pixels (deg != 2) present in skeleton
    node_mask = skel & (deg != 2)
    node_ids = -np.ones_like(skel, dtype=np.int32)

    nodes: Dict[int, Dict] = {}
    nid = 0
    ys, xs = np.where(node_mask)
    for y, x in zip(ys, xs):
        t = "junction" if deg[y, x] >= 3 else "endpoint"
        nodes[nid] = {"y": int(y), "x": int(x), "deg": int(deg[y, x]), "type": t}
        node_ids[y, x] = nid
        nid += 1

    # Visited edges (as set of undirected pixel-pair tuples)
    visited = set()
    edges: List[Dict] = []

    def _add_edge_path(p0, p1, path_pix):
        edges.append({"u": p0, "v": p1, "pixels": path_pix})

    # Walk from each node along unvisited neighbors
    for n_id, n in nodes.items():
        y0, x0 = n["y"], n["x"]
        for ny, nx in _neighbors8_coords(y0, x0, H, W):
            if not skel[ny, nx]:
                continue
            a, b = (y0, x0), (ny, nx)
            key = tuple(sorted([a, b]))
            if key in visited:
                continue

            # start path
            path = []
            py, px = y0, x0
            cy, cx = ny, nx
            visited.add(key)

            while True:
                path.append((cy, cx))
                # if current is a node, we reached end
                if node_mask[cy, cx]:
                    u = n_id
                    v = int(node_ids[cy, cx])
                    # exclude terminal node pixels from "pixels", keep interior only
                    interior = path[:-1]  # exclude the last node pixel (the first node is not included anyway)
                    _add_edge_path(u, v, interior)
                    break

                # find the next skeleton neighbor excluding the previous pixel
                next_candidates = []
                for ty, tx in _neighbors8_coords(cy, cx, H, W):
                    if (ty, tx) == (py, px):
                        continue
                    if skel[ty, tx]:
                        next_candidates.append((ty, tx))

                if not next_candidates:
                    # dead end (shouldn't happen often if proper skeleton)
                    # treat current as endpoint node
                    if node_ids[cy, cx] < 0:
                        # add as a new endpoint node
                        node_ids[cy, cx] = len(nodes)
                        nodes[node_ids[cy, cx]] = {"y": int(cy), "x": int(cx), "deg": 1, "type": "endpoint"}
                    u = n_id
                    v = int(node_ids[cy, cx])
                    interior = path[:-1]
                    _add_edge_path(u, v, interior)
                    break

                # continue along the chain; for degree==2 there should be 1 candidate
                # but if multiple, pick the one that hasn't been visited with (cy,cx)
                chosen = None
                for ty, tx in next_candidates:
                    key2 = tuple(sorted([(cy, cx), (ty, tx)]))
                    if key2 not in visited:
                        chosen = (ty, tx)
                        visited.add(key2)
                        break
                if chosen is None:
                    # all neighbors already visited; stop
                    # connect to current pixel as node
                    if node_ids[cy, cx] < 0:
                        node_ids[cy, cx] = len(nodes)
                        nodes[node_ids[cy, cx]] = {"y": int(cy), "x": int(cx), "deg": int(deg[cy, cx]), "type": "junction" if deg[cy, cx] >= 3 else "endpoint"}
                    u = n_id
                    v = int(node_ids[cy, cx])
                    interior = path[:-1]
                    _add_edge_path(u, v, interior)
                    break

                py, px = cy, cx
                cy, cx = chosen

    return {"nodes": nodes, "edges": edges, "degree_map": deg}


def sample_width_along_skeleton(dist: np.ndarray, graph: Dict, default_width: float = 0.0) -> List[np.ndarray]:
    """
    For each edge in graph, sample width (2*distance) at each path pixel.
    Returns list of 1D arrays aligned to 'pixels' per edge.
    """
    widths_per_edge: List[np.ndarray] = []
    H, W = dist.shape
    for e in graph["edges"]:
        pix = e["pixels"]
        if not pix:
            widths_per_edge.append(np.zeros(0, dtype=np.float32))
            continue
        w = []
        for y, x in pix:
            if 0 <= y < H and 0 <= x < W:
                w.append(2.0 * float(dist[y, x]))
            else:
                w.append(default_width)
        widths_per_edge.append(np.asarray(w, dtype=np.float32))
    return widths_per_edge
