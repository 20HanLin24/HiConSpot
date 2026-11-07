"""
HiConSpot Model
Pointcept detached version

Author: Han Lin (hcl0jf26@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
import math
import numpy as np
from scipy.spatial import cKDTree

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def calculate_angle_between_segments(seg1, seg2):

    vec1 = seg1[2:] - seg1[:2]
    vec2 = seg2[2:] - seg2[:2]

    norm1 = torch.norm(vec1)
    norm2 = torch.norm(vec2)
    if norm1 < 1e-10 or norm2 < 1e-10:
        return torch.tensor(0.0)

    cos_angle = torch.abs(torch.dot(vec1, vec2)) / (norm1 * norm2)
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
    return torch.acos(cos_angle)


def calculate_length_ratio(seg1, seg2):

    len1 = torch.norm(seg1[2:] - seg1[:2])
    len2 = torch.norm(seg2[2:] - seg2[:2])
    if len1 + len2 < 1e-8:
        return torch.tensor(0.5)
    return len1 / (len1 + len2)


def are_segments_connected(a1, a2, b1, b2, tol) -> bool:

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def _distance(p1, p2) -> float:
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def point_on_segment(p, s1, s2):
        seg_len = _distance(s1, s2)
        if seg_len == 0:
            return _distance(p, s1) <= tol
        if abs(cross(p, s1, s2)) / (seg_len + 1e-12) > tol:
            return False
        return (min(s1[0], s2[0]) - tol <= p[0] <= max(s1[0], s2[0]) + tol and
                min(s1[1], s2[1]) - tol <= p[1] <= max(s1[1], s2[1]) + tol)

    def segments_collapse():
        if max(a1[0], a2[0]) < min(b1[0], b2[0]) - tol: return False
        if min(a1[0], a2[0]) > max(b1[0], b2[0]) + tol: return False
        if max(a1[1], a2[1]) < min(b1[1], b2[1]) - tol: return False
        if min(a1[1], a2[1]) > max(b1[1], b2[1]) + tol: return False

        c1 = cross(a1, a2, b1)
        c2 = cross(a1, a2, b2)
        a_len = _distance(a1, a2)
        if a_len == 0:
            cond_res1, cond_res2 = abs(c1), abs(c2)
        else:
            cond_res1, cond_res2 = abs(c1) / a_len, abs(c2) / a_len

        if cond_res1 < tol and cond_res2 < tol:
            a_minx, a_miny = min(a1[0], a2[0]), min(a1[1], a2[1])
            a_maxx, a_maxy = max(a1[0], a2[0]), max(a1[1], a2[1])
            b_minx, b_miny = min(b1[0], b2[0]), min(b1[1], b2[1])
            b_maxx, b_maxy = max(b1[0], b2[0]), max(b1[1], b2[1])
            return not (a_maxx < b_minx - tol or b_maxx < a_minx - tol or
                        a_maxy < b_miny - tol or b_maxy < a_miny - tol)
        return False

    return (point_on_segment(a1, b1, b2) or point_on_segment(a2, b1, b2) or
            point_on_segment(b1, a1, a2) or point_on_segment(b2, a1, a2) or
            segments_collapse())


LINE_T, ARC_T, CIRCLE_T, ELLIPSE_T = 0, 1, 2, 3


def _point_to_segments(p, a, b):
    """点 p (2,) 到一组线段 a->b (m,2) 的最近距离，返回 (m,)。"""
    ab = b - a
    ab2 = np.sum(ab * ab, axis=1)
    safe = np.where(ab2 > 0, ab2, 1.0)
    t = np.where(ab2 > 0, np.sum((p - a) * ab, axis=1) / safe, 0.0)
    t = np.clip(t, 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def _recover_ellipse_radii(length, rx):
    """由 han_process_svg.py 的 length = 2*pi*b + 4*(a-b) 与 rx 反解 ry。"""
    b1 = (length - 4.0 * rx) / (2.0 * np.pi - 4.0)
    if b1 <= rx:
        return b1
    return rx + (length - 2.0 * np.pi * rx) / 4.0


def _ellipse_tree(center, rx, ry, dist_tol):
    a, b = max(rx, ry), min(rx, ry)
    h = ((a - b) / (a + b)) ** 2 if a + b > 0 else 0.0
    perim = math.pi * (a + b) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))
    n = int(min(max(720, math.ceil(perim / (0.1 * dist_tol))), 200_000))
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    pts = np.stack([center[0] + rx * np.cos(t), center[1] + ry * np.sin(t)], axis=1)
    return cKDTree(pts)


def get_knn_with_geometric_relations(nodes, max_degree=128, thresh=0.0025, k=3, tol=1e-5, dist_tol=1e-5, angle_tol=0.03):
    """
    Args:
        nodes: (N, 11) ndarray，
               列 0 length；1-2 mid；3-4 start；5-6 end；7-10 类型 one-hot。
    Returns:
        nns_list, thresh_nns_list, dist_in_thresh_list, geometric_relations_list
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    empty = {"nns_list": [], "thresh_nns_list": [],
             "dist_in_thresh_list": [], "geometric_relations_list": []}
    if nodes.size == 0:
        return empty
    if nodes.ndim != 2 or nodes.shape[1] != 11:
        raise ValueError(f"nodes must be (N, 11), got {nodes.shape}")

    N = nodes.shape[0]
    type_id = np.argmax(nodes[:, 7:11], axis=1)
    is_open = (type_id == LINE_T) | (type_id == ARC_T)     # 线段与圆弧（弦）
    is_line = type_id == LINE_T
    open_idx = np.where(is_open)[0]
    closed_idx = np.where(~is_open)[0]

    C = nodes[:, 1:3]
    S = nodes[:, 3:5]
    E = nodes[:, 5:7]
    radius = np.linalg.norm(E - S, axis=1) / 2.0

    # ---------- 图元到点的距离 ----------
    circ_idx = closed_idx[type_id[closed_idx] == CIRCLE_T]
    ell_idx = closed_idx[type_id[closed_idx] == ELLIPSE_T]
    ell_trees = {}
    for c in ell_idx:
        ry = _recover_ellipse_radii(nodes[c, 0], radius[c])
        ell_trees[c] = _ellipse_tree(C[c], radius[c], ry, dist_tol)

    def dist_point_to_all(p):
        d = np.full(N, np.inf)
        if open_idx.size:
            d[open_idx] = _point_to_segments(p, S[open_idx], E[open_idx])
        if circ_idx.size:
            d[circ_idx] = np.abs(np.linalg.norm(p - C[circ_idx], axis=1) - radius[circ_idx])
        for c, tree in ell_trees.items():
            d[c] = tree.query(p)[0]
        return d

    # Gmat[i, j]：i 的端点到图元 j 的最小距离
    Gmat = np.full((N, N), np.inf)
    for i in open_idx:
        Gmat[i] = np.minimum(dist_point_to_all(S[i]), dist_point_to_all(E[i]))
    Dist = np.minimum(Gmat, Gmat.T)
    np.fill_diagonal(Dist, 0.0)

    # 开放图元之间的端点距离（Level 2）
    EE = np.full((N, N), np.inf)
    if open_idx.size:
        ee = np.full((open_idx.size, open_idx.size), np.inf)
        for a in (0, 1):
            for b in (0, 1):
                pa = S[open_idx] if a == 0 else E[open_idx]
                pb = S[open_idx] if b == 0 else E[open_idx]
                ee = np.minimum(ee, np.linalg.norm(pa[:, None] - pb[None, :], axis=-1))
        EE[np.ix_(open_idx, open_idx)] = ee

    # ---------- 连通关系 ----------
    L1 = Dist < dist_tol          # General：端点到图元的接触
    L2 = EE < dist_tol            # Endpoint：端点与端点接触
    np.fill_diagonal(L1, True)    # 自身视为连通，用于候选集合
    L2_off = L2.copy()
    np.fill_diagonal(L2_off, False)

    # 向量与方向（线段与圆弧的弦）
    chord = E - S
    chord_len = np.linalg.norm(chord, axis=1)
    unit = np.zeros_like(chord)
    valid = (chord_len > tol)
    unit[valid] = chord[valid] / chord_len[valid, None]

    # 与原函数一致：|cos| < angle_tol
    cos_mat = np.abs(unit @ unit.T)
    orth_mat = (cos_mat < angle_tol) & valid[:, None] & valid[None, :]

    # 1) 距离平方：Level 1 接触处记为 0，与原函数零距离处理一致
    D2 = Dist ** 2
    D2[L1] = 0.0
    np.fill_diagonal(D2, 0.0)

    nns_list, thresh_nns_list = [], []
    dist_in_thresh_list, geometric_relations_list = [], []

    # ---------- 几何关系（6 类 one-hot）----------
    def relation(i, j):
        if not (is_open[i] and is_open[j]):
            # 圆或椭圆：仅 Level 1 接触，视为一般相接
            return np.array([0, 0, 0, 0, 1, 0], dtype=np.int64) if L1[i, j] \
                else np.array([0, 0, 0, 0, 0, 1], dtype=np.int64)
        if not (valid[i] and valid[j]):
            return np.array([0, 0, 0, 0, 0, 1], dtype=np.int64)

        v1, v2 = chord[i], chord[j]
        n1, n2 = chord_len[i], chord_len[j]
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        sin_a = np.clip((v1[0] * v2[1] - v1[1] * v2[0]) / (n1 * n2), -1.0, 1.0)

        is_end = bool(L2_off[i, j])
        is_conn = bool(L1[i, j])
        orth = abs(cos_a) < angle_tol
        par = abs(sin_a) < angle_tol and not is_conn

        if orth and is_end:
            code = 0
        elif orth and is_conn:
            code = 1
        elif par:
            code = 2
        elif is_end:
            code = 3
        elif is_conn:
            code = 4
        else:
            code = 5
        rel = np.zeros(6, dtype=np.int64)
        rel[code] = 1
        return rel

    # ---------- 逐节点输出 ----------
    arange_N = np.arange(N)
    for i in range(N):
        dist_min = D2[i]
        near_idx = np.argsort(dist_min, kind="stable")[:max_degree].tolist()

        cand = L1[i]
        values = np.where(cand, dist_min, np.inf)
        thresh_element_num = int(np.sum(values <= thresh))
        distzero_num = int(np.sum(values == 0))

        order = np.argsort(values, kind="stable")[:thresh_element_num]
        thresh_near_idx = order.tolist()
        thresh_near_values = values[order].tolist()

        if thresh_element_num > k:
            if distzero_num > k:
                thresh_near_idx = thresh_near_idx[:distzero_num]
                thresh_near_values = thresh_near_values[:distzero_num]
            else:
                thresh_near_idx = thresh_near_idx[:k + 1]
                thresh_near_values = thresh_near_values[:k + 1]

        geo_rels = [np.zeros(6, dtype=np.int64) if j == i else relation(i, j)
                    for j in thresh_near_idx]

        nns_list.append(near_idx)
        thresh_nns_list.append(thresh_near_idx)
        dist_in_thresh_list.append(thresh_near_values)
        geometric_relations_list.append(geo_rels)

    return {
        "nns_list": nns_list,
        "thresh_nns_list": thresh_nns_list,
        "dist_in_thresh_list": dist_in_thresh_list,
        "geometric_relations_list": geometric_relations_list,
    }





