"""
HiConSpot Model
Pointcept detached version

Author: Han Lin (hcl0jf26@gmail.com)
Please cite our work if the code is helpful to you.
"""
import os
import warnings

from PIL import Image
import math
import numpy as np

import copy

from .han_processSvg_utils import *



def convert_coordinates(x, y, w, h, img_width, img_height):
    left = int((x - w / 2) * img_width)
    top = int((y - h / 2) * img_height)
    right = int((x + w / 2) * img_width)
    bottom = int((y + h / 2) * img_height)
    print("left,top,right,bottom:", left, top, right, bottom)
    return left, top, right, bottom


def correct_orientation(img):
    width, height = img.size
    if height > width:
        img = img.rotate(90, expand=True)
    return img


def get_sted_angle(start_point, end_point, center_point, z):

    x1, y1 = start_point
    x2, y2 = end_point
    x0, y0 = center_point

    start_vec = (x1 - x0, y1 - y0)
    end_vec = (x2 - x0, y2 - y0)

    start_rad = math.atan2(start_vec[1], start_vec[0])
    end_rad = math.atan2(end_vec[1], end_vec[0])

    def to_positive_degrees(rad):
        deg = math.degrees(rad)
        return deg % 360

    start_angle = to_positive_degrees(start_rad)
    end_angle = to_positive_degrees(end_rad)
    stp, edp = start_point, end_point

    if z == 1:   
        if end_angle <= start_angle:
            end_angle += 360
    elif z == -1:
        start_angle, end_angle = end_angle, start_angle
        stp, edp = end_point, start_point
        if end_angle <= start_angle:
            end_angle += 360


    return start_angle, end_angle, stp, edp


INSIDE = 0   
LEFT = 1   
RIGHT = 2   
BOTTOM = 4   
TOP = 8   


def compute_code(x, y, x_min, y_min, x_max, y_max):

    code = INSIDE
    if x < x_min:
        code |= LEFT
    elif x > x_max:
        code |= RIGHT
    if y < y_min:
        code |= BOTTOM
    elif y > y_max:
        code |= TOP
    return code


def cohen_sutherland(x1, y1, x2, y2, x_min, y_min, x_max, y_max):
    code1 = compute_code(x1, y1, x_min, y_min, x_max, y_max)
    code2 = compute_code(x2, y2, x_min, y_min, x_max, y_max)
    inside_code_flag = 0
    accept = False

    while True:   
        if not (code1 | code2):
             
            inside_code_flag += 1
            accept = True
            break
        elif code1 & code2:
             
            break
        else:
            inside_code_flag += 1
             
            x, y = 0, 0
            outside_code = code1 if code1 else code2   

            if outside_code & TOP:   
                x = x1 + (x2 - x1) * (y_max - y1) / (y2 - y1)
                y = y_max
            elif outside_code & BOTTOM:   
                x = x1 + (x2 - x1) * (y_min - y1) / (y2 - y1)
                y = y_min
            elif outside_code & RIGHT:   
                y = y1 + (y2 - y1) * (x_max - x1) / (x2 - x1)
                x = x_max
            elif outside_code & LEFT:   
                y = y1 + (y2 - y1) * (x_min - x1) / (x2 - x1)
                x = x_min

            if outside_code == code1:
                x1, y1 = x, y
                code1 = compute_code(x1, y1, x_min, y_min, x_max, y_max)   
            else:
                x2, y2 = x, y
                code2 = compute_code(x2, y2, x_min, y_min, x_max, y_max)   

    if accept:
        return x1, y1, x2, y2, inside_code_flag
    return None

 
def get_valid_arc_angles(arc, x_min, y_min, x_max, y_max):
    """计算弧线在裁剪窗口内的有效角度范围"""
    cx, cy = arc["center"]
    radius = arc["radius"]
    start_angle, end_angle = arc["angle_scope"]

     
    boundary_angles = []
    for boundary in [(x_min, 'v'), (x_max, 'v'), (y_min, 'h'), (y_max, 'h')]:
        pos, typ = boundary
        if typ == 'v':   
            if abs(cx - pos) <= radius:
                theta = math.acos((pos - cx) / radius)
                boundary_angles.extend([theta, -theta])
        else:   
            if abs(cy - pos) <= radius:
                theta = math.asin((pos - cy) / radius)
                boundary_angles.extend([math.pi / 2 - theta, math.pi / 2 + theta])

     
    valid_angles = [start_angle, end_angle]
    for angle in boundary_angles:
        if start_angle <= angle <= end_angle:
            valid_angles.append(angle)

    valid_angles.sort()
    return valid_angles


def calculate_clipped_angles_backup(
        center: Tuple[float, float],
        radius: float,
        startp: Tuple[float, float],
        endp: Tuple[float, float],
        orig_angles: Tuple[float, float],
        bbox: Tuple[float, float, float, float]
) -> None | tuple[float, float] | tuple[float, float, tuple[float, float], tuple[float, float]]:

    x_min, y_min, x_max, y_max = bbox
    cx, cy = center
    start_angle, end_angle = orig_angles

    if (cx + radius < x_min or cx - radius > x_max or
        cy + radius < y_min or cy - radius > y_max):
        return None
     
    intersections = []
     
    for y_bound in [y_min, y_max]:
        if cy - radius <= y_bound <= cy + radius:
            dx = math.sqrt(radius ** 2 - (y_bound - cy) ** 2)
            for x in [cx - dx, cx + dx]:
                if x_min <= x <= x_max:

                    angle1 = math.degrees(math.atan2(cy - y_bound, x - cx))
                    intersections.append(angle1)

    for x_bound in [x_min, x_max]:
        if cx - radius <= x_bound <= cx + radius:
            dy = math.sqrt(radius ** 2 - (x_bound - cx) ** 2)
            for y in [cy - dy, cy + dy]:
                if y_min <= y <= y_max:

                    angle2 = math.degrees(math.atan2(cy - y, x_bound - cx))
                    intersections.append(angle2)

    valid_angles = []
    norm_start = start_angle % 360
    norm_end = end_angle % 360
    if norm_end < norm_start:
        norm_end += 360
    for angle in intersections:
         
        norm_angle = angle % 360
         
        if norm_start <= norm_end:
             
            if norm_start <= norm_angle <= norm_end:
                valid_angles.append(norm_angle)
        else:
             
            if norm_angle >= norm_start or norm_angle <= norm_end:
                valid_angles.append(norm_angle)

    if (x_min <= startp[0] <= x_max) and (y_min <= startp[1] <= y_max):
        valid_angles.append(start_angle)
    if (x_min <= endp[0] <= x_max) and (y_min <= endp[1] <= y_max):
        valid_angles.append(end_angle)

     
    if not valid_angles:
        if (x_min <= cx <= x_max) and (y_min <= cy <= y_max):
            return start_angle, end_angle, startp, endp
        return None

    norm_valid = [a % 360 for a in valid_angles]
     
    sorted_pairs = sorted(zip(norm_valid, valid_angles), key=lambda x: x[0])
    sorted_angles = [a for (_, a) in sorted_pairs]   

     
    if len(sorted_angles) > 1 and abs(sorted_angles[-1] - sorted_angles[0]) > (norm_end - norm_start):
        split_idx = np.argmax(np.diff(sorted_angles) > (norm_end - norm_start))
        sorted_angles =sorted_angles[split_idx + 1:] + sorted_angles[:split_idx + 1]
    else:
        sorted_angles = sorted_angles

     
    new_stpx = cx + radius * math.cos(math.radians(sorted_angles[0]))
    new_stpy = cy - radius * math.sin(math.radians(sorted_angles[0]))   
    new_edpx = cx + radius * math.cos(math.radians(sorted_angles[-1]))
    new_edpy = cy - radius * math.sin(math.radians(sorted_angles[-1]))   

    return sorted_angles[0], sorted_angles[-1], (new_stpx, new_stpy), (new_edpx, new_edpy)


def calculate_clipped_angles(
        center: Tuple[float, float],
        radius: float,
        startp: Tuple[float, float],
        endp: Tuple[float, float],
        orig_angles: Tuple[float, float],
        bbox: Tuple[float, float, float, float]
) -> None | tuple[float, float] | tuple[float, float, tuple[float, float], tuple[float, float]]:

    x_min, y_min, x_max, y_max = bbox
    cx, cy = center
    start_angle, end_angle = orig_angles

     
    if (cx + radius < x_min or cx - radius > x_max or
        cy + radius < y_min or cy - radius > y_max):
        return None

    intersections = []
     
    for y_bound in [y_min, y_max]:
        if cy - radius <= y_bound <= cy + radius:
            dx = math.sqrt(radius ** 2 - (y_bound - cy) ** 2)
            for x in [cx - dx, cx + dx]:
                if x_min <= x <= x_max:

                    angle1 = math.degrees(math.atan2(cy - y_bound, x - cx))
                    intersections.append(angle1)

    for x_bound in [x_min, x_max]:
        if cx - radius <= x_bound <= cx + radius:
            dy = math.sqrt(radius ** 2 - (x_bound - cx) ** 2)
            for y in [cy - dy, cy + dy]:
                if y_min <= y <= y_max:

                    angle2 = math.degrees(math.atan2(cy - y, x_bound - cx))
                    intersections.append(angle2)

    valid_angles = []
    norm_start = start_angle % 360
    norm_end = end_angle % 360
    angle_range_adjusted = False
    if norm_end < norm_start:
        norm_end += 360
        angle_range_adjusted = True
    for angle in intersections:
         
        norm_angle = angle % 360

        adjusted_angle = norm_angle
        if angle_range_adjusted and norm_angle < norm_start:
            adjusted_angle = norm_angle + 360

        if norm_start <= norm_end:
             
            if norm_start <= adjusted_angle <= norm_end:
                valid_angles.append(adjusted_angle)
        else:
             
            if norm_angle >= norm_start or norm_angle <= norm_end:
                valid_angles.append(norm_angle)

    if (x_min <= startp[0] <= x_max) and (y_min <= startp[1] <= y_max):
        valid_angles.append(start_angle)
    if (x_min <= endp[0] <= x_max) and (y_min <= endp[1] <= y_max):
        valid_angles.append(end_angle)

    if not valid_angles:
        if (x_min <= cx <= x_max) and (y_min <= cy <= y_max):
            return start_angle, end_angle, startp, endp
        return None

    unique_angles = list(set(valid_angles))

     
    norm_valid = [a % 360 for a in unique_angles]
     
    sorted_pairs = sorted(zip(norm_valid, unique_angles), key=lambda x: x[0])
    sorted_angles = [a for (_, a) in sorted_pairs]
     
    if len(sorted_angles) > 1 and abs(sorted_angles[-1] - sorted_angles[0]) > (norm_end - norm_start):
        split_idx = np.argmax(np.diff(sorted_angles) > (norm_end - norm_start))
        sorted_angles =sorted_angles[split_idx + 1:] + sorted_angles[:split_idx + 1]
    else:
        sorted_angles = sorted_angles

     
    new_stpx = cx + radius * math.cos(math.radians(sorted_angles[0]))
    new_stpy = cy - radius * math.sin(math.radians(sorted_angles[0]))   
    new_edpx = cx + radius * math.cos(math.radians(sorted_angles[-1]))
    new_edpy = cy - radius * math.sin(math.radians(sorted_angles[-1]))   

     
    return sorted_angles[0], sorted_angles[-1], (new_stpx, new_stpy), (new_edpx, new_edpy)

def calculate_arc_midpoint(center, radius, start_angle, end_angle):
    """计算弧线段的几何中点（按角度均匀分布）
    center:(cx, cy)，图片坐标轴
    radius:float，单位像素
    start_angle, end_angle：弧线起止角终止角，单位角度
    :return :弧线段中点坐标，单位像素，图片坐标轴
    """
    mid_angle = (start_angle + end_angle) / 2
    return (
        center[0] + radius * math.cos(math.radians(mid_angle)),
        center[1] - radius * math.sin(math.radians(mid_angle))
    )

def get_2d_square_distance(src, dst):
    """Calculate Euclid distance between each two points using pytorch.计算两点间的欧几里得平方距离=两点逐元素相减差的平方和
    """
    return (src[0] - dst[0]) ** 2 + (src[1] - dst[1]) ** 2

def get_dist(seg1, seg2):

    i_start = seg1[:2]   
    i_end = seg1[2:]   

    p_start = seg2[:2]   
    p_end = seg2[2:]   

    dist_istart_pstart = get_2d_square_distance(i_start, p_start)   
    dist_istart_pend = get_2d_square_distance(i_start, p_end)   
    dist_iend_pstart = get_2d_square_distance(i_end, p_start)   
    dist_iend_pend = get_2d_square_distance(i_end, p_end)   

    dist = [dist_istart_pstart, dist_istart_pend, dist_iend_pstart, dist_iend_pend]

    dist_min = min(dist)
    return dist_min


def square_distance(src, dst):

    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


def crop_image_npy(
        image_path,
        npy_dir,
        output_folder,
        out_img,
        output_size=(1600, 1600),
        step=1600,
        thresh=0.0025,
        tol=1e-5,
        distol=1e-5,
        k=3,
        angle_tol=0.03):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
     
    img = Image.open(image_path)
     
    img = correct_orientation(img)
    width, height = img.size
    print("IMG_SIZE:", img.size)

    cols, rows = (width + step - 1) // step, (height + step - 1) // step
     
    filename = os.path.basename(image_path)
    filebasename = filename.split('.')[0]

    label_name = os.path.splitext(filename)[0] + ".npy"   
    ann_path = os.path.join(npy_dir, label_name)
    if not os.path.exists(ann_path):
        warnings.warn("ann_path not exists!")
        return None
    adj_node_classes = np.load(ann_path, allow_pickle=True).item()

    target = adj_node_classes["cat"]   
    center = adj_node_classes["ct_norm_z2o"]   
    nodes = adj_node_classes["nd_ft"]   
    nns = adj_node_classes["nns"]   
    segments = adj_node_classes["segments"]   
    layers = adj_node_classes["layers"]

    for row in range(rows):
        for col in range(cols):
            left = col * step   
            upper = row * step   
            right = min(left + output_size[0], width)     
            lower = min(upper + output_size[1], height)   

            cropped_img = img.crop((left, upper, right, lower))   
            cropped_size = (right - left, lower - upper)

            final_img = Image.new('RGB', cropped_size, (255, 255, 255))
             
            paste_position = (0, 0)
             
            final_img.paste(cropped_img, paste_position)

            final_img.save(fp=os.path.join(out_img, f"{filebasename}_{row}_{col}.jpg"), format='JPEG')

            new_layers = []
            new_target = []
            new_center_norm = []
            new_nodes = []
            new_segments = []
            new2old_idx = {}   
            new_eid = 0

            for idx, seg in enumerate(segments):
                start_x, start_y, end_x, end_y = seg[0], seg[1], seg[2], seg[3]
                pstx, psty, pedx, pedy = start_x*width, (1-start_y)*height, end_x*width, (1-end_y)*height   
                x_min, y_min, x_max, y_max = left, upper, right, lower

                seg_type = nodes[idx][-4:]   
                if seg_type == [1, 0, 0, 0] or seg_type == [0, 1, 0, 0]:

                    new_seg = cohen_sutherland(pstx, psty, pedx, pedy, x_min, y_min, x_max, y_max)   
                    if new_seg:
                        new_sx, new_sy, new_ex, new_ey, flg = new_seg
                         
                        new_sx -= x_min
                        new_sy -= y_min
                        new_ex -= x_min
                        new_ey -= y_min
                         
                        if flg == 1 and seg_type == [0, 1, 0, 0]:
                             
                            arcmid = center[idx][:]
                            a_x, a_y = arcmid[0] * width, (1 - arcmid[1]) * height   
                            mid_newx, mid_newy = (a_x - x_min) / (x_max - x_min), (a_y - y_min) / (y_max - y_min)
                            mid_newy = 1 - mid_newy   
                            new_mid = [mid_newx, mid_newy]
                            new_center_norm.append(new_mid)
                            new_seg = [new_sx / (x_max - x_min), 1 - new_sy / (y_max - y_min),
                                       new_ex / (x_max - x_min), 1 - new_ey / (y_max - y_min)]   
                         
                        elif flg != 1 and seg_type == [0, 1, 0, 0]:
                            mid = (nodes[idx][1], nodes[idx][2])   
                            c_x, c_y = mid[0] * width, (1 - mid[1]) * height   
                            r = math.sqrt(get_2d_square_distance((c_x, c_y), (pstx, psty)))   
                             
                            px_cx, px_cy = mid[0] * width, mid[1] * height
                            px_stx, px_sty, px_edx, px_edy = start_x * width, start_y * height, end_x * width, end_y * height
                            px_st, px_ed = (px_stx, px_sty), (px_edx, px_edy)
                             
                             
                            start_angle, end_angle, stp, edp = get_sted_angle(px_st, px_ed, (px_cx, px_cy),
                                                                              z=1)   

                            res = calculate_clipped_angles((c_x, c_y), r, (pstx, psty), (pedx, pedy),
                                                        (start_angle, end_angle), (x_min, y_min, x_max, y_max))
                            if res:
                                nstart_angle, nend_angle, nstp, nedp = res   
                            else:
                                continue
                            nacr_mid = calculate_arc_midpoint((c_x, c_y), r, nstart_angle, nend_angle)   
                            mid_newx, mid_newy = (nacr_mid[0] - x_min) / (x_max - x_min), (nacr_mid[1] - y_min) / (y_max - y_min)
                            mid_newy = 1 - mid_newy   
                            new_mid = [mid_newx, mid_newy]
                            new_center_norm.append(new_mid)

                            c_nstpx, c_nstpy = (nstp[0] - x_min) / (x_max - x_min), (nstp[1] - y_min) / (y_max - y_min)
                            c_nedpx, c_nedpy = (nedp[0] - x_min) / (x_max - x_min), (nedp[1] - y_min) / (y_max - y_min)

                            new_seg = [c_nstpx, 1 - c_nstpy,
                                       c_nedpx, 1 - c_nedpy]   
                         
                        else:
                             
                            mid_newx, mid_newy = (new_sx+new_ex)/(x_max - x_min)/2, (new_sy+new_ey)/(y_max - y_min)/2
                            mid_newy = 1 - mid_newy   
                            new_mid = [mid_newx, mid_newy]
                            new_center_norm.append(new_mid)
                            new_seg = [new_sx / (x_max - x_min), 1 - new_sy / (y_max - y_min),
                                       new_ex / (x_max - x_min), 1 - new_ey / (y_max - y_min)]   

                        new_segments.append(new_seg)
                         
                        new2old_idx[new_eid] = idx
                        new_target.append(target[idx])
                        new_layers.append(layers[idx])
                        new_eid += 1
                         
                        new_node = nodes[idx][:]
                        new_node[0] = new_node[0]*width/(x_max - x_min)     
                        new_node[1], new_node[2] = mid_newx, mid_newy       
                        new_node[3], new_node[4] = new_seg[0], new_seg[1]   
                        new_node[5], new_node[6] = new_seg[2], new_seg[3]   
                        new_nodes.append(new_node)
                else:
                    mid = center[idx][:]
                    c_x, c_y = mid[0] * width, (1 - mid[1]) * height   
                    if left < c_x < right and upper < c_y < lower:
                         
                        new_cx, new_cy = (c_x - x_min)/(x_max - x_min), (c_y - y_min)/(y_max - y_min)
                        new_cy = 1-new_cy   
                        new_mid = [new_cx, new_cy]
                        new_center_norm.append(new_mid)

                        new2old_idx[new_eid] = idx
                        new_target.append(target[idx])
                        new_layers.append(layers[idx])
                        new_eid += 1

                        new_node = nodes[idx][:]
                        new_node[0] = new_node[0]*width/(x_max-x_min)
                        new_node[1], new_node[2] = new_cx, new_cy
                        new_node[3] = (new_node[3]*width - x_min) / (x_max - x_min)
                        new_node[4] = 1 - ((1 - new_node[4])*height - y_min) / (y_max - y_min)
                        new_node[5] = (new_node[5]*width - x_min) / (x_max - x_min)
                        new_node[6] = 1 - ((1 - new_node[6])*height - y_min) / (y_max - y_min)
                        new_nodes.append(new_node)

                        new_segments.append([new_cx - new_node[0], new_cy, new_cx + new_node[0], new_cy])

            segments_arr = np.array(new_segments)   

            res_with_GR = get_knn_with_geometric_relations(
                copy.deepcopy(segments_arr), max_degree=128, thresh=thresh, tol=tol, dist_tol=distol, k=k, angle_tol=angle_tol)
             
            if isinstance(res_with_GR, dict):
                nns, thresh_nns = res_with_GR['nns_list'], res_with_GR['thresh_nns_list'],
                edge_features = {'dist': res_with_GR['dist_in_thresh_list'],
                                 'relation': res_with_GR['geometric_relations_list'],
                                 }
                res_with_edge_nns_dict = {
                    "nns": nns, "thresh_nns": thresh_nns,
                }
            else:
                edge_features = None
                res_with_edge_nns_dict = None
             
            data_gcn = {
                "nd_ft": new_nodes, "ct_norm_z2o": new_center_norm,
                "cat": new_target, "segments": new_segments, "new2old_idx": new2old_idx, "layers": new_layers,
                "edge_features": edge_features, "res_with_edge_nns_dict": res_with_edge_nns_dict,
            }

            npy_path = os.path.join(output_folder, f"{filebasename}_{row}_{col}.npy")
             
            np.save(npy_path, data_gcn)
            print(f"生成npy文件,{npy_path}")




