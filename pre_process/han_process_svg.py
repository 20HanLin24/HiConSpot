# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by HanLin from  https://github.com/nicehuster/SymPoint. and https://github.com/VITA-Group/CADTransformer.

# THE DATA SOURCE:https://floorplancad.github.io/

import copy
import os, sys
import math, re
import argparse
import signal
from collections import defaultdict

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
project_root = os.path.abspath(ROOT)

sys.path.insert(0, ROOT)

import numpy as np
from glob import glob
import xml.etree.ElementTree as ET

from svgpathtools import parse_path, Arc, Line


from multiprocessing.pool import Pool
from functools import partial

import matplotlib.pyplot as plt


from .han_processSvg_utils import *

COMMANDS = ['Line', 'Arc', 'circle', 'ellipse']


def is_arc_path(path_obj):

     
    for seg in path_obj:
        if isinstance(seg, Arc):
            return True
    return False


def get_arc_segments(path_obj):
 
    arc_segs = []
    for seg in path_obj:
        if isinstance(seg, Arc):
            arc_segs.append(seg)
    return arc_segs


def get_line_segments(path_obj):

    lines = []
    for line in path_obj:
        if isinstance(line, Line):
            lines.append(line)
    return lines


def mysvg2npyV2(svg_path, output_dir, max_degree,):

    tree = ET.parse(svg_path)   
    root = tree.getroot()       
    ns = root.tag[:-3]          

    minx, miny, width, height = [int(float(x)) for x in root.attrib['viewBox'].split(' ')]
    half_width = width / 2
    half_height = height / 2

    segments = []      
    segments_norm = []
    nodes = []
    nd_ft = []         

    centers = []       
    centers_norm = []       
    centers_norm_z2o = []   
    classes = []       
    instances = []     

    args = []
    rotations = []     
    radius = []        
    strokes = []       
    widths = []        
    layerIds = []      
    id = 0

    inst_infos = defaultdict(list)

    for g in root.iter(ns + 'g'):
        id += 1
        for path in g.iter(ns + 'path'):
            try:
                path_repre = parse_path(path.attrib['d'])   
            except Exception as e:
                raise RuntimeError("Parse path failed!{}, {}".format(svg_path, path.attrib['d']))
            has_arc = is_arc_path(path_repre)   

            path_type = path_repre[0].__class__.__name__

            if path_type == 'Arc':
                start = path_repre.point(0)   
                end = path_repre.point(1)   
                segments.append([start.real, start.imag, end.real, end.imag])   

                mid = path_repre.point(0.5)    
                length = path_repre.length()   

                rotation = path_repre[0].rotation           

                rx = getattr(path_repre[0], 'rx',
                             getattr(path_repre[0], 'radius',
                                     getattr(path_repre[0], 'r', 0)))
                ry = getattr(path_repre[0], 'ry', rx)   
                r = (rx, ry)   
                type_onehot = [0, 1, 0, 0]

            else:
                start = path_repre.point(0)   
                end = path_repre.point(1)     
                segments.append([start.real, start.imag, end.real, end.imag])
                mid = path_repre.point(0.5)    
                length = path_repre.length()   

                rotation = 0   
                r = (0, 0)   
                type_onehot = [1, 0, 0, 0]

            nodes.append([length, (mid.real - minx), (mid.imag - miny),
                          (start.real - minx), (start.imag - miny), (end.real - minx), (end.imag - miny)] + type_onehot)
            nd_ft.append([length / width, (mid.real - minx) / width, (mid.imag - miny) / height,
                          (start.real - minx) / width, (start.imag - miny) / height,
                          (end.real - minx) / width, (end.imag - miny) / height] + type_onehot)   
            centers.append([mid.real, mid.imag])

            semanticId = int(path.attrib['semanticId']) if 'semanticId' in path.attrib else 0
            instanceId = int(path.attrib['instanceId']) if 'instanceId' in path.attrib else -1
            classes.append(semanticId)
            instances.append(instanceId)

            rgb = list(map(int, re.findall(r'\d+', path.attrib['stroke'])))
            strokes.append(rgb)
            widths.extend([float(path.attrib["stroke-width"])])
            layerIds.append(id)
            rotations.append(rotation)
            radius.append(r)

            inds = [0, 1 / 3, 2 / 3, 1.0]
            arg = []
            for ind in inds:
                point = path_repre.point(ind)
                arg.extend([point.real, point.imag])
            args.append(arg)
            inst_infos[(instanceId, semanticId)].extend(arg)

        for circle in g.iter(ns + 'circle'):
            cx = float(circle.attrib['cx'])   
            cy = float(circle.attrib['cy'])   
            r = float(circle.attrib['r'])     
            type_onehot = [0, 0, 1, 0]
            rotation = 360   

            segments.append([cx - r, cy, cx + r, cy])   
            nodes.append([r * 2.0 * math.pi, (cx - minx), (cy - miny),
                          (cx - r - minx), (cy - miny), (cx + r - minx), (cy - miny)] + type_onehot)
            nd_ft.append([r * 2.0 * math.pi / width, (cx - minx) / width, (cy - miny) / height,
                          (cx - r - minx) / width, (cy - miny) / height, (cx + r - minx) / width, (cy - miny) / height] + type_onehot)
            centers.append([cx, cy])

            semanticId = int(circle.attrib['semanticId']) if 'semanticId' in circle.attrib else 0
            instanceId = int(circle.attrib['instanceId']) if 'instanceId' in circle.attrib else -1
            classes.append(semanticId)
            instances.append(instanceId)

            rgb = list(map(int, re.findall(r'\d+', circle.attrib['stroke'])))
            strokes.append(rgb)
            widths.extend([float(circle.attrib["stroke-width"])])
            layerIds.append(id)
            rotations.append(rotation)
            radius.append((r, r))

            thetas = [0, math.pi / 2, math.pi, 3 * math.pi / 2, ]
            arg = []
            for theta in thetas:
                x, y = cx + r * math.cos(theta), cy + r * math.sin(theta)
                arg.extend([x, y])
            args.append(arg)
            inst_infos[(instanceId, semanticId)].extend(arg)

        for ellipse in g.iter(ns + 'ellipse'):
            cx = float(ellipse.attrib['cx'])   
            cy = float(ellipse.attrib['cy'])   
            rx = float(ellipse.attrib['rx'])   
            ry = float(ellipse.attrib['ry'])   
            type_onehot = [0, 0, 0, 1]
            rotation = 360   

            if rx > ry:
                a, b = rx, ry
            else:
                a, b = ry, rx

            ellipse_len = 2 * math.pi * b + 4 * (a - b)

            segments.append([cx - rx, cy, cx + rx, cy])
            nodes.append([ellipse_len, (cx - minx), (cy - miny),
                          (cx - rx - minx), (cy - miny), (cx + rx - minx), (cy - miny)] + type_onehot)
            nd_ft.append([ellipse_len / width, (cx - minx) / width, (cy - miny) / height,
                          (cx - rx - minx) / width, (cy - miny) / height, (cx + rx - minx) / width, (cy - miny) / height] + type_onehot)
            centers.append([cx, cy])

            semanticId = int(ellipse.attrib['semanticId']) if 'semanticId' in ellipse.attrib else 0
            instanceId = int(ellipse.attrib['instanceId']) if 'instanceId' in ellipse.attrib else -1
            classes.append(semanticId)
            instances.append(instanceId)

            rgb = list(map(int, re.findall(r'\d+', ellipse.attrib['stroke'])))
            strokes.append(rgb)
            widths.extend([float(ellipse.attrib["stroke-width"])])
            layerIds.append(id)
            rotations.append(rotation)
            radius.append((rx, ry))

            thetas = [0, math.pi / 2, math.pi, 3 * math.pi / 2, ]
            arg = []
            for theta in thetas:
                x, y = cx + a * math.cos(theta), cy + b * math.sin(theta)
                arg.extend([x, y])
            args.append(arg)
            inst_infos[(instanceId, semanticId)].extend(arg)

    segments = np.array(segments)   
    if segments.shape[0] < 2:
        print('Warning: too few segments')
        return

    assert len(classes) == len(instances), 'error'

    for c in centers:
        centers_norm.append([(c[0] - half_width) / half_width, (c[1] - half_height) / half_height])
    for c in centers:
        centers_norm_z2o.append([(c[0] - minx) / width, (c[1] - miny) / height])
    for s in segments:
        segments_norm.append([(s[0] - minx) / width,
                              (s[1] - miny) / height,
                              (s[2] - minx) / width,
                              (s[3] - miny) / height])
    obj_cts = []
    obj_boxes = []
    for (inst_id, sem_id), coords in inst_infos.items():
        if inst_id < 0: continue
        coords = np.array(coords).reshape(-1, 2)
        x1, y1 = np.min(coords[:, 0]), np.min(coords[:, 1])
        x2, y2 = np.max(coords[:, 0]), np.max(coords[:, 1])
        obj_cts.append([(x1 + x2) / 2, (y1 + y2) / 2, 0, inst_id])
        obj_boxes.append([x1, y1, x2, y2, sem_id])

    res_with_GR = get_knn_with_geometric_relations(
        copy.deepcopy(segments), max_degree=max_degree,
        thresh=width*7.5e-3,     
        dist_tol=width*0.00015,
        angle_tol=0.03
    )   
    nns, thresh_nns = res_with_GR['nns_list'], res_with_GR['thresh_nns_list'],
    edge_features = {'dist': res_with_GR['dist_in_thresh_list'],
                     'relation': res_with_GR['geometric_relations_list'],}
    res_with_edge_nns_dict = {
        "nns": nns, "thresh_nns": thresh_nns,
    }
     
    data_gcn = {
        "nd": nodes, "ct": centers,
        "nd_ft": nd_ft, "ct_norm": centers_norm, "ct_norm_z2o": centers_norm_z2o, "segments_norm": segments_norm, "segments":segments,
        "cat": classes, "inst": instances,
        "args": args, "width": width, "height": height, "widths": widths, "rgb": strokes,
        "obj_cts": obj_cts,   
        "boxes": obj_boxes,
        "edge_features": edge_features, "res_with_edge_nns_dict": res_with_edge_nns_dict,
        "layers": layerIds, "rotations": rotations, "radius": radius,
    }

    basename = os.path.basename(svg_path)   

    npy_path = os.path.join(output_dir, basename.replace(".svg", ".npy"))
     
    np.save(npy_path, data_gcn)
    print(f"Generate npy,{npy_path}")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser = argparse.ArgumentParser(description='construct graph')
    parser.add_argument('-i', '--input_dir', type=str,
                        help='the input svg directory')
    parser.add_argument('-o', '--output_dir', type=str,
                        help='the output npy directory')
    parser.add_argument('-d', '--max_degree', type=int,
                        help='the maximum neighbor number of each node', default=128)   
    parser.add_argument('-v', '--visualize', type=bool,
                        help='the visualize flag', default=False)   
    parser.add_argument('--thread_num', type=int,
                        help='multiprocess number', default=48)   
    args = parser.parse_args()
    return args


def init_worker():
    '''
    Catch Ctrl+C signal to termiante workers
    '''
    signal.signal(signal.SIGINT, signal.SIG_IGN)



def main2026():
    args = parse_args()
    args.input_dir = f"{ROOT}/datasets/Floor_Plan_CAD/svg/train"
    args.output_dir = '{}/datasets/Floor_Plan_CAD/npy/train'.format(ROOT)

    args.max_degree = 128
    args.visualize = False
    args.thread_num = 1


    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)   

    if args.visualize:   
        if not os.path.exists(os.path.join(args.output_dir, 'visualize')):
            os.makedirs(os.path.join(args.output_dir, 'visualize'), exist_ok=True)   

    print(f'> svg -> npy')
     
    svg_paths = sorted(glob(os.path.join(args.input_dir, "*.svg")))   


    has_npy_paths = sorted(glob(os.path.join(args.output_dir, "*.npy")))   
    has_npy_paths = [os.path.basename(fpaths) for fpaths in has_npy_paths]
    filter_svg_paths = []   
    for svg_file in svg_paths:
        file = os.path.basename(svg_file)
        if file.replace('.svg', '.npy') not in has_npy_paths:
            filter_svg_paths.append(svg_file)

    print("svg num = ", len(svg_paths), '\n')
    print("processed num = ", len(has_npy_paths), '\n')
    print("remaining num = ", len(filter_svg_paths))

     
    partial_func = partial(mysvg2npyV2, output_dir=args.output_dir, max_degree=args.max_degree, figsize=(8, 8), visualize=args.visualize, enable_edge_features=True)
    p = Pool(args.thread_num, init_worker)   

    try:
        p.map(partial_func, filter_svg_paths)   
    except KeyboardInterrupt:
        print("....\nCaught KeyboardInterrupt, terminating workers")
        p.terminate()   
    else:
        p.close()   
    p.join()   

     


if __name__ == "__main__":
    print("start")
    print(ROOT)
    main2026()




