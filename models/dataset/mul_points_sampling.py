"""
HiConSpot Model

Author: Han Lin (hcl0jf26@gmail.com)
Please cite our work if the code is helpful to you.
"""

import math
import warnings
from collections import defaultdict

from torch.nn import functional as F
import torch
import torch.nn as nn
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np


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



def split_point_sets_to_individual_tensors(point_sets: List[List[Tuple[float, float]]]) -> List[torch.Tensor]:


    point_tensor = torch.tensor(point_sets, dtype=torch.float32)  # (N, max_points, 2)


    individual_tensors = []
    for i in range(point_tensor.shape[1]):

        points_at_position_i = point_tensor[:, i, :]  # (V, 2)
        individual_tensors.append(points_at_position_i)

    return individual_tensors

