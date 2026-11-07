

import numpy as np
import math
import random


def RandomHorizonFilp(args,width=140):
    
    args[:,0::2] = width - args[:,0::2]
    return args


def RandomVerticalFilp(args,Hight=140):
    
    args[:, 1::2] = Hight - args[:, 1::2]
    return args


def rotate_xy(args, width, height, angle):
    """
       对二维坐标点进行旋转变换（以图像中心为原点）
       Args:
           args: 输入坐标点，形状为 (N, 2M)，每行存储 M 个点的坐标（x1, y1, x2, y2, ...）
           width: 图像宽度，用于计算旋转中心 x 坐标（width/2）
           height: 图像高度，用于计算旋转中心 y 坐标（height/2）
           angle: 旋转角度（度），顺时针旋转为正方向
       Returns:
           旋转后的坐标点，形状与输入一致（N, 2M）
    """
    Pi_angle = angle * math.pi / 180.0   
    a, b = width/2, height/2             
    x0, y0 = args[:, ::2], args[:, 1::2]   
     
     
    X0 = (x0-a) * math.cos(Pi_angle) - (y0-b) * math.sin(Pi_angle) + a
    Y0 = (x0-a) * math.sin(Pi_angle) + (y0-b) * math.cos(Pi_angle) + b
     
    return np.concatenate([X0, Y0], axis=1)


def random_rotate(points,width,height):
    """对二维坐标点进行随机旋转变换（以图像中心为原点）

    Args:
        points: 输入坐标点，形状为 (N, 2)，每行存储一个点的 (x, y) 坐标
        width: 图像宽度，用于计算旋转中心 x 坐标（width/2）
        height: 图像高度，用于计算旋转中心 y 坐标（height/2）

    Returns:
        随机旋转后的坐标点，形状与输入一致（N, 2）"""
    angle = random.uniform(-180, 180)   
     
    centroid = [width/2, height/2]
     
     
     
    points_centered = points - centroid

     
     
    angle_radians = np.deg2rad(angle)
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians)],
        [np.sin(angle_radians),  np.cos(angle_radians)]
    ])

     
     
     
    points_rotated = np.dot(points_centered, rotation_matrix.T) + centroid

    return points_rotated


def random_rotate_pt(points,width,height):
    """对二维坐标点进行随机旋转变换（以图像中心为原点）

    Args:
        points: 输入坐标点，形状为 (N, 2)，每行存储一个点的 (x, y) 坐标
        width: 图像宽度，用于计算旋转中心 x 坐标（width/2）
        height: 图像高度，用于计算旋转中心 y 坐标（height/2）

    Returns:
        随机旋转后的坐标点，形状与输入一致（N, 2）"""
    angle = random.uniform(-180, 180)   
     
    centroid = [width/2, height/2]
     
     
     
    points_centered = points - centroid

     
     
    angle_radians = np.deg2rad(angle)
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians)],
        [np.sin(angle_radians),  np.cos(angle_radians)]
    ])

     
     
     
    points_rotated = np.dot(points_centered, rotation_matrix.T) + centroid

    return points_rotated, angle
