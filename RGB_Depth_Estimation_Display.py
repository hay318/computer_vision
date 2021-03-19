import math
import sys
from PIL import Image
import numpy as np
import scipy.io as sio
import os
import time
import re
import cupy as cp
import multiprocessing as mp
from openpyxl import load_workbook
import pandas as pd

#Kinect Camera Calibration Parameters
s_factor = 5000.0
fx_d = 365.3768
fy_d = 365.3768
cx_d = 253.6238
cy_d = 211.5918
fx_rgb = 1054.8082
fy_rgb = 1054.8082
cx_rgb = 965.6725
cy_rgb = 552.0879

RR = np.array([
    [0.99991, -0.013167, -0.0020807],
    [0.013164, 0.99991, -0.0011972],
    [-0.0020963, 0.0011697, 1]
])
TT = np.array([0.052428, 0.0006748, 0.000098668])
extrinsics = np.array([[.99991, -0.013167, -0.0020807, 0.052428], [0.013164, 0.99991, -0.0011972, 0.0006748], [-0.0020963, 0.0011697, 1, 0.000098668], [0, 0, 0, 1]])


def init_max():
    for center in centers3:
        if center == (0, 0):
            continue
        min_xy.append(sys.float_info.max)
        min_vex.append((0, 0))
        p_vex.append((0, 0, 0))


def depth_rgb_registration(rgb, depth):

    init_max()
    rgb = Image.open(rgb)
    depth = Image.open(depth).convert('L')
    if rgb.mode != "RGB":
        raise Exception("Color image is not in RGB format")

    for v in range(depth.size[0]):
        for u in range(depth.size[1]):
            try:
                (p, x, y) = depth_to_xyz_and_rgb(v, u, depth)  
            except:
                continue

            if (x > rgb.size[0] - 1 or y > rgb.size[1] - 1 or x < 1 or y < 1 or np.isnan(x) or np.isnan(y)):
                continue
            x = round(x)
            y = round(y)
            color = rgb.getpixel((x, y))
            min_distance((x, y), p)

            if color == (0, 0, 0):
                p[0] = 0
                p[1] = 0
                p[2] = 0
                continue
            points.append(" %f %f %f %d %d %d 0\n" % (p[0], p[1], p[2], 255, 0, 0))

        i = 0
        x = []
        y = []
        z = []

    for val in min_vex:

        disp_list.append(val[0])
        disp_list.append(val[1])
        disp_list.append(p_vex[i][0])
        disp_list.append(p_vex[i][1])
        disp_list.append(p_vex[i][2])


        points.append(" %f %f %f %d %d %d 0\n" % (p_vex[i][0], p_vex[i][1], p_vex[i][2], 0, 255, 0))
        x.append(p_vex[i][0])
        y.append(p_vex[i][1])
        z.append(p_vex[i][2])
        i = i + 1


def min_distance(val, p):
    i = 0
    for center in centers3:
        if center == (0, 0):
            continue
        temp = math.sqrt(math.pow(center[0] - val[0], 2) + math.pow(center[1] - val[1], 2))
        if temp < min_xy[i]:
            min_xy[i] = temp
            min_vex[i] = val
            p_vex[i] = p
        i = i + 1



def depth_to_xyz_and_rgb(uu, vv, dep):
    pcz = dep.getpixel((uu, vv))
    if pcz == 60:
        return

    pcx = (uu - cx_d) * pcz / fx_d
    pcy = ((vv - cy_d) * pcz / fy_d)

    # Extrinsic Calibration
    P3D = np.array([pcx, pcy, pcz])
    P3Dp = np.dot(RR, P3D) - TT
    uup = (P3Dp[0] * fx_rgb / P3Dp[2] + cx_rgb)
    vvp = (P3Dp[1] * fy_rgb / P3Dp[2] + cy_rgb)

    return P3D, uup, vvp


def convert(list):
    s = [str(i) for i in list]
    res = int("".join(s))

    return res


def display_fun(mat, selected_depth, selected_color, results, excel):
    global min_xy, min_vex, p_vex, p_xyz, image_list, image_list2, points, centers3, disp_list, data, disp_list_b
    writer = pd.ExcelWriter(excel, engine='openpyxl')
    wb = writer.book
    route = mat
    db_lis = os.listdir(route)

    route3 = selected_depth
    included_extensions = ['png']
    db_lis3 = [fn for fn in os.listdir(route3)
                   if any(fn.endswith(ext) for ext in included_extensions)]

    route4 = selected_color
    db_lis4 = os.listdir(route4)

    route6 = results
    db_lis6 = os.listdir(route6)

    disp_list = []
    disp_list_b = []
    data = []

    for idx, list1 in enumerate(db_lis4):
        for i in range(len(db_lis3)):
            if list1.split('.')[0] == db_lis3[i].split('.')[0]: 
                rgb = os.route.join(route4, list1)
                depth = os.route.join(route3, sorted(db_lis3)[i])
                m = sorted(db_lis)[idx]
                mat2 = sio.loadmat(os.route.join(route, m))
                abc = list1.split('.')[0]
                disp_list = []

                fnum = convert(re.findall("(\d+)", abc))
                disp_list.append(fnum)
                centers2 = mat2['b1']
                centers3 = np.array(centers2).tolist()

                min_xy = []
                min_vex = []
                p_vex = []
                p_xyz = []
                image_list = []
                image_list2 = []
                points = []

                depth_rgb_registration(rgb, depth)
                disp_list_b.append(disp_list)
                data = pd.DataFrame(disp_list_b)
                data.to_excel(writer, index=False)
                wb.save(excel)
