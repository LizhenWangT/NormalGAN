import cv2 as cv
import math
import numpy as np
import os
from getnormal_zxc import *

def CalAxisNoise(depth, normal, intrinsics):
    h, w = depth.shape
    ray = np.tile(np.array([0, 0, 1], dtype=np.float), (h, w, 1))
    ray = normalize(ray)
    theta = np.arccos(np.sum(ray*normal, axis=2))
    theta[np.where(depth == 0)] = 0.0
    theta = np.clip(theta, 0.0, 1.5) 
    D = (depth/1000.0).astype(np.float)
    deviation = 1.5-0.5*D+0.3*D**2+0.1*D**1.5*(theta**2/(1.58-theta)**2)
    deviation[np.where(depth==0)]=0.0
    noise = np.random.normal(0, 1, h*w).reshape([h, w])
    noise = noise * deviation + depth
    return noise

depthdir = 'Persedata/DZ0'
savedir = 'Persedata/depth'

if not os.path.isdir(savedir):
    os.mkdir(savedir)

intrinsics = [362.633, 363.378, 258.440, 208.440]
for filename in os.listdir(depthdir):
    print(depthdir + '/' + filename)
    depth = cv.imread(depthdir + '/' + filename, -1)
    m = depth > 0
    mask = np.zeros_like(depth).astype(np.uint8)
    mask[np.where(depth>0)] = 255
    normal = depth2normal(depth.astype(np.float32), intrinsics)
    for times in range(1):
        axisnoise = CalAxisNoise(depth, normal, intrinsics)
        newmask = np.copy(mask)
        lateralnoise = np.zeros_like(depth)
        for k in range(1):
            _, contours, _ = cv.findContours(newmask, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
            edges = np.empty((0, contours[0].shape[1], contours[0].shape[2]),dtype=np.int)
            for cidx in range(len(contours)):
                edges = np.vstack((edges, contours[cidx]))
            sample = np.random.randint(low=0, high=edges.shape[0], size=int(0.1*edges.shape[0]))

            for idx in range(sample.shape[0]):
                x = edges[sample[idx], 0, 0]
                y = edges[sample[idx], 0, 1]
                t = 1
                if y>=t and y<newmask.shape[0]-t and x>=t and x<newmask.shape[1]-t:
                    index = tuple(([y - t, y, y, y, y + t], [x, x - t, x, x + t, x]))
                    if idx%2==0:
                        newmask[index] = 0
                        lateralnoise[index] = 0
                    else:
                        newmask[index] = 255
                        lateralnoise[index] = axisnoise[y, x]
            _, vis_contours, _ = cv.findContours(newmask, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)  
        lateralnoise[np.where(depth>0)]=0
        noisedD = (axisnoise + lateralnoise).astype(np.uint16) * m.astype(np.uint16)
        cv.imwrite(savedir + '/' + filename, noisedD)

