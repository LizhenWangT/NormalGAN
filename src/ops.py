import torch
from torch import nn
import numpy as np
import cv2
import time


def convert_depth_to_0_1(tensor, low_thres, up_thres):
    return (tensor - low_thres) / (up_thres - low_thres)


def convert_color_to_0_1(tensor):
    return tensor / 255.0


def convert_depth_back_from_0_1(tensor, low_thres, up_thres):
    return tensor * (up_thres - low_thres) + low_thres


def convert_color_back_from_0_1(tensor):
    return tensor * 255.0


def convert_depth_to_m1_1(tensor, low_thres, up_thres):
    return (tensor - low_thres) * 2.0 / (up_thres - low_thres) - 1.0


def convert_color_to_m1_1(tensor):
    return (tensor - 127.5) / 127.5


def convert_depth_back_from_m1_1(tensor, low_thres, up_thres):
    return (tensor + 1.0) * (up_thres - low_thres) / 2.0 + low_thres


def convert_color_back_from_m1_1(tensor):
    return (tensor + 1.0) * 127.5


def vector_dot(x, y):
    return x[:, :, :, 0:1] * y[:, :, :, 0:1] + \
           x[:, :, :, 1:2] * y[:, :, :, 1:2] + \
           x[:, :, :, 2:3] * y[:, :, :, 2:3]


def normalize(array):  # array is torch tensor
    return torch.nn.functional.normalize(array, p=2, dim=3, eps=1e-3)


# set gloabal tensor to init X,Y only for the first time
flag_XY = True
X = torch.zeros([0]).cuda()
Y = torch.zeros([0]).cuda()


def depth2normal_perse(depth, intrinsics):
    global flag_XY, X, Y
    # depth: [B,1,H,W]
    # intrinsics: [fx, fy, cx, cy]
    fx, fy, cx, cy = intrinsics
    B, _, H, W = depth.shape
    inv_fx = 1.0 / fx
    inv_fy = 1.0 / fy
    depth = depth[:, 0, :, :]

    if flag_XY:
        Y, X = torch.meshgrid(torch.tensor(range(H)), torch.tensor(range(W)))
        X = X.unsqueeze(0).repeat(B, 1, 1).float().cuda()  # (B,H,W)
        Y = Y.unsqueeze(0).repeat(B, 1, 1).float().cuda()
        flag_XY = False

    x_cord_p = (X - cx) * inv_fx * depth
    y_cord_p = (Y - cy) * inv_fy * depth

    p = torch.stack([x_cord_p, y_cord_p, depth], dim=3)  # (B,H,W,3)

    # vector of p_3d in west, south, east, north direction
    p_ctr = p[:, 1:-1, 1:-1, :]
    vw = p_ctr - p[:, 1:-1, 2:, :]
    vs = p[:, 2:, 1:-1, :] - p_ctr
    ve = p_ctr - p[:, 1:-1, :-2, :]
    vn = p[:, :-2, 1:-1, :] - p_ctr
    normal_1 = torch.cross(vs, vw)  # (B,H-2,W-2,3)
    normal_2 = torch.cross(vn, ve)
    normal_1 = normalize(normal_1)
    normal_2 = normalize(normal_2)
    normal = normal_1 + normal_2
    normal = normalize(normal)
    paddings = (0, 0, 1, 1, 1, 1, 0, 0)
    normal = torch.nn.functional.pad(normal, paddings, 'constant')  # (B,H,W,3)
    return normal  # (B,H,W,3)


def depth2normal_ortho(depth, dx, dy):
    global flag_XY, X, Y
    # depth: [B,1,H,W]
    B, _, H, W = depth.shape
    depth = depth[:, 0, :, :]

    if flag_XY:
        Y, X = torch.meshgrid(torch.tensor(range(H)), torch.tensor(range(W)))
        X = X.unsqueeze(0).repeat(B, 1, 1).float().cuda()  # (B,H,W)
        Y = Y.unsqueeze(0).repeat(B, 1, 1).float().cuda()
        flag_XY = False

    x_cord = X * dx
    y_cord = Y * dy
    p = torch.stack([x_cord, y_cord, depth], dim=3)  # (B,H,W,3)

    # vector of p_3d in west, south, east, north direction
    p_ctr = p[:, 1:-1, 1:-1, :]
    vw = p_ctr - p[:, 1:-1, 2:, :]
    vs = p[:, 2:, 1:-1, :] - p_ctr
    ve = p_ctr - p[:, 1:-1, :-2, :]
    vn = p[:, :-2, 1:-1, :] - p_ctr
    normal_1 = torch.cross(vs, vw)  # (B,H-2,W-2,3)
    normal_2 = torch.cross(vn, ve)
    normal_1 = normalize(normal_1)
    normal_2 = normalize(normal_2)
    normal = normal_1 + normal_2
    normal = normalize(normal)
    paddings = (0, 0, 1, 1, 1, 1, 0, 0)
    normal = torch.nn.functional.pad(normal, paddings, 'constant')  # (B,H,W,3)
    return normal  # (B,H,W,3)


def adjust_learning_rate(optimizer, decay_rate=.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # m.weight.data.normal_(0.0, 0.02)
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.normal_(0.0, 0.02)
    elif classname.find('Linear') != -1:
        # m.weight.data.normal_(0.0, 0.02)
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.normal_(0.0, 0.02)


def pseudo(depth):
    # depth: [B,1,H,W]
    B, _, H, W = depth.shape
    output = np.zeros([B, H, W, 3], dtype=np.uint8)
    for i in range(B):
        gray = (depth[i, 0] / 10.0).detach().cpu().numpy().astype(np.uint8)
        output[i] = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return output


# set gloabal tensor to init X,Y only for the first time
flag_per = True
bat = torch.tensor([0]).cuda()
zet = torch.tensor([0]).cuda()
X1 = torch.zeros([0]).cuda()
Y1 = torch.zeros([0]).cuda()


def perse2ortho(depth_p, color_p, intrinsics_p, intrinsics_o):
    global X1, Y1, flag_per, bat, zet
    t = time.time()
    B, _, H, W = depth_p.shape
    left, top, cx, cy = intrinsics_o
    dx = abs(left) / cx
    dy = top / cy
    depth_o = torch.zeros_like(depth_p)
    color_o = torch.zeros_like(color_p)

    fx, fy, cx, cy = intrinsics_p
    depth = depth_p[:, 0, :, :]
    inv_fx = 1.0 / fx
    inv_fy = 1.0 / fy

    if flag_per:
        Y1, X1 = torch.meshgrid(torch.tensor(range(H)), torch.tensor(range(W)))
        X1 = X1.unsqueeze(0).repeat(B, 1, 1).float().cuda()  # (B,H,W)
        Y1 = Y1.unsqueeze(0).repeat(B, 1, 1).float().cuda()
        bat = torch.arange(B).long()
        zet = torch.arange(3).long()

    x_cord_p = (X1 - cx) * inv_fx * depth
    y_cord_p = (cy - Y1) * inv_fy * depth

    p = torch.stack([x_cord_p, y_cord_p, depth, color_p[:, 0, :, :], color_p[:, 1, :, :], color_p[:, 2, :, :]], dim=3)  # (B,H,W,3)

    p[:, :, :, 0] = torch.round((p[:, :, :, 0] - left) / dx).clamp(0, W - 1)
    p[:, :, :, 1] = torch.round((top - p[:, :, :, 1]) / dy).clamp(0, H - 1)
    p = p.reshape((B, -1, 6))

    for b in range(B):
        mask = (depth_p[b] > 0).float().flatten()
        mask = torch.nonzero(mask).squeeze()
        ti = (torch.Tensor([b]).long(), mask.long())
        tmp = p[ti]
        index1 = (bat[b], zet[0], tmp[:, 1].long(), tmp[:, 0].long())
        index2 = (bat[b], zet[1], tmp[:, 1].long(), tmp[:, 0].long())
        index3 = (bat[b], zet[2], tmp[:, 1].long(), tmp[:, 0].long())
        depth_o[index1] = tmp[:, 2]
        color_o[index1] = tmp[:, 3]
        color_o[index2] = tmp[:, 4]
        color_o[index3] = tmp[:, 5]

    return depth_o, color_o


def dilate(depth, pix):
    # depth: [B, 1, H,W]
    newdepth = torch.tensor(depth)
    for i in range(pix):
        d1 = newdepth[:, :, 1:, :]
        d2 = newdepth[:, :, :-1, :]
        d3 = newdepth[:, :, :, 1:]
        d4 = newdepth[:, :, :, :-1]
        newdepth[:, :, :-1, :] = torch.where(newdepth[:, 0:1, :-1, :] > 0, newdepth[:, :, :-1, :], d1)
        newdepth[:, :, 1:, :] = torch.where(newdepth[:, 0:1, 1:, :] > 0, newdepth[:, :, 1:, :], d2)
        newdepth[:, :, :, :-1] = torch.where(newdepth[:, 0:1, :, :-1] > 0, newdepth[:, :, :, :-1], d3)
        newdepth[:, :, :, 1:] = torch.where(newdepth[:, 0:1, :, 1:] > 0, newdepth[:, :, :, 1:], d4)
        depth = newdepth
    return newdepth


def erode(depth, pix):
    # depth: [B, C, H, W]
    newdepth = torch.tensor(depth)
    for i in range(pix):
        d1 = depth[:, :, 1:, :]
        d2 = depth[:, :, :-1, :]
        d3 = depth[:, :, :, 1:]
        d4 = depth[:, :, :, :-1]
        newdepth[:, :, :-1, :] = torch.where(newdepth[:, :, :-1, :] > 0, d1, newdepth[:, :, :-1, :])
        newdepth[:, :, 1:, :] = torch.where(newdepth[:, :, 1:, :] > 0, d2, newdepth[:, :, 1:, :])
        newdepth[:, :, :, :-1] = torch.where(newdepth[:, :, :, :-1] > 0, d3, newdepth[:, :, :, :-1])
        newdepth[:, :, :, 1:] = torch.where(newdepth[:, :, :, 1:] > 0, d4, newdepth[:, :, :, 1:])
        depth = newdepth
    return newdepth


def set_requires_grad(nets, requires_grad):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad


def crop_tensor(tensor, h, w):
    _, _, H, W = tensor.shape
    i1 = np.int(H/2 - h/2)
    i2 = np.int(H/2 + h/2)
    i3 = np.int(W/2 - w/2)
    i4 = np.int(W/2 + w/2)
    return tensor[:, :, i1:i2, i3:i4]


def remove_points(fp, bp):
    f0 = fp[:, :, 2] - bp[:, :, 2]
    f0 = f0 > 0
    fp[f0, 2] = 0.0
    bp[f0, 2] = 0.0


def getEdgeFaces(mask, fp_idx, bp_idx):
    _, contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    all_boundary_faces_idx = []
    for i in range(len(contours)):
        edges = contours[i][:, 0, :]
        nextedges = np.vstack((edges[1:], edges[0]))
        fp_edge_idx = fp_idx[edges[:, 1], edges[:, 0]].reshape(-1, 1)
        bp_edge_idx = bp_idx[edges[:, 1], edges[:, 0]].reshape(-1, 1)
        bp_nextedge_idx = bp_idx[nextedges[:, 1], nextedges[:, 0]].reshape(-1, 1)
        fp_nextedge_idx = fp_idx[nextedges[:, 1], nextedges[:, 0]].reshape(-1, 1)
        boundary_faces_idx = np.vstack((np.hstack((fp_edge_idx, bp_edge_idx, bp_nextedge_idx)),
                                        np.hstack((fp_edge_idx, bp_nextedge_idx, fp_nextedge_idx))))
        if i == 0:
            all_boundary_faces_idx = boundary_faces_idx
        else:
            all_boundary_faces_idx = np.vstack((all_boundary_faces_idx, boundary_faces_idx))
    return all_boundary_faces_idx


def getbackFaces(mask, p_idx):
    p_valid_idx = p_idx * mask
    p00_idx = p_valid_idx[:-1, :-1].reshape(-1, 1)
    p10_idx = p_valid_idx[1:, :-1].reshape(-1, 1)
    p11_idx = p_valid_idx[1:, 1:].reshape(-1, 1)
    p01_idx = p_valid_idx[:-1, 1:].reshape(-1, 1)
    all_faces = np.vstack((np.hstack((p00_idx, p01_idx, p10_idx)), np.hstack((p01_idx, p11_idx, p10_idx)),
                           np.hstack((p00_idx, p11_idx, p10_idx)), np.hstack((p00_idx, p01_idx, p11_idx))))
    fp_faces = all_faces[np.where(all_faces[:, 0] * all_faces[:, 1] * all_faces[:, 2] > 0)]
    return fp_faces


def getfrontFaces(mask, p_idx):
    p_valid_idx = p_idx * mask
    p00_idx = p_valid_idx[:-1, :-1].reshape(-1, 1)
    p10_idx = p_valid_idx[1:, :-1].reshape(-1, 1)
    p11_idx = p_valid_idx[1:, 1:].reshape(-1, 1)
    p01_idx = p_valid_idx[:-1, 1:].reshape(-1, 1)
    all_faces = np.vstack((np.hstack((p00_idx, p10_idx, p01_idx)), np.hstack((p01_idx, p10_idx, p11_idx)),
                           np.hstack((p00_idx, p10_idx, p11_idx)), np.hstack((p00_idx, p11_idx, p01_idx))))
    fp_faces = all_faces[np.where(all_faces[:, 0] * all_faces[:, 1] * all_faces[:, 2] > 0)]
    return fp_faces

