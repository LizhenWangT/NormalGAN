import argparse
import model
import torch
import numpy as np
import loader
import time
import lighting
import os
import ops
import math
import cv2
import trimesh
import ply_from_array


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modeldir", type=str)
    parser.add_argument("--savedir", type=str, default="../results")
    parser.add_argument("--depth_net_name", type=str)
    parser.add_argument("--f_alb_net_name", type=str)
    parser.add_argument("--b_alb_net_name", type=str)
    parser.add_argument("--back_net_name", type=str)
    parser.add_argument("--dataset_dir", type=str, default="../datasets")
    parser.add_argument("--index_file", type=str, default="test.csv")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--low_thres", type=float, default=500.0)
    parser.add_argument("--up_thres", type=float, default=3000.0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    batch_size = np.int(args.batch_size)
    model_dir = args.modeldir
    save_dir = args.savedir
    low_thres = args.low_thres
    up_thres = args.up_thres

    # input resolution
    W = 512
    H = 424
    # perspective intrinsics
    cx = 256.512
    cy = 207.939
    fx = 364.276
    fy = 364.276
    # Orthographic parameters
    Z = 1650
    left = -Z * cx / fx
    right = Z * (W - cx) / fx
    top = Z * cy / fy
    bottom = -Z * (H - cy) / fy
    dx = np.abs(left) / cx
    dy = top / cy

    # load dataset filenames
    image_lists = loader.load_data(args)
    image_lists.sort()
    
    # init for the face generation
    crop_w = 424
    crop_h = 424
    fp_idx = np.zeros([crop_h, crop_w], dtype=np.int)
    bp_idx = np.ones_like(fp_idx) * (crop_h * crop_w)
    for hh in range(crop_h):
        for ww in range(crop_w):
            fp_idx[hh, ww] = hh * crop_w + ww
            bp_idx[hh, ww] += hh * crop_w + ww
    
    # init X, Y coordinate tensors
    Y, X = torch.meshgrid(torch.tensor(range(crop_h)), torch.tensor(range(crop_w)))
    X = X.unsqueeze(0).unsqueeze(0).float().cuda()  # (B,H,W)
    Y = Y.unsqueeze(0).unsqueeze(0).float().cuda()
    x_cord = X * dx
    y_cord = Y * dy

    with torch.no_grad():
        intrinsics_o = [left, top, cx, cy]
        intrinsics_p = [fx, fy, cx, cy]
        
        # load pretrained models
        ngf = 32

        depth_net = model.UNet(in_channel=4, out_channel=2, ngf=ngf, upconv=False, norm=True).cuda()
        back_net = model.SUNet(in_channel=4, out_channel=1, ngf=ngf, upconv=False, norm=True).cuda()
        f_alb_net = model.FUNet(in_channel=4, out_channel=3, ngf=ngf, upconv=True, norm=True).cuda()
        b_alb_net = model.SUNet(in_channel=4, out_channel=3, ngf=ngf, upconv=True, norm=True).cuda()

        depth_net.load_state_dict(torch.load(os.path.join(model_dir, args.depth_net_name)))
        back_net.load_state_dict(torch.load(os.path.join(model_dir, args.back_net_name)))
        f_alb_net.load_state_dict(torch.load(os.path.join(model_dir, args.f_alb_net_name)))
        b_alb_net.load_state_dict(torch.load(os.path.join(model_dir, args.b_alb_net_name)))

        depth_net = depth_net.eval()
        back_net = back_net.eval()
        f_alb_net = f_alb_net.eval()
        b_alb_net = b_alb_net.eval()

        start_time = time.time()
        max_steps = math.ceil(len(image_lists) / batch_size)
        for step in range(max_steps):
            t = time.time()
            # laod data
            # YOU NEED TO APPLY MASK FOR THE COLOR AND DEPTH DATA BEFORE INPUT
            [color_batch, depth_batch, dlist] = loader.get_batch_test(image_lists, batch_size, H, W)
            # optional for kinect 2 data (bad mask)
            #color_batch = ops.erode(color_batch, 2)
            #color_batch = ops.dilate(color_batch, 3)
            
            # reproject the 3D points to orthographic view
            o_depth_batch, o_color_batch = ops.perse2ortho(depth_batch, color_batch, intrinsics_p, intrinsics_o)
            # crop the data to 424
            o_depth_batch = ops.crop_tensor(o_depth_batch, crop_h, crop_w)
            o_color_batch = ops.crop_tensor(o_color_batch, crop_h, crop_w)
            # fill the small holes
            o_depth_batch = ops.dilate(o_depth_batch, 1)
            o_color_batch = ops.dilate(o_color_batch, 1)
            o_depth_batch = ops.erode(o_depth_batch, 1)
            o_color_batch = ops.erode(o_color_batch, 1)
            o_mask_batch = (o_depth_batch > 0).float()
            
            # fix the averange depth to 1555 mm (for better results)
            fix_p = 1555
            tmp1 = fix_p - torch.sum(o_depth_batch) / torch.sum(o_mask_batch)
            o_depth_batch = o_depth_batch + tmp1
            o_depth = ops.convert_depth_to_m1_1(o_depth_batch, low_thres, up_thres).clamp(-1.0, 1.0)
            o_color = ops.convert_color_to_m1_1(o_color_batch).clamp(-1.0, 1.0)
            
            # front depth net
            dout = depth_net(torch.cat((o_depth, o_color), dim=1), inter_mode='bilinear')
            dz0 = dout[:, 0:1]
            o_mask_batch = ops.dilate(o_mask_batch, 10)
            o_mask_batch = ops.erode(o_mask_batch, 8)
            mask = (dout[:, 1:2] > 0.9).float() * o_mask_batch
            dz0_batch = ops.convert_depth_back_from_m1_1(dz0, low_thres, up_thres) * mask
            normal = ops.depth2normal_ortho(dz0_batch + (1 - mask) * 3000, dx, dy)
            f_depth = ops.convert_depth_to_m1_1(dz0_batch, low_thres, up_thres).clamp(-1.0, 1.0)
            
            # front color net
            depth_sphere = lighting.get_H(normal.permute(0, 3, 1, 2), mask)
            depth_sphere = torch.reshape(depth_sphere, (-1, 9, crop_h, crop_w)) * mask
            cz0 = f_alb_net(torch.cat((o_color, f_depth), dim=1), depth_sphere, inter_mode='nearest')
            cz0_batch = ops.convert_color_back_from_m1_1(cz0) * ops.erode(mask, 3)
            cz0_batch = ops.dilate(cz0_batch, 5) * mask # smooth the edge area
            f_alb = ops.convert_color_to_m1_1(cz0_batch).clamp(-1.0, 1.0)
            
            # back depth net
            dz1 = back_net(torch.cat((f_alb, f_depth), dim=1), inter_mode='bilinear')
            dz1_batch = ops.convert_depth_back_from_m1_1(dz1, low_thres, up_thres) * mask
            
            # back color net
            cz1 = b_alb_net(torch.cat((f_alb, f_depth), dim=1), inter_mode='nearest')
            cz1_batch = ops.convert_color_back_from_m1_1(cz1) * mask
            
            network_time = time.time() - t
            
            # convert the images to 3D mesh
            fpct = torch.cat((x_cord, y_cord, dz0_batch, cz0_batch), dim=1)
            bpct = torch.cat((x_cord, y_cord, dz1_batch, cz1_batch), dim=1)
            # dilate for the edge point interpolation
            fpct = ops.dilate(fpct, 1)
            bpct = ops.dilate(bpct, 1)
            fpc = fpct[0].permute(1, 2, 0).detach().cpu().numpy()
            bpc = bpct[0].permute(1, 2, 0).detach().cpu().numpy()
            
            ops.remove_points(fpc, bpc)
            # get the edge region for the edge point interpolation
            mask_pc = fpc[:, :, 2] > low_thres
            mask_pc = mask_pc.astype(np.float32)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
            eroded = cv2.erode(mask_pc, kernel)
            edge = (mask_pc - eroded).astype(np.bool)
            # interpolate 2 points for each edge point pairs
            fpc[edge, 2:6] = (fpc[edge, 2:6] * 2 + bpc[edge, 2:6] * 1) / 3
            bpc[edge, 2:6] = (fpc[edge, 2:6] * 1 + bpc[edge, 2:6] * 2) / 3
            fpc = fpc.reshape(-1, 6)
            bpc = bpc.reshape(-1, 6)
            if (np.sum(mask_pc) < 100):
                print('noimage')
                continue
            f_faces = ops.getfrontFaces(mask_pc, fp_idx)
            b_faces = ops.getbackFaces(mask_pc, bp_idx)
            edge_faces = ops.getEdgeFaces(mask_pc, fp_idx, bp_idx)
            faces = np.vstack((f_faces, b_faces, edge_faces))
            points = np.concatenate((fpc, bpc), axis=0)
            # reset center point and convert mm to m
            points[:, 0:3] = -(points[:, 0:3] - np.array([[crop_w / 2 * dx, (crop_h - 5) * dy - 700, fix_p]])) / 1000.0
            points[:, 0] = -points[:, 0]
            vertices = points[:, 0:3]
            colors = points[:, 3:6].astype(np.uint8)
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors)
            
            # mkdirs
            filename = image_lists[dlist[0]].strip().split(',')[1].split(os.sep)[-1].split('.')[0]
            #groupname = image_lists[dlist[0]].strip().split(',')[1].split(os.sep)[-3]
            output_ply_name = os.path.join(save_dir, 'ply', filename + '.ply')
            print('Output_file:', output_ply_name)
            if not os.path.isdir(os.path.join(save_dir, 'ply')):
                os.mkdir(os.path.join(save_dir, 'ply'))
            if not os.path.isdir(os.path.join(save_dir, 'dz0')):
                os.mkdir(os.path.join(save_dir, 'dz0'))
            if not os.path.isdir(os.path.join(save_dir, 'dz1')):
                os.mkdir(os.path.join(save_dir, 'dz1'))
            if not os.path.isdir(os.path.join(save_dir, 'cz0')):
                os.mkdir(os.path.join(save_dir, 'cz0'))
            if not os.path.isdir(os.path.join(save_dir, 'cz1')):
                os.mkdir(os.path.join(save_dir, 'cz1'))
            
            # save the results
            cz0_batch = cz0_batch + (1 - mask) * 255
            cz1_batch = cz1_batch + (1 - mask) * 255
            loader.save_output(os.path.join(save_dir, 'dz0', filename + '.png'), dz0_batch)
            loader.save_output(os.path.join(save_dir, 'dz1', filename + '.png'), dz1_batch)
            loader.save_output_color(os.path.join(save_dir, 'cz0', filename + '.png'), cz0_batch)
            loader.save_output_color(os.path.join(save_dir, 'cz1', filename + '.png'), cz1_batch)
            ply_from_array.ply_from_array_color(mesh.vertices, mesh.visual.vertex_colors, mesh.faces, output_ply_name)

            print("Step: {}".format(step + 1),
                  " Network time: {:.3f} s".format(network_time),
                  " Total time: {:.3f} s".format(time.time() - start_time))

