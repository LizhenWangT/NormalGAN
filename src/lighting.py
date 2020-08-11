import torch


def H0(normals):
    return torch.ones_like(normals)[:, 0:1, :]


def H1(normals):
    return normals[:, 1:2, :]


def H2(normals):
    return normals[:, 2:3, :]


def H3(normals):
    return normals[:, 0:1, :]


def H4(normals):
    return torch.mul(normals[:, 0:1, :], normals[:, 1:2, :])


def H5(normals):
    return torch.mul(normals[:, 1:2, :], normals[:, 2:3, :])


def H6(normals):
    return - torch.mul(normals[:, 0:1, :], normals[:, 0:1, :]) \
           - torch.mul(normals[:, 1:2, :], normals[:, 1:2, :]) \
           + 2 * torch.mul(normals[:, 2:3, :], normals[:, 2:3, :])


def H7(normals):
    return torch.mul(normals[:, 2:3, :], normals[:, 0:1, :])


def H8(normals):
    return torch.mul(normals[:, 0:1, :], normals[:, 0:1, :]) \
           - torch.mul(normals[:, 1:2, :], normals[:, 1:2, :])


def get_H(normals, mask):
    B, C, H, W = normals.shape
    normals = torch.reshape(normals, (-1, C, H*W))
    mask_array = torch.reshape(mask, (-1, 1, H*W))
    return torch.mul(torch.cat((H0(normals), H1(normals), H2(normals), H3(normals), H4(normals),
                                  H5(normals), H6(normals), H7(normals), H8(normals)), 1), mask_array)


def get_lighting(normals, image, abd, mask, rm_graz=False): #输入为法向图，灰度图，，二值轮廓图
    """
    :param normals: BCHW
    :param image: B1HW
    :param abd: B1HW
    :param mask: B1HW
    :return: lighting with shape (B,9), LH with shape (B,1,H,W)
    """

    ## Remove normals at high grazing angle
    if rm_graz:
        mask_angle = normals[:, 2:3, :, :].gt(0.5).float()
        mask = torch.mul(mask, mask_angle)

    image = torch.mul(image, mask)
    B, C, H, W = image.shape
    image = image.reshape(-1, H*W, 1)   #(B, H*W, 1)
    A = get_H(normals, mask) #(B, 9, H*W)

    # # Use offline estimated albedo
    # if abd is not None:
    #     abd = torch.reshape(abd, (-1, H*W, 1))
    #     A_t = torch.mul(abd, A.permute(0, 2, 1))
    # else:
    #     A_t = A.permute(0, 2, 1)

    abd = torch.reshape(abd, (-1, 1, H*W))
    A = torch.mul(abd, A) #(B, 9, H*W)

    A_t = A.permute(0, 2, 1) #(B, H*W, 9)

    lighting = torch.squeeze(torch.matmul(torch.matmul((torch.matmul(A, A_t)).inverse(), A), image), dim=2) #9个光照的球谐系数
    LH = torch.reshape(torch.matmul(torch.unsqueeze(lighting, 1), A), (-1, 1, H, W))    #利用球谐函数对于当前图片的近似
    return lighting, LH

