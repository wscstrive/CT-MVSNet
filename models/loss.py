import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt

def ctmvsnet_loss(inputs, depth_gt_ms, depth_src, mask_ms, proj_mats, **kwargs):
    stage_lw = kwargs.get("dlossw", [1.0,1.0,1.0])
    depth_values = kwargs.get("depth_values")
    depth_min, depth_max = depth_values[:,0], depth_values[:,-1]

    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    ce_loss_stages = []
    fm_loss_stages = []

    for stage_idx, (stage_inputs, p_mats, src_gt, stage_key) in enumerate([(inputs[k], proj_mats[k], depth_src[k], k) for k in inputs.keys() if "stage" in k]):
        prob_volume = stage_inputs["prob_volume"]
        depth_values = stage_inputs["depth_values"]
        feature_ori = stage_inputs["feature_ori"]
        feature_amt = stage_inputs["feature_amt"]
        depth_ref = stage_inputs["depth"]

        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key] > 0.5

        # ce loss
        ce_loss = cross_entropy_loss(prob_volume, depth_gt, mask, depth_values)
        ce_loss_stages.append(ce_loss)

        # fm loss
        fm_loss = feature_metric_loss(mask,feature_ori,feature_amt,depth_ref, p_mats, src_gt,depth_gt)
        fm_loss_stages.append(fm_loss)

        # total loss
        lam1, lam2 = 2, 1.2
        total_loss = total_loss + stage_lw[stage_idx] * (lam1 * ce_loss + lam2 * fm_loss)

    depth_gt = depth_gt_ms[stage_key]
    epe = cal_metrics(depth_ref, depth_gt, mask, depth_min, depth_max)

    return total_loss, epe, ce_loss_stages, fm_loss_stages

def feature_metric_loss(mask,feature_ori,feature_amt,depth_ref, p_mats, src_gt,depth_gt):
    batch_size, _, _ = depth_ref.shape
    total_src_views = len(src_gt)

    mask_true = mask
    valid_pixel_num = torch.sum(mask_true, dim=[1,2]) + 1e-6

    feature_ori_0 = feature_ori
    feature_amt_0 = feature_amt

    warped_fea_ori = 0
    warped_fea_amt = 0
    s = 1e-8
    for b in range(batch_size):
        depth_ref_est = depth_ref[b, :, :]
        mask = mask.to(torch.float32)
        ref_extrinsics = p_mats[b, 0, 0, :4, :4]  ## p_mats shape: [batch, nviews, 2, 4, 4]
        ref_intrinsics = p_mats[b, 0, 1, :3, :3]
        mask = mask[b, :, :]
        ## Iterate over each source view to generate mask for it
        for src_idx in range(total_src_views):
            depth_src_est = src_gt[src_idx][b, :, :]
            src_extrinsics = p_mats[b, src_idx + 1, 0, :4, :4]
            src_intrinsics = p_mats[b, src_idx + 1, 1, :3, :3]


            warped_ref_fea, warped_amt_fea = reproject_with_depth(feature_ori, feature_amt, depth_ref_est, ref_intrinsics, ref_extrinsics, depth_src_est, src_intrinsics, src_extrinsics)
            warped_fea_ori += warped_ref_fea
            warped_fea_amt += warped_amt_fea

    upsilon1 = (warped_fea_ori - feature_ori_0)
    upsilon2 = (warped_fea_amt - feature_amt_0)

    fm_weight = torch.div(torch.abs(upsilon2), torch.abs(upsilon1) + 1e-8)
    fm_weight = torch.log(torch.clamp(fm_weight, min=1.2, max=3.5))
    fm_weight_mean = torch.mean(fm_weight, 1)

    # image_array = feature_ori_0.permute(1, 2, 0).cpu().detach().numpy()  # 将通道维度移到最后，并转换为 NumPy 数组
    # plt.imshow(image_array)
    # plt.axis('off')  # 关闭坐标轴
    # plt.show()

    fm_loss = F.smooth_l1_loss(depth_ref * fm_weight_mean, depth_gt * fm_weight_mean, size_average=True)

    return fm_loss

def reproject_with_depth(feature_ori, feature_amt, depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    # reference view x, y
    x_ref, y_ref = torch.meshgrid(torch.arange(0, width),
                                  torch.arange(0, height), indexing='xy')
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])

    # reference 3D space
    xyz_ref = torch.matmul(torch.linalg.inv(intrinsics_ref),
                           torch.vstack((x_ref.to(device='cuda'),
                                         y_ref.to(device='cuda'),
                                         torch.ones_like(x_ref,
                                                         device=torch.device('cuda')))) *
                           depth_ref.reshape([-1]))


    xyz_src = torch.matmul(torch.matmul(extrinsics_src,
                                        torch.linalg.inv(extrinsics_ref)),
                           torch.vstack((xyz_ref.to(device='cuda'),
                                        torch.ones_like(x_ref,
                                                        device=torch.device('cuda')))))[:3]

    K_xyz_src = torch.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## reproject
    x_src = xy_src[0].reshape([height, width]).cpu().detach().numpy()
    y_src = xy_src[1].reshape([height, width]).cpu().detach().numpy()
    sampled_depth_src = cv2.remap(np.squeeze(depth_src.cpu().detach().numpy()),
                                  x_src,
                                  y_src,
                                  interpolation=cv2.INTER_LINEAR)
    sampled_depth_src = torch.from_numpy(sampled_depth_src)

    # 3D space
    xyz_src = torch.matmul(torch.linalg.inv(intrinsics_src),
                           torch.vstack((xy_src, torch.ones_like(x_ref, device=torch.device(
                               'cuda')))) * sampled_depth_src.reshape([-1]).to(device='cuda'))
    # reference 3D space
    xyz_reprojected = torch.matmul(torch.matmul(extrinsics_ref, torch.linalg.inv(extrinsics_src)),
                                   torch.vstack((xyz_src.to(device='cuda'),
                                                 torch.ones_like(x_ref, device=torch.device('cuda')))))[:3]

    K_xyz_reprojected = torch.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]

    warped_ori_fea = F.grid_sample(feature_ori, xy_reprojected.view(1, height, width, 2), mode='bilinear',
                                   padding_mode='zeros', align_corners=True)
    warped_amt_fea = F.grid_sample(feature_amt, xy_reprojected.view(1, height, width, 2), mode='bilinear',
                                   padding_mode='zeros', align_corners=True)


    return warped_ori_fea, warped_amt_fea


def cross_entropy_loss(prob_volume, depth_gt, mask, depth_value, return_prob_map=False):
    # from AA
    mask_true = mask
    valid_pixel_num = torch.sum(mask_true, dim=[1,2]) + 1e-6

    shape = depth_gt.shape          # B,H,W

    depth_num = depth_value.shape[1]
    if len(depth_value.shape) < 3:
        depth_value_mat = depth_value.repeat(shape[1], shape[2], 1, 1).permute(2,3,0,1)     # B,N,H,W
    else:
        depth_value_mat = depth_value

    gt_index_image = torch.argmin(torch.abs(depth_value_mat-depth_gt.unsqueeze(1)), dim=1)

    gt_index_image = torch.mul(mask_true, gt_index_image.type(torch.float))
    gt_index_image = torch.round(gt_index_image).type(torch.long).unsqueeze(1) # B, 1, H, W

    # gt index map -> gt one hot volume (B x 1 x H x W )
    gt_index_volume = torch.zeros(shape[0], depth_num, shape[1], shape[2]).type(mask_true.type()).scatter_(1, gt_index_image, 1)

    # cross entropy image (B x D X H x W)
    cross_entropy_image = -torch.sum(gt_index_volume * torch.log(prob_volume + 1e-6), dim=1).squeeze(1) # B, 1, H, W

    # masked cross entropy loss
    masked_cross_entropy_image = torch.mul(mask_true, cross_entropy_image) # valid pixel
    masked_cross_entropy = torch.sum(masked_cross_entropy_image, dim=[1, 2])

    masked_cross_entropy = torch.mean(masked_cross_entropy / valid_pixel_num) # Origin use sum : aggregate with batch
    # winner-take-all depth map
    wta_index_map = torch.argmax(prob_volume, dim=1, keepdim=True).type(torch.long)
    wta_depth_map = torch.gather(depth_value_mat, 1, wta_index_map).squeeze(1)

    if return_prob_map:
        photometric_confidence = torch.max(prob_volume, dim=1)[0] # output shape dimension B * H * W
        return masked_cross_entropy, wta_depth_map, photometric_confidence
    return masked_cross_entropy


def cal_metrics(depth_pred, depth_gt, mask, depth_min, depth_max):
    depth_pred_norm = depth_pred * 128 / (depth_max - depth_min)[:, None, None]
    depth_gt_norm = depth_gt * 128 / (depth_max - depth_min)[:, None, None]

    abs_err = torch.abs(depth_pred_norm[mask] - depth_gt_norm[mask])
    epe = abs_err.mean()
    err1 = (abs_err <= 1).float().mean() * 100
    err3 = (abs_err <= 3).float().mean() * 100

    return epe  # err1, err3