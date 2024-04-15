import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *
from .amt import AdapMatchawareTransformer
from .dfga import DFGANet
Align_Corners_Range = False

class AggWeightNetVolume(nn.Module):
    def __init__(self, in_channels=32):
        super(AggWeightNetVolume, self).__init__()
        self.w_net = nn.Sequential(
            Conv3d(in_channels, 1, kernel_size=1, stride=1, padding=0),
            Conv3d(1, 1, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        """
        :param x: (b, c, d, h, w)
        :return: (b, 1, d, h, w)
        """
        w = self.w_net(x)
        return w

class DepthNet(nn.Module):
    def __init__(self, in_channels=None):
        super(DepthNet, self).__init__()
        self.weight_net = nn.ModuleList([AggWeightNetVolume(in_channels[i]) for i in range(len(in_channels))])
        self.dfga = DFGANet()

    def forward(self, stage_idx, features, proj_matrices, depth_values, num_depth, cost_regularization, prob_volume_init=None, coarse_cost=None):
        """forward.

        :param stage_idx: int, index of stage, [1, 2, 3], stage_1 corresponds to lowest image resolution
        :param features: torch.Tensor, TODO: [B, C, H, W]
        :param proj_matrices: torch.Tensor,
        :param depth_values: torch.Tensor, TODO: [B, D, H, W]
        :param num_depth: int, Ndepth
        :param cost_regularization: nn.Module, regularization network
        :param prob_volume_init:
        :param view_weights: pixel wise view weights for src views
        """
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(features) == len(proj_matrices), "Different number of images and projection matrices"
        assert depth_values.shape[1] == num_depth, "depth_values.shape[1]:{}  num_depth:{}".format(depth_values.shapep[1], num_depth)

        # step 1. feature extraction
        num_views = len(features)
        ref_feature, src_features = features[0], features[1:] # [B, C, H, W]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:] # [B, 2, 4, 4]
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)

        # step 2. differentiable homograph, build cost volume
        volume_adapt = None
        for i, (src_fea, src_proj) in enumerate(zip(src_features, src_projs)): # src_fea: [B, C, H, W]
            src_proj_new = src_proj[:, 0].clone() # [B, 4, 4]
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone() # [B, 4, 4]
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            warped_volume = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_values)

            warped_volume = (ref_volume - warped_volume).pow_(2)
            weight = self.weight_net[stage_idx](warped_volume)
            if volume_adapt is None:
                volume_adapt = (weight + 1) * warped_volume
            else:
                volume_adapt = volume_adapt + (weight + 1) * warped_volume

            del warped_volume

        # aggregate multiple similarity across all the source views
        cost_agg = volume_adapt / (num_views - 1)

        # dual-feature guided aggregation & cost volume regularization
        if stage_idx == 0:
            coarse_cost = cost_agg
            cost_reg = cost_regularization(cost_agg, stage_idx)
        elif stage_idx == 1:
            cost_dfga = self.dfga(coarse_cost, cost_agg, stage_idx)
            coarse_cost = cost_agg
            del cost_agg
            cost_reg = cost_regularization(cost_dfga, stage_idx)
        else:
            cost_dfga = self.dfga(coarse_cost, cost_agg, stage_idx)
            del cost_agg
            cost_reg = cost_regularization(cost_dfga, stage_idx)

        # step 3. probability map
        prob_volume_pre = cost_reg.squeeze(1)

        if prob_volume_init is not None:
            prob_volume_pre += prob_volume_init

        prob_volume = torch.exp(F.log_softmax(prob_volume_pre, dim=1))
        depth = depth_wta(prob_volume, depth_values=depth_values)

        with torch.no_grad():
            photometric_confidence = torch.max(prob_volume, dim=1)[0]
        if stage_idx < 2:
            return {"depth": depth,  "photometric_confidence": photometric_confidence, "prob_volume": prob_volume, "depth_values": depth_values}, coarse_cost.detach()
        else:
            return {"depth": depth,  "photometric_confidence": photometric_confidence, "prob_volume": prob_volume, "depth_values": depth_values}


class CTMVSNet(nn.Module):
    def __init__(self, refine=False, ndepths=[48, 32, 8], depth_interals_ratio=[4, 2, 1], share_cr=False,
            grad_method="detach", arch_mode="fpn", cr_base_chs=[8, 8, 8]):
        super(CTMVSNet, self).__init__()
        self.refine = refine
        self.share_cr = share_cr
        self.ndepths = ndepths
        self.depth_interals_ratio = depth_interals_ratio
        self.grad_method = grad_method
        self.arch_mode = arch_mode
        self.cr_base_chs = cr_base_chs
        self.num_stage = len(ndepths)
        print("**********netphs:{}, depth_intervals_ratio:{},  grad:{}, chs:{}************".format(ndepths,
            depth_interals_ratio, self.grad_method, self.cr_base_chs))

        assert len(ndepths) == len(depth_interals_ratio)

        self.stage_infos = {
                "stage1":{
                    "scale": 4.0,
                    },
                "stage2": {
                    "scale": 2.0,
                    },
                "stage3": {
                    "scale": 1.0,
                    }
                }

        self.feature = FeatureNet(base_channels=8)

        self.AMT = AdapMatchawareTransformer()

        if self.share_cr:
            self.cost_regularization = CostRegNet(in_channels=1, base_channels=8)
        else:
            self.cost_regularization = nn.ModuleList([CostRegNet(in_channels=self.feature.out_channels[i],
                                                                 base_channels=self.cr_base_chs[i]) for i in range(self.num_stage)])
        if self.refine:
            self.refine_network = RefineNet()

        self.DepthNet = DepthNet(self.feature.out_channels)

    def forward(self, imgs, proj_matrices, depth_values, test_tnt=False):
        depth_min = float(depth_values[0, 0].cpu().numpy())
        depth_max = float(depth_values[0, -1].cpu().numpy())
        depth_interval = (depth_max - depth_min) / depth_values.size(1)

        import copy
        # step 1. feature extraction
        features = []
        features_origin = []
        for nview_idx in range(imgs.size(1)):
            img = imgs[:, nview_idx]
            features.append(self.feature(img))
            # save original features
            if nview_idx == 0:
                features_dict = {}
                for i in range(self.num_stage):
                    features_dict["stage{}".format(i+1)] = torch.clone(features[nview_idx]["stage{}".format(i+1)])
                features_origin.append(features_dict)
                del features_dict

        # amt_ed features
        features_amt0 = self.AMT(features)

        del features

        outputs = {}
        depth, cur_depth = None, None
        coarse_cost = None
        for stage_idx in range(self.num_stage):
            features_ori = [feat["stage{}".format(stage_idx + 1)] for feat in features_origin]
            features_amt = [feat["stage{}".format(stage_idx + 1)] for feat in features_amt0]
            proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
            stage_scale = self.stage_infos["stage{}".format(stage_idx + 1)]["scale"]

            Using_inverse_d = False

            if depth is not None:
                if self.grad_method == "detach":
                    cur_depth = depth.detach()
                else:
                    cur_depth = depth
                cur_depth = F.interpolate(cur_depth.unsqueeze(1), [img.shape[2], img.shape[3]],
                                          mode='bilinear',
                                          align_corners=Align_Corners_Range).squeeze(1)
            else:
                cur_depth = depth_values

            # [B, D, H, W]
            depth_range_samples, stage_interval = get_depth_range_samples(cur_depth=cur_depth,
                                                                          ndepth=self.ndepths[stage_idx],
                                                                          depth_inteval_pixel=self.depth_interals_ratio[stage_idx] * depth_interval,
                                                                          dtype=img[0].dtype,
                                                                          device=img[0].device,
                                                                          shape=[img.shape[0], img.shape[2], img.shape[3]],
                                                                          max_depth=depth_max,
                                                                          min_depth=depth_min,
                                                                          use_inverse_depth=Using_inverse_d)

            if stage_idx < 2:  # stage 1,2 (save coarse cost volume)
                outputs_stage, coarse_cost = self.DepthNet(stage_idx, features_amt, proj_matrices_stage,
                                                           depth_values=F.interpolate(depth_range_samples.unsqueeze(1),[self.ndepths[stage_idx], img.shape[2] // int(stage_scale), img.shape[3] // int(stage_scale)],
                                                                                      mode='trilinear',
                                                                                      align_corners=Align_Corners_Range).squeeze(1),
                                                           num_depth=self.ndepths[stage_idx],
                                                           cost_regularization=self.cost_regularization if self.share_cr else self.cost_regularization[stage_idx],
                                                           coarse_cost=coarse_cost)
            else: # stage 3
                outputs_stage = self.DepthNet(stage_idx, features_amt, proj_matrices_stage,
                                              depth_values=F.interpolate(depth_range_samples.unsqueeze(1), [self.ndepths[stage_idx], img.shape[2] // int(stage_scale),img.shape[3] // int(stage_scale)],
                                                                         mode='trilinear',
                                                                         align_corners=Align_Corners_Range).squeeze(1),
                                              num_depth=self.ndepths[stage_idx],
                                              cost_regularization=self.cost_regularization if self.share_cr else self.cost_regularization[stage_idx],
                                              coarse_cost=coarse_cost)

            depth = outputs_stage['depth']
            # outputs_stage["feature_ori"] = features_ori[0]
            # outputs_stage["feature_amt"] = features_amt[0]

            outputs["stage{}".format(stage_idx + 1)] = outputs_stage
            outputs["stage{}".format(stage_idx + 1)]["feature_ori"] = features_ori[0]
            outputs["stage{}".format(stage_idx + 1)]["feature_amt"] = features_amt[0]
            outputs.update(outputs_stage)

        if self.refine:
            refined_depth = self.refine_network(torch.cat((imgs[:, 0], depth), 1))
            outputs["refined_depth"] = refined_depth

        return outputs
