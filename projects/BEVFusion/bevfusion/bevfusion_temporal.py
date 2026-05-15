from .bevfusion import BEVFusion
from mmdet3d.registry import MODELS
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import DeformConv2dPack


class TemporalAlign(nn.Module):
    """
    可变形时序对齐
    DCN + 粗对齐残差
    """
    def __init__(self, in_channels=256, kernel_size=3):
        super().__init__()
        self.offset_conv = nn.Conv2d(in_channels, 18, kernel_size, padding=1)
        self.dcn = DeformConv2dPack(in_channels, in_channels, kernel_size, padding=1, deform_groups=4)
        self.norm = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU(inplace=True)
        self.offset_scale = nn.Parameter(torch.tensor(0.1), requires_grad=True)

    def forward(self, hist_feat, T_rel, warp_func):
        hist_feat_warped = warp_func(hist_feat, T_rel)
        offset = self.offset_conv(hist_feat_warped) * self.offset_scale
        aligned_feat = self.dcn(hist_feat_warped, offset)
        aligned_feat = aligned_feat + hist_feat_warped
        return self.act(self.norm(aligned_feat))


@MODELS.register_module()
class BEVFusionTemporal(BEVFusion):
    
    def __init__(self, temporal_num_frames=2, use_align="dcn", **kwargs):
        super().__init__(**kwargs)
        self.temporal_num_frames = temporal_num_frames
        self.bev_cache = {}
        self.token_queue = deque(maxlen=temporal_num_frames)
        self.use_align = use_align
        self.pose_cache = {}
        if self.use_align == "dcn":
            self.temporal_align = TemporalAlign(in_channels=256)

    def _get_ego_pose(self, batch_input_metas):
        if batch_input_metas is None or len(batch_input_metas) == 0:
            return None
        meta = batch_input_metas[0]
        pose = None
        for key in ['ego2global', 'lidar2global', 'global_from_ego']:
            if key in meta:
                candidate = meta[key]
                if isinstance(candidate, torch.Tensor):
                    pose = candidate
                    break
                elif isinstance(candidate, list):
                    pose = torch.tensor(candidate)
                    break
        if pose is not None and pose.dim() == 2:
            if pose.shape[0] == 3 and pose.shape[1] == 3:
                T = torch.eye(4, device=pose.device, dtype=pose.dtype)
                T[:3, :3] = pose
                pose = T
            elif pose.shape[0] == 3 and pose.shape[1] == 4:
                T = torch.eye(4, device=pose.device, dtype=pose.dtype)
                T[:3, :4] = pose
                pose = T
        return pose

    def _compute_relative_transform(self, pose_cur, pose_hist):
        return torch.inverse(pose_cur) @ pose_hist

    def _build_affine_matrix(self, T_rel, bev_range, bev_size, device, dtype):
        tx = T_rel[0, 3]
        ty = T_rel[1, 3]
        cos_theta = T_rel[0, 0]
        sin_theta = T_rel[0, 1]
        x_min, x_max = bev_range[0], bev_range[1]
        y_min, y_max = bev_range[2], bev_range[3]
        W, H = bev_size
        scale_x = W / (x_max - x_min)
        scale_y = H / (y_max - y_min)
        delta_u = tx * scale_x
        delta_v = ty * scale_y
        affine = torch.tensor([
            [cos_theta, -sin_theta, -delta_u],
            [sin_theta,  cos_theta, -delta_v]
        ], device=device, dtype=dtype)
        return affine

    def _warp_bev_feature(self, bev_feat, T_rel, bev_range, bev_size):
        '''仿射变换平移对齐'''
        B, C, H, W = bev_feat.shape
        device = bev_feat.device
        dtype = bev_feat.dtype
        affine = self._build_affine_matrix(T_rel, bev_range, (W, H), device, dtype)
        affine = affine.unsqueeze(0)
        grid = F.affine_grid(affine, size=(1, C, H, W), align_corners=False)
        warped = F.grid_sample(bev_feat, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        return warped

    def extract_feat(self, batch_inputs_dict, batch_input_metas, **kwargs):
        x = super().extract_feat(batch_inputs_dict, batch_input_metas, **kwargs)
        if not isinstance(x, torch.Tensor):
            return x

        cur_token = batch_input_metas[0].get('token', None) if batch_input_metas else None
        cur_pose = self._get_ego_pose(batch_input_metas)
        bev_range = [-54.0, 54.0, -54.0, 54.0]
        bev_size = (x.shape[3], x.shape[2]) if isinstance(x, torch.Tensor) else (180, 180)

        if len(self.token_queue) > 0 and cur_token is not None and cur_pose is not None:
            history_bevs = []
            for hist_token in list(self.token_queue):
                if hist_token in self.bev_cache and hist_token in self.pose_cache:
                    hist_bev = self.bev_cache[hist_token]
                    hist_pose = self.pose_cache[hist_token]
                    T_rel = self._compute_relative_transform(cur_pose, hist_pose)

                    if self.use_align == "wrap":
                        hist_bev_aligned = self._warp_bev_feature(hist_bev, T_rel, bev_range, bev_size)
                        history_bevs.append(hist_bev_aligned)

                    elif self.use_align == "dcn":
                        def warp_func(feat, T):
                            return self._warp_bev_feature(feat, T, bev_range, bev_size)
                        hist_bev_aligned = self.temporal_align(hist_bev, T_rel, warp_func)
                        history_bevs.append(hist_bev_aligned)

            if history_bevs:
                all_bevs = [x] + history_bevs
                fused = torch.cat(all_bevs, dim=1)
                if not hasattr(self, '_temporal_conv'):
                    self._temporal_conv = nn.Conv2d(fused.shape[1], x.shape[1], 1).to(x.device)
                x = self._temporal_conv(fused)

        if cur_token is not None:
            self.bev_cache[cur_token] = x.detach().clone()
            self.token_queue.append(cur_token)
            if cur_pose is not None:
                self.pose_cache[cur_token] = cur_pose.detach().clone() if isinstance(cur_pose, torch.Tensor) else cur_pose
            valid_tokens = set(self.token_queue)
            for t in list(self.bev_cache.keys()):
                if t not in valid_tokens:
                    del self.bev_cache[t]
                    if t in self.pose_cache:
                        del self.pose_cache[t]
        return x