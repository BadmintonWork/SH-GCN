import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import RECOGNIZERS, build_head
from .base import BaseRecognizer

@RECOGNIZERS.register_module()
class RecognizerGCN_CSCL(BaseRecognizer):
    """
    GCN-based recognizer for skeleton-based action recognition.
    
    【修改说明】：
    1. 集成了 CSCL。
    2. 维护一个 'class_centers' 内存库，用于存储每个类别的"标准特征"。
    3. 训练时计算两部分损失：骨骼分类(1.0) + 虚拟节点对比损失(lambda)。
    4. 骨骼特征用于分类，虚拟节点用于对比学习。
    """

    def __init__(self,
                 backbone,
                 cls_head,  # ✅ 改回 cls_head，与基础配置兼容
                 neck=None,
                 train_cfg=dict(),
                 test_cfg=dict(),
                 loss_weights=dict(skeleton=1.0, csc=0.1),
                 use_ball_fusion=True):
        
        # 正常传递 cls_head 给父类
        super().__init__(backbone, neck, cls_head=cls_head, train_cfg=train_cfg, test_cfg=test_cfg)
        
        # 支持特征提取模式
        self.feat_ext = test_cfg.get('feat_ext', False)

        # --- CSCL 核心组件 ---
        self.loss_weights = loss_weights
        self.lambda_csc = loss_weights.get('csc', 0.1)
        self.use_ball_fusion = use_ball_fusion

        # 获取类别数和特征维度（从 cls_head 获取）
        self.num_classes = cls_head['num_classes']
        self.feat_dim = cls_head['in_channels']
        
        # 定义类别中心库
        self.register_buffer("class_centers", torch.randn(self.num_classes, self.feat_dim))
        self.class_centers = F.normalize(self.class_centers, p=2, dim=1)

        # 超参数
        self.alpha = 0.9     # 动量系数
        self.T = 0.1         # 温度系数

    def forward_train(self, keypoint, label, ball_trajectory=None, **kwargs):
        """定义训练计算过程"""
        assert self.with_cls_head, "训练时必须配置 cls_head"
        
        losses = dict()

        if keypoint.dtype != torch.float:
            keypoint = keypoint.float()

        if not self.use_ball_fusion:
            ball_trajectory = None    
        
        if ball_trajectory is not None and ball_trajectory.dtype != torch.float:
            ball_trajectory = ball_trajectory.float()

        keypoint = keypoint[:, 0]
        gt_label = label.squeeze(-1)

        # 1. 提取特征
        features = self.extract_feat(keypoint, ball_trajectory)
        virtual_node_feat_raw = features['virtual_node_feature']  # [N, M, C, T]
        skeleton_feat = features['skeleton_feature']              # [N, M, C, T, V]

        # -----------------------------------------------------------
        # Part A: 虚拟节点对比学习损失（用于特征约束，不参与分类）
        # -----------------------------------------------------------
        # 虚拟节点特征聚合：[N, M, C, T] -> [N, C]
        vn_feat_pooled = virtual_node_feat_raw.mean(dim=3).mean(dim=1)
        
        # 计算对比损失并更新内存库
        loss_csc_val = self._compute_cscl_loss_and_update(vn_feat_pooled, gt_label)
        losses['loss_csc'] = self.lambda_csc * loss_csc_val

        # -----------------------------------------------------------
        # Part B: 骨骼点分类分支（权重 1.0）- 唯一的分类损失
        # -----------------------------------------------------------
        if self.with_neck:
            x_skeleton = self.neck(skeleton_feat)
        else:
            x_skeleton = skeleton_feat
        
        # 使用 cls_head 进行分类
        cls_score = self.cls_head(x_skeleton)
        loss_cls = self.cls_head.loss(cls_score, gt_label)
        
        # 添加分类损失（保持原有的命名方式）
        losses.update(loss_cls)
            
        return losses

    def _compute_cscl_loss_and_update(self, features, labels):
        """
        核心辅助函数：计算对比损失 + 动量更新类别中心
        features: [Batch, Dim]
        labels: [Batch]
        """
        # 1. 归一化 (Normalize) - 比较方向一致性
        features_norm = F.normalize(features, p=2, dim=1)
        centers_norm = F.normalize(self.class_centers, p=2, dim=1)

        # 2. 计算相似度 Logits
        logits = torch.matmul(features_norm, centers_norm.t())
        
        # 3. 温度缩放
        logits = logits / self.T 

        # 4. 计算 CrossEntropy Loss
        loss = F.cross_entropy(logits, labels)

        # 5. 动量更新（不计算梯度）
        if self.training:
            with torch.no_grad():
                unique_labels = labels.unique()
                for c in unique_labels:
                    mask = (labels == c)
                    if mask.sum() == 0: 
                        continue

                    f_k_mean = features[mask].mean(dim=0)
                    new_center = self.alpha * self.class_centers[c] + \
                                 (1 - self.alpha) * f_k_mean
                    
                    self.class_centers[c] = F.normalize(new_center, p=2, dim=0)
        
        return loss

    def forward_test(self, keypoint, ball_trajectory=None, **kwargs):
        """定义评估和测试过程 - 使用骨骼点分类"""
        assert self.with_cls_head or self.feat_ext, \
            "测试时必须配置 cls_head 或启用 feat_ext"
        
        bs, nc = keypoint.shape[:2]
        keypoint = keypoint.reshape((bs * nc, ) + keypoint.shape[2:])

        if not self.use_ball_fusion:
            ball_trajectory = None

        if ball_trajectory is not None:
            if ball_trajectory.dtype != torch.float:
                ball_trajectory = ball_trajectory.float()
            assert ball_trajectory.shape[0] == bs
            ball_trajectory = ball_trajectory.unsqueeze(1).repeat(1, nc, 1, 1)
            ball_trajectory = ball_trajectory.reshape(bs * nc, ball_trajectory.shape[2], ball_trajectory.shape[3])

        features = self.extract_feat(keypoint, ball_trajectory)
        skeleton_feat = features['skeleton_feature']

        if self.with_neck:
            x = self.neck(skeleton_feat)
        else:
            x = skeleton_feat
        
        # ========== 特征提取模式 ==========
        feat_ext = self.test_cfg.get('feat_ext', False)
        pool_opt = self.test_cfg.get('pool_opt', 'all')
        score_ext = self.test_cfg.get('score_ext', False)
        
        if feat_ext or score_ext:
            assert bs == 1
            assert isinstance(pool_opt, str)
            dim_idx = dict(n=0, m=1, t=3, v=4)

            if pool_opt == 'all':
                pool_opt = 'nmtv'
            if pool_opt != 'none':
                for digit in pool_opt:
                    assert digit in dim_idx

            if isinstance(x, tuple) or isinstance(x, list):
                x = torch.cat(x, dim=2)
            
            assert len(x.shape) == 5, 'The shape is N, M, C, T, V'
            
            if pool_opt != 'none':
                for d in pool_opt:
                    x = x.mean(dim_idx[d], keepdim=True)

            if score_ext:
                w = self.cls_head.fc_cls.weight
                b = self.cls_head.fc_cls.bias
                x = torch.einsum('nmctv,oc->nmotv', x, w)
                if b is not None:
                    x = x + b[..., None, None]
                x = x[None]
            
            return x.data.cpu().numpy().astype(np.float16)
        # ========== 特征提取模式结束 ==========
        
        # 使用 cls_head 进行分类
        cls_score = self.cls_head(x)
        cls_score = cls_score.reshape(bs, nc, cls_score.shape[-1])
        
        # 默认 average_clips 配置
        if 'average_clips' not in self.test_cfg:
            self.test_cfg['average_clips'] = 'prob'
        
        final_score = self.average_clip(cls_score)
        
        # 支持多 clip 输出
        if isinstance(final_score, tuple) or isinstance(final_score, list):
            final_score = [x.data.cpu().numpy() for x in final_score]
            return [[x[i] for x in final_score] for i in range(bs)]
        
        return final_score.data.cpu().numpy()

    def forward(self, keypoint, label=None, return_loss=True, ball_trajectory=None, **kwargs):
        if return_loss:
            if label is None:
                raise ValueError('Label should not be None.')
            return self.forward_train(keypoint, label, ball_trajectory, **kwargs)

        return self.forward_test(keypoint, ball_trajectory, **kwargs)

    def extract_feat(self, keypoint, ball_trajectory=None):
        return self.backbone(keypoint, ball_trajectory)