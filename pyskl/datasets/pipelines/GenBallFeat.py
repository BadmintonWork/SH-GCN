import numpy as np
from ..builder import PIPELINES

@PIPELINES.register_module()
class GenBallFeat:
    """
    生成球的时序特征：
    - 仅对 (x, y) 计算速度/加速度/角度
    - conf（置信度）可选是否保留为独立 1 通道
    """
    def __init__(self, use_vel=True, use_acc=True, use_angle=True, keep_conf=True, eps=1e-6, normalize=False):
        self.use_vel = use_vel
        self.use_acc = use_acc
        self.use_angle = use_angle
        self.keep_conf = keep_conf
        self.eps = eps
        self.normalize = normalize  # 可选：把xy归一化到[0,1]（需要能拿到W/H）

    def __call__(self, results):
        ball = np.asarray(results['ball_trajectory'], dtype=np.float32)  # (T, C)
        assert ball.ndim == 2 and ball.shape[0] > 0, 'ball_trajectory 应为 (T, C)'

        T = ball.shape[0]
        has_conf = ball.shape[1] >= 3
        xy = ball[:, :2]                         # 只取 (x, y)
        conf = ball[:, 2:3] if has_conf else None

        # 可选：归一化坐标到 [0, 1]，更方便跨视频泛化
        if self.normalize:
            # 尝试从 results 里拿图像宽高
            W = H = None
            if 'img_shape' in results and results['img_shape'] is not None:
                H, W = int(results['img_shape'][0]), int(results['img_shape'][1])
            elif 'meta' in results and isinstance(results['meta'], dict):
                W = results['meta'].get('W', None)
                H = results['meta'].get('H', None)
            if W and H:
                xy = xy.copy()
                xy[:, 0] /= float(W)
                xy[:, 1] /= float(H)

        feats = [xy]                              # 位置：2 通道

        # 速度（对齐到长度 T，最后一帧补0）
        vel = None
        if self.use_vel or self.use_acc or self.use_angle:
            vel = np.zeros_like(xy, dtype=np.float32)
            vel[:-1] = xy[1:] - xy[:-1]

        if self.use_vel:
            feats.append(vel)                     # +2 通道

        # 加速度
        if self.use_acc:
            acc = np.zeros_like(xy, dtype=np.float32)
            if T >= 3:
                acc[:-2] = xy[2:] - 2 * xy[1:-1] + xy[:-2]
            feats.append(acc)                    # +2 通道

        # 角度（速度方向），单通道
        if self.use_angle:
            # 防止 0 速度导致 NaN，用 atan2 直接算角度最稳
            ang = np.arctan2(vel[:, 1], vel[:, 0]) if vel is not None else np.zeros((T,), np.float32)
            feats.append(ang[:, None])            # +1 通道

        # 是否保留 conf（不参与差分）
        if self.keep_conf and conf is not None:
            feats.append(conf)                    # +1 通道

        results['ball_trajectory'] = np.concatenate(feats, axis=-1).astype(np.float32)
        return results