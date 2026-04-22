import numpy as np
from ..builder import PIPELINES

@PIPELINES.register_module()
class RandomTemporalShift:
    """
    时序平移数据增强模块
    --------------------
    用于在时间轴上随机整体平移骨架序列和球轨迹，
    模拟击球帧标注误差或时序抖动，增强模型的时序鲁棒性。

    Args:
        shift_prob (float): 执行平移的概率（0~1），默认 0.3。
        max_shift_ratio (float): 最大平移比例（相对序列长度），默认 0.3。
        pad_mode (str): 补帧模式，'edge' 表示复制边界帧，'zero' 表示补 0。
    """

    def __init__(self, shift_prob=0.3, max_shift_ratio=0.3, pad_mode='edge'):
        assert 0 <= shift_prob <= 1
        assert 0 <= max_shift_ratio <= 1
        assert pad_mode in ['edge', 'zero']
        self.shift_prob = shift_prob
        self.max_shift_ratio = max_shift_ratio
        self.pad_mode = pad_mode

    def __call__(self, results):
        """执行时序平移。"""
        if np.random.rand() > self.shift_prob:
            return results  # 不做增广

        # --- 获取数据 ---
        keypoint = results.get('keypoint', None)
        ball = results.get('ball_trajectory', None)

        if keypoint is None or keypoint.size == 0:
            return results  # 安全退出
        
        # 确定时间维度索引位置
        if keypoint.ndim == 4:  # (M, T, V, C)
            time_axis = 1
            T = keypoint.shape[1]
        elif keypoint.ndim == 3:  # (T, V, C)
            time_axis = 0
            T = keypoint.shape[0]
        else:
            return results  # 不支持的维度
            
        if T <= 1:
            return results  # 单帧不需要平移

        # 计算平移量
        max_shift = max(1, int(T * self.max_shift_ratio))
        shift = np.random.randint(-max_shift, max_shift + 1)
        if shift == 0:
            return results  # 无需平移
            
        # --- 执行关键点平移 ---
        keypoint_shifted = self._shift_array(keypoint, shift, time_axis)
        results['keypoint'] = keypoint_shifted
        
        # --- 球轨迹同步平移 ---
        if ball is not None and len(ball) > 0:
            ball_time_axis = 0  # 球轨迹通常是(T, C)格式
            ball_shifted = self._shift_array(ball, shift, ball_time_axis)
            results['ball_trajectory'] = ball_shifted

        # --- 标注信息同步调整 ---
        if 'contact_local' in results:
            results['contact_local'] = np.clip(results['contact_local'] + shift, 0, T - 1)
        if 'contact_global' in results:
            results['contact_global'] = np.clip(results['contact_global'] + shift, 0, T - 1)

        return results
        
    def _shift_array(self, arr, shift, time_axis):
        """安全地执行数组时间维度平移"""
        # 获取时间维度长度
        time_len = arr.shape[time_axis]
        
        # 创建结果数组
        result = np.zeros_like(arr)
        
        # 处理不同的平移方向
        if shift > 0:  # 右移(延后)
            src_indices = np.arange(0, time_len - shift)
            dst_indices = np.arange(shift, time_len)
            
            # 执行数据复制 - 主体部分
            if time_axis == 0:
                result[dst_indices] = arr[src_indices]
                # 填充前面部分
                if self.pad_mode == 'edge':
                    for i in range(shift):
                        result[i] = arr[0]  # 直接复制第一帧
            elif time_axis == 1:
                result[:, dst_indices] = arr[:, src_indices]
                # 填充前面部分
                if self.pad_mode == 'edge':
                    for i in range(shift):
                        result[:, i] = arr[:, 0]  # 直接复制第一帧
        else:  # 左移(提前)
            shift = abs(shift)
            src_indices = np.arange(shift, time_len)
            dst_indices = np.arange(0, time_len - shift)
            
            # 执行数据复制 - 主体部分
            if time_axis == 0:
                result[dst_indices] = arr[src_indices]
                # 填充后面部分
                if self.pad_mode == 'edge':
                    for i in range(time_len - shift, time_len):
                        result[i] = arr[-1]  # 直接复制最后一帧
            elif time_axis == 1:
                result[:, dst_indices] = arr[:, src_indices]
                # 填充后面部分
                if self.pad_mode == 'edge':
                    for i in range(time_len - shift, time_len):
                        result[:, i] = arr[:, -1]  # 直接复制最后一帧
        
        return result

    def __repr__(self):
        return (f'{self.__class__.__name__}(shift_prob={self.shift_prob}, '
                f'max_shift_ratio={self.max_shift_ratio}, pad_mode={self.pad_mode})')