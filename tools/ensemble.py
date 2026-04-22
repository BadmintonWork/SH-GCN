from mmcv import load, dump
from pyskl.smp import *
import datetime

joint_path = 'work_dirs/ctrgcn/badminton_shuttle_strat/b_100/result.pkl'
bone_path = 'work_dirs/ctrgcn/badminton_shuttle_strat/bm_100/result.pkl'

joint_motion_path = 'work_dirs/ctrgcn/badminton_shuttle_strat/j_100/result.pkl'
bone_motion_path = 'work_dirs/ctrgcn/badminton_shuttle_strat/jm_100/result.pkl'

joint = load(joint_path)
bone = load(bone_path)

joint_motion = load(joint_motion_path)
bone_motion = load(bone_motion_path)


# === 修改开始：自定义标签加载 ===
# 1. 加载原始数据文件
data_file = 'data/badminton_shuttle_true_73_track/test_strat_100.pkl'
print(f"Loading data from: {data_file}")
data_list = load(data_file)

# 2. 检查数据类型并提取标签
if isinstance(data_list, dict):
    if 'annotations' in data_list:
        print("Data is a dict with 'annotations'.")
        label = [x['label'] for x in data_list['annotations']]
    else:
        # 极少情况，看是否有其他键
        print(f"Warning: Unexpected dict keys: {data_list.keys()}")
        label = []
elif isinstance(data_list, list):
    print(f"Data is a raw list with {len(data_list)} samples.")
    # 你的 custom_2d_skeleton_track.py 生成的是这种格式
    label = [x['label'] for x in data_list]
else:
    raise TypeError(f"Unknown data type: {type(data_list)}")

# 3. 强力检查：确保标签数量和预测结果数量严格一致
print(f"Num Labels: {len(label)}")
print(f"Num Predictions: {len(joint)}") # 假设 joint 是其中一个结果

if len(label) != len(joint):
    print("!!!! 严重警告: 标签数量与预测结果数量不匹配 !!!!")
    print("可能的原因: 测试时使用了不同的 split，或者某些样本在 dataloader 中被过滤了。")
    # 如果只差几个，可能是 drop_last 或分布式对其问题，但此处必须解决
    # 尝试截断较长的一方（仅用于调试，不建议用于生产）
    min_len = min(len(label), len(joint))
    label = label[:min_len]
    joint = joint[:min_len]
    bone = bone[:min_len]
    joint_motion = joint_motion[:min_len]
    bone_motion = bone_motion[:min_len]
# === 修改结束 ===

# label = load_label('data/badminton_shuttle_true_73_track/test_100.pkl')
# # label = load_label('/data/nturgbd/ntu60_3danno.pkl', 'xview_val')
# # label = load_label('/data/nturgbd/ntu120_3danno.pkl', 'xsub_val')
# # label = load_label('/data/nturgbd/ntu120_3danno.pkl', 'xset_val')
# # label = load_label('/data/k400/k400_hrnet.pkl', 'val')
# # label = load_label('/data/finegym/gym_hrnet.pkl', 'val')

print('J+B')
fused = comb([joint, bone], [1, 1])
print('Top-1', top1(fused, label))

print('4M')
fused = comb([joint, bone, joint_motion, bone_motion], [2, 2, 1, 1])
print('Top-1', top1(fused, label))

# print('6M')
# fused = comb([joint, bone, kbone, joint_motion, bone_motion, kbone_motion], [2, 2, 2, 1, 1, 1])
# print('Top-1', top1(fused, label))