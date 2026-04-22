import copy as cp
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import load_checkpoint

from ...utils import Graph_aug as Graph
from ...utils import cache_checkpoint
from ..builder import BACKBONES

from .utils import dggcn, dgmstcn, unit_tcn, mstcn, dghgcn, dgphgcn, dgphgcn1, dgmsmlp

EPS = 1e-4


# --- 交叉注意力融合模块 ---
class CrossAttentionFusion(nn.Module):

    def __init__(self, d_model_skel, d_model_ball, nhead=4, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        # PyTorch内置的多头注意力模块，能够处理Q, K, V维度不同的情况
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model_skel,  # Query的维度 (骨骼/虚拟节点)
            kdim=d_model_ball,       # Key的维度 (羽毛球)
            vdim=d_model_ball,       # Value的维度 (羽毛球)
            num_heads=nhead,
            dropout=dropout,
            batch_first=False  # 注意力模块期望 (SeqLen, Batch, Dim)
        )
        self.dropout = nn.Dropout(dropout)
        # LayerNorm用于稳定训练
        self.norm = nn.LayerNorm(d_model_skel)

    def forward(self, x, ball_feature, virtual_node_idx):
        """
        执行融合操作。
        """
        NM, C_skel, T, V_plus_1 = x.shape
        N, C_ball, _ = ball_feature.shape
        M = NM // N

        # 1. 分离批次N和人数M
        x_reshaped = x.view(N, M, C_skel, T, V_plus_1)
        
        # 获取虚拟节点特征 (使用 clone 防止引用同一块内存)
        virtual_node_feat = x_reshaped[:, :, :, :, virtual_node_idx].clone()

        # 2. 准备Query, Key, Value
        # Query: [T, N*M, C_skel]
        query = virtual_node_feat.view(NM, C_skel, T).permute(2, 0, 1)

        # Key & Value: [T, N*M, C_ball]
        ball_feat_expanded = ball_feature.unsqueeze(1).expand(-1, M, -1, -1)
        key = ball_feat_expanded.reshape(NM, C_ball, T).permute(2, 0, 1)
        value = key

        # 3. 执行多头交叉注意力
        attn_output, attn_weights = self.mha(query, key, value)
        if attn_weights.dim() == 4:
            attn_weights = attn_weights.mean(dim=1)
        
        # 4. 残差连接与归一化
        attn_output_reshaped = attn_output.permute(1, 2, 0).view(N, M, C_skel, T)
        fused_feat = virtual_node_feat + self.dropout(attn_output_reshaped)
        fused_feat = self.norm(fused_feat.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)

        # ==========================================
        # 5. 使用 torch.cat 拼接，避免 In-place 报错
        # ==========================================
        
        # 我们需要把原来的特征拆开，去掉旧的虚拟节点，拼上新的融合后的节点
        
        # 取出除虚拟节点以外的所有骨骼特征
        # shape: [N, M, C_skel, T, V]
        skeleton_parts = x_reshaped[:, :, :, :, :virtual_node_idx] 
        
        # 准备拼接的新虚拟节点特征
        # fused_feat shape: [N, M, C_skel, T] -> 增加一个维度变成 [N, M, C_skel, T, 1]
        fused_feat_expanded = fused_feat.unsqueeze(-1)
        
        # 拼接： [骨骼特征, 新虚拟节点特征]

        out = torch.cat([skeleton_parts, fused_feat_expanded], dim=4)

        # 6. 恢复原始形状 [NM, C, T, V+1]
        return out.view(NM, C_skel, T, V_plus_1), attn_weights


class BallTrajectoryTCN(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, kernel_size=5):
        super().__init__()
        padding = (kernel_size - 1) // 2
        
        self.tcn1 = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=kernel_size, padding=padding, stride=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=False) 
        )
        self.tcn2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=kernel_size, padding=padding, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=False) 
        )
        self.tcn3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=kernel_size, padding=padding, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=False) 
        )
        self.tcn4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=kernel_size, padding=padding, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=False) 
        )
        self.tcn5 = nn.Sequential(
            nn.Conv1d(128, mid_channels, kernel_size=kernel_size, padding=padding, stride=2),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=False) 
        )

    def forward(self, ball_trajectory):
        if ball_trajectory.dtype != torch.float:
            ball_trajectory = ball_trajectory.float()
        
        traj_data = ball_trajectory.permute(0, 2, 1)
        
        x1 = self.tcn1(traj_data)
        x2 = self.tcn2(x1)
        x3 = self.tcn3(x2)
        x4 = self.tcn4(x3)
        x5 = self.tcn5(x4)
        
        return {
            'stage3': x2,
            'stage5': x3,
            'stage8': x5
        }

class DGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, edge_type, node_type,stride=1, residual=True, **kwargs):
        super().__init__()
        common_args = ['act', 'norm', 'g1x1']
        for arg in common_args:
            if arg in kwargs:
                value = kwargs.pop(arg)
                kwargs['tcn_' + arg] = value
                kwargs['gcn_' + arg] = value

        gcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'gcn_'}
        tcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'tcn_'}
        kwargs = {k: v for k, v in kwargs.items() if k[1:4] != 'cn_'}
        assert len(kwargs) == 0

        tcn_type = tcn_kwargs.pop('type', 'unit_tcn')
        assert tcn_type in ['unit_tcn', 'mstcn', 'dgmstcn','dgmsmlp']
        if tcn_type == 'unit_tcn':
            self.tcn = unit_tcn(out_channels, out_channels, 9, stride=stride, **tcn_kwargs)
        elif tcn_type == 'mstcn':
            self.tcn = mstcn(out_channels, out_channels, stride=stride, **tcn_kwargs)
        elif tcn_type =='dgmstcn':
            self.tcn = dgmstcn(out_channels, out_channels, stride=stride, **tcn_kwargs)
        elif tcn_type =='dgmsmlp':
            self.tcn = dgmsmlp(out_channels, out_channels, stride=stride, **tcn_kwargs)
        
        gcn_type = gcn_kwargs.pop('type', 'dghgcn')
        assert gcn_type in ['dghgcn', 'dgphgcn', 'dgphgcn1','dggcn']
        if gcn_type == 'dggcn':
            self.gcn = dggcn(in_channels, out_channels, A, **gcn_kwargs)
        if gcn_type =='dghgcn':
            self.gcn = dghgcn(in_channels, out_channels, A, edge_type, node_type, **gcn_kwargs)
        if gcn_type =='dgphgcn':
            self.gcn = dgphgcn(in_channels, out_channels, A, edge_type, node_type, **gcn_kwargs)
        if gcn_type =='dgphgcn1':
            self.gcn = dgphgcn1(in_channels, out_channels, A, edge_type, node_type, **gcn_kwargs)

        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, A=None):
        res = self.residual(x)
        x = self.tcn(self.gcn(x, A)) + res
        return self.relu(x)
        
    def init_weights(self):
        pass


@BACKBONES.register_module()
class SHGCN(nn.Module):
    def __init__(self,
                 graph_cfg,
                 in_channels=3,
                 base_channels=64,
                 ch_ratio=2,
                 num_stages=10,
                 inflate_stages=[5, 8],
                 down_stages=[5, 8],
                 data_bn_type='VC',
                 num_person=2,
                 pretrained=None,
                 fusion_stages=[5, 8], 
                 **kwargs):
        super().__init__()

        self.fusion_stages = fusion_stages


        # 1. 创建处理羽毛球轨迹的TCN (保留)
        self.ball_tcn = BallTrajectoryTCN(
            in_channels=8,
            mid_channels=128
        )
        
        # 2. 创建可学习的虚拟节点基础特征 (保留)
        self.virtual_node_feature = nn.Parameter(
            torch.randn(1, in_channels, 1, 1) * 0.01
        )
        
        # 3. 动态创建交叉注意力融合模块        
        self.fusion_modules = nn.ModuleDict()

        # 定义各阶段的通道数配置
        # Stage 3: GCN=64 (base), Ball=32 (tcn2)
        # Stage 5: GCN=128 (base*2), Ball=64 (tcn3)
        # Stage 8: GCN=256 (base*4), Ball=128 (tcn5)
        # 如果你不改 base_channels(64) 和 ch_ratio(2)，这些值是固定的

        stage_configs = {
            3: {'skel': base_channels, 'ball': 32, 'heads': 2},
            5: {'skel': int(base_channels * ch_ratio ** 1), 'ball': 64, 'heads': 4},
            8: {'skel': int(base_channels * ch_ratio ** 2), 'ball': 128, 'heads': 8}
        }

        print(f"Initializing Fusion Stages: {self.fusion_stages}")
        for stage_idx in self.fusion_stages:
            if stage_idx not in stage_configs:
                raise ValueError(f"暂不支持在 Stage {stage_idx} 进行融合，只支持 [3, 5, 8]")
            
            cfg = stage_configs[stage_idx]
            self.fusion_modules[str(stage_idx)] = CrossAttentionFusion(
                d_model_skel=cfg['skel'],
                d_model_ball=cfg['ball'],
                nhead=cfg['heads']
            )

        # # 3. 确定第5层和第8层GCN的输出通道数
        # stage5_channels = int(base_channels * ch_ratio ** 1) # 128
        # stage8_channels = int(base_channels * ch_ratio ** 2) # 256
        
        # # 4. 【修改】创建两个交叉注意力融合模块
        # self.fusion_stage5 = CrossAttentionFusion(
        #     d_model_skel=stage5_channels,
        #     d_model_ball=64, # TCN第3层输出通道
        #     nhead=4  # 注意力头数，可调整
        # )
        
        # self.fusion_stage8 = CrossAttentionFusion(
        #     d_model_skel=stage8_channels,
        #     d_model_ball=128, # TCN第5层输出通道
        #     nhead=8 # 更深层特征可以用更多头
        # )
        

        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        node_type = torch.tensor(self.graph.node_type, requires_grad=False)
        edge_type = torch.tensor(self.graph.edge_type, dtype=torch.float32, requires_grad=False)
        total_num_nodes = self.graph.num_node
        self.virtual_node_idx = self.graph.virtual_node_idx

        self.data_bn_type = data_bn_type
        self.kwargs = kwargs

        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * total_num_nodes)
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * total_num_nodes)
        else:
            self.data_bn = nn.Identity()

        lw_kwargs = [cp.deepcopy(kwargs) for i in range(num_stages)]
        for k, v in kwargs.items():
            if isinstance(v, tuple) and len(v) == num_stages:
                for i in range(num_stages):
                    lw_kwargs[i][k] = v[i]
        lw_kwargs[0].pop('tcn_dropout', None)
        lw_kwargs[0].pop('g1x1', None)
        lw_kwargs[0].pop('gcn_g1x1', None)
        if 'gcn_stage' in self.kwargs:
            for i in range(num_stages):
                if i in self.kwargs['gcn_stage']:
                    lw_kwargs[i]['gcn_stage'] = True
                else:
                    lw_kwargs[i]['gcn_stage'] = False

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages
        modules = []
        if self.in_channels != self.base_channels:
            modules = [DGBlock(in_channels, base_channels, A.clone(), edge_type, node_type, 1, residual=False, **lw_kwargs[0])]

        inflate_times = 0
        down_times = 0
        for i in range(2, num_stages + 1):
            stride = 1 + (i in down_stages)
            in_channels = base_channels
            if i in inflate_stages:
                inflate_times += 1
            out_channels = int(self.base_channels * self.ch_ratio ** inflate_times + EPS)
            base_channels = out_channels
            modules.append(DGBlock(in_channels, out_channels, A.clone(), edge_type, node_type, stride, **lw_kwargs[i - 1]))
            down_times += (i in down_stages)

        if self.in_channels == self.base_channels:
            num_stages -= 1

        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)
        self.pretrained = pretrained

    def init_weights(self):
        if isinstance(self.pretrained, str):
            self.pretrained = cache_checkpoint(self.pretrained)
            load_checkpoint(self, self.pretrained, strict=False)

    def forward(self, x, ball_trajectory=None):
        N, M, T, V, C = x.size()
        
        ball_feats = {}
        if ball_trajectory is not None:
            if ball_trajectory.dtype != torch.float:
                ball_trajectory = ball_trajectory.float()
            
            assert ball_trajectory.shape[0] == N, f"Mismatch batch size"
            assert ball_trajectory.shape[1] == T, f"Mismatch time dim"
            
            # 获取所有层级的球特征
            ball_feats = self.ball_tcn(ball_trajectory)
            
        # --- 初始化虚拟节点特征  ---
        feat = self.virtual_node_feature.permute(0, 2, 3, 1).unsqueeze(1)
        virtual_feat = feat.expand(N, M, T, -1, -1)
        x = torch.cat([x, virtual_feat], dim=3)
        virtual_node_idx = V
        
        # --- 标准前向处理流程  ---
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        if self.data_bn_type == 'MVC':
            x = self.data_bn(x.view(N, M * (V+1) * C, T))
        else:
            x = self.data_bn(x.view(N * M, (V+1) * C, T))
        x = x.view(N, M, V+1, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V+1)
        
        attention_maps = {}

        # --- 通过GCN网络处理 ---
        for i in range(self.num_stages):
            # i 是索引 (0-9)，对应 Stage 1-10
            # 所以当前是 Stage i+1
            current_stage = i + 1
            
            x = self.gcn[i](x)
            
            # 判断当前 Stage 是否需要融合，且球数据存在
            if ball_trajectory is not None and current_stage in self.fusion_stages:
                stage_key = str(current_stage)
                ball_feat_key = f'stage{current_stage}'
                
                # 获取对应的融合模块和球特征
                fusion_mod = self.fusion_modules[stage_key]
                ball_feat = ball_feats[ball_feat_key]
                
                # 执行融合
                x, attn_w = fusion_mod(x, ball_feat, virtual_node_idx)
                attention_maps[ball_feat_key] = attn_w.detach().cpu()

        # --- 输出处理 (保留) ---
        final_x = x.reshape((N, M) + x.shape[1:])
        virtual_node_feature = final_x[:, :, :, :, -1]
        skeleton_feature = final_x[:, :, :, :, :-1]
        return {'full_feature': final_x, 'virtual_node_feature': virtual_node_feature, 'skeleton_feature': skeleton_feature,'attention_maps': attention_maps}
