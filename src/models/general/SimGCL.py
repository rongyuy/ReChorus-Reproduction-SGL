# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BaseModel import GeneralModel
from models.general.LightGCN import LightGCNBase, LGCNEncoder

class SimGCLEncoder(LGCNEncoder):
    """
    SimGCL 专用的 Encoder，支持在图卷积过程中注入随机噪声。
    """
    def __init__(self, user_count, item_count, emb_size, norm_adj, n_layers=3, eps=0.1):
        super(SimGCLEncoder, self).__init__(user_count, item_count, emb_size, norm_adj, n_layers)
        self.eps = eps

    def forward(self, users, items, perturbed=False):
        # 获取初始 User 和 Item 的 Embedding
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = []

        # 图卷积传播
        for k in range(len(self.layers)):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            
            # 核心逻辑：如果是扰动模式(perturbed=True)，则在当前层加入噪声
            if perturbed:
                # 生成与当前 embedding 同维度的随机噪声
                random_noise = torch.rand_like(ego_embeddings).cuda()
                # 按照论文公式：进行 L2 归一化并乘以系数 eps
                random_noise = F.normalize(random_noise, dim=1) * self.eps
                # 还可以选择性加入论文提到的 sign 约束，但大部分复现仅用上述公式即可达到 SOTA
                # random_noise = torch.sign(ego_embeddings) * random_noise 
                
                ego_embeddings = ego_embeddings + random_noise
            
            all_embeddings.append(ego_embeddings)

        # 层级聚合 (Mean)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)

        # 分离 User 和 Item
        user_all_embeddings = all_embeddings[:self.user_count, :]
        item_all_embeddings = all_embeddings[self.user_count:, :]

        # 获取指定 batch 用户的 Embedding
        user_embeddings = user_all_embeddings[users, :]
        item_embeddings = item_all_embeddings[items, :]

        return user_embeddings, item_embeddings

class SimGCL(GeneralModel, LightGCNBase):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'n_layers', 'eps', 'lam', 'tau']

    @staticmethod
    def parse_model_args(parser):
        # 先加载 LightGCN 的参数 (emb_size, n_layers 等)
        parser = LightGCNBase.parse_model_args(parser)
        
        # 添加 SimGCL 特有的参数
        parser.add_argument('--eps', type=float, default=0.1,
                            help='SimGCL: Noise epsilon (e.g. 0.1)')
        parser.add_argument('--lam', type=float, default=0.1,
                            help='SimGCL: Weight for InfoNCE loss (e.g. 0.1)')
        parser.add_argument('--tau', type=float, default=0.2,
                            help='SimGCL: Temperature parameter for InfoNCE (e.g. 0.2)')
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        GeneralModel.__init__(self, args, corpus)
        
        self.eps = args.eps
        self.lam = args.lam
        self.tau = args.tau
        # 初始化 LightGCN 基础部分（如图结构）
        self._base_init(args, corpus)

    def _base_define_params(self):
        # 覆盖 LightGCNBase 的该方法，使用自定义的 SimGCLEncoder
        self.encoder = SimGCLEncoder(self.user_num, self.item_num, self.emb_size, 
                                     self.norm_adj, self.n_layers, self.eps)

    def forward(self, feed_dict):
        # 1. 正常的前向传播（用于计算推荐分数）
        out_dict = LightGCNBase.forward(self, feed_dict)
        
        # 2. 如果是在训练阶段，计算对比视图
        if self.training:
            # ReChorus 的 feed_dict['user_id'] 是当前 batch 的 user
            # feed_dict['item_id'] 包含了正样本和负样本 [batch_size, 1 + num_neg]
            # 为了计算 CL Loss，我们只需要 batch 中的 unique users 和 items（通常只取正样本 item）
            batch_users = feed_dict['user_id']
            batch_items = feed_dict['item_id'][:, 0] # 取出正样本 Item

            # 生成第一个视图 (View 1)
            u_view1, i_view1 = self.encoder(batch_users, batch_items, perturbed=True)
            # 生成第二个视图 (View 2)
            u_view2, i_view2 = self.encoder(batch_users, batch_items, perturbed=True)
            
            # 将视图保存到输出字典中，传给 loss 函数使用
            out_dict['u_view1'] = u_view1
            out_dict['u_view2'] = u_view2
            out_dict['i_view1'] = i_view1
            out_dict['i_view2'] = i_view2
            
        return out_dict

    def loss(self, out_dict: dict) -> torch.Tensor:
        # 1. 计算基础的 BPR Loss (直接调用父类 GeneralModel 的方法)
        bpr_loss = super().loss(out_dict)

        # 2. 计算 InfoNCE Loss (Contrastive Loss)
        # 只有在训练时 out_dict 里才会有 view 数据
        if 'u_view1' in out_dict:
            u_view1, u_view2 = out_dict['u_view1'], out_dict['u_view2']
            i_view1, i_view2 = out_dict['i_view1'], out_dict['i_view2']

            # 计算 User 侧的 CL Loss
            user_cl_loss = self.info_nce_loss(u_view1, u_view2)
            # 计算 Item 侧的 CL Loss
            item_cl_loss = self.info_nce_loss(i_view1, i_view2)
            
            cl_loss = user_cl_loss + item_cl_loss
            
            # 总 Loss = BPR + lambda * CL
            return bpr_loss + self.lam * cl_loss
        
        return bpr_loss

    def info_nce_loss(self, view1, view2):
        """
        计算 InfoNCE Loss
        view1, view2: [batch_size, emb_size]
        """
        # 对 embedding 进行 L2 归一化
        view1 = F.normalize(view1, dim=1)
        view2 = F.normalize(view2, dim=1)
        
        # 正样本相似度：同一节点在不同视图下的相似度
        # (batch_size,)
        pos_score = (view1 * view2).sum(dim=1)
        
        # 所有样本相似度：view1 节点与 view2 中所有节点的相似度
        # (batch_size, batch_size)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        
        # 计算 LogSumExp
        # 分子：exp(pos / tau)
        # 分母：sum(exp(all / tau))
        pos_score = torch.exp(pos_score / self.tau)
        ttl_score = torch.exp(ttl_score / self.tau).sum(dim=1)
        
        cl_loss = -torch.log(pos_score / ttl_score).mean()
        return cl_loss