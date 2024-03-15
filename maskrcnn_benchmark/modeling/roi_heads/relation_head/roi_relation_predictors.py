# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
import torch
from maskrcnn_benchmark.modeling import registry
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter 
import pickle, math
import os
import json
from torch.distributions import Beta

from maskrcnn_benchmark.layers import smooth_l1_loss, kl_div_loss, entropy_loss, Label_Smoothing_Regression
from maskrcnn_benchmark.modeling.utils import cat
from .model_msg_passing import IMPContext
from .model_vtranse import VTransEFeature
from .model_vctree import VCTreeLSTMContext
from .model_motifs import LSTMContext, FrequencyBias
from .model_motifs_with_attribute import AttributeLSTMContext
from .model_transformer import TransformerContext
from .utils_relation import layer_init, get_box_info, get_box_pair_info
from maskrcnn_benchmark.data import get_dataset_statistics
from .utils_motifs import rel_vectors, obj_edge_vectors, to_onehot, nms_overlaps, encode_box_info 

from .utils_motifs import to_onehot, encode_box_info
from maskrcnn_benchmark.modeling.make_layers import make_fc


@registry.ROI_RELATION_PREDICTOR.register("PrototypeEmbeddingNetwork")
class PrototypeEmbeddingNetwork(nn.Module):
    
    def __init__(self, config, in_channels):
        super(PrototypeEmbeddingNetwork, self).__init__()

        
        self.cfg = config

        assert in_channels is not None
        self.in_channels = in_channels
        self.obj_dim = in_channels
        

        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        statistics = get_dataset_statistics(config)

        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        # assert self.num_obj_cls == len(obj_classes)
        # assert self.num_rel_cls == len(rel_classes)
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes

        self.num_obj_cls = len(obj_classes)
        # self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = len(rel_classes)
        
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM 
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        self.mlp_dim = 2048 # config.MODEL.ROI_RELATION_HEAD.PENET_MLP_DIM
        self.post_emb = nn.Linear(self.obj_dim, self.mlp_dim * 2)  

        self.embed_dim = 300 # config.MODEL.ROI_RELATION_HEAD.PENET_EMBED_DIM
        dropout_p = 0.2 # config.MODEL.ROI_RELATION_HEAD.PENET_DROPOUT

        if self.cfg.GLOBAL.DATASET_NAME == 'vg':
            
            with open('./predicate_dict.pkl', 'rb') as f:
                self.pred_count_dict = pickle.load(f)
            
            self.pred_count_dict['__background__'] = sum([v for v in self.pred_count_dict.values()]) * 20       
            
        elif self.cfg.GLOBAL.DATASET_NAME == 'gqa':
            
            with open('./predicate_dict_gqa.pkl', 'rb') as f:
                self.pred_count_dict = pickle.load(f)
            
            self.pred_count_dict['__background__'] = sum([v for v in self.pred_count_dict.values()]) * 20       



        self.rel_loss_type =  self.cfg.REL_LOSS_TYPE

        if self.rel_loss_type == 'ce':
            self.rel_loss_weight = torch.ones(len(self.rel_classes)).cuda()

        elif self.rel_loss_type == 'ce_rwt':
            self.rel_loss_weight = torch.Tensor([self.pred_count_dict[pred_name] for pred_name in self.rel_classes]).cuda()
            self.rel_loss_weight = (1-self.cfg.REWEIGHT_BETA) / (1-(self.cfg.REWEIGHT_BETA**self.rel_loss_weight))
            median = torch.median(self.rel_loss_weight[1:])
            self.rel_loss_weight = self.rel_loss_weight / median
            self.rel_loss_weight[0] = torch.min(self.rel_loss_weight[1:])
        
        
        obj_embed_vecs = obj_edge_vectors(obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)  # load Glove for objects
        rel_embed_vecs = rel_vectors(rel_classes, wv_dir=config.GLOVE_DIR, wv_dim=self.embed_dim)   # load Glove for predicates
        self.obj_embed = nn.Embedding(self.num_obj_cls, self.embed_dim)
        self.rel_embed = nn.Embedding(self.num_rel_cls, self.embed_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.rel_embed.weight.copy_(rel_embed_vecs, non_blocking=True)
       
        self.W_sub = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_obj = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_pred = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)

        self.gate_sub = nn.Linear(self.mlp_dim*2, self.mlp_dim)  
        self.gate_obj = nn.Linear(self.mlp_dim*2, self.mlp_dim)
        self.gate_pred = nn.Linear(self.mlp_dim*2, self.mlp_dim)

        self.vis2sem = nn.Sequential(*[
            nn.Linear(self.mlp_dim, self.mlp_dim*2), nn.ReLU(True),
            nn.Dropout(dropout_p), nn.Linear(self.mlp_dim*2, self.mlp_dim)
        ])

        self.project_head = MLP(self.mlp_dim, self.mlp_dim, self.mlp_dim*2, 2)

        self.linear_sub = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_obj = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_pred = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_rel_rep = nn.Linear(self.mlp_dim, self.mlp_dim)
        
        self.norm_sub = nn.LayerNorm(self.mlp_dim)
        self.norm_obj = nn.LayerNorm(self.mlp_dim)
        self.norm_rel_rep = nn.LayerNorm(self.mlp_dim)

        self.dropout_sub = nn.Dropout(dropout_p)
        self.dropout_obj = nn.Dropout(dropout_p)
        self.dropout_rel_rep = nn.Dropout(dropout_p)
        
        self.dropout_rel = nn.Dropout(dropout_p)
        self.dropout_pred = nn.Dropout(dropout_p)
       
        self.down_samp = MLP(self.pooling_dim, self.mlp_dim, self.mlp_dim, 2) 

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        ##### refine object labels
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum= 0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True),
        ])

        self.obj_embed1 = nn.Embedding(self.num_obj_cls, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(obj_embed_vecs, non_blocking=True)

        self.obj_dim = in_channels
        self.out_obj = make_fc(self.hidden_dim, self.num_obj_cls) 
        self.lin_obj_cyx = make_fc(self.obj_dim + self.embed_dim + 128, self.hidden_dim)

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'
        
        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

        if self.cfg.TYPE == 'ra_extract':
            self.featurebank_dict = dict()
            self.count_dict = dict()

            self.featurebank_proj_dict = dict()
            self.count_proj_dict = dict()

        self.predict_use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        if self.predict_use_bias:
            self.freq_bias = FrequencyBias(config, statistics)


    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None, cur_iter=None):

        add_losses = {}
        add_data = {}

        # refine object labels
        entity_dists, entity_preds = self.refine_obj_labels(roi_features, proposals)
        ##### 

        entity_rep = self.post_emb(roi_features)   # using the roi features obtained from the faster rcnn
        entity_rep = entity_rep.view(entity_rep.size(0), 2, self.mlp_dim)

        sub_rep = entity_rep[:, 1].contiguous().view(-1, self.mlp_dim)    # xs
        obj_rep = entity_rep[:, 0].contiguous().view(-1, self.mlp_dim)    # xo

        entity_embeds = self.obj_embed(entity_preds) # obtaining the word embedding of entities with GloVe 

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        sub_reps = sub_rep.split(num_objs, dim=0)
        obj_reps = obj_rep.split(num_objs, dim=0)
        entity_preds = entity_preds.split(num_objs, dim=0)
        entity_embeds = entity_embeds.split(num_objs, dim=0)

        fusion_so = []
        pair_preds = []

        for pair_idx, sub_rep, obj_rep, entity_pred, entity_embed, proposal in zip(rel_pair_idxs, sub_reps, obj_reps, entity_preds, entity_embeds, proposals):

            s_embed = self.W_sub(entity_embed[pair_idx[:, 0]])  #  Ws x ts
            o_embed = self.W_obj(entity_embed[pair_idx[:, 1]])  #  Wo x to

            sem_sub = self.vis2sem(sub_rep[pair_idx[:, 0]])  # h(xs)
            sem_obj = self.vis2sem(obj_rep[pair_idx[:, 1]])  # h(xo)
            
            gate_sem_sub = torch.sigmoid(self.gate_sub(cat((s_embed, sem_sub), dim=-1)))  # gs
            gate_sem_obj = torch.sigmoid(self.gate_obj(cat((o_embed, sem_obj), dim=-1)))  # go

            sub = s_embed + sem_sub * gate_sem_sub  # s = Ws x ts + gs · h(xs)  i.e., s = Ws x ts + vs
            obj = o_embed + sem_obj * gate_sem_obj  # o = Wo x to + go · h(xo)  i.e., o = Wo x to + vo

            ##### for the model convergence
            sub = self.norm_sub(self.dropout_sub(torch.relu(self.linear_sub(sub))) + sub)
            obj = self.norm_obj(self.dropout_obj(torch.relu(self.linear_obj(obj))) + obj)
            #####

            fusion_so.append(fusion_func(sub, obj)) # F(s, o)
            pair_preds.append(torch.stack((entity_pred[pair_idx[:, 0]], entity_pred[pair_idx[:, 1]]), dim=1))

        fusion_so = cat(fusion_so, dim=0)  
        pair_pred = cat(pair_preds, dim=0) 

        sem_pred = self.vis2sem(self.down_samp(union_features))  # h(xu)
        gate_sem_pred = torch.sigmoid(self.gate_pred(cat((fusion_so, sem_pred), dim=-1)))  # gp

        rel_rep = fusion_so - sem_pred * gate_sem_pred  #  F(s,o) - gp · h(xu)   i.e., r = F(s,o) - up
        predicate_proto = self.W_pred(self.rel_embed.weight)  # c = Wp x tp  i.e., semantic prototypes
        ##### for the model convergence
        rel_rep = self.norm_rel_rep(self.dropout_rel_rep(torch.relu(self.linear_rel_rep(rel_rep))) + rel_rep)

        if self.cfg.TYPE == 'ra_extract':

            union_reps = rel_rep.split(num_rels, dim=0)
            for pair_idx, sub_rep, obj_rep, union_rep, entity_pred, entity_embed, proposal, rel_label in zip(rel_pair_idxs, sub_reps, obj_reps, union_reps, entity_preds, entity_embeds, proposals, rel_labels):

                for i in range(len(pair_idx)):
                    if rel_label[i].item() == 0:
                        continue
                    tri_key = str(entity_pred[pair_idx[i, 0]].item()) + '_' + str(entity_pred[pair_idx[i, 1]].item()) + '_' + str(rel_label[i].item())
                    if tri_key not in self.featurebank_dict.keys():
                        self.featurebank_dict[tri_key] = []
                        self.count_dict[tri_key] = 0

                    if self.count_dict[tri_key] <= 100:
                        self.count_dict[tri_key] +=1
                        tail_union_feature = union_rep[i].detach().cpu()
                        tail_rel_label = rel_label[i].cpu()
                        tail_sub_feature = sub_rep[pair_idx[i, 0]].detach().cpu()
                        tail_obj_feature = obj_rep[pair_idx[i, 0]].detach().cpu()
                        tail_sub_proposal = proposal[pair_idx[i, 0], None]
                        tail_obj_proposal = proposal[pair_idx[i, 1], None]
                        self.featurebank_dict[tri_key].append(
                        (tail_rel_label, tail_sub_feature, tail_obj_feature, tail_sub_proposal, tail_obj_proposal, tail_union_feature))

        rel_rep = self.project_head(self.dropout_rel(torch.relu(rel_rep)))
        predicate_proto = self.project_head(self.dropout_pred(torch.relu(predicate_proto)))

        # if self.cfg.TYPE == 'ra_extract':
        #     union_reps = rel_rep.split(num_rels, dim=0)
        #     for pair_idx, sub_rep, obj_rep, union_rep, entity_pred, entity_embed, proposal, rel_label in zip(rel_pair_idxs, sub_reps, obj_reps, union_reps, entity_preds, entity_embeds, proposals, rel_labels):
        #         for i in range(len(pair_idx)):
        #             tri_key = str(entity_pred[pair_idx[i, 0]].item()) + '_' + str(entity_pred[pair_idx[i, 1]].item()) + '_' + str(rel_label[i].item())
        #             if tri_key not in self.featurebank_proj_dict.keys():
        #                 self.featurebank_proj_dict[tri_key] = []
        #                 self.count_proj_dict[tri_key] = 0
        #             if self.count_proj_dict[tri_key] <= 100:
        #                 self.count_proj_dict[tri_key] +=1
        #                 tail_union_feature = union_rep[i].detach().cpu()
        #                 tail_rel_label = rel_label[i].cpu()
        #                 tail_sub_feature = sub_rep[pair_idx[i, 0]].detach().cpu()
        #                 tail_obj_feature = obj_rep[pair_idx[i, 0]].detach().cpu()
        #                 tail_sub_proposal = proposal[pair_idx[i, 0], None]
        #                 tail_obj_proposal = proposal[pair_idx[i, 1], None]
        #                 self.featurebank_proj_dict[tri_key].append(
        #                 (tail_rel_label, tail_sub_feature, tail_obj_feature, tail_sub_proposal, tail_obj_proposal, tail_union_feature))

        rel_rep_norm = rel_rep / rel_rep.norm(dim=1, keepdim=True)  # r_norm
        predicate_proto_norm = predicate_proto / predicate_proto.norm(dim=1, keepdim=True)  # c_norm

        ### (Prototype-based Learning  ---- cosine similarity) & (Relation Prediction)
        rel_dists = rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()  #  <r_norm, c_norm> / τ
        # the rel_dists will be used to calculate the Le_sim with the ce_loss

        if (not self.training) & (self.predict_use_bias):
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred)

        entity_dists = entity_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        if self.training:

            ### Prototype Regularization  ---- cosine similarity
            target_rpredicate_proto_norm = predicate_proto_norm.clone().detach() 
            simil_mat = predicate_proto_norm @ target_rpredicate_proto_norm.t()  # Semantic Matrix S = C_norm @ C_norm.T
            l21 = torch.norm(torch.norm(simil_mat, p=2, dim=1), p=1) / (self.num_rel_cls*self.num_rel_cls)  
            add_losses.update({"l21_loss": l21})  # Le_sim = ||S||_{2,1}
            ### end
            
            ### Prototype Regularization  ---- Euclidean distance
            gamma2 = 7.0
            predicate_proto_a = predicate_proto.unsqueeze(dim=1).expand(-1, self.num_rel_cls, -1) 
            predicate_proto_b = predicate_proto.detach().unsqueeze(dim=0).expand(self.num_rel_cls, -1, -1)
            proto_dis_mat = (predicate_proto_a - predicate_proto_b).norm(dim=2) ** 2  # Distance Matrix D, dij = ||ci - cj||_2^2
            sorted_proto_dis_mat, _ = torch.sort(proto_dis_mat, dim=1)
            topK_proto_dis = sorted_proto_dis_mat[:, :2].sum(dim=1) / 1   # obtain d-, where k2 = 1
            dist_loss = torch.max(torch.zeros(self.num_rel_cls).cuda(), -topK_proto_dis + gamma2).mean()  # Lr_euc = max(0, -(d-) + gamma2)
            add_losses.update({"dist_loss2": dist_loss})
            ### end 

            ###  Prototype-based Learning  ---- Euclidean distance
            rel_labels = cat(rel_labels, dim=0)
            gamma1 = 1.0
            rel_rep_expand = rel_rep.unsqueeze(dim=1).expand(-1, self.num_rel_cls, -1)  # r
            predicate_proto_expand = predicate_proto.unsqueeze(dim=0).expand(rel_labels.size(0), -1, -1)  # ci
            distance_set = (rel_rep_expand - predicate_proto_expand).norm(dim=2) ** 2    # Distance Set G, gi = ||r-ci||_2^2
            mask_neg = torch.ones(rel_labels.size(0), self.num_rel_cls).cuda()  
            mask_neg[torch.arange(rel_labels.size(0)), rel_labels] = 0
            distance_set_neg = distance_set * mask_neg
            distance_set_pos = distance_set[torch.arange(rel_labels.size(0)), rel_labels]  # gt i.e., g+
            sorted_distance_set_neg, _ = torch.sort(distance_set_neg, dim=1)
            topK_sorted_distance_set_neg = sorted_distance_set_neg[:, :11].sum(dim=1) / 10  # obtaining g-, where k1 = 10, 
            loss_sum = torch.max(torch.zeros(rel_labels.size(0)).cuda(), distance_set_pos - topK_sorted_distance_set_neg + gamma1).mean()
            add_losses.update({"loss_dis": loss_sum})     # Le_euc = max(0, (g+) - (g-) + gamma1)
            ### end 
 
        return entity_dists, rel_dists, add_losses, add_data


    def refine_obj_labels(self, roi_features, proposals):
        use_gt_label = self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0) if use_gt_label else None
        pos_embed = self.pos_embed(encode_box_info(proposals))

        # label/logits embedding will be used as input
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_labels = obj_labels.long()
            obj_embed = self.obj_embed1(obj_labels)
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight

        assert proposals[0].mode == 'xyxy'

        pos_embed = self.pos_embed(encode_box_info(proposals))
        num_objs = [len(p) for p in proposals]
        obj_pre_rep_for_pred = self.lin_obj_cyx(cat([roi_features, obj_embed, pos_embed], -1))

        if self.mode == 'predcls':
            obj_labels = obj_labels.long()
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_cls)
        else:
            obj_dists = self.out_obj(obj_pre_rep_for_pred)  # 
            use_decoder_nms = self.mode == 'sgdet' and not self.training
            if use_decoder_nms:
                boxes_per_cls = [proposal.get_field('boxes_per_cls') for proposal in proposals]
                obj_preds = self.nms_per_cls(obj_dists, boxes_per_cls, num_objs).long()
            else:
                obj_preds = (obj_dists[:, 1:].max(1)[1] + 1).long()
        
        return obj_dists, obj_preds

    def nms_per_cls(self, obj_dists, boxes_per_cls, num_objs):
        obj_dists = obj_dists.split(num_objs, dim=0)
        obj_preds = []
        for i in range(len(num_objs)):
            is_overlap = nms_overlaps(boxes_per_cls[i]).cpu().numpy() >= self.nms_thresh # (#box, #box, #class)

            out_dists_sampled = F.softmax(obj_dists[i], -1).cpu().numpy()
            out_dists_sampled[:, 0] = -1

            out_label = obj_dists[i].new(num_objs[i]).fill_(0)

            for i in range(num_objs[i]):
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                out_label[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind,:,cls_ind], cls_ind] = 0.0
                out_dists_sampled[box_ind] = -1.0 # This way we won't re-sample

            obj_preds.append(out_label.long())
        obj_preds = torch.cat(obj_preds, dim=0)
        return obj_preds


@registry.ROI_RELATION_PREDICTOR.register("ReTAGPENet")
class ReTAGPENet(nn.Module):
    def __init__(self, config, in_channels):
        super(ReTAGPENet, self).__init__()

        self.cfg = config

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        assert in_channels is not None
        self.in_channels = in_channels
        self.obj_dim = in_channels

        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        statistics = get_dataset_statistics(config)

        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']

        # assert self.num_obj_cls == len(obj_classes)
        # assert self.num_rel_cls == len(rel_classes)

        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_classes = len(self.obj_classes)
        self.num_rel_classes = len(self.rel_classes)

        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM 
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.mlp_dim = 2048 # config.MODEL.ROI_RELATION_HEAD.PENET_MLP_DIM
        self.post_emb = nn.Linear(self.obj_dim, self.mlp_dim * 2)  
        self.embed_dim = 300 # config.MODEL.ROI_RELATION_HEAD.PENET_EMBED_DIM
        dropout_p = 0.2 # config.MODEL.ROI_RELATION_HEAD.PENET_DROPOUT

        ## Retrieval Augmented Parameters
        if self.cfg.GLOBAL.DATASET_NAME == 'vg':
            
            with open('./predicate_dict.pkl', 'rb') as f:
                self.pred_count_dict = pickle.load(f)
            
            self.pred_count_dict['__background__'] = sum([v for v in self.pred_count_dict.values()]) * 20       

            self.head_ids = torch.tensor(self.cfg.HEAD_IDS).cuda()
            self.body_ids = torch.tensor(self.cfg.BODY_IDS).cuda()
            self.tail_ids = torch.tensor(self.cfg.TAIL_IDS).cuda()

            
        elif self.cfg.GLOBAL.DATASET_NAME == 'gqa':
            
            with open('./predicate_dict_gqa.pkl', 'rb') as f:
                self.pred_count_dict = pickle.load(f)
            
            self.pred_count_dict['__background__'] = sum([v for v in self.pred_count_dict.values()]) * 20       

            self.head_ids = torch.tensor(self.cfg.GQA_HEAD_IDS).cuda()
            self.body_ids = torch.tensor(self.cfg.GQA_BODY_IDS).cuda()
            self.tail_ids = torch.tensor(self.cfg.GQA_TAIL_IDS).cuda()

        self.rel_loss_type =  self.cfg.REL_LOSS_TYPE

        if self.rel_loss_type == 'ce':
            self.rel_loss_weight = torch.ones(len(self.rel_classes)).cuda()

            if self.cfg.GLOBAL.DATASET_NAME == 'vg':
                fb = np.load(f'featurebank/{self.mode}_bg_processed_fb_train_{self.cfg.RASGG.MEMORY_SIZE}.npy', allow_pickle=True).tolist()
                print(f"LOADED FEATUREBANK FROM featurebank/{self.mode}_bg_processed_fb_train_{self.cfg.RASGG.MEMORY_SIZE}.npy")
            elif self.cfg.GLOBAL.DATASET_NAME == 'gqa':
                fb = np.load(f'featurebank_gqa/{self.mode}_bg_processed_fb_train.npy', allow_pickle=True).tolist()
                print(f"LOADED FEATUREBANK FROM featurebank_gqa/{self.mode}_bg_processed_fb_train.npy")

            self.freq_rwt = torch.Tensor([self.pred_count_dict[pred_name] for pred_name in self.rel_classes]).cuda()
            self.freq_rwt = (1-self.cfg.REWEIGHT_BETA) / (1-(self.cfg.REWEIGHT_BETA**self.freq_rwt))
            median = torch.median(self.freq_rwt[1:])
            self.freq_rwt = self.freq_rwt / median
            self.freq_rwt[0] = torch.min(self.freq_rwt[1:])
            self.contra_loss_weight = self.freq_rwt.clone()
            # self.contra_loss_weight[0] = 1 / (10 * torch.sum(1/self.rel_loss_weight[1:]))
            self.contra_loss_weight[0] = 1e-6
        
        self.featurebank_key_array = torch.FloatTensor(fb['key']).cuda()
        self.featurebank_key_array_norm = F.normalize(self.featurebank_key_array)
        self.featurebank_value_array = torch.LongTensor(fb['value']).cuda()
            
        obj_embed_vecs = obj_edge_vectors(obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)  # load Glove for objects
        rel_embed_vecs = rel_vectors(rel_classes, wv_dir=config.GLOVE_DIR, wv_dim=self.embed_dim)   # load Glove for predicates
        self.obj_embed = nn.Embedding(self.num_obj_cls, self.embed_dim)
        self.rel_embed = nn.Embedding(self.num_rel_cls, self.embed_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.rel_embed.weight.copy_(rel_embed_vecs, non_blocking=True)
       
        self.W_sub = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_obj = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_pred = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)

        self.gate_sub = nn.Linear(self.mlp_dim*2, self.mlp_dim)  
        self.gate_obj = nn.Linear(self.mlp_dim*2, self.mlp_dim)
        self.gate_pred = nn.Linear(self.mlp_dim*2, self.mlp_dim)

        self.vis2sem = nn.Sequential(*[
            nn.Linear(self.mlp_dim, self.mlp_dim*2), nn.ReLU(True),
            nn.Dropout(dropout_p), nn.Linear(self.mlp_dim*2, self.mlp_dim)
        ])

        self.project_head = MLP(self.mlp_dim, self.mlp_dim, self.mlp_dim*2, 2)
        # self.retriever = MLP(self.mlp_dim, self.mlp_dim, self.mlp_dim, 2)
        

        self.linear_sub = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_obj = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_pred = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_rel_rep = nn.Linear(self.mlp_dim, self.mlp_dim)
        
        self.norm_sub = nn.LayerNorm(self.mlp_dim)
        self.norm_obj = nn.LayerNorm(self.mlp_dim)
        self.norm_rel_rep = nn.LayerNorm(self.mlp_dim)

        self.dropout_sub = nn.Dropout(dropout_p)
        self.dropout_obj = nn.Dropout(dropout_p)
        self.dropout_rel_rep = nn.Dropout(dropout_p)
        
        self.dropout_rel = nn.Dropout(dropout_p)
        self.dropout_pred = nn.Dropout(dropout_p)
       
        self.down_samp = MLP(self.pooling_dim, self.mlp_dim, self.mlp_dim, 2) 

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        ##### refine object labels
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum= 0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True),
        ])

        self.obj_embed1 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(obj_embed_vecs, non_blocking=True)

        self.obj_dim = in_channels
        self.out_obj = make_fc(self.hidden_dim, self.num_obj_classes) 
        self.lin_obj_cyx = make_fc(self.obj_dim + self.embed_dim + 128, self.hidden_dim)

        self.ret_fusion = config.RASGG.RETRIEVAL_FUSION
        self.num_retrievals = config.RASGG.NUM_RETRIEVALS

        self.fusion_attn_q = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.fusion_attn_k = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.fusion_attn_v = nn.Linear(self.mlp_dim * 2, self.mlp_dim * 2)
        self.dropout_fusion = nn.Dropout(dropout_p)
        
        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

        self.predict_use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        if self.predict_use_bias:
            self.freq_bias = FrequencyBias(config, statistics)

        self.pos_masks = {'all_poscnt': 0, 'all_cnt': 0, 'bg_poscnt': 0, 'bg_cnt': 0, 'head_poscnt': 0, 'head_cnt': 0, 'body_poscnt': 0, 'body_cnt': 0, 'tail_poscnt': 0, 'tail_cnt': 0}
        self.pos_retrieval_prop = {'all': [], 'head':[], 'body':[], 'tail': [], 'bg': []}


    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None, cur_iter=None):

        add_losses = {}
        add_data = {}

        # refine object labels
        entity_dists, entity_preds = self.refine_obj_labels(roi_features, proposals)
        ##### 

        entity_rep = self.post_emb(roi_features)   # using the roi features obtained from the faster rcnn
        entity_rep = entity_rep.view(entity_rep.size(0), 2, self.mlp_dim)

        sub_rep = entity_rep[:, 1].contiguous().view(-1, self.mlp_dim)    # xs
        obj_rep = entity_rep[:, 0].contiguous().view(-1, self.mlp_dim)    # xo

        entity_embeds = self.obj_embed(entity_preds) # obtaining the word embedding of entities with GloVe 

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        sub_reps = sub_rep.split(num_objs, dim=0)
        obj_reps = obj_rep.split(num_objs, dim=0)
        entity_preds = entity_preds.split(num_objs, dim=0)
        entity_embeds = entity_embeds.split(num_objs, dim=0)

        fusion_so = []
        pair_preds = []

        for pair_idx, sub_rep, obj_rep, entity_pred, entity_embed, proposal in zip(rel_pair_idxs, sub_reps, obj_reps, entity_preds, entity_embeds, proposals):

            s_embed = self.W_sub(entity_embed[pair_idx[:, 0]])  #  Ws x ts
            o_embed = self.W_obj(entity_embed[pair_idx[:, 1]])  #  Wo x to

            sem_sub = self.vis2sem(sub_rep[pair_idx[:, 0]])  # h(xs)
            sem_obj = self.vis2sem(obj_rep[pair_idx[:, 1]])  # h(xo)
            
            gate_sem_sub = torch.sigmoid(self.gate_sub(cat((s_embed, sem_sub), dim=-1)))  # gs
            gate_sem_obj = torch.sigmoid(self.gate_obj(cat((o_embed, sem_obj), dim=-1)))  # go

            sub = s_embed + sem_sub * gate_sem_sub  # s = Ws x ts + gs · h(xs)  i.e., s = Ws x ts + vs
            obj = o_embed + sem_obj * gate_sem_obj  # o = Wo x to + go · h(xo)  i.e., o = Wo x to + vo

            ##### for the model convergence
            sub = self.norm_sub(self.dropout_sub(torch.relu(self.linear_sub(sub))) + sub)
            obj = self.norm_obj(self.dropout_obj(torch.relu(self.linear_obj(obj))) + obj)
            #####

            fusion_so.append(fusion_func(sub, obj)) # F(s, o)
            pair_preds.append(torch.stack((entity_pred[pair_idx[:, 0]], entity_pred[pair_idx[:, 1]]), dim=1))

        fusion_so = cat(fusion_so, dim=0)  
        pair_pred = cat(pair_preds, dim=0) 

        sem_pred = self.vis2sem(self.down_samp(union_features))  # h(xu)
        gate_sem_pred = torch.sigmoid(self.gate_pred(cat((fusion_so, sem_pred), dim=-1)))  # gp

        rel_rep = fusion_so - sem_pred * gate_sem_pred  #  F(s,o) - gp · h(xu)   i.e., r = F(s,o) - up
        predicate_proto = self.W_pred(self.rel_embed.weight)  # c = Wp x tp  i.e., semantic prototypes

        ##### for the model convergence
        rel_rep = self.norm_rel_rep(self.dropout_rel_rep(torch.relu(self.linear_rel_rep(rel_rep))) + rel_rep)


        # query_rels = self.retriever(rel_rep.detach())
        query_rels = rel_rep
        with torch.no_grad():
            key_rels = self.featurebank_key_array
            sim = torch.matmul(F.normalize(query_rels), F.normalize(key_rels).t())
            
        ret_qk, retrieved_idx = torch.topk(sim, 20*self.cfg.RASGG.NUM_RETRIEVALS)

        rel_rep = self.project_head(self.dropout_rel(torch.relu(rel_rep))) ## self.mlp_dim*2
        rel_rep_norm = rel_rep / rel_rep.norm(dim=1, keepdim=True)  # r_norm

        predicate_proto = self.project_head(self.dropout_pred(torch.relu(predicate_proto)))
        predicate_proto_norm = predicate_proto / predicate_proto.norm(dim=1, keepdim=True)  # c_norm

        target_rpredicate_proto_norm = predicate_proto_norm.clone().detach() 

        retrieved_key_pred = []
        retrieved_value = []
        retrieved_value_sub, retrieved_value_obj, retrieved_value_pred = [], [], [] ## N_rels * N_retrieve
        retrieved_sims = []
        retrieved_dists = []
        retrieved_ipss = []
                
        for i in range(retrieved_idx.shape[0]):
            ret_idx = retrieved_idx[i]
            ret_idx = ret_idx[1:self.cfg.RASGG.NUM_RETRIEVALS+1] ## To remove the self-retrieval case
            ret_sim = sim[i][ret_idx]

            ret_value = self.featurebank_value_array[ret_idx]
            retrieved_sims.append(ret_sim)
            retrieved_key_pred.append(self.featurebank_key_array[ret_idx, :])
            retrieved_value.append(ret_value)
            retrieved_value_sub.append(ret_value[:, 0])
            retrieved_value_obj.append(ret_value[:, 1])
            retrieved_value_pred.append(ret_value[:, 2])
            retrieved_dists.append(torch.sum(F.one_hot(ret_value[:, 2], num_classes=len(self.rel_classes)), 0) / self.cfg.RASGG.NUM_RETRIEVALS)
            retrieved_ipss.append(self.freq_rwt[ret_value[:, 2]])
            
        retrieved_sims = torch.stack(retrieved_sims)
        retrieved_value_sub = torch.stack(retrieved_value_sub)
        retrieved_value_obj = torch.stack(retrieved_value_obj)
        retrieved_value_pred = torch.stack(retrieved_value_pred)
        retrieved_key_pred = torch.stack(retrieved_key_pred)
        retrieved_dists = torch.stack(retrieved_dists)
        retrieved_ipss = torch.stack(retrieved_ipss)
        
        del key_rels        
        
        ### (Prototype-based Learning  ---- cosine similarity) & (Relation Prediction)
        rel_dists = rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()  #  <r_norm, c_norm> / τ
        # the rel_dists will be used to calculate the Le_sim with the ce_loss

        entity_dists = entity_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        ## Contrastive Loss
        rel_labels = cat(rel_labels, dim=0)

        mixup_labels = torch.zeros_like(rel_labels)
        pos_ret_mask = (rel_labels.reshape(-1 ,1) == retrieved_value_pred.reshape(-1, self.cfg.RASGG.NUM_RETRIEVALS))
        rel_label_confidences = pos_ret_mask.sum(-1) / self.cfg.RASGG.NUM_RETRIEVALS

        mask_fg_change = (rel_label_confidences < self.cfg.RASGG.THRESHOLD) & torch.any(torch.isin(retrieved_value_pred, self.tail_ids), dim=-1)
        mask_fg_nochange = (~mask_fg_change) & (rel_labels != 0)
        mask_fg_change = (mask_fg_change) & (rel_labels != 0)

        mask_bg_change = torch.any(torch.isin(retrieved_value_pred, self.tail_ids), dim=-1) & (rel_labels == 0)
            
        mask_bg_change_split = mask_bg_change.split(num_rels)
        mask_bg_change = []

        for m_bg in mask_bg_change_split:

            if m_bg.sum() == 0:
                _mask_bg_change = m_bg
            else:
                _mask_bg_change = torch.zeros(m_bg.shape[0]).bool().cuda()
                # bg_idx = torch.where(m_bg)[0]
                # bg_idx = bg_idx[torch.randperm(len(bg_idx))]

                # for _bg_idx in bg_idx[:min(self.cfg.RASGG.NUM_CORRECT_BG, len(bg_idx))]:
                #     _mask_bg_change[_bg_idx.item()] = True
                _mask_bg_change[torch.where(m_bg)[0][0]] = True

            mask_bg_change.append(_mask_bg_change)

        mask_bg_change = torch.cat(mask_bg_change)
            
        mask_nochange = mask_fg_nochange
        mask_change = mask_fg_change | mask_bg_change

        ## Change Labels
        sample_prob = retrieved_dists[mask_change] * (self.freq_rwt).unsqueeze(0)
        sample_prob = sample_prob / torch.sum(sample_prob, dim=1, keepdim=True)

        mixup_labels[mask_nochange] = rel_labels[mask_nochange]
        mixup_labels[mask_change] = torch.multinomial(sample_prob, 1).squeeze()

    
        if self.cfg.RASGG.MIXUP:
            rel_labels_onehot = F.one_hot(rel_labels, num_classes=len(self.rel_classes))
            mixup_labels_onehot = F.one_hot(mixup_labels, num_classes=len(self.rel_classes))
            beta_dist = Beta(torch.ones(len(rel_labels)) * self.cfg.RASGG.MIXUP_ALPHA, torch.ones(len(rel_labels)) * self.cfg.RASGG.MIXUP_BETA)
            lambda_coef = beta_dist.sample().unsqueeze(1).cuda()
            rel_labels_onehot = lambda_coef * rel_labels_onehot + (1-lambda_coef) * mixup_labels_onehot
            # rel_labels_onehot = self.cfg.RASGG.MIXUP_RATIO * rel_labels_onehot + (1-self.cfg.RASGG.MIXUP_RATIO) * mixup_labels_onehot
            add_data['rel_labels'] = rel_labels_onehot.split(num_rels, dim=0)
        else:
            add_data['rel_labels'] = mixup_labels.split(num_rels, dim=0)
      
        if self.training:
            ### Prototype Regularization  ---- cosine similarity
            simil_mat = predicate_proto_norm @ target_rpredicate_proto_norm.t()  # Semantic Matrix S = C_norm @ C_norm.T
            l21 = torch.norm(torch.norm(simil_mat, p=2, dim=1), p=1) / (self.num_rel_cls*self.num_rel_cls) 
            add_losses.update({"l21_loss": l21})  # Le_sim = ||S||_{2,1}
            ### end
            
            ### Prototype Regularization  ---- Euclidean distance
            gamma2 = 7.0
            predicate_proto_a = predicate_proto.unsqueeze(dim=1).expand(-1, self.num_rel_cls, -1) 
            predicate_proto_b = predicate_proto.detach().unsqueeze(dim=0).expand(self.num_rel_cls, -1, -1)
            proto_dis_mat = (predicate_proto_a - predicate_proto_b).norm(dim=2) ** 2  # Distance Matrix D, dij = ||ci - cj||_2^2
            sorted_proto_dis_mat, _ = torch.sort(proto_dis_mat, dim=1)
            topK_proto_dis = sorted_proto_dis_mat[:, :2].sum(dim=1) / 1   # obtain d-, where k2 = 1
            dist_loss = torch.max(torch.zeros(self.num_rel_cls).cuda(), -topK_proto_dis + gamma2).mean()  # Lr_euc = max(0, -(d-) + gamma2)
            add_losses.update({"dist_loss2": dist_loss})
            ### end 

            ###  Prototype-based Learning  ---- Euclidean distance
            gamma1 = 1.0
            rel_rep_expand = rel_rep.unsqueeze(dim=1).expand(-1, self.num_rel_cls, -1)  # r
            predicate_proto_expand = predicate_proto.unsqueeze(dim=0).expand(rel_labels.size(0), -1, -1)  # ci
            distance_set = (rel_rep_expand - predicate_proto_expand).norm(dim=2) ** 2    # Distance Set G, gi = ||r-ci||_2^2
            
            mask_neg = torch.ones(rel_labels.size(0), self.num_rel_cls).cuda()
            mask_neg[torch.arange(rel_labels.size(0)), rel_labels] = 0
            
            mask_neg_mix = torch.ones(rel_labels.size(0), self.num_rel_cls).cuda()
            mask_neg_mix[torch.arange(rel_labels.size(0)), mixup_labels] = 0

            distance_set_neg = distance_set * mask_neg
            distance_set_neg_mix = distance_set * mask_neg_mix
            distance_set_pos = distance_set[torch.arange(rel_labels.size(0)), rel_labels]  # gt i.e., g+
            distance_set_pos_mix = distance_set[torch.arange(rel_labels.size(0)), mixup_labels]  # gt i.e., g+
            sorted_distance_set_neg, _ = torch.sort(distance_set_neg, dim=1)
            sorted_distance_set_neg_mix, _ = torch.sort(distance_set_neg_mix, dim=1)

            topK_sorted_distance_set_neg = sorted_distance_set_neg[:, :11].sum(dim=1) / 10  # obtaining g-, where k1 = 10, 
            topK_sorted_distance_set_neg_mix = sorted_distance_set_neg_mix[:, :11].sum(dim=1) / 10  # obtaining g-, where k1 = 10, 
            
            proto_euc_loss = lambda_coef * (distance_set_pos - topK_sorted_distance_set_neg) + (1-lambda_coef) * (distance_set_pos_mix - topK_sorted_distance_set_neg_mix) + gamma1

            loss_sum = torch.max(torch.zeros(rel_labels.size(0)).cuda(), proto_euc_loss).mean()
            add_losses.update({"loss_dis": loss_sum})     # Le_euc = max(0, (g+) - (g-) + gamma1)

            num_rel_labs = (rel_labels != 0).sum().item()
            
            with torch.no_grad():
    
                head_mask = torch.isin(rel_labels, self.head_ids)
                num_head_mask = head_mask.sum().item()
                ret_success_head = (head_mask.unsqueeze(1).expand(-1, self.cfg.RASGG.NUM_RETRIEVALS)) & pos_ret_mask
                self.pos_masks['head_poscnt'] += torch.sum(ret_success_head).item() 
                self.pos_masks['head_cnt'] += num_head_mask * pos_ret_mask.shape[1]

                body_mask = torch.isin(rel_labels, self.body_ids)
                num_body_mask = body_mask.sum().item()
                ret_success_body = (body_mask.unsqueeze(1).expand(-1, self.cfg.RASGG.NUM_RETRIEVALS)) & pos_ret_mask
                self.pos_masks['body_poscnt'] += torch.sum(ret_success_body).item() 
                self.pos_masks['body_cnt'] += num_body_mask * pos_ret_mask.shape[1]
                
                
                tail_mask = torch.isin(rel_labels, self.tail_ids)
                num_tail_mask = tail_mask.sum().item()
                ret_success_tail = (tail_mask.unsqueeze(1).expand(-1, self.cfg.RASGG.NUM_RETRIEVALS)) & pos_ret_mask
                self.pos_masks['tail_poscnt'] += torch.sum(ret_success_tail).item() 
                self.pos_masks['tail_cnt'] += num_tail_mask * pos_ret_mask.shape[1]

                bg_mask = (rel_labels == 0)
                num_bg_mask = bg_mask.sum().item()
                ret_success_bg = (bg_mask.unsqueeze(1).expand(-1, self.cfg.RASGG.NUM_RETRIEVALS)) & pos_ret_mask
                self.pos_masks['bg_poscnt'] += torch.sum(ret_success_bg).item()
                self.pos_masks['bg_cnt'] += num_bg_mask * pos_ret_mask.shape[1]

                num_all_mask = (~bg_mask).sum().item()
                ret_success_all = ((~bg_mask).unsqueeze(1).expand(-1, self.cfg.RASGG.NUM_RETRIEVALS)) & pos_ret_mask
                self.pos_masks['all_poscnt'] += torch.sum(ret_success_all).item()
                self.pos_masks['all_cnt'] += num_all_mask * pos_ret_mask.shape[1]
                
                assert num_rel_labs == num_head_mask  + num_body_mask + num_tail_mask 
                # import pdb
                # pdb.set_trace()
                assert self.pos_masks['all_poscnt'] == self.pos_masks['head_poscnt'] + self.pos_masks['body_poscnt'] + self.pos_masks['tail_poscnt']

            if cur_iter % 200 == 0:
                pos_ret_all = self.pos_masks['all_poscnt'] / self.pos_masks['all_cnt']
                pos_ret_bg = self.pos_masks['bg_poscnt'] / self.pos_masks['bg_cnt']
                pos_ret_head = self.pos_masks['head_poscnt'] / self.pos_masks['head_cnt'] if self.pos_masks['head_cnt'] != 0 else -1
                pos_ret_body = self.pos_masks['body_poscnt'] / self.pos_masks['body_cnt'] if self.pos_masks['body_cnt'] != 0 else -1
                pos_ret_tail = self.pos_masks['tail_poscnt'] / self.pos_masks['tail_cnt'] if self.pos_masks['tail_cnt'] != 0 else -1
                
                print(f"Iteration:{cur_iter} ### Prop. of Positive ALL RETRIEVAL: {pos_ret_all:.2f} # BG RETRIEVAL: {pos_ret_bg:.2f}")
                print(f"Prop. of Positive HEAD RETRIEVAL: {pos_ret_head:.2f} # BODY RETRIEVAL: {pos_ret_body:.2f} # TAIL RETRIEVAL: {pos_ret_tail:.2f}")

                
                self.pos_masks = {'all_poscnt': 0, 'all_cnt': 0, 'bg_poscnt': 0, 'bg_cnt': 0, 'head_poscnt': 0, 'head_cnt': 0, 'body_poscnt': 0, 'body_cnt': 0, 'tail_poscnt': 0, 'tail_cnt': 0}

                self.pos_retrieval_prop['all'].append(pos_ret_all)
                self.pos_retrieval_prop['bg'].append(pos_ret_bg)
                self.pos_retrieval_prop['head'].append(pos_ret_head)
                self.pos_retrieval_prop['body'].append(pos_ret_body)
                self.pos_retrieval_prop['tail'].append(pos_ret_tail)

                with open(os.path.join(self.cfg.OUTPUT_DIR, 'pos_ret_dict.pkl'), 'wb') as f:
                    pickle.dump(self.pos_retrieval_prop, f)
                 
            
        return entity_dists, rel_dists, add_losses, add_data


    def refine_obj_labels(self, roi_features, proposals):
        use_gt_label = self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0) if use_gt_label else None
        pos_embed = self.pos_embed(encode_box_info(proposals))

        # label/logits embedding will be used as input
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_labels = obj_labels.long()
            obj_embed = self.obj_embed1(obj_labels)
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight

        assert proposals[0].mode == 'xyxy'

        pos_embed = self.pos_embed(encode_box_info(proposals))
        num_objs = [len(p) for p in proposals]
        obj_pre_rep_for_pred = self.lin_obj_cyx(cat([roi_features, obj_embed, pos_embed], -1))

        if self.mode == 'predcls':
            obj_labels = obj_labels.long()
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_classes)
        else:
            obj_dists = self.out_obj(obj_pre_rep_for_pred) 
            use_decoder_nms = self.mode == 'sgdet' and not self.training
            if use_decoder_nms:
                boxes_per_cls = [proposal.get_field('boxes_per_cls') for proposal in proposals]
                obj_preds = self.nms_per_cls(obj_dists, boxes_per_cls, num_objs).long()
            else:
                obj_preds = (obj_dists[:, 1:].max(1)[1] + 1).long()
        
        return obj_dists, obj_preds

    def nms_per_cls(self, obj_dists, boxes_per_cls, num_objs):
        obj_dists = obj_dists.split(num_objs, dim=0)
        obj_preds = []
        for i in range(len(num_objs)):
            is_overlap = nms_overlaps(boxes_per_cls[i]).cpu().numpy() >= self.nms_thresh # (#box, #box, #class)

            out_dists_sampled = F.softmax(obj_dists[i], -1).cpu().numpy()
            out_dists_sampled[:, 0] = -1

            out_label = obj_dists[i].new(num_objs[i]).fill_(0)

            for i in range(num_objs[i]):
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                out_label[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind,:,cls_ind], cls_ind] = 0.0
                out_dists_sampled[box_ind] = -1.0 # This way we won't re-sample

            obj_preds.append(out_label.long())
        obj_preds = torch.cat(obj_preds, dim=0)
        return obj_preds

    def predict_and_retrieve(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None, cur_iter=None):

        add_losses = {}
        add_data = {}

        # refine object labels
        entity_dists, entity_preds = self.refine_obj_labels(roi_features, proposals)
        ##### 

        entity_rep = self.post_emb(roi_features)   # using the roi features obtained from the faster rcnn
        entity_rep = entity_rep.view(entity_rep.size(0), 2, self.mlp_dim)

        sub_rep = entity_rep[:, 1].contiguous().view(-1, self.mlp_dim)    # xs
        obj_rep = entity_rep[:, 0].contiguous().view(-1, self.mlp_dim)    # xo

        entity_embeds = self.obj_embed(entity_preds) # obtaining the word embedding of entities with GloVe 

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        sub_reps = sub_rep.split(num_objs, dim=0)
        obj_reps = obj_rep.split(num_objs, dim=0)
        entity_preds = entity_preds.split(num_objs, dim=0)
        entity_embeds = entity_embeds.split(num_objs, dim=0)

        fusion_so = []
        pair_preds = []

        for pair_idx, sub_rep, obj_rep, entity_pred, entity_embed, proposal in zip(rel_pair_idxs, sub_reps, obj_reps, entity_preds, entity_embeds, proposals):

            s_embed = self.W_sub(entity_embed[pair_idx[:, 0]])  #  Ws x ts
            o_embed = self.W_obj(entity_embed[pair_idx[:, 1]])  #  Wo x to

            sem_sub = self.vis2sem(sub_rep[pair_idx[:, 0]])  # h(xs)
            sem_obj = self.vis2sem(obj_rep[pair_idx[:, 1]])  # h(xo)
            
            gate_sem_sub = torch.sigmoid(self.gate_sub(cat((s_embed, sem_sub), dim=-1)))  # gs
            gate_sem_obj = torch.sigmoid(self.gate_obj(cat((o_embed, sem_obj), dim=-1)))  # go

            sub = s_embed + sem_sub * gate_sem_sub  # s = Ws x ts + gs · h(xs)  i.e., s = Ws x ts + vs
            obj = o_embed + sem_obj * gate_sem_obj  # o = Wo x to + go · h(xo)  i.e., o = Wo x to + vo

            ##### for the model convergence
            sub = self.norm_sub(self.dropout_sub(torch.relu(self.linear_sub(sub))) + sub)
            obj = self.norm_obj(self.dropout_obj(torch.relu(self.linear_obj(obj))) + obj)
            #####

            fusion_so.append(fusion_func(sub, obj)) # F(s, o)
            pair_preds.append(torch.stack((entity_pred[pair_idx[:, 0]], entity_pred[pair_idx[:, 1]]), dim=1))

        fusion_so = cat(fusion_so, dim=0)  
        pair_pred = cat(pair_preds, dim=0) 

        sem_pred = self.vis2sem(self.down_samp(union_features))  # h(xu)
        gate_sem_pred = torch.sigmoid(self.gate_pred(cat((fusion_so, sem_pred), dim=-1)))  # gp

        rel_rep = fusion_so - sem_pred * gate_sem_pred  #  F(s,o) - gp · h(xu)   i.e., r = F(s,o) - up
        predicate_proto = self.W_pred(self.rel_embed.weight)  # c = Wp x tp  i.e., semantic prototypes

        ##### for the model convergence
        rel_rep = self.norm_rel_rep(self.dropout_rel_rep(torch.relu(self.linear_rel_rep(rel_rep))) + rel_rep)

        # query_rels = rel_rep
        # with torch.no_grad():
        #     key_rels = self.featurebank_key_array
        #     sim = torch.matmul(F.normalize(query_rels), F.normalize(key_rels).t())
        # ret_qk, retrieved_idx = torch.topk(sim, 2*self.cfg.RASGG.NUM_RETRIEVALS)

        rel_rep = self.project_head(self.dropout_rel(torch.relu(rel_rep))) ## self.mlp_dim*2
        rel_rep_norm = rel_rep / rel_rep.norm(dim=1, keepdim=True)  # r_norm

        predicate_proto = self.project_head(self.dropout_pred(torch.relu(predicate_proto)))
        predicate_proto_norm = predicate_proto / predicate_proto.norm(dim=1, keepdim=True)  # c_norm

        target_rpredicate_proto_norm = predicate_proto_norm.clone().detach() 
        
        # retrieved_key_pred = []
        # retrieved_value = []
        # retrieved_value_sub, retrieved_value_obj, retrieved_value_pred = [], [], [] ## N_rels * N_retrieve
        # retrieved_sims = []
        # retrieved_dists = []
        # retrieved_ipss = []
                
        # for i in range(retrieved_idx.shape[0]):
        #     ret_idx = retrieved_idx[i]
        #     ret_idx = ret_idx[1:self.cfg.RASGG.NUM_RETRIEVALS+1] ## To remove the self-retrieval case
        #     ret_sim = sim[i][ret_idx]

        #     ret_value = self.featurebank_value_array[ret_idx]
        #     retrieved_sims.append(ret_sim)
        #     retrieved_key_pred.append(self.featurebank_key_array[ret_idx, :])
        #     retrieved_value.append(ret_value)
        #     retrieved_value_sub.append(ret_value[:, 0])
        #     retrieved_value_obj.append(ret_value[:, 1])
        #     retrieved_value_pred.append(ret_value[:, 2])
        #     retrieved_dists.append(torch.sum(F.one_hot(ret_value[:, 2], num_classes=len(self.rel_classes)), 0) / self.cfg.RASGG.NUM_RETRIEVALS)
        #     retrieved_ipss.append(self.freq_rwt[ret_value[:, 2]])
            
        # retrieved_sims = torch.stack(retrieved_sims)
        # retrieved_value_sub = torch.stack(retrieved_value_sub)
        # retrieved_value_obj = torch.stack(retrieved_value_obj)
        # retrieved_value_pred = torch.stack(retrieved_value_pred)
        # retrieved_key_pred = torch.stack(retrieved_key_pred)
        # retrieved_dists = torch.stack(retrieved_dists)
        # retrieved_ipss = torch.stack(retrieved_ipss)
        
        # del key_rels

        ### (Prototype-based Learning  ---- cosine similarity) & (Relation Prediction)
        rel_dists = rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()  #  <r_norm, c_norm> / τ
        # the rel_dists will be used to calculate the Le_sim with the ce_loss

        if self.predict_use_bias:
            rel_dists = rel_dists + self.cfg.RASGG.FREQUENCY_LOGIT_COEF * self.freq_bias.index_with_labels(pair_pred)

        entity_dists = entity_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # retrieved_value = torch.stack(retrieved_value).cpu().numpy()
        retrieved_value = None

        return entity_dists, rel_dists, add_losses, add_data, retrieved_value





class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)  
        return x
    
    
def fusion_func(x, y):
    return F.relu(x + y) - (x - y) ** 2



def make_roi_relation_predictor(cfg, in_channels):
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg, in_channels)
