import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import torchvision.models as models

from resnet import resnet50
from itertools import chain

from copy import deepcopy


class CompatModel(nn.Module):
    """ 
    # This file includes code from "fashion_compatibility_mcn"
    # Copyright (c) 2019 WangX
    # Licensed under the MIT License.
    # See https://github.com/WangXin93/fashion_compatibility_mcn/blob/master/LICENSE for details.

    Reference: 
    https://github.com/WangXin93/fashion_compatibility_mcn/tree/master
    """
    def __init__(
            self, 
            args,
            embed_size=1000, 
            need_rep=False, 
            vocabulary=None,
            num_style=None,
            vse_off=False,
            pe_off=False,
            sae_off=False,
            sesim_off=True,
            mlp_layers=2,
            conv_feats="1234",
        ):

        super(CompatModel, self).__init__()
        
        ### Args 
        self.args = args
        
        self.vse_off = vse_off
        self.pe_off = pe_off
        self.sae_off = sae_off
        self.sesim_off = sesim_off
        
        self.mlp_layers = mlp_layers
        self.conv_feats = conv_feats
        
        self.num_style = num_style

        ### 이미지 Pretrained model(ResNet50) 로드
        cnn = resnet50(pretrained=True, need_rep=need_rep)
        cnn.fc = nn.Linear(cnn.fc.in_features, embed_size)
        self.cnn = cnn
        self.need_rep = need_rep
        self.relas = int((4*5)/2)
        
        ### Weight initializaton
        nn.init.xavier_uniform_(cnn.fc.weight)
        nn.init.constant_(cnn.fc.bias, 0)
        
        # 단어 임베딩 (word_embedding) 초기화
        self.word_embedding = nn.Embedding(vocabulary, embed_size, padding_idx = 1)

        
        # Visual embedding model (VSE 모듈)
        self.image_embedding = nn.Linear(2048, embed_size)

        
        '''
        Outfit compatibility learning 모듈의 변수
        '''
        ### 이미지 멀티 레이어에 대한 fully connected layer (차원 일치 목적)
        self.fc_l1 = nn.Linear(256, embed_size)
        self.fc_l2 = nn.Linear(512, embed_size)
        self.fc_l3 = nn.Linear(1024, embed_size)
        
        ### 이미지 멀티 레이어에 대한 마스킹 
        self.masks = nn.Embedding(self.relas, embed_size)
        self.masks.weight.data.normal_(0.9, 0.7)
        self.masks_l1 = nn.Embedding(self.relas, 256)
        self.masks_l1.weight.data.normal_(0.9, 0.7)
        self.masks_l2 = nn.Embedding(self.relas, 512)
        self.masks_l2.weight.data.normal_(0.9, 0.7)
        self.masks_l3 = nn.Embedding(self.relas, 1024)
        self.masks_l3.weight.data.normal_(0.9, 0.7)
        
        ### Global average pooling layer - Resnet 멀티 레이어의 중간 feature map(matrix)을 vector형태의 임베딩으로 변환
        self.ada_avgpool2d = nn.AdaptiveAvgPool2d((1, 1))
 
        
        ### outfit embedding에 대한 임베딩 차원
        self.num_rela = self.relas * len(conv_feats) 
        
        '''
        MCA - 싱글, 멀티 레이어 다양성 및 셀프, 크로스 어텐션 다양성
        '''
        ### Cross attention layer (이미지(멀티레이어 이미지)와 텍스트(단어 임베딩) 간의 Attention)
        if self.args.self_attention == True:
            self.img_modules = nn.ModuleList()
            self.txt_modules = nn.ModuleList()
        self.cross_modules = nn.ModuleList()
        
        for _ in range(self.args.num_blocks):
            mca = MCA_Transformer(embed_size, num_heads = 8, qkv_bias=False, attn_drop=0., ffn_drop=0., proj_drop=0.)
            
            if self.args.self_attention == True:
                self.img_modules.append(deepcopy(mca))
                self.txt_modules.append(deepcopy(mca))
            self.cross_modules.append(deepcopy(mca))
        
        ### Batch Normalization 
        self.bn = nn.BatchNorm1d(self.num_rela)
        
        ### Outfit embedding > Outfit compatibility score 
        if self.mlp_layers > 0:
            predictor = []
            for _ in range(self.mlp_layers-1):
                linear = nn.Linear(self.num_rela, self.num_rela)
                nn.init.xavier_uniform_(linear.weight)
                nn.init.constant_(linear.bias, 0)
                predictor.append(linear)
                predictor.append(nn.ReLU())
            linear = nn.Linear(self.num_rela, 1)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.constant_(linear.bias, 0)
            predictor.append(linear)
            self.predictor = nn.Sequential(*predictor)

        self.sigmoid = nn.Sigmoid()

        '''
        스타일 오토인코더 모듈
        '''
        # Style projection
        self.rep2style = nn.Linear(2048, embed_size) 
        self.attn2style = nn.Linear(embed_size, embed_size)
        
        # Style encoder
        self.style_encoder = nn.ModuleList()
        self.style_encoder.append(nn.Linear(embed_size , num_style)) 
        self.style_encoder.append(nn.Softmax(dim=1))
        
        # Style Embedding
        self.style_embedding = nn.Embedding(num_style, embed_size)
        

    def forward(self, images, texts, names, style, is_compat): 
        '''
        1. Compatibility module
        '''
        if self.need_rep:
            out, features, tmasks, rep = self._compute_multiattention_score(images, texts)
        else:
            out, features, tmasks = self._compute_multiattention_score(images, texts)
        
        '''
        2. VSE module
        '''
        if self.vse_off:
            vse_loss = torch.tensor(0.)
        else:
            vse_loss = self._compute_vse_loss(names, rep) 
        
        '''
        3. PE loss 모듈
        '''
        if self.pe_off: 
            tmasks_loss, features_loss = torch.tensor(0.), torch.tensor(0.)
        else:
            tmasks_loss, features_loss = self._compute_type_repr_loss(tmasks, features)
            
        '''
        4. Style loss 모듈
        '''
        CE = nn.CrossEntropyLoss()
        
        if self.sae_off: 
            sae_loss = torch.tensor(0.)
            sclf_loss = torch.tensor(0.)
            sae_basis_loss = torch.tensor(0.)
        else:
            if self.args.rep2sae == True:
                p_, sae_loss, sae_basis_loss = self._compute_rep2style_loss(rep, is_compat)
            if self.args.attn2sae == True:
                p_, sae_loss, sae_basis_loss = self._compute_attn2style_loss(features, is_compat)
            
            sclf_loss = CE(p_, style) 
        
        ### SESIM_OFF 고정
        if self.sesim_off:
            sesim_loss = torch.tensor(0.)
            sesim_basis_loss = torch.tensor(0.)
        else:
            p_, sesim_loss, sesim_basis_loss = self._compute_sesim_loss(rep, style, is_compat) 
        
        if self.sae_off == False:
            basis_loss = sae_basis_loss
        if self.sesim_off == False:
            basis_loss = sesim_basis_loss
            
        if (self.sae_off) & (self.sesim_off):
            p_ = torch.tensor(0.)
            basis_loss = torch.tensor(0.)

        return out, p_, vse_loss, tmasks_loss, features_loss, sclf_loss, sae_loss, basis_loss, sesim_loss
    
    def _compute_multiattention_score(self, images, texts, activate = True):
        '''
         Objective:
         - 멀티레이어 이미지와 상품 텍스트 내의 단어 간의 attention 임베딩을 추출하는 것을 목표로 함
         - 쿼리, 키, 밸류 간의 관계에 따라 여러 방식으로 접목할 수 있음
         - Cross attention 기작뿐만 아니라, Self attention 또한 적용할 수 있음
        
         Input:
         - images: 상품 이미지 (batch_size, item_num, ...)
         - texts: 
         - activate: 시그모이드 적용 여부 (True, False)
        '''
        
        batch_size, item_num, _, _, img_size = images.shape
        self.batch_size, self.item_num = batch_size, item_num
        
        images = torch.reshape(images, (-1, 3, img_size, img_size))
        
        '''
        Image embedding
        '''
        if self.need_rep:
            img_features_l4, *rep = self.cnn(images)
            rep_l1, rep_l2, rep_l3, rep_l4, rep = rep # l1 - [BxI, 256, 56, 56],l2 - [BxI,512, 28, 28], l3 - [BxI, 1024, 14, 14], l4 - [64, 2048, 7, 7]
            
        else:
            img_features_l4 = self.cnn(images)
        
        ### 멀티어텐션을 위하여, rep_l1, rep_l2, rep_l3에 대한  averagepooling 및 fc 레이어 
        rep_list = []
        masks_list = []
        if "1" in self.conv_feats:
            img_features_l1 = self.fc_l1(self.ada_avgpool2d(rep_l1).squeeze()).unsqueeze(dim = 1) # [BxI, 256, 56, 56] -(ada_avgpool2d)> [BxI, 256, 1, 1] -(squeeze)> [BxI, 256] -(fc)> [BxI, 1000] -(unsqueeze)> [BxI, 1, 1000]
            rep_list.append(img_features_l1); masks_list.append(self.masks_l1)
        if "2" in self.conv_feats:
            img_features_l2 = self.fc_l2(self.ada_avgpool2d(rep_l2).squeeze()).unsqueeze(dim = 1)
            rep_list.append(img_features_l2); masks_list.append(self.masks_l2)
        if "3" in self.conv_feats:
            img_features_l3 = self.fc_l3(self.ada_avgpool2d(rep_l3).squeeze()).unsqueeze(dim = 1)
            rep_list.append(img_features_l3); masks_list.append(self.masks_l3)
        if "4" in self.conv_feats:
            rep_list.append(img_features_l4.unsqueeze(dim = 1)); masks_list.append(self.masks)
        
        ### 멀티어텐션을 위한 멀티레이어의 임베딩 병합
        img_features = torch.cat(rep_list, dim=1)
        masks = F.relu(self.masks.weight)        
        
        
        '''
        Word embedding
        '''
        padded_names = rnn_utils.pad_sequence(texts, batch_first=True, padding_value = 1).to(self.args.device) # (BxI, ?) -(pad_sequence)> (BxI, S); 서로 다른 길이의 상품명 토큰의 길이를 배치 사이즈 중 가장 긴 것으로 세팅
        mask = (padded_names != 1) # 단어 토큰의 PAD 여부; True(단어), False(PAD)
        
        word_features = self.word_embedding(padded_names) # 워드 임베딩 추출 (BxI, S, D)
        word_features = word_features * (mask.unsqueeze(dim=2)).float() # 차원: (BxI, S, D)
        
        '''
        Cross Attention - [Q: 이미지 멀티레이어, K,V: 상품명의 단어] (image & words in the product name)
        '''
        
        ## For MCA_Transformer
        for i in range(self.args.num_blocks):
            if self.args.self_attention == True:
                img_mask = torch.BoolTensor([True]).expand(img_features.shape[:2]).to(self.args.device)
                img_features = self.img_modules[i](img_features, img_features, img_features, img_mask)
                word_features = self.txt_modules[i](word_features, word_features, word_features, mask) 

            img_features = self.cross_modules[i](img_features, word_features, word_features, mask) # Output - [BxI, L, D] // 실제로는 attention_feature이나,block으로 인해 다시 self-attention에 들어가야하는 상황을 고려하여 img_features로 명명
        
        img_features = img_features.reshape(batch_size, item_num, len(self.conv_feats), -1) # [B, I, L, D] 

        '''
        Pairwise relation을 활용한 Outfit embedding 추출
        '''
        relations = []
        for l in range(len(self.conv_feats)):
            for mi, (i, j) in enumerate(itertools.combinations_with_replacement([0,1,2,3], 2)):
                '''
                itertools.combinations_with_replacement: 01,02,03,...,44의 콤비네이션을 만들어내는 형태 
                00, 01, ... 을 통해서 outfit내의 상품 간의 pairwise score혹은 similarity를 만드는 것.
                여튼, left, right가 comparison matrix에서의 i,j의 첨자를 의미하는 것으로 파악했음
                '''
                if self.pe_off:
                    left = F.normalize(img_features[:, i:i+1, l, :], dim=-1) 
                    right = F.normalize(img_features[:, j:j+1, l, :], dim=-1)
                else:
                    left = F.normalize(masks[mi] * img_features[:, i:i+1, l, :], dim=-1) 
                    right = F.normalize(masks[mi] * img_features[:, j:j+1, l, :], dim=-1)
                rela = torch.matmul(left, right.transpose(1, 2)).squeeze() 
                relations.append(rela)

        if batch_size == 1: # Inference during evaluation, which input one sample
            relations = torch.stack(relations).unsqueeze(0)
        else:
            relations = torch.stack(relations, dim=1)
        
        '''
        Outfit embedding > Compatibility score
        '''
        relations = self.bn(relations)
        
        # Predictor
        if self.mlp_layers == 0:
            out = relations.mean(dim=-1, keepdim=True)
        else:
            out = self.predictor(relations) 
        
        if activate:
            out = self.sigmoid(out) 
            
        if self.need_rep:
            return out, img_features, masks, rep
        else:
            return out, img_features, masks
    
    '''
    Style 모듈 - 이미지(ResNet representation) 활용
    '''
    def _compute_rep2style_loss(self, rep, is_compat):
        styemb_item = F.normalize(self.rep2style(rep), dim=1)
        embed_size = styemb_item.shape[-1]
        styemb_item = torch.reshape(styemb_item, (self.batch_size, self.item_num, -1))
        
        styemb = styemb_item.mean(axis = 1).squeeze()
        styemb = styemb[is_compat==1]

        p_ = self.style_encoder[0](styemb)
        p = self.style_encoder[1](p_)
        
        rec_styemb = torch.matmul(p, self.style_embedding(torch.LongTensor(range(self.num_style)).to(self.args.device)))
        
        scores = torch.matmul(styemb, rec_styemb.transpose(0, 1))
        diag = scores.diag().unsqueeze(dim=1)
        cost_sae = torch.clamp(0.2 - diag + scores, min=0, max=1e6)  # 0.2 is margin
        cost_sae = cost_sae - torch.diag(cost_sae.diag())
        sae_loss = cost_sae.sum()
        sae_loss = sae_loss / (styemb.shape[0] ** 2)
        
        style_basis = F.normalize(self.style_embedding.weight.clone(), dim=0)
        basis_loss = (torch.matmul(style_basis, style_basis.transpose(1,0)) - torch.eye(self.num_style).to(self.args.device)).norm()
        basis_loss = basis_loss/len(style_basis)

        return p_, sae_loss, basis_loss
    
    '''
    Style 모듈 - 멀티 어텐션 임베딩 활용
    '''
    def _compute_attn2style_loss(self, attn_features, is_compat):
        attn_features = attn_features[:,:,-1,:].squeeze() # (B,I,L,D) > (B,I,D)
        attn_features = attn_features.reshape(-1,attn_features.shape[-1]) # (B,I,D) > (BxI,D)
        
        styemb_item = F.normalize(self.attn2style(attn_features), dim=1) # Style 공간으로 projection
        embed_size = styemb_item.shape[-1]
        styemb_item = torch.reshape(styemb_item, (self.batch_size, self.item_num, -1)) # (BxI,D) > (B,I,D)
        
        styemb = styemb_item.mean(axis = 1).squeeze() # 아웃핏레벨의 임베딩 추출: (B,I,D) > (B,D)
        styemb = styemb[is_compat==1] 

        p_ = self.style_encoder[0](styemb)
        p = self.style_encoder[1](p_)
        
        rec_styemb = torch.matmul(p, self.style_embedding(torch.LongTensor(range(self.num_style)).to(self.args.device)))
        
        scores = torch.matmul(styemb, rec_styemb.transpose(0, 1))
        diag = scores.diag().unsqueeze(dim=1)
        cost_sae = torch.clamp(0.2 - diag + scores, min=0, max=1e6)  # 0.2 is margin
        cost_sae = cost_sae - torch.diag(cost_sae.diag())
        sae_loss = cost_sae.sum()
        sae_loss = sae_loss / (styemb.shape[0] ** 2)
        
        style_basis = F.normalize(self.style_embedding.weight.clone(), dim=0)
        basis_loss = (torch.matmul(style_basis, style_basis.transpose(1,0)) - torch.eye(self.num_style).to(self.args.device)).norm()
        basis_loss = basis_loss/len(style_basis)

        return p_, sae_loss, basis_loss
    
    '''
    VSE 모듈
    '''
    def _compute_vse_loss(self, names, rep):
        """ Visual semantice loss which map both visual embedding and semantic embedding 
        into a common space.

        Reference: 
        https://github.com/xthan/polyvore/blob/e0ca93b0671491564b4316982d4bfe7da17b6238/polyvore/polyvore_model_bi.py#L362
        """
        
        '''
        단어임베딩 추출
        '''
        padded_names = rnn_utils.pad_sequence(names, batch_first=True, padding_value = 1).to(self.args.device)
        mask = (padded_names != 1)
        cap_mask = torch.ge(mask.sum(dim=1), 2)
        word_features = self.word_embedding(padded_names)
        word_features = word_features * (mask.unsqueeze(dim=2)).float()
        word_lengths = mask.sum(dim=1)
        word_lengths = torch.where(
            word_lengths == 0,
            (torch.ones(word_features.shape[0]).float() * 0.1).to(rep.device),
            word_lengths.float(),
        )
        '''
        상품명 임베딩 변환
        '''
        semb = word_features.sum(dim=1) / word_lengths.unsqueeze(dim=1)
        semb = F.normalize(semb, dim=1)
        
        '''
        이미지 임베딩
        ''' 
        # Normalized Visual Embedding
        vemb = F.normalize(self.image_embedding(rep), dim=1)
        embed_size = vemb.size(-1)

        '''
        VSE Loss
        '''
        semb = torch.masked_select(semb, cap_mask.unsqueeze(dim=1))
        vemb = torch.masked_select(vemb, cap_mask.unsqueeze(dim=1))
        semb = semb.reshape([-1, embed_size])
        vemb = vemb.reshape([-1, embed_size])
        scores = torch.matmul(semb, vemb.transpose(0, 1))
        diagnoal = scores.diag().unsqueeze(dim=1)
        cost_s = torch.clamp(0.2 - diagnoal + scores, min=0, max=1e6)  # 0.2 is margin
        cost_im = torch.clamp(0.2 - diagnoal.transpose(0, 1) + scores, min=0, max=1e6)
        cost_s = cost_s - torch.diag(cost_s.diag())
        cost_im = cost_im - torch.diag(cost_im.diag())
        vse_loss = cost_s.sum() + cost_im.sum()
        vse_loss = vse_loss / (semb.shape[0] ** 2)

        return vse_loss
    
    '''
    PE 모듈
    '''
    def _compute_type_repr_loss(self, tmasks, features):
        """ Here adopt two losses to improve the type-spcified represetations.
        `tmasks_loss` expect the masks to be sparse and `features_loss` regularize
        the feature vector to be a unit vector.

        Reference:
        Conditional Similarity Networks: https://arxiv.org/abs/1603.07810
        """
        # Type embedding loss
        tmasks_loss = tmasks.norm(1) / len(tmasks)
        features_loss = features.norm(2) / np.sqrt(
            (features.shape[0] * features.shape[1])
        )
        return tmasks_loss, features_loss 

class MCA_Transformer(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, attn_drop=0., ffn_drop=0., proj_drop=0.):
        super(MCA_Transformer, self).__init__()
        self.mca = MCA_renewal(dim, num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.ln1 = nn.LayerNorm(dim, eps=1e-8)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.ln2 = nn.LayerNorm(dim, eps=1e-8)
        
    def forward(self, x_q, x_k, x_v, mask):
        attn_output = self.mca(x_q, x_k, x_v, mask)
        x = x_q + attn_output
        x = self.ln1(x)
        ffn_output = self.ffn(x)
        x = x + ffn_output
        x = self.ln2(x)
        
        return x
        

class MCA_renewal(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self._reset_parameters()

    def _reset_parameters(self):
        torch.manual_seed(0)
        nn.init.xavier_uniform_(self.q.weight)
        nn.init.xavier_uniform_(self.k.weight)
        nn.init.xavier_uniform_(self.v.weight)
        nn.init.xavier_uniform_(self.proj.weight)
        if self.k.bias is not None:
            nn.init.xavier_normal_(self.k.bias)
        if self.v.bias is not None:
            nn.init.xavier_normal_(self.v.bias)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0.)

    def forward(self, x_q, x_k, x_v, mask):
        B, N_q, C = x_q.shape
        _, N_kv, C = x_k.shape
        _, N_kv, C = x_v.shape
        
        # b, h, n, d
        q = self.q(x_q).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x_k).reshape(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x_v).reshape(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        #
        mask = mask.expand(self.num_heads, B, N_kv).permute(1,0,2) #
        mask = mask.unsqueeze(dim=2).float()
        
        # [b, h, n, d] * [b, h, d, m] -> [b, h, n, m]
        attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn * mask
        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x        

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate, hidden_act='relu', eps=1e-8):
        super(PointWiseFeedForward, self).__init__()
        self.layer_norm = nn.LayerNorm(hidden_units, eps=eps)
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.act_func = self.get_hidden_act(hidden_act)
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)
    
    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": F.relu,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::
            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    

    def forward(self, inputs):
        inputs = self.layer_norm(inputs)
        outputs = self.dropout2(self.conv2(self.act_func(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs
