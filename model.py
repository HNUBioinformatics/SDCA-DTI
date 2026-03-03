import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import Config


class CLSModulation(nn.Module):
    """Lightweight feature modulation using CLS vector (FiLM-style)"""
    def __init__(self, cls_dim, feature_dim):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.Linear(cls_dim, feature_dim * 2),
            nn.GELU()
        )
        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_normal_(self.modulation[0].weight, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.modulation[0].bias)
    
    def forward(self, x, cls_feat):
        modulation = self.modulation(cls_feat)
        scale, shift = modulation.chunk(2, dim=-1)
        scale = torch.sigmoid(scale)
        scale = scale.unsqueeze(-1)
        shift = shift.unsqueeze(-1)
        return scale * x + shift


class DepthwiseSeparableConv1D(nn.Module):
    """Depthwise separable convolution"""
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_normal_(self.depthwise.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.pointwise.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x


class ProteinEncoder(nn.Module):
    """Protein encoder"""
    def __init__(self):
        super().__init__()
        self.esm2_proj = nn.Linear(Config.ESM2_DIM, Config.PROTEIN_TOKEN_DIM)
        self.proj_norm = nn.LayerNorm(Config.PROTEIN_TOKEN_DIM)
        
        self.dwc_layers = nn.ModuleList([
            DepthwiseSeparableConv1D(
                Config.PROTEIN_TOKEN_DIM, 
                Config.PROTEIN_TOKEN_DIM, 
                kernel_size=Config.PROTEIN_CNN_KERNEL, 
                padding=Config.PROTEIN_CNN_KERNEL // 2
            )
            for _ in range(Config.PROTEIN_CNN_LAYERS)
        ])
        
        self.cls_modulation = nn.ModuleList([
            CLSModulation(Config.ESM2_DIM, Config.PROTEIN_TOKEN_DIM)
            for _ in range(Config.PROTEIN_CNN_LAYERS)
        ])
        
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(Config.DROPOUT)
        self.final_norm = nn.LayerNorm(Config.PROTEIN_TOKEN_DIM)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_normal_(self.esm2_proj.weight, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.esm2_proj.bias)
        
    def forward(self, token_feat, token_mask, cls_feat):
        x = self.esm2_proj(token_feat)
        x = self.proj_norm(x)
        x = x.transpose(1, 2)
        
        if token_mask is not None: 
            mask = token_mask.unsqueeze(1).float()
            x = x * mask
        
        layer_outputs = []
        for dwc_layer, modulation in zip(self.dwc_layers, self.cls_modulation):
            x = dwc_layer(x)
            x = modulation(x, cls_feat)
            x = self.activation(x)
            layer_outputs.append(x)
        
        x = sum(layer_outputs)
        x = x.transpose(1, 2)
        x = self.dropout(x)
        x = self.final_norm(x)
        
        if token_mask is not None: 
            mask = token_mask.unsqueeze(-1).float()
            x_masked = x.clone()
            x_masked[~token_mask.unsqueeze(-1).expand_as(x)] = float('-inf')
            global_feat = x_masked.max(dim=1)[0]
        else:
            global_feat = x.max(dim=1)[0]
        
        return x, global_feat


class DrugEncoder(nn.Module):
    """Drug encoder"""
    def __init__(self):
        super().__init__()
        self.chembert_proj = nn.Linear(Config.CHEMBERT_DIM, Config.DRUG_TOKEN_DIM)
        self.proj_norm = nn.LayerNorm(Config.DRUG_TOKEN_DIM)
        
        self.dwc_layers = nn.ModuleList([
            DepthwiseSeparableConv1D(
                Config.DRUG_TOKEN_DIM, 
                Config.DRUG_TOKEN_DIM,
                kernel_size=Config.DRUG_CNN_KERNEL,
                padding=Config.DRUG_CNN_KERNEL // 2
            )
            for _ in range(Config.DRUG_CNN_LAYERS)
        ])
        
        self.cls_modulation = nn.ModuleList([
            CLSModulation(Config.CHEMBERT_DIM, Config.DRUG_TOKEN_DIM)
            for _ in range(Config.DRUG_CNN_LAYERS)
        ])
        
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(Config.DROPOUT)
        self.final_norm = nn.LayerNorm(Config.DRUG_TOKEN_DIM)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_normal_(self.chembert_proj.weight, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.chembert_proj.bias)
        
    def forward(self, token_feat, token_mask, cls_feat):
        x = self.chembert_proj(token_feat)
        x = self.proj_norm(x)
        x = x.transpose(1, 2)
        
        if token_mask is not None:
            mask = token_mask.unsqueeze(1).float()
            x = x * mask
        
        layer_outputs = []
        for dwc_layer, modulation in zip(self.dwc_layers, self.cls_modulation):
            x = dwc_layer(x)
            x = modulation(x, cls_feat)
            x = self.activation(x)
            layer_outputs.append(x)
        
        x = sum(layer_outputs)
        x = x.transpose(1, 2)
        x = self.dropout(x)
        x = self.final_norm(x)
        
        if token_mask is not None:
            mask = token_mask.unsqueeze(-1).float()
            x_masked = x.clone()
            x_masked[~token_mask.unsqueeze(-1).expand_as(x)] = float('-inf')
            global_feat = x_masked.max(dim=1)[0]
        else:
            global_feat = x.max(dim=1)[0]
        
        return x, global_feat


class DMPNNLayer(nn.Module):
    """D-MPNN message passing layer"""
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.message_nn = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT)
        )
        self.update_nn = nn.Sequential(
            nn.Linear(hidden_dim + node_dim, node_dim),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT)
        )
        self.norm = nn.LayerNorm(node_dim)
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.message_nn, self.update_nn]:
            for m in module: 
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
    def forward(self, node_feat, edge_index, edge_feat):
        src, tgt = edge_index[0], edge_index[1]
        src_feat = node_feat[src]
        messages = torch.cat([src_feat, edge_feat], dim=-1)
        messages = self.message_nn(messages)
        
        num_nodes = node_feat.size(0)
        aggregated = torch.zeros(num_nodes, messages.size(-1), device=node_feat.device)
        aggregated.index_add_(0, tgt, messages)
        
        update_input = torch.cat([aggregated, node_feat], dim=-1)
        new_node_feat = self.update_nn(update_input)
        new_node_feat = node_feat + new_node_feat
        new_node_feat = self.norm(new_node_feat)
        
        return new_node_feat


class MolecularGraph(nn.Module):
    """Molecular graph encoder: D-MPNN"""
    def __init__(self):
        super().__init__()
        self.atom_embed = nn.Linear(Config.ATOM_FEAT_DIM, Config.GRAPH_NODE_DIM)
        self.edge_embed = nn.Linear(Config.BOND_FEAT_DIM, Config.GRAPH_NODE_DIM)
        self.embed_norm = nn.LayerNorm(Config.GRAPH_NODE_DIM)
        
        self.dmpnn_layers = nn.ModuleList([
            DMPNNLayer(Config.GRAPH_NODE_DIM, Config.GRAPH_NODE_DIM, Config.DMPNN_HIDDEN_DIM)
            for _ in range(Config.DMPNN_STEPS)
        ])
        
        self.atom_proj = nn.Linear(Config.GRAPH_NODE_DIM, Config.ATOM_FINAL_DIM)
        self.proj_norm = nn.LayerNorm(Config.ATOM_FINAL_DIM)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_normal_(self.atom_embed.weight, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.atom_embed.bias)
        nn.init.kaiming_normal_(self.edge_embed.weight, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.edge_embed.bias)
        nn.init.kaiming_normal_(self.atom_proj.weight, mode='fan_in', nonlinearity='linear')
        nn.init.zeros_(self.atom_proj.bias)
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch
        
        node_feat = self.atom_embed(x)
        node_feat = self.embed_norm(node_feat)
        edge_feat = self.edge_embed(edge_attr) if edge_attr.size(0) > 0 else torch.zeros(0, Config.GRAPH_NODE_DIM, device=x.device)
        
        for layer in self.dmpnn_layers:
            if edge_index.size(1) > 0:
                node_feat = layer(node_feat, edge_index, edge_feat)
        
        atom_feat = self.atom_proj(node_feat)
        atom_feat = self.proj_norm(atom_feat)
        
        num_graphs = int(batch.max().item()) + 1 if atom_feat.size(0) > 0 else 0
        
        if num_graphs > 0:
            global_feats = []
            for b in range(num_graphs):
                indices = (batch == b).nonzero(as_tuple=False).view(-1)
                if indices.numel() > 0:
                    batch_atoms = atom_feat[indices]
                    global_feat = batch_atoms.max(dim=0)[0]
                    global_feats.append(global_feat)
                else:
                    global_feats.append(torch.zeros(Config.ATOM_FINAL_DIM, device=atom_feat.device))
            global_feat = torch.stack(global_feats, dim=0)
        else: 
            global_feat = torch.zeros(1, Config.ATOM_FINAL_DIM, device=atom_feat.device)
        
        return atom_feat, global_feat


class CoAttentionModule(nn.Module):
    """Multi-Head Co-Attention Mechanism"""
    def __init__(self, feature_dim, num_heads=None):
        super().__init__()
        
        self.num_heads = num_heads if num_heads is not None else Config.NUM_ATTENTION_HEADS
        self.feature_dim = feature_dim
        
        assert feature_dim % self.num_heads == 0, \
            f"feature_dim ({feature_dim}) must be divisible by num_heads ({self.num_heads})"
        
        self.head_dim = feature_dim // self.num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Query, Key, Value projections
        self.Wq = nn.Linear(feature_dim, feature_dim)
        self.Wk = nn.Linear(feature_dim, feature_dim)
        self.Wv1 = nn.Linear(feature_dim, feature_dim)
        self.Wv2 = nn.Linear(feature_dim, feature_dim)
        
        # Output projections
        self.out_proj1 = nn.Linear(feature_dim, feature_dim)
        self.out_proj2 = nn.Linear(feature_dim, feature_dim)
        
        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(feature_dim, feature_dim)
        )
        
        self.dropout = nn.Dropout(Config.DROPOUT)
        self.norm = nn.LayerNorm(feature_dim)
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.Wq, self.Wk, self.Wv1, self.Wv2, self.out_proj1, self.out_proj2]:
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='linear')
            nn.init.zeros_(module.bias)
        
        for m in self.fusion_mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def split_heads(self, x):
        """Split features into multiple heads"""
        B, N, D = x.shape
        x = x.view(B, N, self.num_heads, self.head_dim)
        return x.transpose(1, 2)
    
    def combine_heads(self, x):
        """Combine multiple heads back to single dimension"""
        B, num_heads, N, head_dim = x.shape
        x = x.transpose(1, 2)
        return x.contiguous().view(B, N, num_heads * head_dim)
    
    def forward(self, feat1, feat2, mask1=None, mask2=None):
        # Handle 2D input
        squeeze_dim1 = False
        squeeze_dim2 = False
        if feat1.dim() == 2:
            feat1 = feat1.unsqueeze(1)
            squeeze_dim1 = True
            if mask1 is not None: 
                mask1 = mask1.unsqueeze(1)
        
        if feat2.dim() == 2:
            feat2 = feat2.unsqueeze(1)
            squeeze_dim2 = True
            if mask2 is not None: 
                mask2 = mask2.unsqueeze(1)
        
        B, N, D = feat1.shape
        M = feat2.size(1)
        
        # Project to Q, K, V
        Q1 = self.Wq(feat1)
        K2 = self.Wk(feat2)
        V1 = self.Wv1(feat1)
        V2 = self.Wv2(feat2)
        Q2 = self.Wq(feat2)
        K1 = self.Wk(feat1)
        
        # Split into multiple heads
        Q1 = self.split_heads(Q1)
        K2 = self.split_heads(K2)
        V2 = self.split_heads(V2)
        Q2 = self.split_heads(Q2)
        K1 = self.split_heads(K1)
        V1 = self.split_heads(V1)
        
        # Path 1: feat1 attends to feat2
        attn_scores_1to2 = torch.matmul(Q1, K2.transpose(-2, -1)) / self.scale
        
        if mask2 is not None:
            mask2_expanded = mask2.unsqueeze(1).unsqueeze(2)
            attn_scores_1to2 = attn_scores_1to2.masked_fill(mask2_expanded, float('-inf'))
        
        attn_1to2 = F.softmax(attn_scores_1to2, dim=-1)
        attn_1to2_vis = attn_1to2.mean(dim=1).detach()
        attn_1to2 = self.dropout(attn_1to2)
        context_from_feat2 = torch.matmul(attn_1to2, V2)
        context_from_feat2 = self.combine_heads(context_from_feat2)
        context_from_feat2 = self.out_proj1(context_from_feat2)
        
        # Path 2: feat2 attends to feat1
        attn_scores_2to1 = torch.matmul(Q2, K1.transpose(-2, -1)) / self.scale
        
        if mask1 is not None:
            mask1_expanded = mask1.unsqueeze(1).unsqueeze(2)
            attn_scores_2to1 = attn_scores_2to1.masked_fill(mask1_expanded, float('-inf'))
        
        attn_2to1 = F.softmax(attn_scores_2to1, dim=-1)
        attn_2to1_vis = attn_2to1.mean(dim=1).detach()
        attn_2to1 = self.dropout(attn_2to1)
        context_from_feat1 = torch.matmul(attn_2to1, V1)
        context_from_feat1 = self.combine_heads(context_from_feat1)
        context_from_feat1 = self.out_proj2(context_from_feat1)
        
        # Feature Fusion
        feat1_fused = self.fusion_mlp(
            torch.cat([feat1, context_from_feat2], dim=-1)
        )
        feat1_fused = feat1 + feat1_fused
        feat1_fused = self.norm(feat1_fused)
        
        feat2_fused = self.fusion_mlp(
            torch.cat([feat2, context_from_feat1], dim=-1)
        )
        feat2_fused = feat2 + feat2_fused
        feat2_fused = self.norm(feat2_fused)
        
        # Restore original dimensions
        if squeeze_dim1:
            feat1_fused = feat1_fused.squeeze(1)
        if squeeze_dim2:
            feat2_fused = feat2_fused.squeeze(1)
        
        return feat1_fused, feat2_fused, attn_1to2_vis, attn_2to1_vis


class GatedFusion(nn.Module):
    """Gated fusion module"""
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(dim)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.gate: 
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, feat1, feat2):
        concat = torch.cat([feat1, feat2], dim=-1)
        gate = self.gate(concat)
        fused = gate * feat1 + (1 - gate) * feat2
        fused = self.norm(fused)
        return fused


class ContrastiveLoss(nn.Module):
    """Contrastive learning loss"""
    def __init__(self):
        super().__init__()
        self.temperature = Config.CONTRASTIVE_TEMP
        
    def forward(self, feat1, feat2):
        feat1 = F.normalize(feat1, p=2, dim=-1)
        feat2 = F.normalize(feat2, p=2, dim=-1)
        sim_matrix = torch.matmul(feat1, feat2.t()) / self.temperature
        labels = torch.arange(feat1.size(0), device=feat1.device)
        loss = F.cross_entropy(sim_matrix, labels)
        return loss


class DrugProteinInteractionModel(nn.Module):
    """Drug-Protein Interaction Prediction Model"""
    def __init__(self):
        super().__init__()
        
        # Encoders
        self.protein_encoder = ProteinEncoder()
        self.drug_encoder = DrugEncoder()
        self.molecular_graph = MolecularGraph()
        
        # Feature normalization
        self.protein_tokens_norm = nn.LayerNorm(Config.PROTEIN_TOKEN_DIM)
        self.protein_global_norm = nn.LayerNorm(Config.PROTEIN_TOKEN_DIM)
        self.drug_tokens_norm = nn.LayerNorm(Config.DRUG_TOKEN_DIM)
        self.drug_global_norm = nn.LayerNorm(Config.DRUG_TOKEN_DIM)
        self.atom_feat_norm = nn.LayerNorm(Config.ATOM_FINAL_DIM)
        self.drug_struct_norm = nn.LayerNorm(Config.ATOM_FINAL_DIM)
        
        # Projection layers
        self.protein_tokens_proj = nn.Identity()
        self.protein_global_proj = nn.Identity()
        self.drug_tokens_proj = nn.Identity()
        self.drug_global_proj = nn.Identity()
        
        # Co-attention modules
        self.co_attn1 = CoAttentionModule(Config.CROSS_ATTN_DIM)
        self.co_attn2 = CoAttentionModule(Config.CROSS_ATTN_DIM)
        
        # Channel-wise gated fusion
        self.channel_fusion = GatedFusion(Config.CROSS_ATTN_DIM * 2)
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(Config.CROSS_ATTN_DIM * 2, Config.FINAL_DIM),
            nn.LayerNorm(Config.FINAL_DIM),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.FINAL_DIM, Config.FINAL_DIM // 2),
            nn.LayerNorm(Config.FINAL_DIM // 2),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.FINAL_DIM // 2, 1)
        )
        
        # Contrastive learning
        self.contrastive_loss = ContrastiveLoss()
        self._initialize_predictor()
    
    def _initialize_predictor(self):
        for m in self.predictor:
            if isinstance(m, nn.Linear):
                if m.out_features == 1:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                    m.weight.data.mul_(0.1)
                    nn.init.zeros_(m.bias)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(self, drug_graphs, chembert_token_feat, chembert_cls_feat, chembert_token_mask,
                esm2_token_feat, esm2_cls_feat, esm2_token_mask, return_attention=False):
        
        # Protein feature encoding
        protein_tokens, protein_global = self.protein_encoder(
            esm2_token_feat, esm2_token_mask, esm2_cls_feat
        )
        protein_tokens = self.protein_tokens_norm(protein_tokens)
        protein_global = self.protein_global_norm(protein_global)
        protein_tokens_proj = self.protein_tokens_proj(protein_tokens)
        protein_global_proj = self.protein_global_proj(protein_global)
        
        # Drug feature encoding
        drug_tokens, drug_global = self.drug_encoder(
            chembert_token_feat, chembert_token_mask, chembert_cls_feat
        )
        drug_tokens = self.drug_tokens_norm(drug_tokens)
        drug_global = self.drug_global_norm(drug_global)
        drug_tokens_proj = self.drug_tokens_proj(drug_tokens)
        drug_global_proj = self.drug_global_proj(drug_global)
        
        # Drug graph feature encoding
        atom_feat, drug_struct_global = self.molecular_graph(drug_graphs)
        atom_feat = self.atom_feat_norm(atom_feat)
        drug_struct_global = self.drug_struct_norm(drug_struct_global)
        
        attention_dict = {} if return_attention else None
        
        # Interaction Path 1: Atom ↔ Protein tokens
        batch = drug_graphs.batch
        num_graphs = int(batch.max().item()) + 1 if atom_feat.size(0) > 0 else 0
        
        if num_graphs > 0:
            atom_features = []
            max_atoms = 0
            for b in range(num_graphs):
                indices = (batch == b).nonzero(as_tuple=False).view(-1)
                if indices.numel() > 0:
                    feat = atom_feat[indices]
                    atom_features.append(feat)
                    max_atoms = max(max_atoms, feat.size(0))
            
            if max_atoms > 0:
                padded_atoms = []
                atom_masks = []
                for feat in atom_features:
                    num_atoms = feat.size(0)
                    if num_atoms < max_atoms:
                        mask = torch.cat([
                            torch.zeros(num_atoms, dtype=torch.bool, device=feat.device),
                            torch.ones(max_atoms - num_atoms, dtype=torch.bool, device=feat.device)
                        ])
                        pad = torch.zeros(max_atoms - num_atoms, Config.CROSS_ATTN_DIM, device=feat.device)
                        feat = torch.cat([feat, pad], dim=0)
                    else:
                        mask = torch.zeros(max_atoms, dtype=torch.bool, device=feat.device)
                    padded_atoms.append(feat.unsqueeze(0))
                    atom_masks.append(mask.unsqueeze(0))
                
                batched_atoms = torch.cat(padded_atoms, dim=0)
                batched_atom_masks = torch.cat(atom_masks, dim=0)
                
                protein_token_mask = ~esm2_token_mask
                
                drug_inter1, protein_inter1, attn_atom_to_protein, attn_protein_to_atom = self.co_attn1(
                    batched_atoms,
                    protein_tokens_proj,
                    mask1=batched_atom_masks,
                    mask2=protein_token_mask
                )
                
                if return_attention:
                    attention_dict['attn_co1_atom_to_protein'] = attn_atom_to_protein
                    attention_dict['attn_co1_protein_to_atom'] = attn_protein_to_atom
                    attention_dict['atom_mask'] = batched_atom_masks
                    attention_dict['protein_token_mask'] = protein_token_mask
                
                drug_inter1_masked = drug_inter1.clone()
                drug_inter1_masked[batched_atom_masks] = float('-inf')
                drug_inter1_global = drug_inter1_masked.max(dim=1)[0]
                
                protein_mask = esm2_token_mask.unsqueeze(-1).float()
                protein_mask_sum = torch.clamp(protein_mask.sum(dim=1), min=1.0)
                protein_inter1_global = (protein_inter1 * protein_mask).sum(dim=1) / protein_mask_sum
                
                drug_inter1_global = drug_inter1_global + drug_struct_global
                protein_inter1_global = protein_inter1_global + protein_global_proj
            else:
                drug_inter1_global = drug_struct_global
                protein_inter1_global = protein_global_proj
        else:
            drug_inter1_global = drug_struct_global
            protein_inter1_global = protein_global_proj
        
        # Interaction Path 2: Drug tokens ↔ Protein tokens
        drug_token_mask = ~chembert_token_mask
        protein_token_mask = ~esm2_token_mask
        
        drug_inter2, protein_inter2, attn_drug_to_protein, attn_protein_to_drug = self.co_attn2(
            drug_tokens_proj,
            protein_tokens_proj,
            mask1=drug_token_mask,
            mask2=protein_token_mask
        )
        
        if return_attention:
            attention_dict['attn_co2_drugtok_to_protein'] = attn_drug_to_protein
            attention_dict['attn_co2_protein_to_drugtok'] = attn_protein_to_drug
            attention_dict['drug_token_mask'] = drug_token_mask
            attention_dict['protein_token_mask_co2'] = protein_token_mask
        
        drug_mask = chembert_token_mask.unsqueeze(-1).float()
        drug_mask_sum = torch.clamp(drug_mask.sum(dim=1), min=1.0)
        drug_inter2_global = (drug_inter2 * drug_mask).sum(dim=1) / drug_mask_sum
        drug_inter2_global = drug_inter2_global + drug_global_proj
        
        protein_mask = esm2_token_mask.unsqueeze(-1).float()
        protein_mask_sum = torch.clamp(protein_mask.sum(dim=1), min=1.0)
        protein_inter2_global = (protein_inter2 * protein_mask).sum(dim=1) / protein_mask_sum
        protein_inter2_global = protein_inter2_global + protein_global_proj
        
        # Channel-wise concatenation and gated fusion
        channel1_combined = torch.cat([drug_inter1_global, protein_inter1_global], dim=-1)
        channel2_combined = torch.cat([drug_inter2_global, protein_inter2_global], dim=-1)
        combined = self.channel_fusion(channel1_combined, channel2_combined)
        
        # Prediction
        logits = self.predictor(combined).squeeze(-1)
        
        # Contrastive learning
        contrastive_loss = None
        if self.training:
            contrastive_loss = self.contrastive_loss(drug_global_proj, drug_struct_global)
        
        if return_attention:
            return logits, contrastive_loss, attention_dict
        else:
            return logits, contrastive_loss