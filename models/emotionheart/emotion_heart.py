# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
)

from .graphormer_graph_encoder import GraphormerGraphEncoder
from .contrastive_loss import NACL_loss, MIM_loss, infoNCE_loss

logger = logging.getLogger(__name__)
logging.basicConfig(force=True, level=logging.INFO)


@register_model("graphormer")
class EmotionHeartModel(FairseqEncoderModel):
    def __init__(self, args, encoder):
        super().__init__(encoder)
        self.args = args
        self.pretrained_encoder = None
        self.encoder_embed_dim = args.encoder_embed_dim
        self.modalities = args.modalities
        self.n_modalities = len(self.modalities)

        self.encoder = encoder

        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(args.encoder_embed_dim)

        data_embedding_dims = args.dataset_embedding_dims[args.dataset]
        if 'a' in args.modalities:
            self.input_projection_a = nn.Sequential(
                nn.Linear(data_embedding_dims['a'], args.encoder_embed_dim),
            )

        if 't' in args.modalities:
            self.input_projection_t = nn.Sequential(
                nn.Linear(data_embedding_dims['t'], args.encoder_embed_dim),
            )

        if 'v' in args.modalities:
            self.input_projection_v = nn.Sequential(
                nn.Linear(data_embedding_dims['v'], args.encoder_embed_dim),
            )

        if args.do_NACL == True:
            self.NACLloss = NACL_loss(args.temperature)
        if args.do_DGI == True:
            self.DGIloss = MIM_loss(args.encoder_embed_dim, args.temperature)
        if args.do_CLIP == True:
            self.CLIPloss = infoNCE_loss(args.temperature)

        self.linear_fusion = nn.Sequential(
            nn.LayerNorm(self.n_modalities * args.encoder_embed_dim),
            nn.Linear(self.n_modalities * args.encoder_embed_dim, args.encoder_embed_dim),
            self.activation,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Linear(args.encoder_embed_dim, args.num_classes)
        )

    def forward(self, data, n_max_utterances, train=False):
        if self.args.dataset == "iemocap":
            data['utterance_order'] = torch.zeros_like(data['utterance_order'], dtype=torch.long,
                                                       device=data['utterance_order'].device)

        within_modality_loss = 0.
        between_modality_loss = 0.
        cross_entropy_loss = 0.

        mask = data['mask'].clone()
        B, N = mask.shape
        inverted_mask = mask.logical_not()

        sim_mask = mask.unsqueeze(1) | mask.unsqueeze(2)
        data['mask'] = mask.repeat(1, self.n_modalities)

        proj_list = []
        if 'a' in self.modalities:
            proj_list.append(self.input_projection_a(data['audio']))
        if 't' in self.modalities:
            proj_list.append(self.input_projection_t(data['text']))
        if 'v' in self.modalities:
            proj_list.append(self.input_projection_v(data['visual']))

        org_x = torch.stack(proj_list, dim=1).permute(0,2,3,1).contiguous()
        proj = torch.stack(proj_list, dim=1).view(B, -1, self.args.encoder_embed_dim)
        data['x'] = self.layer_norm(proj)

        representation = self.encoder(data, self.modalities)

        if self.args.specific and not self.args.hybrid:
            cls = representation[:, :, 0, :]
            nodes = representation[:, :, 1:, :]  # B, M, N, D
            graphs = nodes.permute(0, 2, 3, 1).contiguous()[inverted_mask, :] # B, M, N, D -> B, N, D, M -> B*valid N, D, M
            fused_emb = nodes.permute(0, 2, 1, 3).contiguous().view(B, N, -1) # B, N, D*M

        else:
            cls = representation[:, 0, :]
            nodes = representation[:, 1:, :]  # real nodes (i.e., utterance tokens)

            fused_emb = nodes.reshape(B, self.n_modalities, N, -1)
            graphs = fused_emb.permute(0, 2, 3, 1).contiguous()[inverted_mask, :]
            fused_emb = fused_emb.permute(0, 2, 1, 3).contiguous().view(B, N, -1)

        if train:
            if self.args.do_CLIP:
                cnt = 0 if self.n_modalities != 1 else 1
                modals = nodes
                if not self.args.specific or self.args.hybrid:
                    modals = nodes.reshape(B, self.n_modalities, N, -1)
                for i in range(self.n_modalities):
                    for j in range(self.n_modalities):
                        if i == j:
                            continue
                        cnt += 1
                        source = modals[:, i, :, :]
                        target = modals[:, j, :, :]

                        between_modality_loss += self.CLIPloss(source, target, sim_mask)
                between_modality_loss /= cnt
                between_modality_loss *= self.args.CLIP_lambda

            if self.args.do_DGI:
                if self.args.specific and not self.args.hybrid:
                    for i in range(self.n_modalities):
                        m_cls = cls[:, i, :]
                        m_embed = nodes[:, i, :, :]
                        within_modality_loss += self.DGIloss(m_cls, m_embed, inverted_mask)
                    within_modality_loss /= self.n_modalities

                else:
                    within_modality_loss += self.DGIloss(cls, nodes, inverted_mask.repeat(1,self.n_modalities))

                within_modality_loss *= self.args.DGI_lambda

            if self.args.do_NACL:
                cnt = 0
                if not self.args.specific or self.args.hybrid:
                    nodes = nodes.reshape(B, self.n_modalities, N, -1)
                for i in range(self.n_modalities):
                    for j in range(self.n_modalities):
                        if i == j:
                            continue
                        cnt += 1
                        source = nodes[:, i, :, :]
                        target = nodes[:, j, :, :]

                        between_modality_loss += self.NACLloss(source, target, sim_mask, self.args.topk,
                                                               self.args.num_classes)
                between_modality_loss /= cnt
                between_modality_loss *= self.args.NACL_lambda

        if self.args.unimodal_inference:
            if self.args.modalities == 'a':
                start_row = 0
                end_row = self.args.encoder_embed_dim
            elif self.args.modalities == 't':
                start_row = self.args.encoder_embed_dim
                end_row = self.args.encoder_embed_dim*2
            elif self.args.modalities == 'v':
                start_row = self.args.encoder_embed_dim*2
                end_row = self.args.encoder_embed_dim*3
            else:
                raise NotImplementedError
            sliced_weight = self.linear_fusion[1].weight[:, start_row:end_row]  # shape: (20, 300)
            sliced_bias = self.linear_fusion[1].bias # shape: (20,)
            fused_emb = F.linear(fused_emb, sliced_weight, sliced_bias)
            fused_emb = self.linear_fusion[2](fused_emb)
        else:
            fused_emb = self.linear_fusion(fused_emb)

        logits = self.classifier(fused_emb)[inverted_mask]
        labels = data['y'][inverted_mask].view(-1)

        if self.args.do_CE:
            class_weights = None

            if self.args.do_WCE:
                class_sample_count = torch.bincount(labels,
                minlength=self.args.num_classes).float()  # non-exist labels will be padded by zero
                class_sample_count[class_sample_count == 0] = float('inf')
                class_weights = 1.0 / class_sample_count  # non-exist labels' weights will be zero

                beta = 0.99
                eps = 1e-8
                class_weights = (1 - beta) / (1 - (beta ** class_sample_count) + eps)
                class_weights = torch.where(class_sample_count > 0, class_weights, torch.zeros_like(class_sample_count))

                class_weights /= class_weights.sum()

            cross_entropy_loss = 0.5*nn.functional.cross_entropy(logits, labels, weight=class_weights)

        # Total Loss
        loss = cross_entropy_loss + within_modality_loss + between_modality_loss

        return loss, logits, labels, graphs, fused_emb[inverted_mask, :], org_x[inverted_mask,:]


class EmotionHeartEncoder(FairseqEncoder):
    def __init__(self, args, n_nodes=None):
        super().__init__(dictionary=None)
        self.args = args
        self.n_modalities = len(args.modalities)

        num_nodes = args.n_max_utterances
        if n_nodes is not None:
            if num_nodes < n_nodes:
                num_nodes = n_nodes

        n_max_speakers = args.n_max_speakers

        def make_graph_encoder(num_nodes, num_speakers, num_degree, num_modalities):
            return GraphormerGraphEncoder(
                num_nodes=num_nodes,
                num_speakers=num_speakers,
                num_degree=num_degree,
                num_edges=args.num_edges,
                num_modalities=num_modalities,
                num_spatial=args.max_dist,
                num_edge_dis=args.num_edge_dis,
                edge_type=args.edge_type,
                multi_hop_max_dist=args.multi_hop_max_dist,
                num_encoder_layers=args.encoder_layers,
                embedding_dim=args.encoder_embed_dim,
                ffn_embedding_dim=args.ffn_embed_dim,
                num_attention_heads=args.encoder_attention_heads,
                dropout=args.dropout,
                attention_dropout=args.attention_dropout,
                activation_dropout=args.act_dropout,
                encoder_normalize_before=args.encoder_normalize_before,
                pre_layernorm=args.pre_layernorm,
                apply_graphormer_init=args.apply_graphormer_init,
                activation_fn=args.activation_fn,
            )

        if args.specific:
            self.modality_encoder = nn.ModuleDict()
            for m in args.modalities:
                self.modality_encoder[m] = make_graph_encoder(
                    num_nodes=num_nodes,
                    num_speakers=n_max_speakers,
                    num_degree=args.num_degree,
                    num_modalities=1,
                )
        else:
            self.graph_encoder = make_graph_encoder(
                num_nodes=None,
                num_speakers=None,
                num_degree=None,
                num_modalities=self.n_modalities,
            )
        if args.hybrid:
            # In the hybrid setting a shared encoder is stacked on top of the
            # modality-specific ones (see forward); with specific=False it
            # replaces the encoder created above.
            self.graph_encoder = make_graph_encoder(
                num_nodes=None,
                num_speakers=None,
                num_degree=args.num_degree,
                num_modalities=self.n_modalities,
            )


    def forward(self, batched_data, modality, perturb=None, masked_tokens=None, **unused):

        graphs = []
        nodes = []

        if self.n_modalities == 1:
            batched_data['modality_position'] = None

        if self.args.specific:
            for i, m in enumerate(modality):
                modality_batched_data = {}

                max_utterances = int(batched_data['x'].shape[1] // self.n_modalities)
                start = i * max_utterances
                end = (i + 1) * max_utterances

                modality_batched_data['x'] = batched_data['x'][:, start:end, :].clone()
                modality_batched_data['mask'] = batched_data['mask'][:, start:end].clone()
                modality_batched_data['utterance_order'] = batched_data['utterance_order'][:, start:end].clone()
                modality_batched_data['speaker_identity'] = batched_data['speaker_identity'][:, start:end].clone()

                modality_batched_data['in_degree'] = batched_data['in_degree'][:, start:end].clone()
                modality_batched_data['out_degree'] = batched_data['out_degree'][:, start:end].clone()

                new_attn_bias = torch.zeros(
                    (batched_data['attn_bias'].shape[0], max_utterances + 1, max_utterances + 1)).to(
                    batched_data['attn_bias'].device)
                new_attn_bias[:, 0, 0] = batched_data['attn_bias'][:, 0, 0].clone()
                new_attn_bias[:, 0, 1:] = batched_data['attn_bias'][:, 0, 1 + start:1 + end].clone()
                new_attn_bias[:, 1:, 0] = batched_data['attn_bias'][:, 1 + start:1 + end, 0].clone()
                new_attn_bias[:, 1:, 1:] = batched_data['attn_bias'][:, 1 + start:1 + end, 1 + start:1 + end].clone()
                modality_batched_data['attn_bias'] = new_attn_bias

                modality_batched_data['attn_edge_type'] = batched_data['attn_edge_type'][:, start:end, start:end,
                                                          :].clone()
                modality_batched_data['spatial_pos'] = batched_data['spatial_pos'][:, start:end, start:end, :].clone()
                modality_batched_data['edge_input'] = batched_data['edge_input'][:, start:end, start:end, :, :].clone()
                modality_batched_data['modality_position'] = None

                inner_states = self.modality_encoder[m](modality_batched_data, perturb=perturb,
                                                        n_modalities=self.n_modalities, use_attn_bias = True)
                inner_states = inner_states[-1].transpose(0, 1)
                graphs.append(inner_states[:, 0, :])
                nodes.append(inner_states[:, 1:, :])

            graphs = torch.stack(graphs, dim=1).unsqueeze(2)  # b, m, 1, e


            nodes = torch.stack(nodes, dim=1)  # b, m, u, e
            b, _, _, e = nodes.shape

            z = torch.cat([graphs, nodes], dim=2)
            if self.args.hybrid:
                batched_data['x'] = nodes.reshape(b, -1, e)
                z = self.graph_encoder(batched_data, perturb=perturb, n_modalities=self.n_modalities)
                z = z[-1].transpose(0, 1)  # b, m*u, e
        else:
            z = self.graph_encoder(batched_data, perturb=perturb, n_modalities=self.n_modalities)
            z = z[-1].transpose(0, 1)  # b, m*u, e

        if masked_tokens is not None:
            raise NotImplementedError

        return z
