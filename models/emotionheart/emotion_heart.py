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
    register_model_architecture,
)
from fairseq.modules import (
    LayerNorm,
)
from fairseq.utils import safe_hasattr

from .graphormer_graph_encoder import init_graphormer_params, GraphormerGraphEncoder
from .contrastive_loss import NACL_loss, infoNCE_loss  #

logger = logging.getLogger(__name__)
logging.basicConfig(force=True, level=logging.INFO)

@register_model("graphormer")
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class EmotionHeartModel(FairseqEncoderModel):
    def __init__(self, args, encoder):
        super(EmotionHeartModel, self).__init__()
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
                # self.activation,
                # nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim)
            )

        if 't' in args.modalities:
            self.input_projection_t = nn.Sequential(
                nn.Linear(data_embedding_dims['t'], args.encoder_embed_dim),
                # self.activation,
                # nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim)
            )

        if 'v' in args.modalities:
            self.input_projection_v = nn.Sequential(
                nn.Linear(data_embedding_dims['v'], args.encoder_embed_dim),
                # self.activation,
                # nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim)
            )

        if args.do_NACL == True:
            self.NCALloss = NACL_loss(self.args.temperature)
        if args.do_DGI == True:
            self.DGIloss = None
        if args.do_CLIP == True:
            self.CLIPloss = None
        
        self.linear_fusion = nn.Linear(self.n_modalities, 1)
        self.classifier = None
        if self.n_modalities == 1:
            self.classifier = nn.Linear(args.encoder_embed_dim, args.num_classes)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim//4),
                self.activation,
                nn.Linear(args.encoder_embed_dim//4, args.num_classes)
            )


    def freeze_encoder(self):
        for name, param in self.model.named_parameters():
            # param.requires_grad = False
            # print(f" Layer '{name}' is frozen.")
            if not name.startswith("encoder.graph_encoder.layers.1"):
                param.requires_grad = False
                print(f" Layer '{name}' is frozen.")
            # if not name.startswith("encoder.graph_encoder.layers.3"):
            #     param.requires_grad = False
            #     print(f" Layer '{name}' is frozen.")


    def forward(self, data, n_max_utterances):
        if self.args.dataset =="iemocap":
            data['utterance_order'] = torch.zeros_like(data['utterance_order'], dtype=torch.long, device=data['utterance_order'].device)
        mask = data['mask'].clone()
        sim_mask = mask.unsqueeze(1) | mask.unsqueeze(2)
        data['mask'] = mask.repeat(1, self.n_modalities)

        proj_list = []
        if 'a' in self.modalities:
            proj_list.append(self.input_projection_a(data['audio']))
        if 't' in self.modalities:
            proj_list.append(self.input_projection_t(data['text']))
        if 'v' in self.modalities:
            proj_list.append(self.input_projection_v(data['visual']))

        B = data['y'].shape[0]
        proj = torch.stack(proj_list, dim=1).view(B, -1, self.encoder_embed_dim)
        data['x'] = self.layer_norm(proj)

        representation = self.encoder(data, self.modalities)

        embeddings = representation[:, 1:, :]  # real nodes (i.e., utterance tokens)
        # summary = representation[:, 0, :]  # virtual nodes (i.e., graph tokens)

        # Global-Local Mutual Information Maximization
        
        B, N = embeddings.shape[:2]
        fused_emb = None

        multimodal_NCE_loss =0.

        graphs = list()
        for i in range(self.n_modalities):
            graphs.append(embeddings[:, i*n_max_utterances:(i+1)*n_max_utterances, :])

        cnt = 0
        if self.args.do_NCE:
            for i, m_source in enumerate(graphs):
                for j, m_target in enumerate(graphs):
                    if i == j:
                        continue
                    cnt += 1
                    multimodal_NCE_loss += self.NCALloss(m_source, m_target, sim_mask, self.args.topk, self.args.num_classes)
            multimodal_NCE_loss /= cnt

                # Supervised (Cross Entropy) Loss
        inverted_mask = mask.logical_not()
        # fused_emb = torch.cat(graphs, dim=-1)
        # logits = self.classifier(fused_emb)[inverted_mask].view(-1, self.args.num_classes)

        # (M, B, N, D) -> (B, N, M, D) -> (B, N, MD)
        # fused_emb = torch.stack(graphs).permute(1, 2, 0, 3).contiguous().view(B,n_max_utterances,-1)
        fused_emb = torch.stack(graphs, dim=-1)
        graphs = fused_emb[inverted_mask,:]

        fused_emb = self.linear_fusion(fused_emb).squeeze()

        logits = self.classifier(self.activation(fused_emb))[inverted_mask]
        labels = data['y'][inverted_mask].view(-1)

        if self.args.do_CE:
            class_weights = None
            if self.args.do_WCE:
                # class_sample_count = torch.bincount(labels,
                                                    # minlength=self.args.num_classes).float()  # non-exist labels will be padded by zero
                # class_sample_count[class_sample_count == 0] = float('inf')
                # class_weights = 1.0 / class_sample_count  # non-exist labels' weights will be zero

                # beta = 0.99
                # eps = 1e-8
                # class_weights = (1 - beta) / (1 - (beta ** class_sample_count) + eps)
                # class_weights = torch.where(class_sample_count > 0, class_weights, torch.zeros_like(class_sample_count))
                #
                # class_weights /= class_weights.sum()
                # class_weights = torch.FloatTensor([1 / 0.086747,
                #                           1 / 0.144406,
                #                           1 / 0.227883,
                #                           1 / 0.160585,
                #                           1 / 0.127711,
                #                           1 / 0.252668]).to(logits.device)
                class_weights = torch.FloatTensor([1 / 0.167904,
                                          1 / 0.151342,
                                          1 / 0.169617,
                                          1 / 0.165620,
                                          1 / 0.173615,
                                          1 / 0.171902]).to(logits.device)


            cross_entropy_loss = nn.functional.cross_entropy(logits, labels, weight=class_weights)

        # Total Loss
        loss = cross_entropy_loss + multimodal_NCE_loss*self.args.multimodal_MNA_lambda

        return loss, logits, labels, graphs


class EmotionHeartEncoder(FairseqEncoder):
    def __init__(self, args, args2=None):
        super().__init__(dictionary=None)
        self.args = args
        if args2 == None:
            args2 = args
        self.n_modalities = len(args.modalities)
        data_embedding_dims = args.dataset_embedding_dims[args.dataset]

        num_nodes = args.n_max_utterances if args.n_max_utterances > args2.n_max_utterances else args2.n_max_utterances
        n_max_speakers = args.n_max_speakers if args.n_max_speakers > args2.n_max_speakers else args2.n_max_speakers
        if args.specific:
            self.graph_encoder = nn.ModuleList([None, None, None])
            for m in args.modalities:
                if m == 'a':
                    i = 0
                elif m == 't':
                    i = 1
                elif m == 'v':
                    i = 2
                else:
                    raise NotImplementedError

                self.graph_encoder[i] = GraphormerGraphEncoder(
                    # < for graphormer
                    num_nodes=args.n_max_utterances,
                    num_speakers=args.n_max_speakers,
                    num_degree=args.num_degree,
                    num_edges=args.num_edges,
                    num_spatial=args.max_dist,
                    num_edge_dis=args.num_edge_dis,
                    edge_type=args.edge_type,
                    multi_hop_max_dist=args.multi_hop_max_dist,
                    # >
                    num_encoder_layers=args.encoder_layers,
                    embedding_dim=data_embedding_dims[m],
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

        else:
            self.graph_encoder = GraphormerGraphEncoder(
                # < for graphormer
                num_nodes=num_nodes,
                num_speakers=n_max_speakers,
                num_degree=args.num_degree,
                num_edges=args.num_edges,
                num_modalities=len(args.modalities),
                num_spatial=args.max_dist,
                num_edge_dis=args.num_edge_dis,
                edge_type=args.edge_type,
                multi_hop_max_dist=args.multi_hop_max_dist,
                # >
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


    def forward(self, batched_data, modality, perturb=None, masked_tokens=None, **unused):

        inner_states = self.graph_encoder(batched_data, perturb=perturb,n_modalities =self.n_modalities)

        z = inner_states[-1].transpose(0, 1)


        # project masked tokens only
        if masked_tokens is not None:
            raise NotImplementedError

        return z
