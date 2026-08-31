
import torch
import torch.nn as nn
import torch.nn.functional as F

class MIM_loss(nn.Module):
    def __init__(self, n_h, temperature=1.0):
        super(MIM_loss, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        self.temperature = temperature
        self.BCEloss = nn.BCEWithLogitsLoss(reduction='none')

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # def forward(self, c, embed, positive_valid_mask):
    #     B, N, _ = embed.shape
    #     device = embed.device
    #
    #     # L2 정규화
    #     c = F.normalize(c, p=2, dim=-1)
    #     embed = F.normalize(embed, p=2, dim=-1)
    #
    #     # 모든 노드와 모든 그래프 요약 간의 점수(코사인 유사도) 계산
    #     all_logits = torch.einsum('bnf,df->bnd', embed, c)
    #
    #     # Temperature 적용
    #     all_logits /= self.temperature
    #
    #     # CrossEntropyLoss를 위한 형태로 변환 (B*N, B)
    #     all_logits = all_logits.view(-1, B)
    #
    #     # 정답 레이블 생성
    #     labels = torch.arange(B, device=device).repeat_interleave(N)
    #
    #     # 손실 계산 (CrossEntropyLoss는 Softmax와 log를 모두 포함)
    #     loss = F.cross_entropy(all_logits, labels, reduction='none')
    #
    #     # 유효한 노드에 대해서만 마스킹
    #     valid_mask = positive_valid_mask.view(-1)
    #     masked_loss = loss[valid_mask]
    #
    #     return torch.mean(masked_loss) if masked_loss.numel() > 0 else torch.tensor(0.0).to(device)


    # def forward(self, c, embed, positive_valid_mask):
    #     B, N, _ = embed.shape
    #     device = embed.device
    #
    #     # ❗❗❗ L2 정규화(Normalization) 추가 ❗❗❗
    #     # 각 임베딩 벡터의 크기를 1로 만들어 줍니다.
    #     c = F.normalize(c, p=2, dim=-1)
    #     embed = F.normalize(embed, p=2, dim=-1)
    #
    #     # 1. 모든 노드와 모든 그래프 요약 간의 점수(코사인 유사도) 계산
    #     #    (이제 내적 연산이 코사인 유사도와 같아짐)
    #     all_logits = torch.einsum('bnf,df->bnd', embed, c)
    #
    #     # 2. Temperature 적용
    #     all_logits /= self.temperature
    #
    #     # 3. CrossEntropyLoss를 위한 형태로 변환
    #     all_logits = all_logits.view(-1, B)
    #
    #     # 4. 정답 레이블 생성
    #     labels = torch.arange(B, device=device).repeat_interleave(N)
    #
    #     # 5. 손실 계산
    #     loss = F.cross_entropy(all_logits, labels, reduction='none')
    #
    #     # 6. 유효한 노드에 대해서만 손실을 평균내기 위한 마스크 적용
    #     valid_mask = positive_valid_mask.view(-1)
    #     masked_loss = loss[valid_mask]
    #
    #     return torch.mean(masked_loss) if masked_loss.numel() > 0 else torch.tensor(0.0).to(device)

    def forward(self, c, embed, positive_valid_mask):

        B, N,_ = embed.shape
        device = embed.device

        # 1. 긍정 쌍 점수 계산 (기존과 동일)
        c_expanded = c.unsqueeze(1).expand_as(embed)
        positive_logits = torch.squeeze(self.f_k(embed, c_expanded), -1)  # Shape: (B, N)

        # 2. 부정 쌍 샘플링 (배치를 섞는 방식)
        # 배치 내에서 순서를 섞어 '잘못된' 노드 임베딩 생성
        shuffled_indices = torch.randperm(B).to(device)
        negative_embed = embed[shuffled_indices]

        # 3. 부정 쌍 점수 계산
        negative_logits = torch.squeeze(self.f_k(negative_embed, c_expanded), -1)  # Shape: (B, N)

        # 4. 긍정/부정 점수 및 레이블 통합
        # 긍정 점수와 부정 점수를 하나로 합침
        all_logits = torch.cat((positive_logits, negative_logits), dim=0)  # Shape: (2*B, N)
        all_logits /= self.temperature

        # 긍정 레이블(1)과 부정 레이블(0) 생성
        positive_labels = torch.ones_like(positive_logits)
        negative_labels = torch.zeros_like(negative_logits)
        all_labels = torch.cat((positive_labels, negative_labels), dim=0)  # Shape: (2*B, N)

        # 5. 손실 계산
        loss = self.BCEloss(all_logits, all_labels)

        # 6. 유효한 노드에 대해서만 손실을 평균내기 위한 마스크 적용
        valid_mask = positive_valid_mask.repeat(2, 1)  # 긍정/부정 쌍 모두에 마스크 적용

        # 마스크가 True인 위치의 loss 값만 가져와 평균 계산
        masked_loss = loss[valid_mask]

        return torch.mean(masked_loss) if masked_loss.numel() > 0 else torch.tensor(0.0).to(device)


class infoNCE_loss(nn.Module):
    def __init__(self, temperature):
        super(infoNCE_loss, self).__init__()
        self.temperature = temperature

    def batch_sim(self, z1: torch.Tensor, z2: torch.Tensor,eps=1e-12):

        # Normalize the vectors to avoid repeated computation of norms
        z1_norm = z1 / (torch.norm(z1, dim=-1, keepdim=True)+eps)  # Shape: (B, N, F)
        z2_norm = z2 / (torch.norm(z2, dim=-1, keepdim=True)+eps)  # Shape: (B, N, F)

        # Compute the cosine similarity using batch matrix multiplication
        cosine_similarity = torch.bmm(z1_norm, z2_norm.transpose(1, 2))  # Shape: (B, N, N)

        return cosine_similarity

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, mask, eps=1e-12):
        f = lambda x: torch.exp(x / self.temperature)

        # Clone하여 새로운 텐서 생성 (in-place 연산 방지)
        within_sim = f(self.batch_sim(z1, z1)).clone()
        between_sim = f(self.batch_sim(z1, z2)).clone()

        # in-place 연산 대신 `masked_fill_()` 사용
        within_sim = within_sim.masked_fill(mask, 0.0)
        between_sim = between_sim.masked_fill(mask, 0.0)

        numerator = between_sim.diagonal(dim1=-2, dim2=-1)
        denominator = between_sim.sum(dim=-1) + within_sim.sum(dim=-1) - within_sim.diagonal(dim1=-2, dim2=-1)

        loss = -torch.log((numerator + eps) / (denominator + eps))

        return torch.nanmean(loss)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, mask):
        ret = self.semi_loss(z1, z2, mask)
        return ret


class NACL_loss(nn.Module):
    def __init__(self, temperature):
        super(NACL_loss, self).__init__()
        self.temperature = temperature


    def topK_masks(self, z, mask,n_classes, k=3):
        """
        Returns masks indicating the positions of top-K positive and negative samples,
        with adjustments for cases where the number of valid samples is less than K.

        Args:
            sim (torch.Tensor): (B, N, N) similarity matrix.
            mask (torch.Tensor): (B, N, N) padding mask (True = ignored values).
            k (int): Number of top-K nearest neighbors.

        Returns:
            tuple(torch.Tensor, torch.Tensor):
                - positive_mask: (B, N, N) mask for positive pairs.
                - negative_mask: (B, N, N) mask for negative pairs.
        """
        _sim = torch.cdist(z,z, p=2)
        B, N, _ = _sim.shape
        sim = _sim.clone()  # sim dist and  Avoid modifying the original similarity matrix

        self_mask = torch.eye(N, device=sim.device).bool().unsqueeze(0)  # (1, N, N), self-similarity mask
        all_mask = mask | self_mask

        sim.masked_fill_(all_mask, float('inf'))  # remove self similarity and padded one

        # ✅ Count valid samples in each batch (excluding padding)
        valid_sample_count = (~all_mask).sum(dim=-1).max(dim=-1).values # (B,) number of non-padding samples per batch

        # ✅ Dynamic K: Adjust top-K for batches with fewer valid samples
        dynamic_k = torch.where(valid_sample_count <= k, 1, k)

        # ✅ Initialize positive_mask
        positive_mask = torch.zeros((B, N, N), dtype=torch.bool, device=sim.device)  # (B, N, N)

        for b in range(B):
            if valid_sample_count[b] == 0:
                continue
            top_k_indices = torch.topk(sim[b], dynamic_k[b].item(), dim=-1, largest=False).indices  # (N, dynamic_k)
            positive_mask[b].scatter_(1, top_k_indices, True)  # Mark top-K positions as True

        positive_mask.masked_fill_(all_mask, False)

        # ✅ Create Negative Mask (excluding padding, self-similarity, and positive samples)
        negative_mask = ~(all_mask | positive_mask)  # (B, N, N)
        negative_mask.masked_fill_(self_mask, False)

        return positive_mask, negative_mask


    def batch_sim(self, z1: torch.Tensor, z2: torch.Tensor,eps=1e-12):

        # Normalize the vectors to avoid repeated computation of norms
        z1_norm = z1 / (torch.norm(z1, dim=-1, keepdim=True)+eps)  # Shape: (B, N, F)
        z2_norm = z2 / (torch.norm(z2, dim=-1, keepdim=True)+eps)  # Shape: (B, N, F)

        # Compute the cosine similarity using batch matrix multiplication
        cosine_similarity = torch.bmm(z1_norm, z2_norm.transpose(1, 2))  # Shape: (B, N, N)

        return cosine_similarity


    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, z1_all_mask, k, n_classes,eps=1e-12, mode=None):
        f = lambda x: torch.exp(x / self.temperature)
        within_sim = self.batch_sim(z2, z2)

        within_sim_positive_mask, within_sim_negative_mask = self.topK_masks(z1, z1_all_mask, n_classes, k=k)

        within_sim = f(within_sim)

        within_positive_sim = within_sim*within_sim_positive_mask

        within_negative_sim = within_sim*within_sim_negative_mask

        numerator = within_positive_sim.sum(dim=-1)
        denominator = numerator + within_negative_sim.sum(dim=-1)

        loss = -torch.log((numerator+eps)/(denominator+eps))

        return torch.nanmean(loss)


    def forward(self, z1: torch.Tensor, z2: torch.Tensor, z1_all_mask, k, n_classes, mode=None):
        return self.semi_loss(z1, z2, z1_all_mask, k, n_classes, mode=mode)
