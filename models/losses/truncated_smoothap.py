# Implemented as per "Recall@k Surrogate Loss with Large Batches and Similarity Mixup" paper
# but only the fixed number of the closest positives is considered

import numpy as np
import torch

from models.losses.loss_utils import sigmoid, compute_aff


class TruncatedSmoothAP:
    def __init__(self, tau1: float = 0.01, similarity: str = 'cosine', positives_per_query: int = 4):
        # We reversed the notation compared to the paper (tau1 is sigmoid on similarity differences)
        # tau1: sigmoid temperature applied on similarity differences
        # positives_per_query: number of positives per query to consider
        # negatives_only: if True in denominator we consider positives and negatives; if False we consider all elements
        #                 (with except to the anchor itself)

        self.tau1 = tau1
        self.similarity = similarity
        self.positives_per_query = positives_per_query

    def __call__(self, embeddings, positives_mask, negatives_mask):
        device = embeddings.device

        positives_mask = positives_mask.to(device)
        negatives_mask = negatives_mask.to(device)

        # Ranking of the retrieval set
        # For each element we ignore elements that are neither positives nor negatives

        # Compute cosine similarity scores
        # 1st dimension corresponds to q, 2nd dimension to z
        s_qz = compute_aff(embeddings, similarity=self.similarity)

        # Find the positives_per_query closest positives for each query
        s_positives = s_qz.detach().clone()
        s_positives.masked_fill_(torch.logical_not(positives_mask), np.NINF)
        #closest_positives_ndx = torch.argmax(s_positives, dim=1).view(-1, 1)  # Indices of closests positives for each query
        closest_positives_ndx = torch.topk(s_positives, k=self.positives_per_query, dim=1, largest=True, sorted=True)[1]
        # closest_positives_ndx is (batch_size, positives_per_query)  with positives_per_query closest positives
        # per each batch element

        n_positives = positives_mask.sum(dim=1)     # Number of positives for each anchor

        # Compute the rank of each example x with respect to query element q as per Eq. (2)
        s_diff = s_qz.unsqueeze(1) - s_qz.gather(1, closest_positives_ndx).unsqueeze(2)
        s_sigmoid = sigmoid(s_diff, temp=self.tau1)

        # Compute the nominator in Eq. 2 and 5 - for q compute the ranking of each of its positives with respect to other positives of q
        # Filter out z not in Positives
        pos_mask = positives_mask.unsqueeze(1)
        pos_s_sigmoid = s_sigmoid * pos_mask

        # Filter out z on the same position as the positive (they have value = 0.5, as the similarity difference is zero)
        mask = torch.ones_like(pos_s_sigmoid).scatter(2, closest_positives_ndx.unsqueeze(2), 0.)
        pos_s_sigmoid = pos_s_sigmoid * mask

        # Compute the rank for each query and its positives_per_query closest positive examples with respect to other positives
        r_p = torch.sum(pos_s_sigmoid, dim=2) + 1.
        # r_p is (batch_size, positives_per_query) matrix

        # Consider only positives and negatives in the denominator
        # Compute the denominator in Eq. 5 - add sum of Indicator function for negatives (or non-positives)
        neg_mask = negatives_mask.unsqueeze(1)
        neg_s_sigmoid = s_sigmoid * neg_mask
        r_omega = r_p + torch.sum(neg_s_sigmoid, dim=2)

        # Compute R(i, S_p) / R(i, S_omega) ration in Eq. 2
        r = r_p / r_omega

        # Compute metrics              mean ranking of the positive example, recall@1
        stats = {}
        # Mean number of positives per query
        stats['positives_per_query'] = n_positives.float().mean(dim=0).item()
        # Mean ranking of selected positive examples (closests positives)
        temp = s_diff.detach() > 0
        temp = torch.logical_and(temp[:, 0], negatives_mask)        # Take the best positive
        hard_ranking = temp.sum(dim=1)
        stats['best_positive_ranking'] = hard_ranking.float().mean(dim=0).item()
        # Recall at 1
        stats['recall'] = {1: (hard_ranking <= 1).float().mean(dim=0).item()}

        # r is (N, positives_per_query) tensor
        # Zero entries not corresponding to real positives - this happens when the number of true positives is lower than positives_per_query
        valid_positives_mask = torch.gather(positives_mask, 1, closest_positives_ndx)   # () tensor
        masked_r = r * valid_positives_mask
        n_valid_positives = valid_positives_mask.sum(dim=1)

        # Filter out rows (queries) without any positive to avoid division by zero
        valid_q_mask = n_valid_positives > 0
        masked_r = masked_r[valid_q_mask]

        ap = (masked_r.sum(dim=1) / n_valid_positives[valid_q_mask]).mean()
        loss = 1. - ap

        stats['loss'] = loss.item()
        stats['ap'] = ap.item()
        stats['avg_embedding_norm'] = embeddings.norm(dim=1).mean().item()
        return loss, stats
