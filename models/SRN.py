import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F


# Semantic Reasoning Network
class SRN(nn.Module):

    def __init__(self):
        super(SRN, self).__init__()

        self.scaler = 20.0
        self.device = torch.device('cuda')

    """
    Args:
        fts: features(fts) for 5 support images and 1 query image, shape: 1 6 1 C h w
        t: threshold(t) for 5 support images and 1 query image, shape: 6 1 1
        sup_masks: masks for 5 support images, shape: 1 5 1 H W
    """

    def forward(self, fts, t, sup_masks):

        n_way = len(fts)
        n_shot = len(fts[0]) - 1
        path_num = n_shot - 1
        img_size = fts.shape[-2:]
        channel = fts.shape[-3]

        prior_protopytes = torch.zeros(1, 4, 1, channel).to(self.device)
        count_max = 0
        count_min = 0

        for path_index in range(path_num):

            protopyte = torch.add(self.getPrototypes(fts[0, path_index], sup_masks[0, path_index])
                                  * (path_num - count_min), self.getPrototypes(fts[0, path_index + 1], sup_masks[0,
                                path_index + 1]) * (count_max + 1)) / n_shot
            prior_protopytes[:, path_index] = protopyte

            count_min = count_min + 1
            count_max = count_max + 1

        ######################## Prototype Contrastive Learning Module (PCLM) ########################

        qry_preds = torch.stack([torch.stack([self.getPred(fts[way][-1], prior_protopytes[way][shot], t[[-1]])
             for shot in range(path_num)], dim=1) for way in range(n_way)], dim=0).view(n_way, path_num,
                *img_size)
        qry_preds = F.interpolate(qry_preds, size=sup_masks.shape[-2:], mode='bilinear')

        qry_prototypes = torch.stack(
            [torch.stack([self.getPrototypes(fts[way, -1], qry_preds[way, [shot]])
                          for shot in range(path_num)], dim=0) for way in range(n_way)], dim=0)

        prototypes_sim = F.cosine_similarity(prior_protopytes, qry_prototypes, dim=3)

        contribution_factor = torch.stack(
            [torch.stack([(1.0 / (torch.sum(prototypes_sim, dim=1) + 1e-5)) * prototypes_sim[way, [shot]]
                          for shot in range(path_num)], dim=0) for way in range(n_way)], dim=0)

        qry_seg = torch.sum(qry_preds * contribution_factor, dim=1)
        qry_segs = torch.stack((1.0 - qry_seg, qry_seg), dim=1)

        return qry_segs


    def getPrototypes(self, fts, mask):
        """
        Args:
            fts: input features, expect shape: 1 C h w
            mask: binary mask, expect shape: 1 H W
        """

        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')
        prototypes = torch.sum(fts * mask[None, ...], dim=(-2, -1)) / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)

        return prototypes


    def getPred(self, fts, prototype, threshold):
        """
        Args:
            fts: input features, expect shape: 1 C h w
            prototype: prototype of one semantic class, expect shape: 1 C
            threshold: expect shape: 1 1 1
        """

        sim = -F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler
        pred = 1.0 - torch.sigmoid(0.5 * (sim - threshold))

        return pred

