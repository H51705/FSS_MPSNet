import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F

class SRN(nn.Module):

    def __init__(self):
        super(SRN, self).__init__()

        self.scaler = 20.0
        self.device = torch.device('cuda')

    def forward(self, fts, t, sup_masks):

        n_way = len(fts)
        n_shot = len(fts[0]) - 1
        path_num = n_shot - 1
        img_size = fts.shape[-2:]
        c_channel = fts.shape[-3]
        protopytes = torch.zeros(1, path_num, 1, c_channel).to(self.device)
        count_max = 0
        count_min = 0

        for path_index in range(path_num):

            protopyte = torch.add(self.getFeatures(fts[0, path_index], sup_masks[0, path_index]) * (path_num - count_min),
                                   self.getFeatures(fts[0, path_index + 1], sup_masks[0, path_index + 1]) * (count_max + 1)) / n_shot
            protopytes[:, path_index] = protopyte

            count_min = count_min + 1
            count_max = count_max + 1

        qry_preds = torch.stack([torch.stack(
            [self.getPred(fts[way][-1], protopytes[way][shot], t[[-1]])
             for shot in range(path_num)], dim=1) for way in range(n_way)], dim=0).view(n_way, path_num, *img_size)
        qry_preds = F.interpolate(qry_preds, size=sup_masks.shape[-2:], mode='bilinear')

        qry_prototypes = torch.stack(
            [torch.stack([self.getFeatures(fts[way, -1], qry_preds[way, [shot]])
                          for shot in range(path_num)], dim=0) for way in range(n_way)], dim=0)

        prototypes_sim = F.cosine_similarity(protopytes, qry_prototypes, dim=3)

        factor = torch.stack(
            [torch.stack([(1.0 / (torch.sum(prototypes_sim, dim=1) + 1e-5)) * prototypes_sim[way, [shot]]
                          for shot in range(path_num)], dim=0) for way in range(n_way)], dim=0)

        qry_seg = torch.sum(qry_preds * factor, dim=1)
        qry_segs = torch.stack((1.0 - qry_seg, qry_seg), dim=1)

        return qry_segs

    def getFeatures(self, fts, mask):

        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')
        masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) \
                     / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)

        return masked_fts


    def getPred(self, fts, prototype, thresh):

        sim = -F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler
        pred = 1.0 - torch.sigmoid(0.5 * (sim - thresh))

        return pred

















