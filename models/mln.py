import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F


class MLN(nn.Module):

    def __init__(self):
        super(MLN, self).__init__()

        self.scaler = 20.0
        self.device = torch.device('cuda')
        self.criterion = nn.NLLLoss()

    def forward(self, sup_fts, t_sup, sup_masks):

        n_way = len(sup_fts)
        n_shot = len(sup_fts[0]) - 1
        n_image = sup_fts[0].shape[0]
        img_size = sup_fts[0][0].shape[-2:]
        bce_loss = torch.zeros(1).to(self.device)

        for path_index in range(n_image):

            fake_qry_fts = sup_fts[:, [path_index]]

            fake_qry_masks = sup_masks[0][[path_index]]
            fake_qry_masks = torch.cat([fake_qry_masks.long() for fake_qry_masks in fake_qry_masks], dim=0)

            fake_sup_fts = torch.cat((sup_fts[:, :path_index], sup_fts[:, (path_index + 1):]), dim=1)

            fake_sup_masks = torch.cat((sup_masks[:, :path_index], sup_masks[:, (path_index + 1):]), dim=1)

            fake_sup_prototypes = torch.stack([torch.stack([self.getFeatures(fake_sup_fts[way, shot], fake_sup_masks[way, shot])
                                    for shot in range(n_shot)], dim=0) for way in range(n_way)], dim=0)

            fake_qry_preds = torch.stack([torch.stack(
                [self.getPred(fake_qry_fts[way][0], fake_sup_prototypes[way][shot], t_sup[[path_index]])
                 for shot in range(n_shot)], dim=1) for way in range(n_way)], dim=0).view(1, n_shot, *img_size)
            fake_qry_preds = F.interpolate(fake_qry_preds, size=fake_sup_masks.shape[-2:], mode='bilinear')

            fake_qry_prototypes = torch.stack([torch.stack([self.getFeatures(fake_qry_fts[0, 0], fake_qry_preds[way, [shot]])
                                    for shot in range(n_shot)], dim=0) for way in range(n_way)], dim=0)

            prototypes_sim = F.cosine_similarity(fake_sup_prototypes, fake_qry_prototypes, dim=3)

            factor = torch.stack([torch.stack([(1.0 / (torch.sum(prototypes_sim, dim=1) + 1e-5)) * prototypes_sim[way, [shot]]
                       for shot in range(n_shot)], dim=0) for way in range(n_way)], dim=0)

            fake_qry_seg = torch.sum(fake_qry_preds * factor, dim=1)
            fake_qry_segs = torch.stack((1.0 - fake_qry_seg, fake_qry_seg), dim=1)

            fake_qry_loss = self.criterion(torch.log(torch.clamp(fake_qry_segs, torch.finfo(torch.float32).eps,
                                                         1 - torch.finfo(torch.float32).eps)), fake_qry_masks)

            bce_loss = bce_loss + fake_qry_loss

        return bce_loss


    def getFeatures(self, fts, mask):

        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')  # 1 c H W
        masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) \
                     / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)

        return masked_fts


    def getPred(self, fts, prototype, thresh):

        sim = -F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler
        pred = 1.0 - torch.sigmoid(0.5 * (sim - thresh))

        return pred