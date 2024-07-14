import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F


# Multi-shot Learning Network
class MLN(nn.Module):

    def __init__(self):
        super(MLN, self).__init__()

        self.scaler = 20.0
        self.device = torch.device('cuda')
        self.criterion = nn.NLLLoss()

    """
    Args:
        sup_fts: support features for 5 support images, shape: 1 5 1 C h w
        sup_t: threshold(t) for support images, shape: 5 1 1
        sup_masks: support masks, shape: 1 5 1 H W
    """

    def forward(self, sup_fts, sup_t, sup_masks):

        n_way = len(sup_fts)
        n_shot = len(sup_fts[0]) - 1
        n_image = sup_fts[0].shape[0]
        img_size = sup_fts.shape[-2:]

        MLN_loss = torch.zeros(1).to(self.device)

        for path_index in range(n_image):

            fake_qry_fts = sup_fts[:, [path_index]]
            fake_qry_masks = sup_masks[0][[path_index]]
            fake_qry_masks = torch.cat([fake_qry_masks.long() for fake_qry_masks in fake_qry_masks], dim=0)

            fake_sup_fts = torch.cat((sup_fts[:, :path_index], sup_fts[:, (path_index + 1):]), dim=1)
            fake_sup_masks = torch.cat((sup_masks[:, :path_index], sup_masks[:, (path_index + 1):]), dim=1)

            fake_sup_prototypes = torch.stack([torch.stack([self.getPrototypes(fake_sup_fts[way, shot],
                fake_sup_masks[way, shot]) for shot in range(n_shot)], dim=0) for way in range(n_way)], dim=0)

            ######################## Prototype Contrastive Learning Module (PCLM) ########################

            fake_qry_preds = torch.stack([torch.stack([self.getPred(fake_qry_fts[way][0],
                  fake_sup_prototypes[way][shot], sup_t[[path_index]]) for shot in range(n_shot)], dim=1) for way in
                                          range(n_way)], dim=0).view(1, n_shot, *img_size)
            fake_qry_preds = F.interpolate(fake_qry_preds, size=fake_qry_masks.shape[-2:], mode='bilinear')

            fake_qry_prototypes = torch.stack([torch.stack([self.getPrototypes(fake_qry_fts[0, 0], fake_qry_preds[way, [shot]])
                                    for shot in range(n_shot)], dim=0) for way in range(n_way)], dim=0)

            prototypes_sim = F.cosine_similarity(fake_sup_prototypes, fake_qry_prototypes, dim=3)

            contribution_factor = torch.stack([torch.stack([(1.0 / (torch.sum(prototypes_sim, dim=1) + 1e-5)) *
                    prototypes_sim[way, [shot]] for shot in range(n_shot)], dim=0) for way in range(n_way)], dim=0)

            fake_qry_seg = torch.sum(fake_qry_preds * contribution_factor, dim=1)
            fake_qry_segs = torch.stack((1.0 - fake_qry_seg, fake_qry_seg), dim=1)

            fake_qry_loss = self.criterion(torch.log(torch.clamp(fake_qry_segs, torch.finfo(torch.float32).eps,
                                                         1 - torch.finfo(torch.float32).eps)), fake_qry_masks)

            MLN_loss = MLN_loss + fake_qry_loss

        return MLN_loss


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


