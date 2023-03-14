"""
Query-Informed FSS
Extended from ADNet code by Hansen et al.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .encoder import Res101Encoder
from .mln import MLN
from .srn import SRN
import numpy as np


class FewShotSeg(nn.Module):

    def __init__(self, pretrained_weights="deeplabv3"):
        super().__init__()

        # Encoder
        self.encoder = Res101Encoder(replace_stride_with_dilation=[True, True, False],
                                     pretrained_weights=pretrained_weights)
        self.mln = MLN()
        self.srn = SRN()


    def forward(self, supp_imgs, supp_masks, qry_imgs):

        self.n_ways = len(supp_imgs)
        self.n_shots = len(supp_imgs[0])
        batch_size = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]

        ###### Extract features ######
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0), ], dim=0)
        img_fts, tao = self.encoder(imgs_concat)
        img_fts = torch.stack([img_fts[dic].view(self.n_shots + 1, batch_size, -1, *img_fts[dic].shape[-2:])
                               for _, dic in enumerate(img_fts)], dim=0)
        supp_masks = torch.stack([torch.stack([shot.float() for shot in way], dim=0) for way in supp_masks], dim=0)

        loss_mln = self.mln(img_fts[:, :self.n_shots], tao[:self.n_shots, None], supp_masks)
        out_srn = self.srn(img_fts, tao[..., None], supp_masks)

        return out_srn, loss_mln

