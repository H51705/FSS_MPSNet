import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .encoder import Res101EncoderMC23
from .MLN import MLN
from .SRN import SRN
import numpy as np


class FewShotSeg5shot(nn.Module):

    def __init__(self, pretrained_weights="deeplabv3"):
        super().__init__()

        self.encoder = Res101EncoderMC23(replace_stride_with_dilation=[True, True, False],
                                     pretrained_weights=pretrained_weights)
        self.MLN = MLN()
        self.SRN = SRN()

    def forward(self, sup_imgs, sup_masks, qry_imgs):
        """
        Args:
            sup_imgs: support images way x shot x [B x 3 x H x W], list of lists of tensors  1 5 1 3 H W
            sup_masks: masks for support images way x shot x [B x H x W], list of lists of tensors  1 5 1 H W
            qry_imgs: query images N x [B x 3 x H x W], list of tensors  1 1 3 H W
        """

        self.n_way = len(sup_imgs)
        self.n_shot = len(sup_imgs[0])
        batch_size = sup_imgs[0][0].shape[0]

        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in sup_imgs]
                                + [torch.cat(qry_imgs, dim=0), ], dim=0)
        img_fts, tao = self.encoder(imgs_concat)

        img_fts = torch.stack([img_fts[dic].view(self.n_shot + 1, batch_size, -1, *img_fts[dic].shape[-2:])
                               for _, dic in enumerate(img_fts)], dim=0)

        sup_masks = torch.stack([torch.stack([shot.float() for shot in way], dim=0) for way in sup_masks], dim=0)

        # MLN
        """
        Args:
            sup_fts: support features for 5 support images, shape: 1 5 1 C h w
            sup_t: threshold(t) for support images, shape: 5 1 1
            sup_masks: support masks, shape: 1 5 1 H W
        """

        MLN_loss = self.MLN(img_fts[:, :self.n_shot], tao[:self.n_shot, None], sup_masks)

        # SRN
        """
        Args:
            fts: features(fts) for 5 support images and 1 query image, shape: 1 6 1 C h w
            t: threshold(t) for 5 support images and 1 query image, shape: 6 1 1
            sup_masks: masks for 5 support images, shape: 1 5 1 H W
        """

        qry_segs = self.SRN(img_fts, tao[..., None], sup_masks)  # 1 2 H W

        return MLN_loss, qry_segs



