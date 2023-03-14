#!/usr/bin/env python
import SimpleITK as sitk
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader

from dataloaders.dataset_specifics import *
from dataloaders.datasets import TestDataset
from models.fewshot import FewShotSeg
from utils import *
import pandas as pd
import torch.nn as nn


def main():

    for epoch_index in range(1, 31):

        _config = {
            'seed': 2021,
            'gpu_id': 0,
            'alpha': 0.9,
            'n_part': 3,
            'n_iters': 7,
            'eval_fold': 0,
            'supp_idx': 4,
            'dataset': '***',
            'test_label': '***',
            'reload_model_path': '***',
            'data_dir': '***'
        }

        # Deterministic setting for reproduciablity.
        if _config['seed'] is not None:
            random.seed(_config['seed'])
            torch.manual_seed(_config['seed'])
            torch.cuda.manual_seed_all(_config['seed'])
            cudnn.deterministic = True

        # Enable cuDNN benchmark mode to select the fastest convolution algorithm.
        cudnn.enabled = True
        cudnn.benchmark = True
        torch.cuda.set_device(device=_config['gpu_id'])
        torch.set_num_threads(1)

        model = FewShotSeg()
        model.cuda()
        model.load_state_dict(torch.load(_config['reload_model_path'], map_location='cpu'))

        data_config = {
            'data_dir': _config['data_dir'],
            'dataset': _config['dataset'],
            'eval_fold': _config['eval_fold'],
            'supp_idx': _config['supp_idx'],
        }
        test_dataset = TestDataset(data_config)
        test_loader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=0,
                                 pin_memory=True,
                                 drop_last=True)

        # Get unique labels (classes).
        labels = get_label_names(_config['dataset'])

        # Loop over classes.
        class_dice = {}
        class_iou = {}

        for label_val, label_name in labels.items():

            # Skip BG class.
            if label_name == 'BG':
                continue
            elif (not np.intersect1d([label_val], _config['test_label'])):
                continue
            print('Test Class: {}'.format(label_name))
            test_dataset.label = label_val

            # Test.
            with torch.no_grad():
                model.eval()

                # Loop through query volumes.
                scores = Scores()
                for i, sample in enumerate(test_loader):
                    # Get support sample + mask for current class.
                    support_sample = test_dataset.getSupport(label=label_val, all_slices=False, N=_config['n_part'])
                    support_sample_sup_image = torch.from_numpy(support_sample['sup_image'])
                    support_sample_sup_label = torch.from_numpy(support_sample['sup_label'])
                    # Unpack support data.
                    support_image = [support_sample_sup_image[[i]].float().cuda() for i in
                                     range(support_sample_sup_image.shape[0])]

                    support_fg_mask = [support_sample_sup_label[[i]].float().cuda() for i in
                                       range(support_sample_sup_image.shape[0])]

                    # Unpack query data.
                    query_image = [sample['query_image'][i].float().cuda() for i in
                                   range(sample['query_image'].shape[0])]

                    query_label = sample['query_label'].long()

                    query_id = sample['id'][0].split('image_')[1][:-len('.nii.gz')]

                    # Compute output.
                    # Match support slice and query sub-chunc
                    query_pred = torch.zeros(query_label.shape[-3:])
                    C_q = sample['query_image'].shape[1]
                    idx_ = np.linspace(0, C_q, _config['n_part'] + 1).astype('int')
                    for sub_chunck in range(_config['n_part']):
                        support_image_s = support_image[sub_chunck]
                        support_image_s = [[shot.float().cuda() for shot in way] for way in support_image_s]
                        support_fg_mask_s = support_fg_mask[sub_chunck]
                        support_fg_mask_s = [[shot.float().cuda() for shot in way] for way in support_fg_mask_s]
                        query_image_s = query_image[0][idx_[sub_chunck]:idx_[sub_chunck + 1]]

                        query_pred_s = []
                        for i in range(query_image_s.shape[0]):
                            _pred_s, _ = model(support_image_s, support_fg_mask_s, [query_image_s[[i]]])

                            query_pred_s.append(_pred_s)
                        query_pred_s = torch.cat(query_pred_s, dim=0)
                        query_pred_s = query_pred_s.argmax(dim=1).cpu()
                        query_pred[idx_[sub_chunck]:idx_[sub_chunck + 1]] = query_pred_s
                    # Record scores.
                    scores.record(query_pred, query_label)


                # Log class-wise results
                class_dice[label_name] = torch.tensor(scores.patient_dice).mean().item()
                class_iou[label_name] = torch.tensor(scores.patient_iou).mean().item()
        print('Final results...')
        print('Mean IoU: {}'.format(class_iou))
        print('Mean Dice: {}'.format(class_dice))
        print('End of validation!')

if __name__ == '__main__':
    main()
