import SimpleITK as sitk
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
from dataloaders.dataset_specifics import *
from dataloaders.datasets import TestDataset5shot
from modules.fewshot_mc23 import FewShotSeg5shot
from utils import *
import pandas as pd
import torch.nn as nn


def main():
    result_dice_all = []
    result_iou_all = []

    for epoch_index in range(1, 51):
        print('************** epoch_index************: ', epoch_index)

        result_dice = []
        result_iou = []
        _config = {
            'seed': 2021,
            'gpu_id': 0,
            'n_part': 3,
            'eval_fold': 0,
            'dataset': '***',
            'test_label': '***',
            'reload_model_path': '***',
            'data_dir': '***'
        }

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

        model = FewShotSeg5shot()
        model.cuda()
        model.load_state_dict(torch.load(_config['reload_model_path'], map_location='cpu'))

        data_config = {
            'data_dir': _config['data_dir'],
            'dataset': _config['dataset'],
            'eval_fold': _config['eval_fold'],
        }
        test_dataset = TestDataset5shot(data_config)
        test_loader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=0,
                                 pin_memory=True,
                                 drop_last=True)

        labels = get_label_names(_config['dataset'])

        class_dice = {}
        class_iou = {}

        for label_val, label_name in labels.items():

            if label_name == 'BG':
                continue
            elif (not np.intersect1d([label_val], _config['test_label'])):
                continue

            print('Test Class: {}'.format(label_name))
            test_dataset.label = label_val

            with torch.no_grad():
                model.eval()
                scores = Scores()

                for i, query_sample in enumerate(test_loader):

                    support_sample = test_dataset.getSupport(label=label_val, all_slices=False, N=_config['n_part'])
                    support_sample_img = torch.from_numpy(support_sample['support_image'])
                    support_sample_lbl = torch.from_numpy(support_sample['support_label'])

                    support_img = [support_sample_img[[i]].float().cuda() for i in
                                     range(support_sample_img.shape[0])]
                    support_lbl = [support_sample_lbl[[i]].float().cuda() for i in
                                       range(support_sample_lbl.shape[0])]

                    query_img = [query_sample['query_image'][i].float().cuda() for i in
                                   range(query_sample['query_image'].shape[0])]
                    query_lbl = query_sample['query_label'].long()
                    query_id = query_sample['id'][0].split('image_')[1][:-len('.nii.gz')]

                    query_pred = torch.zeros(query_lbl.shape[-3:])
                    C_q = query_sample['query_image'].shape[1]
                    idx_ = np.linspace(0, C_q, _config['n_part'] + 1).astype('int')

                    for sub_chunck in range(_config['n_part']):

                        support_img_sub = support_img[sub_chunck]
                        support_img_sub = [[shot.float().cuda() for shot in way] for way in support_img_sub]
                        support_lbl_sub = support_lbl[sub_chunck]
                        support_lbl_sub = [[shot.float().cuda() for shot in way] for way in support_lbl_sub]

                        query_img_sub = query_img[0][idx_[sub_chunck]:idx_[sub_chunck + 1]]

                        query_pred_sub = []
                        for i in range(query_img_sub.shape[0]):
                            MLN_loss, query_pred_subsub = model(support_img_sub, support_lbl_sub, [query_img_sub[[i]]])

                            query_pred_sub.append(query_pred_subsub)
                        query_pred_sub = torch.cat(query_pred_sub, dim=0)
                        query_pred_sub = query_pred_sub.argmax(dim=1).cpu()
                        query_pred[idx_[sub_chunck]:idx_[sub_chunck + 1]] = query_pred_sub

                    scores.record(query_pred, query_lbl)

                class_dice[label_name] = torch.tensor(scores.patient_dice).mean().item()
                class_iou[label_name] = torch.tensor(scores.patient_iou).mean().item()
                print('Mean class Dice: {}'.format(class_dice[label_name]))
                print('Mean class IoU: {}'.format(class_iou[label_name]))

                result_dice.append(class_dice[label_name])
                result_iou.append(class_iou[label_name])

        result_dice_all.append(result_dice)
        result_iou_all.append(result_iou)

    print('write to file...')
    result_dice = pd.DataFrame(result_dice_all, columns=['***'])
    result_dice.to_csv("dice.csv", index=False, sep=',')
    result_iou = pd.DataFrame(result_iou_all, columns=['***'])
    result_iou.to_csv("iou.csv", index=False, sep=',')


if __name__ == '__main__':
    main()
