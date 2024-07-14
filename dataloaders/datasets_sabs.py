from . import image_transforms as myit
import torch
from torch.utils.data import Dataset
import torchvision.transforms as deftfx
import glob
import os
import SimpleITK as sitk
import random
import numpy as np
from .dataset_specifics import *


class TestDataset5shot(Dataset):

    def __init__(self, args):

        if args['dataset'] == 'CMR':
            self.image_dirs = glob.glob(os.path.join(args['data_dir'], 'cmr_MR_normalized/image*'))
        elif args['dataset'] == 'CHAOST2':
            self.image_dirs = glob.glob(os.path.join(args['data_dir'], 'chaos_MR_T2_normalized/image*'))
        elif args['dataset'] == 'SABS':
            self.image_dirs = glob.glob(os.path.join(args['data_dir'], 'sabs_CT_normalized/image*'))

        self.image_dirs = sorted(self.image_dirs, key=lambda x: int(x.split('_')[-1].split('.nii.gz')[0]))

        self.FOLD = get_folds(args['dataset'])
        self.image_dirs = [elem for idx, elem in enumerate(self.image_dirs) if idx in self.FOLD[args['eval_fold']]]

        self.label = None

    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, pat_idx):

        query_img_path = self.image_dirs[pat_idx]

        query_img = sitk.GetArrayFromImage(sitk.ReadImage(query_img_path))
        query_img = (query_img - query_img.mean()) / query_img.std()
        query_img = np.stack(3 * [query_img], axis=1)

        query_lbl = sitk.GetArrayFromImage(
            sitk.ReadImage(query_img_path.split('image_')[0] + 'label_' + query_img_path.split('image_')[-1]))

        query_lbl[query_lbl == 200] = 1
        query_lbl[query_lbl == 500] = 2
        query_lbl[query_lbl == 600] = 3
        query_lbl = 1 * (query_lbl == self.label)

        query_sample = {'id': query_img_path}

        idx = query_lbl.sum(axis=(1, 2)) > 0
        query_sample['query_image'] = torch.from_numpy(query_img[idx])
        query_sample['query_label'] = torch.from_numpy(query_lbl[idx])

        self.support_dir = self.image_dirs.copy()
        index = np.arange(len(self.support_dir))
        self.support_dir.pop(index[pat_idx])

        return query_sample

    def get_support_index(self, n_chunck, C):

        if n_chunck == 1:
            pcts = [0.5]
        else:
            half_part = 1 / (n_chunck * 2)
            part_interval = (1.0 - 1.0 / n_chunck) / (n_chunck - 1)
            pcts = [half_part + part_interval * ii for ii in range(n_chunck)]

        return (np.array(pcts) * C).astype('int')

    def getSupport(self, label=None, all_slices=False, N=None):
        print('label: ', label)

        if label is None:
            raise ValueError('Need to specify label class!')

        support_sample = {}
        support_img = []
        support_lbl = []

        for support_pat_idx in range(len(self.support_dir)):
            img_path = self.support_dir[support_pat_idx]
            img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
            img = (img - img.mean()) / img.std()
            img = np.stack(3 * [img], axis=1)

            lbl = sitk.GetArrayFromImage(
                sitk.ReadImage(img_path.split('image_')[0] + 'label_' + img_path.split('image_')[-1]))

            lbl[lbl == 200] = 1
            lbl[lbl == 500] = 2
            lbl[lbl == 600] = 3
            lbl = 1 * (lbl == label)

            if all_slices:
                support_img.append(img)
                support_lbl.append(lbl)
            else:
                if N is None:
                    raise ValueError('Need to specify number of labeled slices!')

                idx = lbl.sum(axis=(1, 2)) > 0
                idx_ = self.get_support_index(N, idx.sum())
                support_img.append(img[idx][idx_])
                support_lbl.append(lbl[idx][idx_])

        support_sample['support_image'] = np.stack([np.stack(support_img, axis=1)], axis=2)
        support_sample['support_label'] = np.stack([np.stack(support_lbl, axis=1)], axis=2)

        return support_sample


class TrainDataset5shot(Dataset):

    def __init__(self, args):
        self.n_shot = args['n_shot']
        self.n_way = args['n_way']
        self.n_query = args['n_query']
        self.max_iter = args['max_iter']
        self.read = True
        self.min_size = args['min_size']
        self.test_label = args['test_label']
        self.exclude_label = args['exclude_label']
        self.use_gt = args['use_gt']

        if args['dataset'] == 'CMR':
            self.image_dirs = glob.glob(os.path.join(args['data_dir'], 'cmr_MR_normalized/image*'))
            self.label_dirs = glob.glob(os.path.join(args['data_dir'], 'cmr_MR_normalized/label*'))
        elif args['dataset'] == 'CHAOST2':
            self.image_dirs = glob.glob(os.path.join(args['data_dir'], 'chaos_MR_T2_normalized/image*'))
            self.label_dirs = glob.glob(os.path.join(args['data_dir'], 'chaos_MR_T2_normalized/label*'))
        elif args['dataset'] == 'SABS':
            self.image_dirs = glob.glob(os.path.join(args['data_dir'], 'sabs_CT_normalized/image*'))
            self.label_dirs = glob.glob(os.path.join(args['data_dir'], 'sabs_CT_normalized/label*'))

        self.image_dirs = sorted(self.image_dirs, key=lambda x: int(x.split('_')[-1].split('.nii.gz')[0]))
        self.label_dirs = sorted(self.label_dirs, key=lambda x: int(x.split('_')[-1].split('.nii.gz')[0]))
        self.sprvxl_dirs = glob.glob(os.path.join(args['data_dir'], 'supervoxels_' + str(args['n_sv']), 'super*'))
        self.sprvxl_dirs = sorted(self.sprvxl_dirs, key=lambda x: int(x.split('_')[-1].split('.nii.gz')[0]))

        self.FOLD = get_folds(args['dataset'])
        self.image_dirs = [elem for idx, elem in enumerate(self.image_dirs) if idx not in self.FOLD[args['eval_fold']]]
        self.label_dirs = [elem for idx, elem in enumerate(self.label_dirs) if idx not in self.FOLD[args['eval_fold']]]
        self.sprvxl_dirs = [elem for idx, elem in enumerate(self.sprvxl_dirs) if idx not in self.FOLD[args['eval_fold']]]

        if self.read:
            self.images = {}
            self.labels = {}
            self.sprvxls = {}
            for image_dir, label_dir, sprvxl_dir in zip(self.image_dirs, self.label_dirs, self.sprvxl_dirs):
                self.images[image_dir] = sitk.GetArrayFromImage(sitk.ReadImage(image_dir))
                self.labels[label_dir] = sitk.GetArrayFromImage(sitk.ReadImage(label_dir))
                self.sprvxls[sprvxl_dir] = sitk.GetArrayFromImage(sitk.ReadImage(sprvxl_dir))

    def __len__(self):
        return self.max_iter

    def gamma_tansform(self, img):

        gamma_range = (0.5, 1.5)
        gamma = np.random.rand() * (gamma_range[1] - gamma_range[0]) + gamma_range[0]
        cmin = img.min()
        irange = (img.max() - cmin + 1e-5)

        img = img - cmin + 1e-5
        img = irange * np.power(img * 1.0 / irange, gamma)
        img = img + cmin

        return img

    def geom_transform(self, img, mask):

        affine = {'rotate': 5, 'shift': (5, 5), 'shear': 5, 'scale': (0.9, 1.2)}
        alpha = 10
        sigma = 5
        order = 3

        tfx = []
        tfx.append(myit.RandomAffine(affine.get('rotate'),
                                     affine.get('shift'),
                                     affine.get('shear'),
                                     affine.get('scale'),
                                     affine.get('scale_iso', True),
                                     order=order))
        tfx.append(myit.ElasticTransform(alpha, sigma))
        transform = deftfx.Compose(tfx)

        if len(img.shape) > 4:
            n_shot = img.shape[0]
            for shot in range(n_shot):
                cat = np.concatenate((img[shot, 0], mask[shot, :])).transpose(1, 2, 0)
                cat = transform(cat).transpose(2, 0, 1)
                img[shot, 0] = cat[:3, :, :]
                mask[shot, :] = np.rint(cat[3:, :, :])

        else:
            for q in range(img.shape[0]):
                cat = np.concatenate((img[q], mask[q][None])).transpose(1, 2, 0)
                cat = transform(cat).transpose(2, 0, 1)
                img[q] = cat[:3, :, :]
                mask[q] = np.rint(cat[3:, :, :].squeeze())

        return img, mask

    def __getitem__(self, pat_idx):

        pat_idx = random.choice(range(len(self.image_dirs)))

        if self.read:
            img = self.images[self.image_dirs[pat_idx]]
            gt = self.labels[self.label_dirs[pat_idx]]
            sprvxl = self.sprvxls[self.sprvxl_dirs[pat_idx]]
        else:
            img = sitk.GetArrayFromImage(sitk.ReadImage(self.image_dirs[pat_idx]))
            gt = sitk.GetArrayFromImage(sitk.ReadImage(self.label_dirs[pat_idx]))
            sprvxl = sitk.GetArrayFromImage(sitk.ReadImage(self.sprvxl_dirs[pat_idx]))

        if self.exclude_label is not None:
            idx = np.arange(gt.shape[0])
            exclude_idx = np.full(gt.shape[0], True, dtype=bool)
            for i in range(len(self.exclude_label)):
                exclude_idx = exclude_idx & (np.sum(gt == self.exclude_label[i], axis=(1, 2)) > 0)
            exclude_idx = idx[exclude_idx]
        else:
            exclude_idx = []

        img = (img - img.mean()) / img.std()

        if self.use_gt:
            lbl = gt.copy()
        else:
            lbl = sprvxl.copy()

        unique = list(np.unique(lbl))
        unique.remove(0)
        if self.use_gt:
            unique = list(set(unique) - set(self.test_label))

        size = 0
        while size < self.min_size:
            n_slices = (self.n_shot * self.n_way) + self.n_query - 1

            while n_slices < ((self.n_shot * self.n_way) + self.n_query):
                cls_idx = random.choice(unique)

                sli_idx = np.sum(lbl == cls_idx, axis=(1, 2)) > 0
                idx = np.arange(lbl.shape[0])
                sli_idx = idx[sli_idx]
                sli_idx = list(set(sli_idx) - set(np.intersect1d(sli_idx, exclude_idx)))
                n_slices = len(sli_idx)

            subsets = []
            for i in range(len(sli_idx)):
                if not subsets:
                    subsets.append([sli_idx[i]])
                elif sli_idx[i - 1] + 1 == sli_idx[i]:
                    subsets[-1].append(sli_idx[i])
                else:
                    subsets.append([sli_idx[i]])

            i = 0
            while i < len(subsets):
                if len(subsets[i]) < (self.n_shot * self.n_way + self.n_query):
                    del subsets[i]
                else:
                    i += 1
            if not len(subsets):
                return self.__getitem__(idx + np.random.randint(low=0, high=self.max_iter - 1, size=(1,)))

            i = random.choice(np.arange(len(subsets)))
            i = random.choice(subsets[i][:-(self.n_shot * self.n_way + self.n_query - 1)])
            sample = np.arange(i, i + (self.n_shot * self.n_way) + self.n_query)

            lbl_cls = 1 * (lbl == cls_idx)

            size = max(np.sum(lbl_cls[sample[0]]), np.sum(lbl_cls[sample[1]]))

        if np.random.random(1) > 0.5:
            sample = sample[::-1]

        support_lbl = np.stack([lbl_cls[sample[:self.n_shot * self.n_way]]], axis=1)
        query_lbl = lbl_cls[sample[self.n_shot * self.n_way:]]

        support_img = img[sample[:self.n_shot * self.n_way]]
        support_img = np.stack([np.stack((support_img, support_img, support_img), axis=1)], axis=1)
        query_img = img[sample[self.n_shot * self.n_way:]]
        query_img = np.stack((query_img, query_img, query_img), axis=1)

        if np.random.random(1) > 0.5:
            query_img = self.gamma_tansform(query_img)
        else:
            support_img = self.gamma_tansform(support_img)

        if np.random.random(1) > 0.5:
            query_img, query_lbl = self.geom_transform(query_img, query_lbl)
        else:
            support_img, support_lbl = self.geom_transform(support_img, support_lbl)

        sample = {'support_img': support_img,
                  'support_lbl': support_lbl,
                  'query_img': query_img,
                  'query_lbl': query_lbl}

        return sample