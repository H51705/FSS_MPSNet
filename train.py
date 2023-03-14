import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from dataloaders.datasets import TrainDataset as TrainDataset
from models.fewshot import FewShotSeg
from utils import *
from torch.nn import functional as F
import time


def main():
    _config = {
        'seed': 2021,
        'gpu_id': 0,
        'dataset': '***',
        'data_dir': '***',
        'n_shot': 5,
        'n_way': 1,
        'n_query': 1,
        'n_sv': 5000,
        'eval_fold': 0,
        'min_size': 200,
        'exclude_label': None,
        'test_label': '***',
        'use_gt': False,
        'lr_step_gamma': 0.98,
        'max_iters_per_load': 1000,
        'n_steps': 30000,
        'print_interval': 100,
        'save_snapshot_every': 1000,
        'optim': {
            'lr': 1e-3,
            'momentum': 0.9,
            'weight_decay': 0.0005,
        }
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
    model = model.cuda()
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
    lr_milestones = [(ii + 1) * _config['max_iters_per_load'] for ii in
                     range(_config['n_steps'] // _config['max_iters_per_load'] - 1)]
    scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=_config['lr_step_gamma'])

    my_weight = torch.FloatTensor([0.1, 1.0]).cuda()
    criterion = nn.NLLLoss(ignore_index=255, weight=my_weight)

    data_config = {
        'data_dir': _config['data_dir'],
        'dataset': _config['dataset'],
        'n_shot': _config['n_shot'],
        'n_way': _config['n_way'],
        'n_query': _config['n_query'],
        'n_sv': _config['n_sv'],
        'max_iter': _config['max_iters_per_load'],
        'eval_fold': _config['eval_fold'],
        'test_label': _config['test_label'],
        'min_size': _config['min_size'],
        'exclude_label': _config['exclude_label'],
        'use_gt': _config['use_gt'],
    }
    train_dataset = TrainDataset(data_config)
    train_loader = DataLoader(train_dataset,
                              batch_size=1,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=True,
                              drop_last=True)

    n_sub_epochs = _config['n_steps'] // _config['max_iters_per_load']
    log_loss = {'total_loss': 0, 'query_loss': 0, 'loss_mln': 0}

    i_iter = 0
    for sub_epoch in range(n_sub_epochs):
        print('This is epoch {} of {} epochs.'.format(sub_epoch, n_sub_epochs))
        for _, sample in enumerate(train_loader):

            support_images = [[shot.float().cuda() for shot in way] for way in sample['support_images']]
            support_fg_mask = [[shot.float().cuda() for shot in way] for way in sample['support_fg_labels']]

            query_images = [query_image.float().cuda() for query_image in sample['query_images']]
            query_labels = torch.cat([query_label.long().cuda() for query_label in sample['query_labels']], dim=0)

            query_pred, loss_mln = model(support_images, support_fg_mask, query_images)
            query_loss = criterion(torch.log(torch.clamp(query_pred, torch.finfo(torch.float32).eps, 1 - torch.finfo(torch.float32).eps)), query_labels)
            total_loss = query_loss + loss_mln

            for param in model.parameters():
                param.grad = None

            total_loss.backward()
            optimizer.step()
            scheduler.step()

            query_loss = query_loss.detach().data.cpu().numpy()
            loss_mln = loss_mln.detach().data.cpu().numpy()

            log_loss['total_loss'] += total_loss.item()
            log_loss['query_loss'] += query_loss
            log_loss['loss_mln'] += loss_mln

            if (i_iter + 1) % _config['print_interval'] == 0:
                total_loss = log_loss['total_loss'] / _config['print_interval']
                query_loss = log_loss['query_loss'] / _config['print_interval']
                loss_mln = log_loss['loss_mln'] / _config['print_interval']

                log_loss['total_loss'] = 0
                log_loss['query_loss'] = 0
                log_loss['loss_mln'] = 0

                print('step {}: total_loss: {}, query_loss: {}, loss_mln: {}'
                      .format(i_iter + 1, total_loss, query_loss, loss_mln))

            if (i_iter + 1) % _config['save_snapshot_every'] == 0:
                print('save model...')

                torch.save(model.state_dict(), '***'
                           .format(i_iter + 1))

            i_iter += 1

    print('End of training!')


if __name__ == '__main__':
    main()
