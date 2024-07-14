import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from dataloaders.datasets import TrainDataset5shot
from modules.fewshot_mc23 import FewShotSeg5shot
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
        'n_sv': '***',
        'eval_fold': 0,
        'min_size': 200,
        'exclude_label': None,
        'test_label': '***',
        'use_gt': False,
        'max_iters_per_load': 1000,
        'n_steps': 50000,
        'print_interval': 100,
        'save_snapshot_every': 1000,
        'lr_step_gamma': 0.98,
        'optim': {
            'lr': 1e-3,
            'momentum': 0.9,
            'weight_decay': 0.0005,
        }
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
        'eval_fold': _config['eval_fold'],
        'n_shot': _config['n_shot'],
        'n_way': _config['n_way'],
        'n_query': _config['n_query'],
        'n_sv': _config['n_sv'],
        'max_iter': _config['max_iters_per_load'],
        'test_label': _config['test_label'],
        'min_size': _config['min_size'],
        'exclude_label': _config['exclude_label'],
        'use_gt': _config['use_gt'],
    }

    train_dataset = TrainDataset5shot(data_config)
    train_loader = DataLoader(train_dataset,
                              batch_size=1,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=True,
                              drop_last=True)

    epochs = _config['n_steps'] // _config['max_iters_per_load']

    log_loss = {'total_loss': 0, 'query_loss': 0, 'MLN_loss': 0}

    i_iter = 0

    for sub_epoch in range(epochs):
        print('This is epoch {} of {} epochs.'.format(sub_epoch, epochs))

        for _, sample in enumerate(train_loader):

            support_img = [[shot.float().cuda() for shot in way] for way in sample['support_img']]
            support_lbl = [[shot.float().cuda() for shot in way] for way in sample['support_lbl']]
            query_img = [query_img.float().cuda() for query_img in sample['query_img']]
            query_lbl = torch.cat([query_lbl.long().cuda() for query_lbl in sample['query_lbl']], dim=0)

            MLN_loss, qry_segs = model(support_img, support_lbl, query_img)
            print('MLN_loss: ', MLN_loss)

            query_loss = criterion(torch.log(torch.clamp(qry_segs, torch.finfo(torch.float32).eps,
                                                         1 - torch.finfo(torch.float32).eps)), query_lbl)
            print('query_loss: ', query_loss)

            total_loss = query_loss + MLN_loss
            print('total_loss: ', total_loss)

            for param in model.parameters():
                param.grad = None
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            query_loss = query_loss.detach().data.cpu().numpy()
            MLN_loss = MLN_loss.detach().data.cpu().numpy()

            log_loss['total_loss'] += total_loss.item()
            log_loss['query_loss'] += query_loss
            log_loss['MLN_loss'] += MLN_loss

            if (i_iter + 1) % _config['print_interval'] == 0:
                total_loss = log_loss['total_loss'] / _config['print_interval']
                query_loss = log_loss['query_loss'] / _config['print_interval']
                MLN_loss = log_loss['MLN_loss'] / _config['print_interval']

                log_loss['total_loss'] = 0
                log_loss['query_loss'] = 0
                log_loss['MLN_loss'] = 0

                print('step {}: total_loss: {}, query_loss: {}, MLN_loss: {}'
                      .format(i_iter + 1, total_loss, query_loss, MLN_loss))

            if (i_iter + 1) % _config['save_snapshot_every'] == 0:
                print('save model...')
                torch.save(model.state_dict(), '***'
                           .format(i_iter + 1))

            i_iter += 1
    print('End of training!')


if __name__ == '__main__':
    main()
