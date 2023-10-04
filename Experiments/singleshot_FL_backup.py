import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Utils import load_fl
from Utils import generator
from Utils import metrics
from train import *
from prune import *
import timeit
import collections
from torch.utils.tensorboard import SummaryWriter


class Local():
    def __init__(self, prune_loader, train_loader, test_loader, label_split):
        self.prune_loader = prune_loader
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.label_split = label_split

    def local_update(self, global_parameters, args, device, epoch, local_mask=None):
        mask_changed = False
        input_shape, num_classes = load_fl.dimension(args.dataset)
        print('Creating {}-{} model.'.format(args.model_class, args.model))
        local_model = load_fl.model(args.model, args.model_class)(input_shape,
                                                                   num_classes,
                                                                   args.dense_classifier,
                                                                   args.pretrained).to(device)
        # local_model.load_state_dict(global_parameters)
        if local_mask == None:
            local_model.load_state_dict(global_parameters)
        else:
            tmp = collections.OrderedDict()
            for key in global_parameters:
                if 'mask' not in key:
                    tmp[key] = global_parameters[key]
                else:
                    tmp[key] = local_mask[key]
            local_model.load_state_dict(tmp)

        loss = nn.CrossEntropyLoss()
        opt_class, opt_kwargs = load_fl.optimizer(args.optimizer)
        optimizer = opt_class(generator.parameters(local_model), lr=args.lr, weight_decay=args.weight_decay, **opt_kwargs)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drops, gamma=args.lr_drop_rate)

        if (epoch - 1) % args.prune_itv == 0:
            ## Prune ##
            print('Pruning with {} for {} epochs.'.format(args.pruner, args.prune_epochs))
            pruner = load_fl.pruner(args.pruner)(
                generator.masked_parameters(local_model, args.prune_bias, args.prune_batchnorm, args.prune_residual))
            # sparsity = 10 ** (-float(args.compression))
            sparsity = float(args.compression)
            prune_loop(local_model, loss, pruner, self.prune_loader, device, sparsity,
                       args.compression_schedule, args.mask_scope, args.prune_epochs, args.reinitialize,
                       args.prune_train_mode, args.shuffle, args.invert)
            mask_changed = True


        start = timeit.default_timer()
        # The module that you try to calculate the running time
        print('Post-Training for {} epochs.'.format(args.post_epochs))
        post_result = train_eval_loop(local_model, loss, optimizer, scheduler, self.train_loader, self.test_loader, device,
                                      args.post_epochs, args.verbose)
        stop = timeit.default_timer()
        print('------------------Time: ------------', stop - start)

        print(post_result)

        return local_model.state_dict(), post_result, mask_changed


def run(args):
    ## Random Seed and Device ##
    torch.manual_seed(args.seed)
    device = load_fl.device(args.gpu)

    # Split dataset
    print('Loading {} dataset.'.format(args.dataset))
    input_shape, num_classes = load_fl.dimension(args.dataset)
    local = [None for _ in range(args.user_num)]
    dataset = load_fl.fetch_dataset(args.dataset, 'label')
    data_split, label_split = load_fl.split_datasets(args.dataset, dataset, args.user_num, 'non-iid')
    user_idx = torch.arange(args.user_num).tolist()
    for m in range(args.user_num):
        prune_loader_m = load_fl.make_data_loader(load_fl.SplitDataset(dataset['train'], data_split['train'][user_idx[m]]),
                                               args.prune_batch_size, True, args.workers, args.prune_dataset_ratio * num_classes)
        train_loader_m = load_fl.make_data_loader(load_fl.SplitDataset(dataset['train'], data_split['train'][user_idx[m]]),
                                         args.train_batch_size, True, args.workers)
        test_loader_m = load_fl.make_data_loader(load_fl.SplitDataset(dataset['test'], data_split['test'][user_idx[m]]),
                                         args.test_batch_size, False, args.workers)
        local[m] = Local(prune_loader_m, train_loader_m, test_loader_m, label_split[user_idx[m]])

    ## Model, Loss, Optimizer ##
    print('Creating {}-{} model.'.format(args.model_class, args.model))
    global_model = load_fl.model(args.model, args.model_class)(input_shape,
                                                     num_classes, 
                                                     args.dense_classifier, 
                                                     args.pretrained).to(device)
    global_parameters = global_model.state_dict()
    local_masks = []

    p = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # # for i in range(0, len(p)):
    # #     p[i] = 0.9 * p[i]

    select_flag = np.zeros([args.user_num, args.post_epochs + 1])
    for c in range(0, args.user_num):
        select_flag[c, :] = np.random.binomial(1, p[c], args.user_num + 1)

    for c in range(0, args.user_num):
        select_flag[c, 0] = 1

    for global_epoch in range(1, args.post_epochs + 1):
        sum_parameters = None
        res = None
        for m in range(args.user_num):
            if global_epoch == 1:
                local_parameters, result, mask_changed = local[m].local_update(global_parameters, args, device,
                                                                               global_epoch)
            else:
                local_parameters, result, mask_changed = local[m].local_update(global_parameters, args, device,
                                                                               global_epoch, local_masks[m])

            if mask_changed:
                tmp = {}
                for key, var in local_parameters.items():
                    if 'mask' in key:
                        tmp[key] = var.clone()
                local_masks.append(tmp)

            if sum_parameters is None:
                sum_parameters = {}
                res = result
                for key, var in local_parameters.items():
                    if 'mask' not in key:
                        sum_parameters[key] = var.clone()
            else:
                for key in sum_parameters:
                    if 'mask' not in key:
                        sum_parameters[key] = sum_parameters[key] + local_parameters[key]
                res += result

        for key in global_parameters:
            if 'mask' not in key:
                global_parameters[key] = (sum_parameters[key] / args.user_num)
        res = res / args.user_num
        # loss_history.append(LOCAL_LOSS)
        # acc_history.append(LOCAL_ACC)
        print("This is result.")
        print(res)
        # np.savez(file_name, loss=loss_history, Bit=Bit)




