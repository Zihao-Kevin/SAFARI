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
from cal_similarity import *
import datetime
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
        # local_model.load_state_dict(global_parameters)d
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

        return local_model.state_dict(), post_result, mask_changed,


def run(args):
    ## Random Seed and Device ##
    torch.manual_seed(args.seed)
    device = load_fl.device(args.gpu)
    start_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    # Split dataset
    input_shape, num_classes = load_fl.dimension(args.dataset)
    local = [None for _ in range(args.user_num)]
    alpha = 1
    dataset = load_fl.fetch_dataset(args.dataset, 'label')
    idx = [torch.where(dataset.train_labels == i) for i in range(num_classes)]
    data = [dataset.data[idx[i][0]] for i in range(num_classes)]
    label = [torch.ones(len(data[i])) * i for i in range(num_classes)]

    s = np.random.dirichlet(np.ones(num_classes) * alpha, args.user_num)
    data_dist = np.zeros((args.user_num, num_classes))

    for j in range(args.user_num):
        data_dist[j] = ((s[j] * len(data[0])).astype('int') / (s[j] * len(data[0])).astype('int').sum() * len(
            data[0])).astype('int')
        data_num = data_dist[j].sum()
        data_dist[j][np.random.randint(low=0, high=num_classes)] += ((len(data[0]) - data_num))
        data_dist = data_dist.astype('int')

    X = []
    Y = []
    for j in range(args.user_num):
        x_data = []
        y_data = []
        for i in range(num_classes):
            if data_dist[j][i] != 0:
                d_index = np.random.randint(low=0, high=len(data[i]), size=data_dist[j][i])
                x_data.append(data[i][d_index])
                y_data.append(label[i][d_index])
        X.append(torch.cat(x_data))
        Y.append(torch.cat(y_data))
    Non_iid_dataset = [load_fl.Non_iid(X[j], Y[j]) for j in range(args.user_num)]

    print('Loading {} dataset.'.format(args.dataset))

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
    res_list = []

    # p = torch.rand(args.user_num, 1)
    p = [1] * args.user_num
    # # for i in range(0, len(p)):
    # #     p[i] = 0.9 * p[i]
    distance_matrix = None
    select_flag = np.zeros([args.user_num, args.global_epoch + 1])

    for c in range(0, args.user_num):
        select_flag[c, :] = np.random.binomial(1, p[c], args.global_epoch + 1)

    for c in range(0, args.user_num):
        select_flag[c, 0] = 1
    # select_flag = np.array([[1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0.,  0., 1.,], [1., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,  0., 0.,], [1., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,  1., 0.,], [1., 1., 1., 1., 1., 0., 0., 1., 1., 0., 0., 0., 1., 1., 0., 0., 1., 0., 0., 1., 1., 1., 1., 1.,  1., 0.,], [1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1.,  1., 1.,], [1., 1., 1., 1., 0., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1.,  1., 1.,], [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 1., 1., 1., 1., 0., 1., 0., 1.,  1., 1.,], [1., 1., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 0., 0., 1., 0., 1., 0., 0., 0.,  0., 1.,], [1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0.,  0., 1.,], [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0.,  0., 0.,]])

    for global_epoch in range(1, args.global_epoch + 1):
        print("--------------This is global epoch {}.--------------".format(global_epoch))
        res_dict = {}

        select_list = []
        for j in range(0, args.user_num):
            if select_flag[j, global_epoch - 1] == 1:
                select_list.append(j)

        param_dict = {}

        for m in range(args.user_num):
            if m in select_list:
                if global_epoch == 1:
                    local_parameters, result, mask_changed = local[m].local_update(global_parameters, args, device, global_epoch)
                else:
                    local_parameters, result, mask_changed = local[m].local_update(global_parameters, args, device, global_epoch, local_masks[m])

                if mask_changed:
                    tmp = {}
                    for key, var in local_parameters.items():
                        if 'mask' in key:
                            tmp[key] = var.clone()
                    local_masks.append(tmp)

                param_dict[m] = local_parameters
                res_dict[m] = result

        if args.distanced:
            _list_local_masks = []
            for i in range(args.user_num):
                _list_local_masks.append(dict_2_list(local_masks[i]))

            if global_epoch == 1:
                distance_matrix = cal_similarity(_list_local_masks)

            #TODO: if one client failed in the pruning interval
            # if global_epoch != 1 and (global_epoch - 1) % args.prune_itv == 0:
            #     complete_list = list(range(args.user_num))
            #     for user_index in [item for item in complete_list if item not in select_list]:
            #         similar_id = np.argsort(distance_matrix[user_index, select_list])[0]
            #         usr_masks[user_index] = usr_masks[select_list[similar_id]]

            res, sum_parameters = aug_avg_parameter(args.distanced, args.user_num, select_list, distance_matrix, param_dict, res_dict)
        else:
            res, sum_parameters = aug_avg_parameter(args.distanced, args.user_num, select_list, distance_matrix, param_dict, res_dict)

        for key in global_parameters:
            if 'mask' not in key:
                global_parameters[key] = sum_parameters[key]

        res_list.append(res)

        print("-------------------Saving results-------------------")
        res.to_csv('./Results/{}_{}_{}.csv'.format(args.pruner, args.distanced, start_time), sep='\t',
                            encoding='utf-8')

    ## Display Results ##
    frames = []
    for i in range(args.global_epoch):
        frames.append(res_list[i])
    train_result = pd.concat(frames, keys=[i for i in range(1, args.global_epoch + 1)])

    train_result.to_csv('./Results/{}_{}_{}.csv'.format(args.pruner, args.distanced, start_time), sep='\t', encoding='utf-8')





