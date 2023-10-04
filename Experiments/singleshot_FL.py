import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Utils import load_fl
from Utils import generator
from Utils import metrics
from utils import model_sub, model_sum
from train import *
from prune import *
import timeit
import collections
from cal_similarity import *
import datetime
from torch.utils.tensorboard import SummaryWriter

pd.set_option('display.width', None)

class Local():
    def __init__(self, prune_loader, train_loader, test_loader, label_split, aggre_method):
        self.prune_loader = prune_loader
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.label_split = label_split
        self.aggre_method = aggre_method
        if aggre_method in ["scaffold"]:
            self.c_i = None

    def local_update(self, global_parameters, args, device, epoch, local_mask=None, c_g=None):
        mask_changed = False
        input_shape, num_classes = load_fl.dimension(args.dataset)
        print('Creating {}-{} model.'.format(args.model_class, args.model))
        local_model = load_fl.model(args.model, args.model_class)(input_shape,
                                                                   num_classes,
                                                                   args.dense_classifier,
                                                                   args.pretrained).to(device)
        # local_model.load_state_dict(global_parameters)

        loss = nn.CrossEntropyLoss()
        opt_class, opt_kwargs = load_fl.optimizer(args.optimizer)
        optimizer = opt_class(generator.parameters(local_model), lr=args.lr, weight_decay=args.weight_decay, **opt_kwargs)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drops, gamma=args.lr_drop_rate)
        local_model.load_state_dict(global_parameters)
        # print('**********************The global test results are**********************')
        # eval(local_model, loss, self.test_loader, device, True)
        # print('***********************************************************************')

        if local_mask != None:
            tmp = collections.OrderedDict()
            for key in global_parameters:
                if 'mask' not in key:
                    tmp[key] = global_parameters[key]
                else:
                    tmp[key] = local_mask[key]
            local_model.load_state_dict(tmp)

        # Start pruning
        if (epoch - 1) % args.prune_itv == 0:
            print('Pruning with {} for {} epochs.'.format(args.pruner, args.prune_epochs))
            pruner = load_fl.pruner(args.pruner)(
                generator.masked_parameters(local_model, args.prune_bias, args.prune_batchnorm, args.prune_residual))
            sparsity = float(args.compression)
            prune_loop(local_model, loss, pruner, self.prune_loader, device, sparsity,
                       args.compression_schedule, args.mask_scope, args.prune_epochs, args.reinitialize,
                       args.prune_train_mode, args.shuffle, args.invert)
            mask_changed = True

        start = timeit.default_timer()
        print('Post-Training for {} epochs.'.format(args.post_epochs))

        aggre_method = args.aggre_method
        local_updated_parameters = None
        if aggre_method in ["fedavg", "fedprox"]:
            post_result, local_updated_parameters = train_eval_loop(aggre_method, local_model, loss, optimizer, scheduler, self.train_loader,
                                          self.test_loader, device, args.post_epochs, args.verbose)
        elif aggre_method in ["scaffold"]:
            post_result, local_updated_parameters, dy, dc, c_i = train_eval_loop(aggre_method, local_model, loss, optimizer, scheduler,
                        self.train_loader, self.test_loader, device, args.post_epochs, args.verbose, c_g, self.c_i)
            self.c_i = copy.deepcopy(c_i)

        stop = timeit.default_timer()
        print('------------------Time: ------------', stop - start)

        print(post_result)

        if args.save and epoch == 18:
            print('Saving results.')
            finish_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            torch.save(global_parameters, "{}/model_{}_{}.pt".format(args.result_dir, args.pruner, finish_time))
            torch.save(optimizer.state_dict(), "{}/optimizer_{}_{}.pt".format(args.result_dir, args.pruner, finish_time))
            torch.save(scheduler.state_dict(), "{}/scheduler_{}_{}.pt".format(args.result_dir, args.pruner, finish_time))
            args.save = False
        # prune_result = metrics.summary(local_model,
        #                                pruner.scores,
        #                                metrics.flop(local_model, input_shape, device),
        #                                lambda p: generator.prunable(p, args.prune_batchnorm, args.prune_residual))
        # total_params = int((prune_result['sparsity'] * prune_result['size']).sum())
        # possible_params = prune_result['size'].sum()
        # total_flops = int((prune_result['sparsity'] * prune_result['flops']).sum())
        # possible_flops = prune_result['flops'].sum()
        # print("Prune results:\n", prune_result)
        # print(
        #     "Parameter Sparsity: {}/{} ({:.4f})".format(total_params, possible_params, total_params / possible_params))
        # print("FLOP Sparsity: {}/{} ({:.4f})".format(total_flops, possible_flops, total_flops / possible_flops))

        if aggre_method in ["fedavg", "fedprox"]:
            return local_updated_parameters, post_result, mask_changed
        elif aggre_method in ["scaffold"]:
            return local_updated_parameters, post_result, mask_changed, dy, dc


def eval(model, loss, dataloader, device, verbose):
    model.eval()
    total = 0
    correct1 = 0
    correct5 = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            total += loss(output, target).item() * data.size(0)
            _, pred = output.topk(5, dim=1)
            correct = pred.eq(target.view(-1, 1).expand_as(pred))
            correct1 += correct[:, :1].sum().item()
            correct5 += correct[:, :5].sum().item()
    average_loss = total / len(dataloader.dataset)
    accuracy1 = 100. * correct1 / len(dataloader.dataset)
    accuracy5 = 100. * correct5 / len(dataloader.dataset)
    if verbose:
        print('Evaluation: Average loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%)'.format(
            average_loss, correct1, len(dataloader.dataset), accuracy1))
    return average_loss, accuracy1, accuracy5


def run(args):
    ## Random Seed and Device ##
    torch.manual_seed(args.seed)
    if args.gpu == 1:
        device = 1
    elif args.gpu == 0:
        device = 0
    start_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    # Split dataset
    print('Loading {} dataset.'.format(args.dataset))
    input_shape, num_classes = load_fl.dimension(args.dataset)
    local = [None for _ in range(args.user_num)]
    dataset = load_fl.fetch_dataset(args.dataset, 'label')
    data_split_mode = args.data_split_mode
    data_split, label_split = load_fl.split_datasets(args, dataset, args.user_num, data_split_mode)
    user_idx = torch.arange(args.user_num).tolist()
    aggre_method = args.aggre_method

    for m in range(args.user_num):
        train_data_m = load_fl.SplitDataset(dataset['train'], data_split['train'][user_idx[m]])
        test_data_m = load_fl.SplitDataset(dataset['test'], data_split['test'][user_idx[m]])

        # generator an extra dataset for sparsification
        prune_loader_m = load_fl.make_data_loader(train_data_m, args.prune_batch_size, True, args.workers,
                                                  args.prune_dataset_ratio * num_classes)

        train_loader_m = load_fl.make_data_loader(train_data_m, args.train_batch_size, True, args.workers)
        test_loader_m = load_fl.make_data_loader(test_data_m, args.test_batch_size, False, args.workers)
        local[m] = Local(prune_loader_m, train_loader_m, test_loader_m, label_split[user_idx[m]], aggre_method)

    global_test_data = load_fl.SplitDataset(dataset['test'], data_split['test'][args.user_num])
    global_test_loader = load_fl.make_data_loader(global_test_data, args.test_batch_size, False, args.workers)
    ## Model, Loss, Optimizer ##
    print('Creating {}-{} model.'.format(args.model_class, args.model))
    global_model = load_fl.model(args.model, args.model_class)(input_shape,
                                                     num_classes, 
                                                     args.dense_classifier, 
                                                     args.pretrained).to(device)
    global_parameters = global_model.state_dict()
    local_masks = []
    res_list = []
    distance_matrix = None

    # p = torch.rand(args.user_num, 1)
    # p = [1, 1, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
    # p = [1] * args.user_num
    # select_flag = np.zeros([args.user_num, args.global_epoch + 1])
    #
    # for c in range(0, args.user_num):
    #     select = np.random.binomial(1, p[c], args.global_epoch + 1)
    #
    #     select_flag[c, :] = select
    #
    # for c in range(0, args.user_num):
    #     select_flag[c, 0] = 1
    #
    # for i in range(args.global_epoch + 1):
    #     if np.any(select_flag[:, i]) == False:
    #         select_flag[0, i] = 1
    #
    # np.save('select_flag_2_groups', select_flag)
    if args.client_drop_rate != 1:
        p = []
        for i in range(args.user_num):
            p.append(args.client_drop_rate)

        for i in range(args.user_num//2):
            p[i] = p[i] * 0.3 # 0.3

        select_flag = np.zeros([args.user_num, args.global_epoch + 1])

        seed_list = [2022 + i for i in range(args.user_num)]
        for c in range(0, args.user_num):
            np.random.seed(seed_list[c])
            select = np.random.binomial(1, p[c], args.global_epoch + 1)

            select_flag[c, :] = select

        for c in range(0, args.user_num):
            select_flag[c, 0] = 1

        for i in range(args.global_epoch + 1):
            select_flag[args.user_num-1, i] = 1

            if np.any(select_flag[:args.user_num//2, i]) == False:
                rand_num = np.random.randint(0, args.user_num//2)
                select_flag[rand_num, i] = 1

            elif np.any(select_flag[args.user_num//2:, i]) == False:
                rand_num = np.random.randint(args.user_num//2, args.user_num)
                select_flag[rand_num, i] = 1

        # select_flag = np.load('./select_flag_2_groups.npy')
    else:
        p = [1] * args.user_num
        select_flag = np.zeros([args.user_num, args.global_epoch + 1])

        for c in range(0, args.user_num):
            select = np.random.binomial(1, p[c], args.global_epoch + 1)
            select_flag[c, :] = select

        for c in range(0, args.user_num):
            select_flag[c, 0] = 1

    c_g, dy_sum, dc_sum = None, None, None
    zeros_model = copy.deepcopy(global_model)
    tmp = collections.OrderedDict()
    for k in global_model.state_dict():
        tmp[k] = torch.zeros_like(global_model.state_dict()[k]).to(device)
    zeros_model.load_state_dict(tmp)

    if aggre_method in ["scaffold"]:
        c_g = copy.deepcopy(zeros_model)

    train_result_dict = {}
    for global_epoch in range(1, args.global_epoch + 1):
        print("--------------This is global epoch {}.--------------".format(global_epoch))
        # train_result_dict = {}
        if aggre_method in ["scaffold"]:
            dy_sum = copy.deepcopy(zeros_model)
            dc_sum = copy.deepcopy(zeros_model)

        select_list = []
        for j in range(0, args.user_num):
            if select_flag[j, global_epoch - 1] == 1:
                select_list.append(j)

        print('*********************The Select List is***********************')
        print(select_list)
        print('**************************************************************')
        param_dict = {}
        c_param_dict = {}

        if aggre_method in ["scaffold"]:
            param_dict[args.user_num] = copy.deepcopy(dy_sum.state_dict())
            c_param_dict[args.user_num] = copy.deepcopy(dc_sum.state_dict())

        for m in range(args.user_num):
            mask_changed, train_result, local_parameters = None, None, None
            if m in select_list:
                if aggre_method in ["fedavg", "fedprox"]:
                    if global_epoch == 1:
                        local_parameters, train_result, mask_changed = local[m].local_update(global_parameters, args,
                                                                                      device, global_epoch, None)
                    else:
                        local_parameters, train_result, mask_changed = local[m].local_update(global_parameters, args,
                                                                                      device, global_epoch, local_masks[m])
                elif aggre_method in ["scaffold"]:

                    if global_epoch == 1:


                        local_parameters, train_result, mask_changed, dy, dc = local[m].local_update(global_parameters,
                                                                                args, device, global_epoch, None, c_g)
                        dy_sum = model_sum(dy, dy_sum)
                        dc_sum = model_sum(dc, dc_sum)
                    else:
                        local_parameters, train_result, mask_changed, dy, dc = local[m].local_update(global_parameters,
                                                                        args, device, global_epoch, local_masks[m], c_g)
                        dy_sum = model_sum(dy, dy_sum)
                        dc_sum = model_sum(dc, dc_sum)

                if mask_changed and global_epoch<=1:
                    tmp = {}
                    for key, var in local_parameters.items():
                        if 'mask' in key:
                            tmp[key] = var.clone()
                    local_masks.append(tmp)

                if aggre_method in ["fedavg", "fedprox"]:
                    param_dict[m] = local_parameters
                    train_result_dict[m] = train_result

                elif aggre_method in ["scaffold"]:
                    param_dict[m] = dy.state_dict()
                    c_param_dict[m] = dc.state_dict()
                    train_result_dict[m] = train_result



        if global_epoch == 1 and args.distanced == 1:

            if distance_matrix == None:
                if aggre_method in ["scaffold"]:
                    distance_matrix = cal_similarity(param_dict, is_scaffold = True)
                else:
                    distance_matrix = cal_similarity(param_dict)

            else:
                distance_matrix = cal_similarity(param_dict, distance_matrix)
            print('***************The distance matrix is******************')
            distance_matrix = torch.abs(distance_matrix)
            print(distance_matrix)
            print('*******************************************************')

            #TODO: if one client failed in the pruning interval
            # if global_epoch != 1 and (global_epoch - 1) % args.prune_itv == 0:
            #     complete_list = list(range(args.user_num))
            #     for user_index in [item for item in complete_list if item not in select_list]:
            #         similar_id = np.argsort(distance_matrix[user_index, select_list])[0]
            #         usr_masks[user_index] = usr_masks[select_list[similar_id]]

        complete_list = list(range(args.user_num))
        for m in [item for item in complete_list if item not in select_list]:
            if args.distanced:
                # aggregation for those who are not in the select list by SAFARI similarity
                similar_id = select_list[np.argsort(distance_matrix[m, select_list])[0]]
                if aggre_method in ["scaffold"]:
                    local[m].c_i = local[similar_id].c_i

        aggre_res, new_aggre_parameters = None, None
        if aggre_method in ["fedavg", "fedprox"]:
            aggre_res, new_aggre_parameters = aggre_avg_parameter(args.distanced, args.user_num, select_list,
                                                param_dict, train_result_dict, distance_matrix)
            for key in global_parameters:
                if 'mask' not in key:
                    global_parameters[key] = copy.deepcopy(new_aggre_parameters[key])

        elif aggre_method in ["scaffold"]:

            #
            # param_dict[args.user_num] = dy_sum.state_dict()
            # c_param_dict[args.user_num] = dc_sum.state_dict()
            train_result_dict[args.user_num] = train_result_dict[args.user_num - 1]

            aggre_res, new_aggre_parameters = aggre_avg_parameter(args.distanced, args.user_num, select_list,
                                                                  param_dict, train_result_dict, distance_matrix)

            # print("!!")
            # print(new_aggre_parameters)
            # print("!!")
            # print(dy_sum.state_dict())
            _, new_c_aggre_parameters = aggre_avg_parameter(args.distanced, args.user_num, select_list,
                                                                  c_param_dict, train_result_dict, distance_matrix)

            # for key in global_parameters:
            #
            #     if 'mask' not in key:
            #         global_parameters[key] = global_parameters[key].float() + \
            #                                  new_aggre_parameters[key].float() / float(args.user_num) * args.neta_g
            #
            # tmp = copy.deepcopy(c_g.state_dict())
            # for key in c_g.state_dict():
            #     if 'mask' not in key:
            #         tmp[key] = c_g.state_dict()[key].float() + new_c_aggre_parameters[key].float() / float(args.user_num)
            # c_g.load_state_dict(tmp)

            for key in global_parameters:
                if 'mask' not in key:
                    global_parameters[key] = global_parameters[key].float() + \
                                             copy.deepcopy(new_aggre_parameters[key].float()) * args.neta_g

            tmp = copy.deepcopy(c_g.state_dict())
            for key in c_g.state_dict():
                if 'mask' not in key:
                    tmp[key] = c_g.state_dict()[key].float() + copy.deepcopy(new_c_aggre_parameters[key]).float()
            c_g.load_state_dict(tmp)

            # for key in dy_sum.state_dict():
            #     dy_sum.state_dict()[key] /= float(len(select_list))
            #
            # for key in dc_sum.state_dict():
            #     dc_sum.state_dict()[key] /= float(len(select_list))

            param_dict[args.user_num] = copy.deepcopy(dy_sum.state_dict())
            c_param_dict[args.user_num] = copy.deepcopy(dc_sum.state_dict())


        ################
            # for key in global_parameters:
            #     if 'mask' not in key:
            #         global_parameters[key] = global_parameters[key].float() + \
            #                                  dy_sum.state_dict()[key].float() / float(len(select_list)) * args.neta_g
            #
            # tmp = copy.deepcopy(c_g.state_dict())
            # for key in c_g.state_dict():
            #     if 'mask' not in key:
            #         tmp[key] = c_g.state_dict()[key].float() + dc_sum.state_dict()[key].float() / float(args.user_num)
            # c_g.load_state_dict(tmp)
            # cnt = 0
            # for m in param_dict:
            #     if cnt == 0:
            #         aggre_res = train_result_dict[m]
            #     else:
            #         aggre_res += train_result_dict[m]
            #     cnt += 1
            # aggre_res = aggre_res / len(param_dict)
            #
            # # print("!!")
            # # print(new_aggre_parameters)
            # print("!!")
            # print(dy_sum.state_dict())


        loss = nn.CrossEntropyLoss()
        global_model.load_state_dict(global_parameters)
        global_test_loss, global_acc1, global_acc5 = eval(global_model, loss, global_test_loader, device, 0)

        columns = ['global_test_loss', 'global_acc1', 'global_acc5']
        global_res = pd.DataFrame([[global_test_loss, global_acc1, global_acc5]], columns=columns)

        # res_list.append(pd.concat([aggre_res[:-1], global_res], axis=1))
        res_list.append(pd.concat([aggre_res, global_res], axis=1))
        print(("-------------------Epoch results-------------------"))
        print(pd.concat([aggre_res, global_res], axis=1))
        print("-------------------Saving results-------------------")
        ## Display Results ##
        frames = []
        for i in range(len(res_list)):
            frames.append(res_list[i])
        output = pd.concat(frames, keys=[i for i in range(1, args.global_epoch + 1)])

        output.to_csv(
            './Results/{}_{}_{}_{}_{}_{}.csv'.format(args.aggre_method, args.pruner, args.compression, args.distanced, data_split_mode, start_time),
            sep='\t', encoding='utf-8')




