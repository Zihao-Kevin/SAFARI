import numpy as np
import copy
import torch

# def cal_similarity(usr_masks):
#     similarity_matrix = np.zeros([len(usr_masks), len(usr_masks)])
#     mask_list = []
#     for i in range(0, len(usr_masks)):
#         x_vec = torch.cat([usr_masks[i][k].reshape(-1, 1) for k in range(len(usr_masks[i]))], dim=0)
#         mask_list.append(x_vec)
#
#     for i in range(0, len(mask_list)):
#         for j in range(0, len(mask_list)):
#             similarity_matrix[i, j] = torch.linalg.norm(mask_list[i] - mask_list[j])
#
#     return similarity_matrix

def hamming_distance(mask_a, mask_b):
    dis = 0; total = 0
    for key in mask_a:
        # if 'mask' in key:
        # if 'conv' in key and 'mask' in key:
        # dis += torch.sum((mask_a[key]*1000).int() ^  (mask_b[key]*1000).int())
        dis += torch.sum((mask_a[key]).int() ^ (mask_b[key]).int())
        total += mask_a[key].numel()
    return dis, total


def cal_similarity(param_dict, similarity_matrix=None, is_scaffold = False):
    if is_scaffold == False:
        if similarity_matrix == None:
            similarity_matrix = torch.zeros([len(param_dict), len(param_dict)])
        for i in param_dict.keys():
            for j in param_dict.keys():
                similarity_matrix[i, j], _ = hamming_distance(param_dict[i], param_dict[j])
    else:
        if similarity_matrix == None:
            similarity_matrix = torch.zeros([len(param_dict)-1, len(param_dict)-1])
        for i in param_dict.keys():
            if i == len(param_dict) - 1:
                continue
            for j in param_dict.keys():
                if j == len(param_dict) - 1:
                    continue
                similarity_matrix[i, j], _ = hamming_distance(param_dict[i], param_dict[j])
    return similarity_matrix

## before 4.29 revision
# def cal_similarity(param_dict, similarity_matrix=None, is_scaffold = True):
#     if similarity_matrix == None:
#         similarity_matrix = torch.zeros([len(param_dict), len(param_dict)])
#     for i in param_dict.keys():
#         for j in param_dict.keys():
#             res = 0
#             for key in param_dict[i]:
#                 if param_dict[i][key].dtype != torch.int64:
#                     res += torch.linalg.norm(param_dict[i][key] - param_dict[j][key])
#                 else:
#                     res += torch.linalg.norm(torch.tensor(float(param_dict[i][key] - param_dict[j][key])))
#             similarity_matrix[i, j] = res
#
#     return similarity_matrix


def dict_2_list(dict):
    tmp = []
    for key in dict:
        tmp.append(dict[key])
    return tmp


def aggre_avg_parameter(is_similarity, usr_num, select_list, param_dict, train_result_dict, similarity=None):
    sum_parameters = None
    res = None
    for m in param_dict:
        if sum_parameters == None:
            sum_parameters = {}
            res = train_result_dict[m]
            for key, var in param_dict[m].items():
                if 'mask' not in key:
                    sum_parameters[key] = var.clone()
        else:
            for key in sum_parameters:
                if 'mask' not in key:
                    sum_parameters[key] = sum_parameters[key] + param_dict[m][key]
            res += train_result_dict[m]
    res = res / len(param_dict)
    if is_similarity:
        # aggregation for those who are not in the select list by SAFARI similarity
        complete_list = list(range(usr_num))
        for user_id in [item for item in complete_list if item not in select_list]:
            if len(select_list) == 1:
                similar_id = select_list[0]
            else:
                similar_id = select_list[np.argsort(similarity[user_id, select_list])[0]]
            try:
                print("!!!!")
                print(similar_id)
                for key, var in param_dict[similar_id].items():
                    if 'mask' not in key:
                        sum_parameters[key] = sum_parameters[key] + var.clone()
            except KeyError:
                print("KeyError")
        for key in sum_parameters:
            if 'mask' not in key:
                sum_parameters[key] = (sum_parameters[key] / usr_num)
    else:
        l = len(select_list)
        for key in sum_parameters:
            if 'mask' not in key:
                sum_parameters[key] = (sum_parameters[key] / l)
    return res, sum_parameters


def find_closest_id(usr_num, select_list, distance):
    complete_list = list(range(usr_num))
    for user_id in [item for item in complete_list if item not in select_list]:
        similar_id = np.argsort(distance[user_id, select_list])[0]
    return similar_id