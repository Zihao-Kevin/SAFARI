import copy
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import model_sub, model_sum
import collections


def freeze_grad(x):
    for p in x.parameters():
        p.requires_grad = False
    return x

def train_fedavg(model, loss, optimizer, train_loader, device, epochs, verbose, log_interval=1):
    model.train()
    # rows = []
    train_loss = 0
    for epoch in tqdm(range(epochs)):
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            train_loss = loss(output, target)
            total += train_loss.item() * data.size(0)
            train_loss.backward()
            optimizer.step()
        train_loss = total / len(train_loader.dataset)

    return train_loss, model


def train_fedprox(mu, model, loss, optimizer, train_loader, test_loader, device, epochs, verbose, log_interval=1):
    model.train()
    ori_model = copy.deepcopy(model)
    ori_model = freeze_grad(ori_model)
    # rows = []
    train_loss = 0
    for epoch in tqdm(range(epochs)):
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            train_loss = loss(output, target)
            diff = model_sub(model, ori_model)
            loss_proximal = 0
            for k in diff.state_dict():
                loss_proximal += torch.norm(diff.state_dict()[k].float())
            train_loss += 0.5 * mu * loss_proximal
            train_loss.backward()
            optimizer.step()
            total += train_loss.item() * data.size(0)

        train_loss = total / len(train_loader.dataset)
    return train_loss, model


def train_scaffold(c_g, c_i, model, loss, optimizer, train_loader, test_loader, device, epochs, verbose, log_interval=1):
    model.train()
    if c_i == None:
        c_i = copy.deepcopy(model)
        tmp = collections.OrderedDict()
        for k in c_i.state_dict():
            tmp[k] = torch.zeros_like(c_i.state_dict()[k]).to(device)
        c_i.load_state_dict(tmp)
    c_i = freeze_grad(c_i)
    c_g = freeze_grad(c_g)
    ori_model = copy.deepcopy(model)
    ori_model = freeze_grad(ori_model)
    rows = []
    for epoch in tqdm(range(epochs)):
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            train_loss = loss(output, target)
            train_loss.backward()
            # y_i <-- y_i - eta_l ( g_i(y_i)-c_i+c )  =>  g_i(y_i)' <-- g_i(y_i)-c_i+c
            for pm, pcg, pc in zip(model.parameters(), c_g.parameters(), c_i.parameters()):
                pm.grad = pm.grad - pc + pcg
            optimizer.step()

            total += train_loss.item() * data.size(0)

        train_loss = total / len(train_loader.dataset)
        test_loss, accuracy1, accuracy5 = eval(model, loss, test_loader, device, verbose)
        row = [train_loss, test_loss, accuracy1, accuracy5]
        # scheduler.step()
        rows.append(row)

    # dy = y - x
    dy = model_sub(model, ori_model)

    # K = epochs
    K = len(train_loader.dataset.idx) / train_loader.batch_size

    factor = - 1.0 / (K * optimizer.defaults['lr'])
    factor_mul = copy.deepcopy(dy)
    tmp = collections.OrderedDict()
    for k in factor_mul.state_dict():
        tmp[k] = factor * factor_mul.state_dict()[k]
    factor_mul.load_state_dict(tmp)
    dc = model_sub(factor_mul, c_g)

    c_i = model_sum(c_i, dc)
    return rows, model.state_dict(), dy, dc, c_i


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


def train_eval_loop(aggre_method, model, loss, optimizer, scheduler, train_loader, test_loader, device, epochs,
                    verbose, c_g=None, c_i=None):
    # test_loss, accuracy1, accuracy5 = eval(model, loss, test_loader, device, verbose)
    # rows = [[np.nan, test_loss, accuracy1, accuracy5]]
    rows = []
    model_parameters = None
    if aggre_method in ['fedavg', 'fedprox']:
        if aggre_method == 'fedavg':
            train_loss, new_model = train_fedavg(model, loss, optimizer, train_loader, device, epochs, verbose)
            test_loss, accuracy1, accuracy5 = eval(new_model, loss, test_loader, device, verbose)
            row = [train_loss, test_loss, accuracy1, accuracy5]
            rows.append(row)
            model_parameters = copy.deepcopy(new_model.state_dict())
        elif aggre_method == 'fedprox':
            mu = 0.001
            # rows, model_parameters = train_fedprox(mu, model, loss, optimizer, train_loader, test_loader, device, epochs, verbose)
            train_loss, new_model = train_fedprox(mu, model, loss, optimizer, train_loader, test_loader, device, epochs, verbose)
            test_loss, accuracy1, accuracy5 = eval(new_model, loss, test_loader, device, verbose)
            row = [train_loss, test_loss, accuracy1, accuracy5]
            rows.append(row)
            model_parameters = copy.deepcopy(new_model.state_dict())
    elif aggre_method == 'scaffold':
        rows, model_parameters, dy, dc, c_i = train_scaffold(c_g, c_i, model, loss, optimizer, train_loader, test_loader, device, epochs, verbose)

    columns = ['train_loss', 'test_loss', 'top1_accuracy', 'top5_accuracy']

    if aggre_method in ['fedavg', 'fedprox']:
        return pd.DataFrame(rows, columns=columns), model_parameters
    elif aggre_method == 'scaffold':
        return pd.DataFrame(rows, columns=columns), model_parameters, dy, dc, c_i


