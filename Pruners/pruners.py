import torch
import numpy as np
from torch import nn
import copy
import math
from Layers import layers

class Pruner:
    def __init__(self, masked_parameters):
        self.masked_parameters = list(masked_parameters)
        self.scores = {}

    def score(self, model, loss, dataloader, device):
        raise NotImplementedError

    def _global_mask(self, sparsity):
        r"""Updates masks of model with scores by sparsity level globally.
        """
        # # Set score for masked parameters to -inf 
        # for mask, param in self.masked_parameters:
        #     score = self.scores[id(param)]
        #     score[mask == 0.0] = -np.inf

        # Threshold scores
        global_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        k = int((1.0 - sparsity) * global_scores.numel())
        if not k < 1:
            threshold, _ = torch.kthvalue(global_scores, k)
            for mask, param in self.masked_parameters:
                score = self.scores[id(param)] 
                zero = torch.tensor([0.], dtype=torch.float).to(mask.device)
                one = torch.tensor([1.], dtype=torch.float).to(mask.device)
                mask.copy_(torch.where(score <= threshold, zero, one))
    
    def _local_mask(self, sparsity):
        r"""Updates masks of model with scores by sparsity level parameter-wise.
        """
        for mask, param in self.masked_parameters:
            score = self.scores[id(param)]
            k = int((1.0 - sparsity) * score.numel())
            if not k < 1:
                threshold, _ = torch.kthvalue(torch.flatten(score), k)
                zero = torch.tensor([0.], dtype=torch.float).to(mask.device)
                one = torch.tensor([1.], dtype=torch.float).to(mask.device)
                mask.copy_(torch.where(score <= threshold, zero, one))

    def mask(self, sparsity, scope):
        r"""Updates masks of model with scores by sparsity according to scope.
        """
        if scope == 'global':
            self._global_mask(sparsity)
        if scope == 'local':
            self._local_mask(sparsity)

    @torch.no_grad()
    def apply_mask(self):
        r"""Applies mask to prunable parameters.
        """
        for mask, param in self.masked_parameters:
            param.mul_(mask)

    def alpha_mask(self, alpha):
        r"""Set all masks to alpha in model.
        """
        for mask, _ in self.masked_parameters:
            mask.fill_(alpha)

    # Based on https://github.com/facebookresearch/open_lth/blob/master/utils/tensor_utils.py#L43
    def shuffle(self):
        for mask, param in self.masked_parameters:
            shape = mask.shape
            perm = torch.randperm(mask.nelement())
            mask = mask.reshape(-1)[perm].reshape(shape)

    def invert(self):
        for v in self.scores.values():
            v.div_(v**2)

    def stats(self):
        r"""Returns remaining and total number of prunable parameters.
        """
        remaining_params, total_params = 0, 0 
        for mask, _ in self.masked_parameters:
             remaining_params += mask.detach().cpu().numpy().sum()
             total_params += mask.numel()
        return remaining_params, total_params


class Rand(Pruner):
    def __init__(self, masked_parameters):
        super(Rand, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.randn_like(p)


class Mag(Pruner):
    def __init__(self, masked_parameters):
        super(Mag, self).__init__(masked_parameters)
    
    def score(self, model, loss, dataloader, device):
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.data).detach().abs_()


# Based on https://github.com/mi-lad/snip/blob/master/snip.py#L18
class SNIP(Pruner):
    def __init__(self, masked_parameters):
        super(SNIP, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):

        # allow masks to have gradient
        for m, _ in self.masked_parameters:
            m.requires_grad = True

        # compute gradient
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            model.zero_grad()
            output = model(data)
            loss(output, target).backward()

        # calculate score |g * theta|
        for m, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(m.grad).detach().abs_()
            p.grad.data.zero_()
            m.grad.data.zero_()
            m.requires_grad = False

        # normalize score
        all_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        norm = torch.sum(all_scores)
        for _, p in self.masked_parameters:
            self.scores[id(p)].div_(norm)


# Based on https://github.com/alecwangcq/GraSP/blob/master/pruner/GraSP.py#L49
class GraSP(Pruner):
    def __init__(self, masked_parameters):
        super(GraSP, self).__init__(masked_parameters)
        self.temp = 200
        self.eps = 1e-10

    def score(self, model, loss, dataloader, device):

        # first gradient vector without computational graph
        stopped_grads = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data) / self.temp
            L = loss(output, target)

            grads = torch.autograd.grad(L, [p for (_, p) in self.masked_parameters], create_graph=False)
            flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])
            stopped_grads += flatten_grads

        # second gradient vector with computational graph
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            model.zero_grad()
            output = model(data) / self.temp
            L = loss(output, target)

            grads = torch.autograd.grad(L, [p for (_, p) in self.masked_parameters], create_graph=True)
            flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])
            
            gnorm = (stopped_grads * flatten_grads).sum()
            gnorm.backward()
        
        # calculate score Hg * theta (negate to remove top percent)
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.grad * p.data).detach()
            p.grad.data.zero_()

        # normalize score
        all_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        norm = torch.abs(torch.sum(all_scores)) + self.eps
        for _, p in self.masked_parameters:
            self.scores[id(p)].div_(norm)


class SynFlow(Pruner):
    def __init__(self, masked_parameters):
        super(SynFlow, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):
      
        @torch.no_grad()
        def linearize(model):
            # model.double()
            signs = {}
            for name, param in model.state_dict().items():
                signs[name] = torch.sign(param)
                param.abs_()
            return signs

        @torch.no_grad()
        def nonlinearize(model, signs):
            # model.float()
            for name, param in model.state_dict().items():
                param.mul_(signs[name])
        
        signs = linearize(model)

        (data, _) = next(iter(dataloader))
        input_dim = list(data[0,:].shape)
        input = torch.ones([1] + input_dim).to(device)
        model.zero_grad()

        output = model(input)
        torch.sum(output).backward()
        
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.grad * p).detach().abs_()
            p.grad.data.zero_()

        nonlinearize(model, signs)


class DispFL(Pruner):
    def __init__(self, masked_parameters):
        super(DispFL, self).__init__(masked_parameters)
        self.sparsity = 0.5

    def func_masks(self, module):
        r"""Returns an iterator over modules masks, yielding the mask.
        """
        for name, buf in module.named_buffers():
            if "mask" in name:
                yield name

    def prunable(self, module, batchnorm, residual):
        r"""Returns boolean whether a module is prunable.
        """
        isprunable = isinstance(module, (layers.Linear, layers.Conv2d))
        if batchnorm:
            isprunable |= isinstance(module, (layers.BatchNorm1d, layers.BatchNorm2d))
        if residual:
            isprunable |= isinstance(module, (layers.Identity1d, layers.Identity2d))
        return isprunable


    def screen_gradients(self, train_data, device, model):
        # model = self.model
        # model.to(device)

        model.eval()
        # # # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        # # sample one epoch  of data
        model.zero_grad()
        (x, labels) = next(iter(train_data))
        x, labels = x.to(device), labels.to(device)
        log_probs = model.forward(x)
        loss = criterion(log_probs, labels.long())
        loss.backward()
        gradient={}

        name = 'any_mask'
        k = 0
        for module in filter(lambda p: self.prunable(p, False, False), model.modules()):
            for _, param in zip(self.func_masks(module), module.parameters(recurse=False)):
                if param is not module.bias:
                    tmp_name = str(k) + '+' + name
                    gradient[tmp_name.split('_')[0]] = param.grad
                    k = k + 1

        return gradient

    def fire_mask(self, masks, weights, drop_ratio, device):
        # drop_ratio = self.args.anneal_factor / 2 * (1 + np.cos((round * np.pi) / self.args.comm_round))
        new_masks = copy.deepcopy(masks)

        num_remove = {}
        for name in masks:
            num_non_zeros = torch.sum(masks[name])
            num_remove[name] = math.ceil(drop_ratio * num_non_zeros)
            temp_weights = torch.where(masks[name] > 0, torch.abs(weights[name.split('_')[0]]),
                                       100000 * torch.ones_like(weights[name.split('_')[0]]))
            x, idx = torch.sort(temp_weights.view(-1).to(device))
            new_masks[name].view(-1)[idx[:num_remove[name]]] = 0

        # for name in masks:
        #     num_non_zeros = torch.sum(masks[name])
        #     num_remove[name] = math.ceil(drop_ratio * num_non_zeros)
        #     temp_weights = torch.where(masks[name] > 0, torch.abs(weights[name]), 100000 * torch.ones_like(weights[name]))
        #     x, idx = torch.sort(temp_weights.view(-1).to(device))
        #     new_masks[name].view(-1)[idx[:num_remove[name]]] = 0
        return new_masks, num_remove


    # we only update the private components of client's mask
    def regrow_mask(self, masks,  num_remove, device, gradient=None):
        new_masks = copy.deepcopy(masks)
        for name in masks:
            temp = torch.where(masks[name] == 0, torch.abs(gradient[name.split('_')[0]]), -100000 * torch.ones_like(gradient[name.split('_')[0]]))
            sort_temp, idx = torch.sort(temp.view(-1).to(device), descending=True)
            new_masks[name].view(-1)[idx[:num_remove[name]]] = 1

        # if name not in public_layers:
                # if "conv" in name:
                # if not self.args.dis_gradient_check:
                #     temp = torch.where(masks[name] == 0, torch.abs(gradient[name]), -100000 * torch.ones_like(gradient[name]))
                #     sort_temp, idx = torch.sort(temp.view(-1).to(device), descending=True)
                #     new_masks[name].view(-1)[idx[:num_remove[name]]] = 1
                # else:
                #     temp = torch.where(masks[name] == 0, torch.ones_like(masks[name]),torch.zeros_like(masks[name]) )
                #     idx = torch.multinomial( temp.flatten().to(device),num_remove[name], replacement=False)
                #     new_masks[name].view(-1)[idx]=1
        return new_masks

    def score(self, model, loss, dataloader, device):

        # gradient = self.screen_gradients(dataloader, device, model)

        masks = {}
        weights = {}
        name = 'any_mask'
        k = 0
        for mask, param in self.masked_parameters:
            tmp_name = str(k)+'+'+name
            masks[tmp_name] = mask
            weights[tmp_name.split('_')[0]] = param
            k = k + 1

        # for k in model.state_dict():
        #     if 'mask' in k:
        #         masks[k] = model.state_dict()[k]
        #     else:
        #         weights[k] = model.state_dict()[k]

        masks, num_remove = self.fire_mask(masks, weights, self.sparsity, device)
        # masks = self.regrow_mask(masks, num_remove, device, gradient)
        self.scores = masks
        #
        #
        # for _, p in self.masked_parameters:
        #     self.scores[id(p)] = torch.clone(p.data).detach().abs_()
        #
    def mask(self, sparsity, scope):
        r"""Updates masks of model with scores by sparsity level parameter-wise.
        """
        k = 0
        name = 'any_mask'
        for mask, param in self.masked_parameters:
            tmp_name = str(k)+'+'+name
            zero = torch.tensor([0.], dtype=torch.float).to(mask.device)
            one = torch.tensor([1.], dtype=torch.float).to(mask.device)
            mask.copy_(torch.where(self.scores[tmp_name] == 0, zero, one))
            k = k + 1