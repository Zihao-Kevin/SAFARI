import torch
import numpy as np
from torchvision import transforms
import torch.optim as optim
from Models import mlp
from Models import lottery_vgg
from Models import lottery_resnet
from Models import tinyimagenet_vgg
from Models import tinyimagenet_resnet
from Models import imagenet_vgg
from Models import imagenet_resnet
from Pruners import pruners
from Utils import custom_datasets
from torch.utils.data import Dataset
import datasets
from torch.utils.data.dataloader import default_collate


def fetch_dataset(data_name, subset):
    dataset = {}
    print('fetching data {}...'.format(data_name))
    root = './data/{}'.format(data_name)
    if data_name == 'mnist':
        dataset['train'] = datasets.MNIST(root=root, split='train', subset=subset, transform=datasets.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        dataset['test'] = datasets.MNIST(root=root, split='test', subset=subset, transform=datasets.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))

    elif data_name == 'cifar10':
        dataset['train'] = datasets.CIFAR10(root=root, split='train', subset=subset, transform=datasets.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
        dataset['test'] = datasets.CIFAR10(root=root, split='test', subset=subset, transform=datasets.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
    elif data_name in ['PennTreebank', 'WikiText2', 'WikiText103']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\')'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\')'.format(data_name))
    else:
        raise ValueError('Not valid dataset name')
    print('data ready')
    return dataset


class SplitDataset(Dataset):
    def __init__(self, dataset, idx):
        super().__init__()
        self.dataset = dataset
        self.idx = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        input = self.dataset[self.idx[index]]
        return input


def iid(dataname, dataset, num_users):
    if dataname in ['mnist', 'cifar10']:
        label = torch.tensor(dataset.target)
    elif dataname in ['WikiText2']:
        label = dataset.token
    else:
        raise ValueError('Not valid data name')
    num_items = int(len(dataset) / num_users)
    data_split, idx = {}, list(range(len(dataset)))
    label_split = {}
    for i in range(num_users):
        num_items_i = min(len(idx), num_items)
        data_split[i] = torch.tensor(idx)[torch.randperm(len(idx))[:num_items_i]].tolist()
        label_split[i] = torch.unique(label[data_split[i]]).tolist()
        idx = list(set(idx) - set(data_split[i]))
    return data_split, label_split


def non_iid(dataname, dataset, num_users, label_split=None):
    label = np.array(dataset.target)
    shared_per_user = 5
    classes_size = 10
    data_split = {i: [] for i in range(num_users)}
    label_idx_split = {}
    for i in range(len(label)):
        label_i = label[i].item()
        if label_i not in label_idx_split:
            label_idx_split[label_i] = []
        label_idx_split[label_i].append(i)
    shard_per_class = int(shared_per_user * num_users / classes_size)
    for label_i in label_idx_split:
        label_idx = label_idx_split[label_i]
        num_leftover = len(label_idx) % shard_per_class
        leftover = label_idx[-num_leftover:] if num_leftover > 0 else []
        new_label_idx = np.array(label_idx[:-num_leftover]) if num_leftover > 0 else np.array(label_idx)
        new_label_idx = new_label_idx.reshape((shard_per_class, -1)).tolist()
        for i, leftover_label_idx in enumerate(leftover):
            new_label_idx[i] = np.concatenate([new_label_idx[i], [leftover_label_idx]])
        label_idx_split[label_i] = new_label_idx
    if label_split is None:
        label_split = list(range(classes_size)) * shard_per_class
        label_split = torch.tensor(label_split)[torch.randperm(len(label_split))].tolist()
        label_split = np.array(label_split).reshape((num_users, -1)).tolist()
        for i in range(len(label_split)):
            label_split[i] = np.unique(label_split[i]).tolist()
    for i in range(num_users):
        for label_i in label_split[i]:
            idx = torch.arange(len(label_idx_split[label_i]))[torch.randperm(len(label_idx_split[label_i]))[0]].item()
            data_split[i].extend(label_idx_split[label_i].pop(idx))
    return data_split, label_split


class Non_iid(Dataset):
    def __init__(self, x, y):
        self.x_data = x.unsqueeze(1).to(torch.float32)
        self.y_data = y.to(torch.int64)
        self.batch_size = 32  # set batchsize in here
        self.cuda_available = torch.cuda.is_available()

    # Return the number of data
    def __len__(self):
        return len(self.x_data)

    # Sampling
    def __getitem__(self):
        idx = np.random.randint(low=0, high=len(self.x_data), size=self.batch_size)  # random_index
        x = self.x_data[idx]
        y = self.y_data[idx]
        if self.cuda_available:
            return x.cuda(), y.cuda()
        else:
            return x, y


def dirichlet(dataname, dataset, num_users, alpha):
    classes_size = 10  # number of class in the dataset
    labelList = np.array(dataset.target)

    min_size = 0
    N = len(labelList)
    np.random.seed(2022)

    idx_batch = None
    client_dataidx_map = {}
    while min_size < classes_size:
        idx_batch = [[] for _ in range(num_users)]
        # for each class in the dataset
        for k in range(classes_size):
            idx_k = np.where(labelList == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_users))
            # Balance, if num is alreadly larger than N / num_users, stop incorporating.
            proportions = np.array([p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            # split and incorporate two lists
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(num_users):
        np.random.shuffle(idx_batch[j])
        client_dataidx_map[j] = idx_batch[j]

    client_cls_counts = {}
    label_split = []
    for client_i, dataidx in client_dataidx_map.items():
        unq, unq_cnt = np.unique(labelList[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        client_cls_counts[client_i] = tmp
        label_split.append(list(client_cls_counts[client_i].keys()))
    print('Data statistics: %s' % str(client_cls_counts))

    local_sizes = []
    for i in range(num_users):
        local_sizes.append(len(client_dataidx_map[i]))
    local_sizes = np.array(local_sizes)
    weights = local_sizes / np.sum(local_sizes)
    print(weights)

    # return idx_batch, weights, client_cls_counts
    return idx_batch, label_split

def split_datasets(args, dataset, num_users, data_split_mode):
    data_split = {}
    if data_split_mode == 'iid':
        data_split['train'], label_split = iid(args.dataset, dataset['train'], num_users)
        data_split['test'], _ = iid(args.dataset, dataset['test'], num_users + 1)
    elif 'non-iid' in data_split_mode:
        label_split = []
        # for i in range(int(num_users/5)):
        #     label_split.append([0, 1])
        #     label_split.append([2, 3])
        #     label_split.append([4, 5])
        #     label_split.append([6, 7])
        #     label_split.append([8, 9])

        for i in range(int(num_users/2)):
            label_split.append([0, 1, 2, 3, 4])
        for i in range(int(num_users/2)):
            label_split.append([5, 6, 7, 8, 9])

        data_split['train'], _ = non_iid(args.dataset, dataset['train'], num_users, label_split)
        data_split['test'], _ = iid(args.dataset, dataset['test'], num_users + 1)
    elif data_split_mode == 'dir':
        alpha = args.dir_alpha
        data_split['train'], label_split = dirichlet(args.dataset, dataset['train'], num_users, alpha)
        data_split['test'], _ = iid(args.dataset, dataset['test'], num_users + 1)
    else:
        raise ValueError('Not valid data split mode')
    return data_split, label_split


def input_collate(batch):
    if isinstance(batch[0], dict):
        output = {key: [] for key in batch[0].keys()}
        for b in batch:
            for key in b:
                output[key].append(b[key])
        return output
    else:
        return default_collate(batch)


def make_data_loader(dataset, batch_size, train, workers, length=None):
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': workers, 'pin_memory': True} if use_cuda else {}
    shuffle = train is True
    if length is not None:
        indices = torch.randperm(len(dataset))[:length]
        dataset = torch.utils.data.Subset(dataset, indices)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, shuffle=shuffle,
                                                 batch_size=batch_size, **kwargs, collate_fn=input_collate)

    return data_loader


def device(gpu):
    use_cuda = torch.cuda.is_available()
    return torch.device(("cuda:" + str(gpu)) if use_cuda else "cpu")


def dimension(dataset):
    if dataset == 'mnist':
        input_shape, num_classes = (1, 28, 28), 10
    if dataset == 'cifar10':
        input_shape, num_classes = (3, 32, 32), 10
    if dataset == 'cifar100':
        input_shape, num_classes = (3, 32, 32), 100
    if dataset == 'tiny-imagenet':
        input_shape, num_classes = (3, 64, 64), 200
    if dataset == 'imagenet':
        input_shape, num_classes = (3, 224, 224), 1000
    return input_shape, num_classes


def get_transform(size, padding, mean, std, preprocess):
    transform = []
    if preprocess:
        transform.append(transforms.RandomCrop(size=size, padding=padding))
        transform.append(transforms.RandomHorizontalFlip())
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(mean, std))
    return transforms.Compose(transform)


def model(model_architecture, model_class):
    default_models = {
        'fc': mlp.fc,
        'conv': mlp.conv,
    }
    lottery_models = {
        'vgg11': lottery_vgg.vgg11,
        'vgg11-bn': lottery_vgg.vgg11_bn,
        'vgg13': lottery_vgg.vgg13,
        'vgg13-bn': lottery_vgg.vgg13_bn,
        'vgg16': lottery_vgg.vgg16,
        'vgg16-bn': lottery_vgg.vgg16_bn,
        'vgg19': lottery_vgg.vgg19,
        'vgg19-bn': lottery_vgg.vgg19_bn,
        'resnet20': lottery_resnet.resnet20,
        'resnet32': lottery_resnet.resnet32,
        'resnet44': lottery_resnet.resnet44,
        'resnet56': lottery_resnet.resnet56,
        'resnet110': lottery_resnet.resnet110,
        'resnet1202': lottery_resnet.resnet1202,
        'wide-resnet20': lottery_resnet.wide_resnet20,
        'wide-resnet32': lottery_resnet.wide_resnet32,
        'wide-resnet44': lottery_resnet.wide_resnet44,
        'wide-resnet56': lottery_resnet.wide_resnet56,
        'wide-resnet110': lottery_resnet.wide_resnet110,
        'wide-resnet1202': lottery_resnet.wide_resnet1202
    }
    tinyimagenet_models = {
        'vgg11': tinyimagenet_vgg.vgg11,
        'vgg11-bn': tinyimagenet_vgg.vgg11_bn,
        'vgg13': tinyimagenet_vgg.vgg13,
        'vgg13-bn': tinyimagenet_vgg.vgg13_bn,
        'vgg16': tinyimagenet_vgg.vgg16,
        'vgg16-bn': tinyimagenet_vgg.vgg16_bn,
        'vgg19': tinyimagenet_vgg.vgg19,
        'vgg19-bn': tinyimagenet_vgg.vgg19_bn,
        'resnet18': tinyimagenet_resnet.resnet18,
        'resnet34': tinyimagenet_resnet.resnet34,
        'resnet50': tinyimagenet_resnet.resnet50,
        'resnet101': tinyimagenet_resnet.resnet101,
        'resnet152': tinyimagenet_resnet.resnet152,
        'wide-resnet18': tinyimagenet_resnet.wide_resnet18,
        'wide-resnet34': tinyimagenet_resnet.wide_resnet34,
        'wide-resnet50': tinyimagenet_resnet.wide_resnet50,
        'wide-resnet101': tinyimagenet_resnet.wide_resnet101,
        'wide-resnet152': tinyimagenet_resnet.wide_resnet152,
    }
    imagenet_models = {
        'vgg11': imagenet_vgg.vgg11,
        'vgg11-bn': imagenet_vgg.vgg11_bn,
        'vgg13': imagenet_vgg.vgg13,
        'vgg13-bn': imagenet_vgg.vgg13_bn,
        'vgg16': imagenet_vgg.vgg16,
        'vgg16-bn': imagenet_vgg.vgg16_bn,
        'vgg19': imagenet_vgg.vgg19,
        'vgg19-bn': imagenet_vgg.vgg19_bn,
        'resnet18': imagenet_resnet.resnet18,
        'resnet34': imagenet_resnet.resnet34,
        'resnet50': imagenet_resnet.resnet50,
        'resnet101': imagenet_resnet.resnet101,
        'resnet152': imagenet_resnet.resnet152,
        'wide-resnet50': imagenet_resnet.wide_resnet50_2,
        'wide-resnet101': imagenet_resnet.wide_resnet101_2,
    }
    models = {
        'default': default_models,
        'lottery': lottery_models,
        'tinyimagenet': tinyimagenet_models,
        'imagenet': imagenet_models
    }
    if model_class == 'imagenet':
        print("WARNING: ImageNet models do not implement `dense_classifier`.")
    return models[model_class][model_architecture]


def pruner(method):
    prune_methods = {
        'rand': pruners.Rand,
        'mag': pruners.Mag,
        'snip': pruners.SNIP,
        'grasp': pruners.GraSP,
        'synflow': pruners.SynFlow,
        'dispfl': pruners.DispFL,
    }
    return prune_methods[method]


def optimizer(optimizer):
    optimizers = {
        'adam': (optim.Adam, {}),
        'sgd': (optim.SGD, {}),
        'momentum': (optim.SGD, {'momentum': 0.9, 'nesterov': True}),
        'rms': (optim.RMSprop, {}),
        'LBFGS': (optim.LBFGS, {})
    }
    return optimizers[optimizer]

