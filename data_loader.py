import logging

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from datasets import CIFAR10_truncated

#logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# generate the non-IID distribution for all methods
def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

def _data_transforms_cifar10():
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, valid_transform

def load_cifar10_data(datadir):
    train_transform, test_transform = _data_transforms_cifar10()

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=train_transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=test_transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)

def partition_data(dataset, datadir, partition, n_nets, alpha):
    logging.info("*********partition data***************")
    X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    if partition == "hetero":
        K = 10
        N = y_train.shape[0]
        logging.info("N = " + str(N))
        net_dataidx_map = {}

        idx_batch = [[] for _ in range(n_nets)]
        # for each class in the dataset
        for k in range(K):
            min_size = 0
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            while min_size < 100:
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = proportions / proportions.sum()
                min_size = np.min(proportions) * len(idx_k)
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    else:
        print("partition = homo")
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts

# for centralized training
def get_dataloader_CIFAR10(datadir, train_bs, test_bs, dataidxs=None):
    dl_obj = CIFAR10_truncated

    transform_train, transform_test = _data_transforms_cifar10()

    train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl

# for local devices
def get_dataloader_test_CIFAR10(datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None):
    dl_obj = CIFAR10_truncated

    transform_train, transform_test = _data_transforms_cifar10()

    train_ds = dl_obj(datadir, dataidxs=dataidxs_train, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(datadir, dataidxs=dataidxs_test, train=False, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl

def load_partition_data_cifar10(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size):

    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts \
        = partition_data(dataset, data_dir, partition_method, client_number, partition_alpha)
    class_num = len(np.unique(y_train))
    logging.debug("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    train_data_global, test_data_global = get_dataloader_CIFAR10(data_dir, batch_size, batch_size)
    logging.debug("train_dl_global number = " + str(len(train_data_global)))
    logging.debug("test_dl_global number = " + str(len(test_data_global)))
    test_data_num = len(test_data_global)

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        dataidxs = net_dataidx_map[client_idx]
        local_data_num = len(dataidxs)
        data_local_num_dict[client_idx] = local_data_num
        logging.debug("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

        train_data_local, test_data_local = get_dataloader_CIFAR10(data_dir, batch_size, batch_size, dataidxs)
        logging.debug("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
    
    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num


def partition_data_ssl(dataset, datadir, partition, n_nets, alpha, plabel=0.1):
    logging.info("*********partition data***************")
    X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    K = 10
    N = y_train.shape[0]
    logging.info("N = " + str(N))
    net_dataidx_map_label = {}
    net_dataidx_map_unlabel = {}

    idx_batch_label = [[] for _ in range(n_nets)]
    idx_batch_unlabel = [[] for _ in range(n_nets)]
    # for each class in the dataset
    for k in range(K):
        min_size = 0
        idx_k = np.where(y_train == k)[0]
        np.random.shuffle(idx_k)
        while min_size < 100:
            proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
            ## Balance
            proportions = proportions / proportions.sum()
            min_size = np.min(proportions) * len(idx_k)
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        idxs = np.split(idx_k, proportions)
        idx_batch_label = [idx_j + idx[:int(len(idx)*plabel)].tolist() for idx_j, idx in zip(idx_batch_label, idxs)]
        idx_batch_unlabel = [idx_j + idx[int(len(idx)*plabel):].tolist() for idx_j, idx in zip(idx_batch_unlabel, idxs)]

    for j in range(n_nets):
        np.random.shuffle(idx_batch_label[j])
        np.random.shuffle(idx_batch_unlabel[j])
        net_dataidx_map_label[j] = idx_batch_label[j]
        net_dataidx_map_unlabel[j] = idx_batch_unlabel[j]

    traindata_cls_counts_l = record_net_data_stats(y_train, net_dataidx_map_label)
    traindata_cls_counts_u = record_net_data_stats(y_train, net_dataidx_map_unlabel)

    return X_train, y_train, X_test, y_test, net_dataidx_map_label, net_dataidx_map_unlabel, traindata_cls_counts_l, traindata_cls_counts_u


def load_partition_data_cifar10_ssl(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size, plabel):
    X_train, y_train, X_test, y_test, net_dataidx_map_label, net_dataidx_map_unlabel, traindata_cls_counts_l, traindata_cls_counts_u \
        = partition_data_ssl(dataset, data_dir, partition_method, client_number, partition_alpha, plabel)
    class_num = len(np.unique(y_train))
    logging.debug("traindata_cls_counts_label = " + str(traindata_cls_counts_l))
    logging.debug("traindata_cls_counts_unlabel = " + str(traindata_cls_counts_u))

    label_data_num = sum([len(net_dataidx_map_label[r]) for r in range(client_number)])
    unlabel_data_num = sum([len(net_dataidx_map_unlabel[r]) for r in range(client_number)])

    train_data_global, test_data_global = get_dataloader_CIFAR10(data_dir, batch_size, batch_size)
    logging.debug("train_dl_global number = " + str(len(train_data_global)))
    logging.debug("test_dl_global number = " + str(len(test_data_global)))
    test_data_num = len(test_data_global)

    # get local dataset
    label_local_num_dict = dict()
    label_local_dict = dict()
    unlabel_local_num_dict = dict()
    unlabel_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(client_number):
        dataidxs_l = net_dataidx_map_label[client_idx]
        local_label_num = len(dataidxs_l)
        label_local_num_dict[client_idx] = local_label_num
        logging.debug("client_idx = %d, local_label_number = %d" % (client_idx, local_label_num))

        dataidxs_u = net_dataidx_map_unlabel[client_idx]
        local_unlabel_num = len(dataidxs_u)
        unlabel_local_num_dict[client_idx] = local_unlabel_num
        logging.debug("client_idx = %d, local_unlabel_number = %d" % (client_idx, local_unlabel_num))

        label_data_local, test_data_local = get_dataloader_CIFAR10(data_dir, batch_size, batch_size, dataidxs_l)
        unlabel_data_local, _ = get_dataloader_CIFAR10(data_dir, batch_size, batch_size, dataidxs_u)

        logging.debug("client_idx = %d, batch_num_label_local = %d, batch_num_unlabel_local = %d, batch_num_test_local = %d" % (
            client_idx, len(label_data_local), len(unlabel_data_local), len(test_data_local)))

        label_local_dict[client_idx] = label_data_local
        unlabel_local_dict[client_idx] = unlabel_data_local
        test_data_local_dict[client_idx] = test_data_local

    return label_data_num, unlabel_data_num, test_data_num, train_data_global, test_data_global, \
        label_local_num_dict, unlabel_local_num_dict, label_local_dict, unlabel_local_dict, \
        test_data_local_dict, class_num