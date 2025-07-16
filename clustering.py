import torch
import sys
sys.path.append("..")
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from collections import defaultdict

def cluster_kmeans(data, feature, k, seed):
    cluster_datasets = []
    cluster_fea_sets = []
    for i in range(k):
        cluster_datasets.append([])
        cluster_fea_sets.append([])
    # 设置 K-means 聚类的数量
    kmeans = KMeans(n_clusters=k, random_state=seed)
    # 进行聚类
    kmeans.fit(np.array(feature))
    cluster_idx_list = kmeans.labels_
    centroids = kmeans.cluster_centers_

    for i in range(len(cluster_idx_list)):
        cluster_datasets[cluster_idx_list[i]].append(data[i])
        cluster_fea_sets[cluster_idx_list[i]].append(feature[i])

    return cluster_datasets, cluster_fea_sets, centroids

def init_thresholds(cluster_dataset, k, centroids, percent=None):

    thresholds = defaultdict(int) # for each i-th cluster
    for i in range(k):
        dist_set = []
        # cluster-->dist-->mediam
        for feature in cluster_dataset[i]:
            dist = np.linalg.norm(feature - centroids[i])
            dist_set.append(dist)
        if percent is not None:
            thresholds[i] = np.percentile(dist_set, percent)
        else:
            thresholds[i] = np.median(dist_set)
    return thresholds

def drop(cluster_dataset, cluster_fea_set, device, k, centroids, thresholds, batch_size):
    new_dataset = []
    new_fea_set = []
    for i in range(k):
        new_dataset.append([])
        new_fea_set.append([])
        # cluster-->dist-->drop-->save
        for id, feature in enumerate(cluster_fea_set[i]):
            dist = np.linalg.norm(feature - centroids[i])
            if dist < thresholds[i]:
                new_dataset[i].append(cluster_dataset[i][id])
                new_fea_set[i].append(cluster_fea_set[i][id])

    return new_dataset, new_fea_set

def vote(cluster_dataset, cluster_fea_set, model, device, k, centroids, thresholds, batch_size):
    results = []
    model.to(device)
    model.eval()
    cluster_dataset, cluster_fea_set = drop(cluster_dataset, cluster_fea_set, device, k, centroids, thresholds, batch_size)
    with torch.no_grad():
        for i in range(k): # the i-th cluster
            results.append(defaultdict(int))
            if len(cluster_dataset[i]) == 0:
                continue
            dl_i = DataLoader(dataset=cluster_dataset[i], batch_size=batch_size, shuffle=True, drop_last=True)
            for batch_idx, (feature, target) in enumerate(dl_i):
                feature, target = feature.to(device), target.to(device)
                output = model(feature)
                pred = output.argmax(dim=1)
                for j in range(pred.size(0)):
                    results[i][pred[j].item()] += 1
    return results, cluster_dataset, cluster_fea_set

def truth_count(cluster_dataset, k):
    truth_counts = []
    for i in range(k): # the i-th cluster
        truth_counts.append(defaultdict(int))
        for batch_idx, (feature, target) in enumerate(cluster_dataset[i]):
            truth_counts[i][target.item()] += 1

    # check main class rate for each cluster
    print("truth_counts:")
    print(truth_counts)
    columns = ['cluster_id', 'sum', 'label1', 'prop1', 'label2', 'prop2']
    df = pd.DataFrame(columns=columns)

    for i in range(k):
        total = sum(truth_counts[i].values())
        unique_values = list(set(truth_counts[i].values()))
        unique_values.sort()
        label1 = max(truth_counts[i], key=truth_counts[i].get)
        prop1 = truth_counts[i][label1] / total
        if len(unique_values) >= 2:
            value2 = unique_values[-2]
            label2 = [key for key, value in truth_counts[i].items() if value == value2]
            prop2 = value2 / total
        else:
            label2, prop2 = 'none', 0  # 如果没有第二大的值

        df.loc[len(df)] = [i, total, label1, prop1, label2, prop2]

    return df # predicted result for i-th cluster

def estep(args, model, device, labeled_dataset, cluster_dataset, cluster_fea_set,
          k, centroids, thresholds, batch_size, r1, r2, r3, t1, t2):
    """
    dist_threshold_drop-->predict-->vote (and drop)-->pseudo
    """
    mixed_dataset = []
    for batch_idx, (data, target) in enumerate(labeled_dataset):
        mixed_dataset.append((data, target))

    columns = ['cluster_id', 'sum', 'label1', 'prop1', 'label2', 'prop2']
    df = pd.DataFrame(columns=columns)
    # vote
    results, cluster_dataset, cluster_fea_set = (
        vote(cluster_dataset, cluster_fea_set, model, device, k, centroids, thresholds, batch_size))
    print("pred_vote:")
    print(results)
    # add pseudo
    pseudo_label_dataset = []
    for i in range(k):
        # record voting results
        total = sum(results[i].values())
        if total == 0:
            # thresholds[i] = thresholds[i] * 1.1
            df.loc[len(df)] = [i, 0, 'none', 0, 'none', 0]
            continue

        unique_values = list(set(results[i].values()))
        unique_values.sort()
        pseudo = max(results[i], key=results[i].get)
        prop1 = results[i][pseudo] / total
        if len(unique_values) >= 2:
            value2 = unique_values[-2]
            label2 = [key for key, value in results[i].items() if value == value2]
            prop2 = value2 / total
        else:
            label2, prop2 = 'none', 0  # 如果没有第二大的值

        df.loc[len(df)] = [i, total, pseudo, prop1, label2, prop2]

        if prop1 < t1:
            # thresholds[i] = thresholds[i] * r1
            continue

        # add pseudo
        with torch.no_grad():
            dl_i = DataLoader(dataset=cluster_dataset[i], batch_size=batch_size)
            for batch_idx, (feature, target) in enumerate(dl_i):
                feature, target = feature.to(device), target.to(device)
                output = model(feature)
                pred = output.argmax(dim=1).cpu()
                for j in range(target.size(0)):
                    # flattened = feature[j].cpu().numpy().reshape(-1)
                    # dist = np.linalg.norm(flattened - centroids[i])
                    if pred[j] == pseudo:
                    # if dist < thresholds[i]: # use this --> more robust
                        pseudo_label_dataset.append((feature[j].cpu(), pseudo))

        # when majority ratio reach 80%, adapt threshold by 10%
        # if prop1 >= t2:
        #     thresholds[i] = thresholds[i] * r3
        # else:
        #     thresholds[i] = thresholds[i] * r2

    if len(pseudo_label_dataset) > 0:
        pseudo_label_loader = DataLoader(pseudo_label_dataset, batch_size=batch_size)
        for batch_idx, (data, target) in enumerate(pseudo_label_loader):
            mixed_dataset.append((data, target))

    return mixed_dataset, df, thresholds