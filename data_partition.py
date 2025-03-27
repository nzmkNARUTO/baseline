import numpy as np
import torch
import torchvision
import random
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils.data import ConcatDataset

# 设置 matplotlib 显示中文和负号
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号
plt.rcParams["font.size"] = 25


def missing_class_split(train_labels, n_clients, n_missing_classes):
    # 获取类别数量
    n_classes = train_labels.max() + 1

    if n_missing_classes >= n_classes:
        raise ValueError(
            f"缺失类别数量 {n_missing_classes} 不能大于或等于总类别数量 {n_classes}"
        )

    # 为每个客户端确定缺失的类别
    client_missing_classes = []
    for i in range(n_clients):
        # 为每个客户端随机选择 n_missing_classes 个类别作为缺失类别
        missing_classes = random.sample(range(n_classes), n_missing_classes)
        client_missing_classes.append(missing_classes)

    # 确定每个类别被哪些客户端持有
    class_to_clients = defaultdict(list)
    for client_id in range(n_clients):
        for class_id in range(n_classes):
            if class_id not in client_missing_classes[client_id]:
                class_to_clients[class_id].append(client_id)

    # 记录每个类别对应的样本索引
    class_idcs = [np.argwhere(train_labels == y).flatten() for y in range(n_classes)]

    # 为每个客户端分配样本
    client_idcs = [[] for _ in range(n_clients)]

    for class_id, idcs in enumerate(class_idcs):
        # 找出拥有该类别的客户端
        available_clients = class_to_clients[class_id]
        if not available_clients:
            print(f"警告: 类别 {class_id} 没有被任何客户端持有")
            continue

        # 将该类别的样本平均分配给拥有该类别的客户端
        n_available_clients = len(available_clients)
        samples_per_client = len(idcs) // n_available_clients

        for i, client_id in enumerate(available_clients):
            # 最后一个客户端可能会获得额外的样本（如果不能整除）
            if i == n_available_clients - 1:
                client_idcs[client_id] += idcs[i * samples_per_client :].tolist()
            else:
                client_idcs[client_id] += idcs[
                    i * samples_per_client : (i + 1) * samples_per_client
                ].tolist()

    return client_idcs, client_missing_classes


def dirichlet_split_noniid(train_labels, alpha, n_clients):
    """
    按照参数为alpha的Dirichlet分布将样本索引集合划分为n_clients个子集
    """
    n_classes = train_labels.max() + 1
    # (K, N) 类别标签分布矩阵X，记录每个类别划分到每个client去的比例
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    # (K, ...) 记录K个类别对应的样本索引集合
    class_idcs = [np.argwhere(train_labels == y).flatten() for y in range(n_classes)]

    # 记录N个client分别对应的样本索引集合
    client_idcs = [[] for _ in range(n_clients)]
    for k_idcs, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例fracs将类别为k的样本索引k_idcs划分为了N个子集
        # i表示第i个client，idcs表示其对应的样本索引集合idcs
        for i, idcs in enumerate(
            np.split(k_idcs, (np.cumsum(fracs)[:-1] * len(k_idcs)).astype(int))
        ):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs


n_clients = 5
dirichlet_alpha = 1.0
seed = 42


if __name__ == "__main__":
    np.random.seed(seed)
    train_data = datasets.MNIST(root="./data", download=True, train=True)
    test_data = datasets.MNIST(root="./data", download=True, train=False)

    classes = train_data.classes
    n_classes = len(classes)

    labels = np.concatenate(
        [np.array(train_data.targets), np.array(test_data.targets)], axis=0
    )
    dataset = ConcatDataset([train_data, test_data])

    # 我们让每个client不同label的样本数量不同，以此做到Non-IID划分
    # client_idcs = dirichlet_split_noniid(
    #     labels, alpha=dirichlet_alpha, n_clients=n_clients
    # )

    client_idcs, _ = missing_class_split(labels, n_clients, 5)

    # 展示不同label划分到不同client的情况
    plt.figure(figsize=(12, 8))
    label_distribution = [[] for _ in range(n_classes)]
    for c_id, idc in enumerate(client_idcs):
        for idx in idc:
            label_distribution[labels[idx]].append(c_id)

    plt.hist(
        label_distribution,
        stacked=True,
        bins=np.arange(-0.5, n_clients + 0.5, 1),
        label=classes,
        rwidth=0.5,
    )
    plt.xticks(
        np.arange(n_clients),
        ["Client %d" % c_id for c_id in range(n_clients)],
    )
    plt.xlabel("客户端", fontsize=30)
    plt.ylabel("数据量", fontsize=30)
    # plt.legend()
    # plt.title("Display Label Distribution on Different Clients")
    plt.show()
    plt.savefig("label_distribution.png")
