from fed_baselines.client_base import FedClient
import copy
from collections import Counter
from utils.models import *

from torch.utils.data import DataLoader


class FedAvgMinusClient(FedClient):

    def __init__(self, name, epoch, dataset_id, model_name, batch_size, lr):
        super().__init__(name, epoch, dataset_id, model_name, batch_size, lr)

    def load_trainset(self, trainset):
        """
        Client loads the training dataset.
        :param trainset: Dataset for training.
        """
        self.trainset = trainset
        self.n_data = len(trainset)
        labels = [int(trainset.dataset.targets[i]) for i in trainset.indices]
        self.V = Counter(labels)
        pass

    def get_data_distribution(self):
        return dict(self.V)
