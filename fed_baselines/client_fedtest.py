from fed_baselines.client_base import FedClient
import copy
from collections import Counter
from utils.models import *

from torch.utils.data import DataLoader


class FedTestClient(FedClient):
    def __init__(self, name, epoch, dataset_id, model_name):
        super().__init__(name, epoch, dataset_id, model_name)

    def load_trainset(self, trainset):
        """
        Client loads the training dataset.
        :param trainset: Dataset for training.
        """
        self.trainset = trainset
        self.n_data = len(trainset)
        for _, target in DataLoader(self.trainset, batch_size=self.n_data):
            self.V = Counter(target.numpy())

    def get_data_distribution(self):
        return dict(self.V)
