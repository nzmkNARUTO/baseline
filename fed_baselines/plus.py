from collections import Counter


class ServerPlus:
    def __init__(self):
        pass


class ClientPlus:
    def load_trainset(self, trainset):
        """
        Client loads the training dataset.
        :param trainset: Dataset for training.
        """
        self.trainset = trainset
        self.n_data = len(trainset)
        labels = [int(trainset.dataset.targets[i]) for i in trainset.indices]
        self.V = Counter(labels)

    def get_data_distribution(self):
        return dict(self.V)
