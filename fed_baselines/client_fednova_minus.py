import copy
from fed_baselines.client_base import FedClient
from utils.models import *
from torch.utils.data import DataLoader
from collections import Counter


class FedNovaMinusClient(FedClient):

    def __init__(self, name, epoch, dataset_id, model_name, batch_size, lr):
        super().__init__(name, epoch, dataset_id, model_name, batch_size, lr)
        self.rho = 0.9
        self._momentum = self.rho

    def train(self):
        """
        Client trains the model on local dataset using FedNova
        :return: Local updated model, number of local data points, training loss, normalization coefficient, normalized gradients
        """
        train_loader = DataLoader(
            self.trainset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        self.model.to(self._device)
        global_weights = copy.deepcopy(self.model.state_dict())

        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self._lr, momentum=self._momentum
        )
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr, weight_decay=1e-4)
        loss_func = nn.CrossEntropyLoss()

        tau = 0
        for epoch in range(self._epoch):
            for step, (x, y) in enumerate(train_loader):
                with torch.no_grad():
                    b_x = x.to(self._device)  # Tensor on GPU
                    b_y = y.to(self._device)  # Tensor on GPU

                with torch.enable_grad():
                    self.model.train()
                    output = self.model(b_x)
                    loss = loss_func(output, b_y.long())
                    optimizer.zero_grad()

                    loss.backward()
                    optimizer.step()

                    tau += 1

        coeff = (tau - self.rho * (1 - pow(self.rho, tau)) / (1 - self.rho)) / (
            1 - self.rho
        )

        state_dict = self.model.state_dict()
        norm_grad = copy.deepcopy(global_weights)
        for key in norm_grad:
            norm_grad[key] = torch.div(global_weights[key] - state_dict[key], coeff)

        # nonzero_count, total_count, sparsity = self.get_sparsity()
        # print(f"Client {self.name} sparsity: {sparsity:.2f}%")

        return (
            self.model.state_dict(),
            self.n_data,
            loss.data.cpu().numpy(),
            coeff,
            norm_grad,
        )

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
