from fed_baselines.server_base import FedServer
from utils.models import *
import torch
from torch.utils.data import DataLoader, random_split
from utils.fed_utils import assign_dataset, init_model


class FedTestServer(FedServer):
    def __init__(self, client_list, dataset_id, model_name):
        super().__init__(client_list, dataset_id, model_name)

    def agg(self):
        """
        Server aggregates models from connected clients.
        :return: model_state: Updated global model after aggregation
        :return: avg_loss: Averaged loss value
        :return: n_data: Number of the local data points
        """
        client_num = len(self.selected_clients)
        if client_num == 0 or self.n_data == 0:
            return self.model.state_dict(), 0, 0

        model_state = self.model.state_dict()
        avg_loss = 0

        sensitivities = []
        loss_func = nn.CrossEntropyLoss().to(self._device)
        dataloader = DataLoader(
            self.analysisset, batch_size=len(self.analysisset.indices), shuffle=True
        )
        for i in range(self.analysisset.dataset.targets.max() + 1):
            sensitivity = {}
            for data, target in dataloader:
                data = data.to(self._device)
                target = target.to(self._device)
                output = self.model(data)
                loss = loss_func(
                    output,
                    torch.zeros_like(target) + i,
                )
                self.model.zero_grad()
                loss.backward()
            for name, param in self.model.named_parameters():
                sensitivity.update({name: param.grad})
            sensitivities.append(sensitivity)

        delta_state = {}
        for name in self.selected_clients:
            delta_state[name] = self.model.state_dict()
        for name in self.selected_clients:
            if name not in self.client_state:
                continue
            for key in self.client_state[name]:
                delta_state[name][key] = self.client_state[name][key] - model_state[key]

        # Aggregate the local updated models from selected clients
        for i, name in enumerate(self.selected_clients):
            if name not in self.client_state:
                continue
            for key in self.client_state[name]:
                model_state[key] += (
                    delta_state[name][key] * self.client_n_data[name] / self.n_data
                )

            avg_loss = (
                avg_loss
                + self.client_loss[name] * self.client_n_data[name] / self.n_data
            )

        # Server load the aggregated model as the global model
        self.model.load_state_dict(model_state)
        self.round = self.round + 1
        n_data = self.n_data

        return model_state, avg_loss, n_data

    def load_testset(self, testset):
        """
        Server loads the test dataset.
        :param data: Dataset for testing.
        """
        test_size = int(len(testset) * 0.9)
        analysis_size = len(testset) - test_size
        self.testset, self.analysisset = random_split(
            testset, [test_size, analysis_size]
        )
