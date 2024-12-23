from fed_baselines.server_base import FedServer
import copy
import torch
from torch.utils.data import DataLoader, random_split
from utils.models import *
import math


class FedNovaPlusServer(FedServer):

    def __init__(self, client_list, dataset_id, model_name, len_class, num_round, x):
        super().__init__(client_list, dataset_id, model_name)
        # Normalized coefficient
        self.client_coeff = {}
        # Normalized gradients
        self.client_norm_grad = {}
        self.len_class = len_class
        self.num_round = num_round
        self.x = x
        self.all_clients_data_distribution = {}
        self.selected_clients_data_distribution = {}

    def calculate_data_distribution(self):
        for name in self.selected_clients:
            self.selected_clients_data_distribution[name] = (
                self.all_clients_data_distribution[name]
            )
            # Fill in the missing classes
            for i in range(self.len_class):
                if i not in self.selected_clients_data_distribution[name]:
                    self.selected_clients_data_distribution[name][i] = 0
        # Calculate the data distribution
        for name in self.selected_clients:
            for i in range(self.len_class):
                self.selected_clients_data_distribution[name][i] = (
                    self.selected_clients_data_distribution[name][i]
                    / self.client_n_data[name]
                )
        pass

    def analysis_sensitivity(self):
        self.calculate_data_distribution()
        sensitivities = {}
        loss_func = nn.CrossEntropyLoss().to(self._device)
        for i in range(self.len_class):
            dataloader = DataLoader(
                self.analysisset,
                batch_size=len(self.analysisset.indices),
                shuffle=False,
            )
            sensitivity = {}
            for data, target in dataloader:
                data = data.to(self._device)
                target = target.to(self._device)
                output = self.model(data)
                loss = loss_func(
                    output,
                    (torch.zeros_like(target) + i).long(),
                )
                self.model.zero_grad()
                loss.backward()
            for name, param in self.model.named_parameters():
                sensitivity.update({name: param.grad})
            sensitivities[i] = copy.deepcopy(sensitivity)
        return sensitivities

    def calculate_gradient(self, model_state):
        # calculate the gradient
        delta_state = {}
        for name in self.selected_clients:
            delta_state[name] = model_state[name]
        return delta_state

    def reweight_gradient(self, sensitivities, delta_state):
        # reweight the gradient
        for name in self.selected_clients:
            mask = {}
            for i in range(self.len_class):
                x = self.x
                ratio = self.selected_clients_data_distribution[name][i] * (1 - x) + x
                # ratio = 1
                mask[i] = self.mask(sensitivities[i], self.toOne(ratio))
            for key in delta_state[name]:
                delta_state[name][key] = sum(
                    [
                        delta_state[name][key]
                        * mask[i][key]
                        * self.toUniform(
                            self.selected_clients_data_distribution[name][i]
                        )
                        for i in range(self.len_class)
                    ]
                )
        return delta_state

    def agg(self):
        """
        Server aggregates normalized models from connected clients using FedNova
        :return: Updated global model after aggregation, Averaged loss value, Number of the local data points
        """
        client_num = len(self.selected_clients)
        if client_num == 0 or self.n_data == 0:
            return self.model.state_dict(), 0, 0

        self.model.to(self._device)

        model_state = self.model.state_dict()
        nova_model_state = copy.deepcopy(model_state)
        avg_loss = 0
        coeff = 0.0

        # # analysis the sensitivity of each class
        sensitivities = self.analysis_sensitivity()

        # calculate the gradient
        delta_state = self.calculate_gradient(self.client_norm_grad)

        # # reweight the gradient
        delta_state = self.reweight_gradient(sensitivities, delta_state)

        for i, name in enumerate(self.selected_clients):
            coeff = (
                coeff + self.client_coeff[name] * self.client_n_data[name] / self.n_data
            )
            for key in self.client_state[name]:
                if i == 0:
                    nova_model_state[key] = (
                        delta_state[name][key] * self.client_n_data[name] / self.n_data
                    )
                else:
                    nova_model_state[key] += (
                        delta_state[name][key] * self.client_n_data[name] / self.n_data
                    )
            avg_loss = (
                avg_loss
                + self.client_loss[name] * self.client_n_data[name] / self.n_data
            )

        for key in model_state:
            model_state[key] -= coeff * nova_model_state[key]

        self.model.load_state_dict(model_state)

        self.round = self.round + 1

        return model_state, avg_loss, self.n_data

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

    def topk(self, params: dict, ratio):
        topkParams = {}
        for key in params.keys():
            topkParams[key] = torch.topk(
                torch.abs(params[key].reshape(-1)),
                math.ceil(params[key].numel() * ratio),
                largest=True,
            )
        return topkParams

    def mask(self, sensitivities, ratio):
        sensitivities = self.topk(sensitivities, ratio)
        mask = {}
        for key in sensitivities.keys():
            mask[key] = torch.zeros_like(self.model.state_dict()[key])
            mask[key].reshape(-1)[sensitivities[key][1]] = 1
        return mask

    def toUniform(self, data_distribution):
        # return data_distribution
        average = 1 / self.len_class
        return (
            data_distribution
            + (average - data_distribution) / self.num_round * self.round
        )

    def toOne(self, data_distribution):
        # return data_distribution
        return data_distribution + (1 - data_distribution) / self.num_round * self.round

    def rec(
        self,
        name,
        state_dict,
        n_data,
        loss,
        coeff,
        norm_grad,
        client_id,
        client_data_distribution,
    ):
        """
        Server receives the local updates from the connected client k.
        :param name: Name of client k
        :param state_dict: Model dict from the client k
        :param n_data: Number of local data points in the client k
        :param loss: Loss of local training in the client k
        :param coeff: Normalization coefficient
        :param norm_grad: Normalized gradients
        """
        self.n_data = self.n_data + n_data
        self.client_state[name] = {}
        self.client_n_data[name] = {}
        self.client_coeff[name] = -1
        self.client_norm_grad[name] = {}

        self.client_state[name].update(state_dict)
        self.client_n_data[name] = n_data
        self.client_loss[name] = {}
        self.client_loss[name] = loss
        self.client_coeff[name] = coeff
        self.client_norm_grad[name].update(norm_grad)
        self.all_clients_data_distribution[client_id] = client_data_distribution

    def flush(self):
        """
        Flushing the client information in the server
        """
        self.n_data = 0
        self.client_state = {}
        self.client_n_data = {}
        self.client_loss = {}
        self.client_coeff = {}
        self.client_norm_grad = {}
        self.selected_clients_data_distribution = {}
