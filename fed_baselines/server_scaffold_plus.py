from utils.fed_utils import init_model
from fed_baselines.server_base import FedServer
import copy
import torch
import math
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, random_split


class ScaffoldPlusServer(FedServer):

    def __init__(self, client_list, dataset_id, model_name, len_class, num_round, x):
        super().__init__(client_list, dataset_id, model_name)
        # server control variate
        self.scv = init_model(
            model_name=self.model_name,
            num_class=self._num_class,
            image_channel=self._image_channel,
            image_dim=self._image_dim,
        )
        # Dict of all client control variates
        self.client_ccv_state = {}
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
                    torch.zeros_like(target) + i,
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
            delta_state[name] = self.model.state_dict()
        for name in self.selected_clients:
            if name not in self.client_state:
                continue
            for key in self.client_state[name]:
                delta_state[name][key] = self.client_state[name][key] - model_state[key]
        return delta_state

    def reweight_gradient(self, sensitivities, delta_state):
        # reweight the gradient
        for name in self.selected_clients:
            mask = {}
            for i in range(self.len_class):
                x = 1
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
        Server aggregates normalized models from connected clients using SCAFFOLD
        :return: Updated global model after aggregation, Averaged loss value, Number of the local data points, server control variate
        """
        client_num = len(self.selected_clients)
        if client_num == 0 or self.n_data == 0:
            return self.model.state_dict(), 0, 0

        self.scv.to(self._device)
        self.model.to(self._device)
        model_state = self.model.state_dict()
        new_scv_state = copy.deepcopy(model_state)
        avg_loss = 0

        # analysis the sensitivity of each class
        sensitivities = self.analysis_sensitivity()

        # calculate the gradient
        delta_state = self.calculate_gradient(model_state)

        # reweight the gradient
        delta_state = self.reweight_gradient(sensitivities, delta_state)

        # print('number of selected clients in Cloud: ' + str(client_num))
        for i, name in enumerate(self.selected_clients):
            if name not in self.client_state:
                continue
            for key in self.client_state[name]:

                if i == 0:
                    new_scv_state[key] = (
                        self.client_ccv_state[name][key]
                        * self.client_n_data[name]
                        / self.n_data
                    )
                else:
                    new_scv_state[key] = (
                        new_scv_state[key]
                        + self.client_ccv_state[name][key]
                        * self.client_n_data[name]
                        / self.n_data
                    )

                model_state[key] += (
                    delta_state[name][key] * self.client_n_data[name] / self.n_data
                )

            avg_loss = (
                avg_loss
                + self.client_loss[name] * self.client_n_data[name] / self.n_data
            )

        scv_state = self.scv.state_dict()

        self.model.load_state_dict(model_state)
        self.scv.load_state_dict(new_scv_state)
        self.round = self.round + 1

        return model_state, avg_loss, self.n_data, scv_state

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
        average = 1 / self._num_class
        return (
            data_distribution
            + (average - data_distribution) / self.num_round * self.round
        )

    def toOne(self, data_distribution):
        return data_distribution + (1 - data_distribution) / self.num_round * self.round

    def rec(
        self,
        name,
        state_dict,
        n_data,
        loss,
        ccv_state,
        client_id,
        client_data_distribution,
    ):
        """
        Server receives the local updates from the connected client k.
        :param name: Name of client k
        :param state_dict: Model dict from the client k
        :param n_data: Number of local data points in the client k
        :param loss: Loss of local training in the client k
        :param ccv_state: Normalization coefficient
        """
        self.n_data = self.n_data + n_data
        self.client_state[name] = {}
        self.client_n_data[name] = {}
        self.client_ccv_state[name] = {}

        self.client_state[name].update(state_dict)
        self.client_n_data[name] = n_data
        self.client_loss[name] = {}
        self.client_loss[name] = loss
        self.client_ccv_state[name].update(ccv_state)
        self.all_clients_data_distribution[client_id] = client_data_distribution

    def flush(self):
        """
        Flushing the client information in the server
        """
        self.n_data = 0
        self.client_state = {}
        self.client_n_data = {}
        self.client_loss = {}
        self.client_ccv_state = {}
        self.selected_clients_data_distribution = {}
