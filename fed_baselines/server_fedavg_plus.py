from collections import defaultdict
from fed_baselines.server_base import FedServer
import torch
from torch import nn
import math
import copy
import pandas as pd
from torch.utils.data import DataLoader, random_split, Subset


class FedAvgPlusServer(FedServer):
    def __init__(self, client_list, dataset_id, model_name, len_class, num_round, x):
        super().__init__(client_list, dataset_id, model_name)
        self.len_class = len_class
        self.num_round = num_round
        self.x = x
        self.all_clients_data_distribution = {}
        self.selected_clients_data_distribution = {}
        self.global_data_distribution = {}

    def calculate_data_distribution(self):
        # for name in self.selected_clients:
        #     self.selected_clients_data_distribution[name] = (
        #         self.all_clients_data_distribution[name]
        #     )
        #     # Fill in the missing classes
        #     for i in range(self.len_class):
        #         if i not in self.selected_clients_data_distribution[name]:
        #             self.selected_clients_data_distribution[name][i] = 0
        selected_clients_data_distribution = (
            pd.DataFrame(self.all_clients_data_distribution).fillna(0).sort_index()
        )
        for i in range(self.len_class):
            if i not in selected_clients_data_distribution.index:
                selected_clients_data_distribution.loc[i] = 0

        # Calculate the data distribution
        for name in self.selected_clients:
            self.selected_clients_data_distribution[name] = dict(
                selected_clients_data_distribution[name]
                / selected_clients_data_distribution[name].sum()
            )
        for i in range(self.len_class):
            self.global_data_distribution[i] = dict(
                selected_clients_data_distribution.loc[i]
                / selected_clients_data_distribution.loc[i].sum()
            )
        for i in self.global_data_distribution:
            for j in self.global_data_distribution[i]:
                if math.isnan(self.global_data_distribution[i][j]):
                    self.global_data_distribution[i][j] = 0
        pass

    def analysis_sensitivity(self):
        self.calculate_data_distribution()
        sensitivities = {}
        loss_func = nn.CrossEntropyLoss().to(self._device)
        for i in self.analysis_dataset.keys():
            dataloader = DataLoader(
                self.analysis_dataset[i],
                batch_size=len(self.analysis_dataset[i].indices),
                shuffle=True,
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
                x = self.x
                ratio = self.selected_clients_data_distribution[name][i] * (1 - x) + x
                # ratio = self.global_data_distribution[i][name] * (1 - x) + x
                mask[i] = self.mask(sensitivities[i], self.toOne(ratio))
                # mask[i] = self.mask(sensitivities[i], ratio)
            for key in mask[i]:
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
        Server aggregates models from connected clients.
        :return: model_state: Updated global model after aggregation
        :return: avg_loss: Averaged loss value
        :return: n_data: Number of the local data points
        """
        client_num = len(self.selected_clients)
        if client_num == 0 or self.n_data == 0:
            return self.model.state_dict(), 0, 0
        # self.calculate_data_distribution()
        model_state = self.model.state_dict()
        avg_loss = 0

        # # analysis the sensitivity of each class
        sensitivities = self.analysis_sensitivity()

        # calculate the gradient
        delta_state = self.calculate_gradient(model_state)

        # # reweight the gradient
        delta_state = self.reweight_gradient(sensitivities, delta_state)

        # Aggregate the local updated models from selected clients
        for i, name in enumerate(self.selected_clients):
            if name not in self.client_state:
                continue
            for key in self.client_state[name]:
                model_state[key] = (
                    model_state[key]
                    + delta_state[name][key] * self.client_n_data[name] / self.n_data
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

    def load_testset(self, test_dataset):
        """
        Server loads the test dataset.
        :param data: Dataset for testing.
        """
        test_dataset_size = int(len(test_dataset) * 0.9)
        analysis_dataset_size = len(test_dataset) - test_dataset_size
        self.testset, analysis_dataset = random_split(
            test_dataset, [test_dataset_size, analysis_dataset_size]
        )
        self.analysis_dataset = {}
        index = defaultdict(list)
        for i in analysis_dataset.indices:
            index[int(analysis_dataset.dataset.targets[i])].append(i)
        if 0 in index.keys():
            for i in range(self.len_class):
                self.analysis_dataset[i] = Subset(test_dataset, index[i])
        else:
            for i in range(self.len_class):
                self.analysis_dataset[i] = Subset(test_dataset, index[i + 1])

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

    def rec(self, name, state_dict, n_data, loss, client_id, client_data_distribution):
        """
        Server receives the local updates from the connected client k.
        :param name: Name of client k
        :param state_dict: Model dict from the client k
        :param n_data: Number of local data points in the client k
        :param loss: Loss of local training in the client k
        """
        self.n_data = self.n_data + n_data
        self.client_state[name] = {}
        self.client_n_data[name] = {}

        self.client_state[name].update(state_dict)
        self.client_n_data[name] = n_data
        self.client_loss[name] = {}
        self.client_loss[name] = loss
        self.all_clients_data_distribution[client_id] = client_data_distribution

    def flush(self):
        """
        Flushing the client information in the server
        """
        self.n_data = 0
        self.client_state = {}
        self.client_n_data = {}
        self.client_loss = {}
        self.selected_clients_data_distribution = {}
