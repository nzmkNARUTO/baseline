from collections import defaultdict
from fed_baselines.server_base import FedServer
import torch
import math
import pandas as pd
from torch.utils.data import random_split, Subset
from random import random


class PruneFLServer(FedServer):
    def __init__(self, client_list, dataset_id, model_name, x):
        super().__init__(client_list, dataset_id, model_name)
        self.x = x

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

    def prune_weight(self, weight):
        # prune the weight
        mask = self.mask(weight, self.x)
        for key in mask:
            weight[key] = weight[key] * mask[key]
        return weight

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

        # calculate the gradient
        delta_state = self.calculate_gradient(model_state)

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

        model_state = self.prune_weight(model_state)

        return model_state, avg_loss, n_data

    def mask(self, weight, ratio):
        mask = {}
        for key in weight.keys():
            mask[key] = torch.zeros_like(weight[key])
            for i in range(mask[key].numel()):
                mask[key].reshape(-1)[i] = 1 if random() > ratio else 0
        return mask
