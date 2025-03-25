from fed_baselines.server_fedavg_minus import FedAvgMinusServer
import copy


class FedNovaMinusServer(FedAvgMinusServer):

    def __init__(self, client_list, dataset_id, model_name, len_class, num_round, x):
        super().__init__(client_list, dataset_id, model_name, len_class, num_round, x)
        # Normalized coefficient
        self.client_coeff = {}
        # Normalized gradients
        self.client_norm_grad = {}

    def calculate_gradient(self, model_state):
        # calculate the gradient
        delta_state = {}
        for name in self.selected_clients:
            delta_state[name] = model_state[name]
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
                    nova_model_state[key] = (
                        nova_model_state[key]
                        + delta_state[name][key]
                        * self.client_n_data[name]
                        / self.n_data
                    )
            avg_loss = (
                avg_loss
                + self.client_loss[name] * self.client_n_data[name] / self.n_data
            )

        for key in model_state:
            model_state[key] = model_state[key] - coeff * nova_model_state[key]

        self.model.load_state_dict(model_state)

        self.round = self.round + 1

        model_state = self.prune_weight(sensitivities, model_state)

        return model_state, avg_loss, self.n_data

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
