from fed_baselines.server_fedavg_minus import FedAvgMinusServer


class FedProxMinusServer(FedAvgMinusServer):
    def __init__(self, client_list, dataset_id, model_name, len_class, num_round, x):
        super().__init__(client_list, dataset_id, model_name, len_class, num_round, x)
