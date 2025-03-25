from fed_baselines.server_base import FedServer


class FedProxServer(FedServer):
    def __init__(self, client_list, dataset_id, model_name):
        super().__init__(client_list, dataset_id, model_name)
