#!/usr/bin/env python
import os
import random
import json
import pickle
import argparse
import yaml
from json import JSONEncoder
from tqdm import tqdm

from fed_baselines.server_base import FedServer
from fed_baselines.client_base import FedClient
from fed_baselines.server_fedavg_plus import FedAvgPlusServer
from fed_baselines.client_fedavg_plus import FedAvgPlusClient
from fed_baselines.server_fedavg_minus import FedAvgMinusServer
from fed_baselines.client_fedavg_minus import FedAvgMinusClient

from fed_baselines.server_fednova import FedNovaServer
from fed_baselines.client_fednova import FedNovaClient
from fed_baselines.server_fednova_plus import FedNovaPlusServer
from fed_baselines.client_fednova_plus import FedNovaPlusClient
from fed_baselines.server_fednova_minus import FedNovaMinusServer
from fed_baselines.client_fednova_minus import FedNovaMinusClient

from fed_baselines.server_fedprox import FedProxServer
from fed_baselines.client_fedprox import FedProxClient
from fed_baselines.server_fedprox_plus import FedProxPlusServer
from fed_baselines.client_fedprox_plus import FedProxPlusClient
from fed_baselines.server_fedprox_minus import FedProxMinusServer
from fed_baselines.client_fedprox_minus import FedProxMinusClient

from fed_baselines.server_scaffold import ScaffoldServer
from fed_baselines.client_scaffold import ScaffoldClient
from fed_baselines.server_scaffold_plus import ScaffoldPlusServer
from fed_baselines.client_scaffold_plus import ScaffoldPlusClient
from fed_baselines.server_scaffold_minus import ScaffoldMinusServer
from fed_baselines.client_scaffold_minus import ScaffoldMinusClient

from fed_baselines.server_prunefl import PruneFLServer
from fed_baselines.server_fedrolex import FedRolexServer
from fed_baselines.server_fiarse import FIARSEServer
from fed_baselines.client_fiarse import FIARSEClient

from postprocessing.recorder import Recorder
from preprocessing.baselines_dataloader import divide_data
from utils.models import *

json_types = (list, dict, str, int, float, bool, type(None))


class PythonObjectEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, json_types):
            return super().default(self, obj)
        return {"_python_object": pickle.dumps(obj).decode("latin-1")}


def as_python_object(dct):
    if "_python_object" in dct:
        return pickle.loads(dct["_python_object"].encode("latin-1"))
    return dct


def fed_args():
    """
    Arguments for running federated learning baselines
    :return: Arguments for federated learning baselines
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config", type=str, required=True, help="Yaml file for configuration"
    )
    parser.add_argument("--no_tqdm", action="store_true", help="Disable tqdm bar")

    args = parser.parse_args()
    return args


def fed_run():
    """
    Main function for the baselines of federated learning
    """
    args = fed_args()
    with open(args.config, "r") as yaml_file:
        try:
            config = yaml.safe_load(yaml_file)
        except yaml.YAMLError as exc:
            print(exc)

    algo_list = [
        "FedAvg",
        "FedAvg_Plus",
        "FedAvg_Minus",
        "FedNova",
        "FedNova_Plus",
        "FedNova_Minus",
        "FedProx",
        "FedProx_Plus",
        "FedProx_Minus",
        "SCAFFOLD",
        "SCAFFOLD_PLUS",
        "SCAFFOLD_MINUS",
        "PruneFL",
        "FedRolex",
        "FIARSE",
    ]
    assert (
        config["client"]["fed_algo"] in algo_list
    ), "The federated learning algorithm is not supported"

    dataset_list = ["MNIST", "EMNIST", "FashionMNIST", "CIFAR10", "SVHN", "CIFAR100"]
    assert config["system"]["dataset"] in dataset_list, "The dataset is not supported"

    model_list = [
        "LeNet",
        "AlexCifarNet",
        "ResNet18",
        "ResNet34",
        "ResNet50",
        "ResNet101",
        "ResNet152",
        "VGG11",
        "VGG13",
        "VGG16",
        "VGG19",
        "CNN",
        "MNISTCNN",
        "Linear",
    ]
    assert config["system"]["model"] in model_list, "The model is not supported"

    divide_method = ["IID", "DropClass", "Dirichlet"]
    assert (
        config["system"]["divide_method"] in divide_method
    ), "The divide method is not supported"

    np.random.seed(config["system"]["i_seed"])
    torch.manual_seed(config["system"]["i_seed"])
    random.seed(config["system"]["i_seed"])

    client_dict = {}
    recorder = Recorder()

    if config["system"]["divide_method"] == "DropClass":
        assert config["system"][
            "num_local_class"
        ], "The number of local classes is not specified"
        divide_config = {"num_local_class": config["system"]["num_local_class"]}
    elif config["system"]["divide_method"] == "IID":
        divide_config = None
    elif config["system"]["divide_method"] == "Dirichlet":
        divide_config = {"alpha": config["system"]["alpha"]}

    trainset_config, testset, len_class = divide_data(
        num_client=config["system"]["num_client"],
        divide_method=config["system"]["divide_method"],
        divide_config=divide_config,
        dataset_name=config["system"]["dataset"],
        i_seed=config["system"]["i_seed"],
    )
    max_acc = 0
    # Initialize the clients w.r.t. the federated learning algorithms and the specific federated settings
    for client_id in trainset_config["users"]:
        if config["client"]["fed_algo"] == "FedAvg":
            client_dict[client_id] = FedClient(
                client_id,
                dataset_id=config["system"]["dataset"],
                epoch=config["client"]["num_local_epoch"],
                model_name=config["system"]["model"],
                lr=config["client"]["lr"],
                batch_size=config["client"]["batch_size"],
            )
        elif config["client"]["fed_algo"] == "FedAvg_Plus":
            client_dict[client_id] = FedAvgPlusClient(
                client_id,
                dataset_id=config["system"]["dataset"],
                epoch=config["client"]["num_local_epoch"],
                model_name=config["system"]["model"],
                lr=config["client"]["lr"],
                batch_size=config["client"]["batch_size"],
            )
        elif config["client"]["fed_algo"] == "FedAvg_Minus":
            client_dict[client_id] = FedAvgMinusClient(
                client_id,
                dataset_id=config["system"]["dataset"],
                epoch=config["client"]["num_local_epoch"],
                model_name=config["system"]["model"],
                lr=config["client"]["lr"],
                batch_size=config["client"]["batch_size"],
            )
        elif config["client"]["fed_algo"] == "FedNova":
            client_dict[client_id] = FedNovaClient(
                client_id,
                dataset_id=config["system"]["dataset"],
                epoch=config["client"]["num_local_epoch"],
                model_name=config["system"]["model"],
                lr=config["client"]["lr"],
                batch_size=config["client"]["batch_size"],
            )
        elif config["client"]["fed_algo"] == "FedNova_Plus":
            client_dict[client_id] = FedNovaPlusClient(
                client_id,
                dataset_id=config["system"]["dataset"],
                epoch=config["client"]["num_local_epoch"],
                model_name=config["system"]["model"],
                lr=config["client"]["lr"],
                batch_size=config["client"]["batch_size"],
            )
        elif config["client"]["fed_algo"] == "FedNova_Minus":
            client_dict[client_id] = FedNovaMinusClient(
                client_id,
                dataset_id=config["system"]["dataset"],
                epoch=config["client"]["num_local_epoch"],
                model_name=config["system"]["model"],
                lr=config["client"]["lr"],
                batch_size=config["client"]["batch_size"],
            )
        elif config["client"]["fed_algo"] == "FedProx":
            client_dict[client_id] = FedProxClient(
                client_id,
                dataset_id=config["system"]["dataset"],
                epoch=config["client"]["num_local_epoch"],
                model_name=config["system"]["model"],
                lr=config["client"]["lr"],
                batch_size=config["client"]["batch_size"],
            )
        elif config["client"]["fed_algo"] == "FedProx_Plus":
            client_dict[client_id] = FedProxPlusClient(
                client_id,
                dataset_id=config["system"]["dataset"],
                epoch=config["client"]["num_local_epoch"],
                model_name=config["system"]["model"],
                lr=config["client"]["lr"],
                batch_size=config["client"]["batch_size"],
            )
        elif config["client"]["fed_algo"] == "FedProx_Minus":
            client_dict[client_id] = FedProxMinusClient(
                client_id,
                dataset_id=config["system"]["dataset"],
                epoch=config["client"]["num_local_epoch"],
                model_name=config["system"]["model"],
                lr=config["client"]["lr"],
                batch_size=config["client"]["batch_size"],
            )
        elif config["client"]["fed_algo"] == "SCAFFOLD":
            client_dict[client_id] = ScaffoldClient(
                client_id,
                dataset_id=config["system"]["dataset"],
                epoch=config["client"]["num_local_epoch"],
                model_name=config["system"]["model"],
                lr=config["client"]["lr"],
                batch_size=config["client"]["batch_size"],
            )
        elif config["client"]["fed_algo"] == "SCAFFOLD_PLUS":
            client_dict[client_id] = ScaffoldPlusClient(
                client_id,
                dataset_id=config["system"]["dataset"],
                epoch=config["client"]["num_local_epoch"],
                model_name=config["system"]["model"],
                lr=config["client"]["lr"],
                batch_size=config["client"]["batch_size"],
            )
        elif config["client"]["fed_algo"] == "SCAFFOLD_MINUS":
            client_dict[client_id] = ScaffoldMinusClient(
                client_id,
                dataset_id=config["system"]["dataset"],
                epoch=config["client"]["num_local_epoch"],
                model_name=config["system"]["model"],
                lr=config["client"]["lr"],
                batch_size=config["client"]["batch_size"],
            )
        elif config["client"]["fed_algo"] == "PruneFL":
            client_dict[client_id] = FedClient(
                client_id,
                dataset_id=config["system"]["dataset"],
                epoch=config["client"]["num_local_epoch"],
                model_name=config["system"]["model"],
                lr=config["client"]["lr"],
                batch_size=config["client"]["batch_size"],
            )
        elif config["client"]["fed_algo"] == "FedRolex":
            client_dict[client_id] = FedClient(
                client_id,
                dataset_id=config["system"]["dataset"],
                epoch=config["client"]["num_local_epoch"],
                model_name=config["system"]["model"],
                lr=config["client"]["lr"],
                batch_size=config["client"]["batch_size"],
            )
        elif config["client"]["fed_algo"] == "FIARSE":
            client_dict[client_id] = FIARSEClient(
                client_id,
                dataset_id=config["system"]["dataset"],
                epoch=config["client"]["num_local_epoch"],
                model_name=config["system"]["model"],
                lr=config["client"]["lr"],
                batch_size=config["client"]["batch_size"],
            )
        client_dict[client_id].load_trainset(trainset_config["user_data"][client_id])

    # Initialize the clients w.r.t. the federated learning algorithms and the specific federated settings
    if config["client"]["fed_algo"] == "FedAvg":
        fed_server = FedServer(
            trainset_config["users"],
            dataset_id=config["system"]["dataset"],
            model_name=config["system"]["model"],
        )
    elif config["client"]["fed_algo"] == "FedAvg_Plus":
        fed_server = FedAvgPlusServer(
            trainset_config["users"],
            dataset_id=config["system"]["dataset"],
            model_name=config["system"]["model"],
            len_class=len_class,
            num_round=config["system"]["num_round"],
            x=config["system"]["x"],
        )
    elif config["client"]["fed_algo"] == "FedAvg_Minus":
        fed_server = FedAvgMinusServer(
            trainset_config["users"],
            dataset_id=config["system"]["dataset"],
            model_name=config["system"]["model"],
            len_class=len_class,
            num_round=config["system"]["num_round"],
            x=config["system"]["x"],
        )
    elif config["client"]["fed_algo"] == "FedNova":
        fed_server = FedNovaServer(
            trainset_config["users"],
            dataset_id=config["system"]["dataset"],
            model_name=config["system"]["model"],
        )
    elif config["client"]["fed_algo"] == "FedNova_Plus":
        fed_server = FedNovaPlusServer(
            trainset_config["users"],
            dataset_id=config["system"]["dataset"],
            model_name=config["system"]["model"],
            len_class=len_class,
            num_round=config["system"]["num_round"],
            x=config["system"]["x"],
        )
    elif config["client"]["fed_algo"] == "FedNova_Minus":
        fed_server = FedNovaMinusServer(
            trainset_config["users"],
            dataset_id=config["system"]["dataset"],
            model_name=config["system"]["model"],
            len_class=len_class,
            num_round=config["system"]["num_round"],
            x=config["system"]["x"],
        )
    elif config["client"]["fed_algo"] == "FedProx":
        fed_server = FedProxServer(
            trainset_config["users"],
            dataset_id=config["system"]["dataset"],
            model_name=config["system"]["model"],
        )
    elif config["client"]["fed_algo"] == "FedProx_Plus":
        fed_server = FedProxPlusServer(
            trainset_config["users"],
            dataset_id=config["system"]["dataset"],
            model_name=config["system"]["model"],
            len_class=len_class,
            num_round=config["system"]["num_round"],
            x=config["system"]["x"],
        )
    elif config["client"]["fed_algo"] == "FedProx_Minus":
        fed_server = FedProxMinusServer(
            trainset_config["users"],
            dataset_id=config["system"]["dataset"],
            model_name=config["system"]["model"],
            len_class=len_class,
            num_round=config["system"]["num_round"],
            x=config["system"]["x"],
        )
    elif config["client"]["fed_algo"] == "SCAFFOLD":
        fed_server = ScaffoldServer(
            trainset_config["users"],
            dataset_id=config["system"]["dataset"],
            model_name=config["system"]["model"],
        )
        scv_state = fed_server.scv.state_dict()
    elif config["client"]["fed_algo"] == "SCAFFOLD_PLUS":
        fed_server = ScaffoldPlusServer(
            trainset_config["users"],
            dataset_id=config["system"]["dataset"],
            model_name=config["system"]["model"],
            len_class=len_class,
            num_round=config["system"]["num_round"],
            x=config["system"]["x"],
        )
        scv_state = fed_server.scv.state_dict()
    elif config["client"]["fed_algo"] == "SCAFFOLD_MINUS":
        fed_server = ScaffoldMinusServer(
            trainset_config["users"],
            dataset_id=config["system"]["dataset"],
            model_name=config["system"]["model"],
            len_class=len_class,
            num_round=config["system"]["num_round"],
            x=config["system"]["x"],
        )
        scv_state = fed_server.scv.state_dict()
    elif config["client"]["fed_algo"] == "PruneFL":
        fed_server = PruneFLServer(
            trainset_config["users"],
            dataset_id=config["system"]["dataset"],
            model_name=config["system"]["model"],
            x=config["system"]["x"],
        )
    elif config["client"]["fed_algo"] == "FedRolex":
        fed_server = FedRolexServer(
            trainset_config["users"],
            dataset_id=config["system"]["dataset"],
            model_name=config["system"]["model"],
            x=config["system"]["x"],
        )
    elif config["client"]["fed_algo"] == "FIARSE":
        fed_server = FIARSEServer(
            trainset_config["users"],
            dataset_id=config["system"]["dataset"],
            model_name=config["system"]["model"],
            x=config["system"]["x"],
        )
    fed_server.load_testset(testset)
    global_state_dict = fed_server.state_dict()

    # Main process of federated learning in multiple communication rounds
    if args.no_tqdm:
        pbar = range(config["system"]["num_round"])
    else:
        pbar = tqdm(range(config["system"]["num_round"]))
    for global_round in pbar:
        for client_id in trainset_config["users"]:
            # Local training
            if config["client"]["fed_algo"] == "FedAvg":
                client_dict[client_id].update(global_state_dict)
                state_dict, n_data, loss = client_dict[client_id].train()
                fed_server.rec(client_dict[client_id].name, state_dict, n_data, loss)
            elif config["client"]["fed_algo"] == "FedAvg_Plus":
                client_dict[client_id].update(global_state_dict)
                state_dict, n_data, loss = client_dict[client_id].train()
                fed_server.rec(
                    client_dict[client_id].name,
                    state_dict,
                    n_data,
                    loss,
                    client_id,
                    client_dict[client_id].get_data_distribution(),
                )
            elif config["client"]["fed_algo"] == "FedAvg_Minus":
                if fed_server.round == 0:
                    client_dict[client_id].update(global_state_dict)
                else:
                    client_dict[client_id].update(global_state_dict[client_id])
                state_dict, n_data, loss = client_dict[client_id].train()
                fed_server.rec(
                    client_dict[client_id].name,
                    state_dict,
                    n_data,
                    loss,
                    client_id,
                    client_dict[client_id].get_data_distribution(),
                )
            elif config["client"]["fed_algo"] == "FedNova":
                client_dict[client_id].update(global_state_dict)
                state_dict, n_data, loss, coeff, norm_grad = client_dict[
                    client_id
                ].train()
                fed_server.rec(
                    client_dict[client_id].name,
                    state_dict,
                    n_data,
                    loss,
                    coeff,
                    norm_grad,
                )
            elif config["client"]["fed_algo"] == "FedNova_Plus":
                client_dict[client_id].update(global_state_dict)
                state_dict, n_data, loss, coeff, norm_grad = client_dict[
                    client_id
                ].train()
                fed_server.rec(
                    client_dict[client_id].name,
                    state_dict,
                    n_data,
                    loss,
                    coeff,
                    norm_grad,
                    client_id,
                    client_dict[client_id].get_data_distribution(),
                )
            elif config["client"]["fed_algo"] == "FedNova_Minus":
                if fed_server.round == 0:
                    client_dict[client_id].update(global_state_dict)
                else:
                    client_dict[client_id].update(global_state_dict[client_id])
                state_dict, n_data, loss, coeff, norm_grad = client_dict[
                    client_id
                ].train()
                fed_server.rec(
                    client_dict[client_id].name,
                    state_dict,
                    n_data,
                    loss,
                    coeff,
                    norm_grad,
                    client_id,
                    client_dict[client_id].get_data_distribution(),
                )
            elif config["client"]["fed_algo"] == "FedProx":
                client_dict[client_id].update(global_state_dict)
                state_dict, n_data, loss = client_dict[client_id].train()
                fed_server.rec(client_dict[client_id].name, state_dict, n_data, loss)
            elif config["client"]["fed_algo"] == "FedProx_Plus":
                client_dict[client_id].update(global_state_dict)
                state_dict, n_data, loss = client_dict[client_id].train()
                fed_server.rec(
                    client_dict[client_id].name,
                    state_dict,
                    n_data,
                    loss,
                    client_id,
                    client_dict[client_id].get_data_distribution(),
                )
            elif config["client"]["fed_algo"] == "FedProx_Minus":
                if fed_server.round == 0:
                    client_dict[client_id].update(global_state_dict)
                else:
                    client_dict[client_id].update(global_state_dict[client_id])
                state_dict, n_data, loss = client_dict[client_id].train()
                fed_server.rec(
                    client_dict[client_id].name,
                    state_dict,
                    n_data,
                    loss,
                    client_id,
                    client_dict[client_id].get_data_distribution(),
                )
            elif config["client"]["fed_algo"] == "SCAFFOLD":
                client_dict[client_id].update(global_state_dict, scv_state)
                state_dict, n_data, loss, delta_ccv_state = client_dict[
                    client_id
                ].train()
                fed_server.rec(
                    client_dict[client_id].name,
                    state_dict,
                    n_data,
                    loss,
                    delta_ccv_state,
                )
            elif config["client"]["fed_algo"] == "SCAFFOLD_PLUS":
                client_dict[client_id].update(global_state_dict, scv_state)
                state_dict, n_data, loss, delta_ccv_state = client_dict[
                    client_id
                ].train()
                fed_server.rec(
                    client_dict[client_id].name,
                    state_dict,
                    n_data,
                    loss,
                    delta_ccv_state,
                    client_id,
                    client_dict[client_id].get_data_distribution(),
                )
            elif config["client"]["fed_algo"] == "SCAFFOLD_MINUS":
                if fed_server.round == 0:
                    client_dict[client_id].update(global_state_dict, scv_state)
                else:
                    client_dict[client_id].update(
                        global_state_dict[client_id], scv_state[client_id]
                    )

                state_dict, n_data, loss, delta_ccv_state = client_dict[
                    client_id
                ].train()
                fed_server.rec(
                    client_dict[client_id].name,
                    state_dict,
                    n_data,
                    loss,
                    delta_ccv_state,
                    client_id,
                    client_dict[client_id].get_data_distribution(),
                )
            elif config["client"]["fed_algo"] == "PruneFL":
                client_dict[client_id].update(global_state_dict)
                state_dict, n_data, loss = client_dict[client_id].train()
                fed_server.rec(client_dict[client_id].name, state_dict, n_data, loss)
            elif config["client"]["fed_algo"] == "FedRolex":
                if fed_server.round == 0:
                    client_dict[client_id].update(global_state_dict)
                else:
                    client_dict[client_id].update(global_state_dict[client_id])
                state_dict, n_data, loss = client_dict[client_id].train()
                fed_server.rec(client_dict[client_id].name, state_dict, n_data, loss)
            elif config["client"]["fed_algo"] == "FIARSE":
                if fed_server.round == 0:
                    client_dict[client_id].update(global_state_dict, {})
                else:
                    client_dict[client_id].update(global_state_dict, mask)
                state_dict, n_data, loss = client_dict[client_id].train()
                fed_server.rec(client_dict[client_id].name, state_dict, n_data, loss)

        # Global aggregation
        fed_server.select_clients()
        if config["client"]["fed_algo"] == "FedAvg":
            global_state_dict, avg_loss, _ = fed_server.agg()
        elif config["client"]["fed_algo"] == "FedAvg_Plus":
            global_state_dict, avg_loss, _ = fed_server.agg()
        elif config["client"]["fed_algo"] == "FedAvg_Minus":
            global_state_dict, avg_loss, _ = fed_server.agg()
        elif config["client"]["fed_algo"] == "FedNova":
            global_state_dict, avg_loss, _ = fed_server.agg()
        elif config["client"]["fed_algo"] == "FedNova_Plus":
            global_state_dict, avg_loss, _ = fed_server.agg()
        elif config["client"]["fed_algo"] == "FedNova_Minus":
            global_state_dict, avg_loss, _ = fed_server.agg()
        elif config["client"]["fed_algo"] == "FedProx":
            global_state_dict, avg_loss, _ = fed_server.agg()
        elif config["client"]["fed_algo"] == "FedProx_Plus":
            global_state_dict, avg_loss, _ = fed_server.agg()
        elif config["client"]["fed_algo"] == "FedProx_Minus":
            global_state_dict, avg_loss, _ = fed_server.agg()
        elif config["client"]["fed_algo"] == "SCAFFOLD":
            global_state_dict, avg_loss, _, scv_state = fed_server.agg()  # scarffold
        elif config["client"]["fed_algo"] == "SCAFFOLD_PLUS":
            global_state_dict, avg_loss, _, scv_state = fed_server.agg()
        elif config["client"]["fed_algo"] == "SCAFFOLD_MINUS":
            global_state_dict, avg_loss, _, scv_state = fed_server.agg()
        elif config["client"]["fed_algo"] == "PruneFL":
            global_state_dict, avg_loss, _ = fed_server.agg()
        elif config["client"]["fed_algo"] == "FedRolex":
            global_state_dict, avg_loss, _ = fed_server.agg()
        elif config["client"]["fed_algo"] == "FIARSE":
            global_state_dict, avg_loss, _, mask = fed_server.agg()

        # Testing and flushing
        accuracy = fed_server.test()
        fed_server.flush()

        # Record the results
        recorder.res["server"]["iid_accuracy"].append(accuracy)
        recorder.res["server"]["train_loss"].append(avg_loss)

        if max_acc < accuracy:
            max_acc = accuracy
        if not args.no_tqdm:
            pbar.set_description(
                "Global Round: %d" % global_round
                + "| Train loss: %.4f " % avg_loss
                + "| Accuracy: %.4f" % accuracy
                + "| Max Acc: %.4f" % max_acc
            )

        # Save the results
        if not os.path.exists(config["system"]["res_root"]):
            os.makedirs(config["system"]["res_root"])

        file_path = os.path.join(
            config["system"]["res_root"],
            str(config["client"]["fed_algo"])
            + "_"
            + str(config["system"]["dataset"])
            + "_"
            + str(config["system"]["model"])
            + "_"
            + str(config["system"]["divide_method"])
            + "_"
            + str(
                "a=" + str(config["system"]["alpha"])
                if config["system"]["divide_method"] == "Dirichlet"
                else "n=" + str(config["system"]["num_local_class"])
            )
            + "_x="
            + str(config["system"]["x"]),
        )
        with open(file_path, "w") as jsfile:
            json.dump(recorder.res, jsfile, cls=PythonObjectEncoder)


if __name__ == "__main__":
    fed_run()
