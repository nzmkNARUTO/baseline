import yaml
import os
from multiprocessing import Pool
from copy import deepcopy

config = {
    "system": {
        "num_client": 5,
        "num_round": 50,
        "i_seed": 235235,
        "x": 0,
    },
    "client": {
        "lr": 0.1,
        "batch_size": 256,
        "num_local_epoch": 5,
        "momentum": 0.9,
        "num_worker": 4,
    },
}


algo_list = [
    "FedAvg",
    "FedAvg_Plus",
    "FedProx",
    "FedProx_Plus",
    "FedNova",
    "FedNova_Plus",
    "SCAFFOLD",
    "SCAFFOLD_PLUS",
]

dataset_list = {
    "MNIST": ["Linear", "LeNet"],
    "CIFAR10": ["CNN", "ResNet18", "AlexCifarNet"],
    # "CIFAR100": ["CNN", "ResNet18"],
}
divide_method_list = {"Dirichlet": [0.1, 0.5, 1.0], "DropClass": [1, 5, 10]}


def run(config):
    file_address = (
        "config/"
        + str(config["client"]["fed_algo"])
        + "_"
        + str(config["system"]["dataset"])
        + "_"
        + str(config["system"]["model"])
        + "_"
        + str(config["system"]["divide_method"])
    )
    if config["system"]["divide_method"] == "Dirichlet":
        file_address += "_a=" + str(config["system"]["alpha"]) + ".yaml"
    else:
        file_address += "_n=" + str(config["system"]["num_local_class"]) + ".yaml"
    with open(file_address, "w") as f:
        yaml.dump(config, f)
    print(f"python fl_main.py --config {file_address}")
    os.system(f"python fl_main.py --config {file_address}")


if __name__ == "__main__":
    p = Pool(5)
    for algo in algo_list:
        config["client"]["fed_algo"] = algo
        algo_name = algo.replace("_Plus", "").replace("_PLUS", "")
        for dataset in dataset_list:
            config["system"]["dataset"] = dataset
            for model in dataset_list[dataset]:
                config["system"]["model"] = model
                for divide_method in divide_method_list:
                    config["system"]["divide_method"] = divide_method
                    for alpha_or_local_num_class in divide_method_list[divide_method]:
                        if divide_method == "Dirichlet":
                            config["system"]["alpha"] = alpha_or_local_num_class
                            config["system"][
                                "res_root"
                            ] = f"/home/airadmin/Share/baseline/results/{algo_name}/{dataset}/{model}/{divide_method}/a={alpha_or_local_num_class}"
                        else:
                            config["system"][
                                "num_local_class"
                            ] = alpha_or_local_num_class
                            config["system"][
                                "res_root"
                            ] = f"/home/airadmin/Share/baseline/results/{algo_name}/{dataset}/{model}/{divide_method}/n={alpha_or_local_num_class}"
                        # if "plus" in algo.lower():
                        #     for x in [0.1, 0.3, 0.5, 0.8, 1.0]:
                        #         config["system"]["x"] = x
                        #         p.apply_async(
                        #             run,
                        #             args=(deepcopy(config),),
                        #         )
                        # else:
                        p.apply_async(
                            run,
                            args=(deepcopy(config),),
                        )
    p.close()
    p.join()
    print("All Done!")
