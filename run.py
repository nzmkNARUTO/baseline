import yaml
import os
from multiprocessing import Pool

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

dataset_list = ["MNIST", "CIFAR10"]

model_list = [
    "LeNet",
    "AlexCifarNet",
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "ResNet101",
    "ResNet152",
    "CNN",
    "Linear",
]


def run(config):
    file_address = f"config/{config['client']['fed_algo']}_Linear_{config['system']['dataset']}_a={config['system']['alpha']}_x={config['system']['x']}.yaml"
    with open(file_address, "w") as f:
        yaml.dump(config, f)
    print(f"python fl_main.py --config {file_address}")
    os.system(f"python fl_main.py --config {file_address}")


if __name__ == "__main__":
    p = Pool(20)
    for algo in algo_list:
        for dataset in dataset_list:
            for alpha in [0.1, 0.5, 1.0]:
                if "plus" in algo.lower():
                    for x in [0.1, 0.3, 0.5, 0.8, 1.0]:
                        config = {
                            "system": {
                                "num_client": 5,
                                "dataset": dataset,
                                "divide_method": "Dirichlet",
                                "num_local_class": 1,
                                "alpha": alpha,
                                "model": "Linear",
                                "i_seed": 235235,
                                "num_round": 50,
                                "res_root": f"/home/airadmin/Share/baseline/results/{algo[:-5]}/{dataset}/a={alpha}",
                                "x": x,
                            },
                            "client": {
                                "fed_algo": algo,
                                "lr": 0.1,
                                "batch_size": 256,
                                "num_local_epoch": 5,
                                "momentum": 0.9,
                                "num_worker": 4,
                            },
                        }
                        p.apply_async(run, args=(config,))
                else:
                    config = {
                        "system": {
                            "num_client": 5,
                            "dataset": dataset,
                            "divide_method": "Dirichlet",
                            "num_local_class": 1,
                            "alpha": alpha,
                            "model": "Linear",
                            "i_seed": 235235,
                            "num_round": 50,
                            "res_root": f"/home/airadmin/Share/baseline/results/{algo}/{dataset}/a={alpha}",
                            "x": 1,
                        },
                        "client": {
                            "fed_algo": algo,
                            "lr": 0.1,
                            "batch_size": 256,
                            "num_local_epoch": 5,
                            "momentum": 0.9,
                            "num_worker": 4,
                        },
                    }
                    p.apply_async(run, args=(config,))
    p.close()
    p.join()
    print("All Done!")
