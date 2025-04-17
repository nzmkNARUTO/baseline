import numpy as np
import matplotlib.pyplot as plt
import json
from json import JSONEncoder
import pickle

json_types = (list, dict, str, int, float, bool, type(None))
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 或者使用其他支持中文的字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
plt.rcParams["font.size"] = 16


class PythonObjectEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, json_types):
            return super().default(self, obj)
        return {"_python_object": pickle.dumps(obj).decode("latin-1")}


def as_python_object(dct):
    if "_python_object" in dct:
        return pickle.loads(dct["_python_object"].encode("latin-1"))
    return dct


class Recorder(object):
    def __init__(self):
        self.res_list = []
        self.res = {
            "server": {"iid_accuracy": [], "train_loss": []},
            "clients": {"iid_accuracy": [], "train_loss": []},
        }

    def load(self, filename, label):
        """
        Load the result files
        :param filename: Name of the result file
        :param label: Label for the result file
        """
        with open(filename) as json_file:
            res = json.load(json_file, object_hook=as_python_object)
        self.res_list.append((res, label))

    def plot(self):
        """
        Plot the testing accuracy and training loss on number of epochs or communication rounds
        """
        fig, axes = plt.subplots(2)
        for i, (res, label) in enumerate(self.res_list):
            axes[0].plot(
                np.array(res["server"]["iid_accuracy"]),
                label=label,
                alpha=1,
                linewidth=2,
            )
            axes[1].plot(
                np.array(res["server"]["train_loss"]), label=label, alpha=1, linewidth=2
            )

        for i, ax in enumerate(axes):
            if i == 0:
                ax.set_ylabel("准确率")
            if i == 1:
                ax.set_ylabel("Loss")
                ax.set_xlabel("训练轮次")
                ax.legend(prop={"size": 12})
            ax.tick_params(axis="both")
            ax.grid()

    def plot2(self):
        """
        Plot the testing accuracy and training loss on number of epochs or communication rounds
        """
        plt.figure(figsize=(10, 6))
        for i, (res, label) in enumerate(self.res_list):
            plt.plot(
                np.array(res["server"]["iid_accuracy"]),
                label=label,
                alpha=1,
                linewidth=2,
            )

        plt.ylabel("准确率")
        plt.xlabel("训练轮次")
        plt.legend()
        plt.tick_params(axis="both")
        plt.grid()
