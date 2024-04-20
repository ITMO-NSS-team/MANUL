from datetime import datetime
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class SingletonClass(type):
    _instances = {}

    def __call__(cls, *args, **kwds):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwds)
            cls._instances[cls] = instance
        return cls._instances[cls]


class Casher(metaclass=SingletonClass):
    def __init__(self, cash_directory=None) -> None:
        if cash_directory is not None:
            self.name_of_dir = cash_directory
        else:
            self.name_of_dir = f"info_log/{datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')}"

        if not os.path.exists(self.name_of_dir):
            os.makedirs(self.name_of_dir)
        print(f'Log folder set as {self.name_of_dir}')


    def save_confusion_matrix(self, name: str, data, data2 = None):
        target_true = data[0]
        target_predict = data[1]
        cm = confusion_matrix(target_true.reshape(-1), target_predict.reshape(-1))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)


        if data2 is not None:
            f, axes = plt.subplots(1, 2, sharey='row')
            target_true = data2[0]
            target_predict = data2[1]
            cm2 = confusion_matrix(target_true.reshape(-1), target_predict.reshape(-1))
            disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2)
            disp.plot(ax=axes[0])
            disp.im_.colorbar.remove()
            disp2.plot(ax=axes[1])
            disp2.im_.colorbar.remove()

            f.colorbar(disp.im_, ax=axes)
        else:
            disp.plot()

        plt.savefig(f"{self.name_of_dir}/{name}")
        plt.close()

    def save_plot(self, name, data):
        plt.plot(data)
        plt.savefig(f"{self.name_of_dir}/{name}")
        plt.close()

    def save_plots(self, name, data, labels=None):
        for i, data_i in enumerate(data):
            label_data = i
            if labels is not None:
                label_data = labels[i]
            plt.scatter(np.arange(len(data_i)), data_i, label=label_data)
        plt.legend()
        plt.savefig(f"{self.name_of_dir}/{name}")
        plt.close()

    def save_graph(self, data, name='graph.txt'):
        graph_data = []
        for i, edges in enumerate(data):
            graph_data.append(list(edges.values()))
        with open(f"{self.name_of_dir}/{name}", "w") as fl:
            fl.write(str(graph_data))

    def save_end_graph(self, data, name='graph.txt'):
        with open(f"{self.name_of_dir}/{name}", "w") as fl:
            fl.write(str(data))

    def save_pickle(self, data, name):
        with open(f"{self.name_of_dir}/{name}", "wb") as pkl:
            pickle.dump(data, pkl)

    def save_boxplot(self, name, data):
        data = np.array(data)
        jg = data[:, 0, :]
        eag = data[:, 1, :]

        box_data = np.concatenate((jg, eag), axis=1)
        plt.boxplot(box_data, labels=["1 класс,\nbase", "2 класс,\nbase", "1 класс,\nman", "2 класс,\nman"])
        plt.savefig(f"{self.name_of_dir}/{name}")
        plt.close()

    def save_model(self, name, model):
        torch.save(model.state_dict(), f"{self.name_of_dir}/{name}.pt")

    def load_model(self, name):
        the_model = torch.load(name)
        return the_model

    def get_path(self):
        return self.name_of_dir