from IPython.display import clear_output
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import os

class ExperimentLogger:
    def log(self, values):
        for k, v in values.items():
            if k not in self.__dict__:
                self.__dict__[k] = [v]
            else:
                self.__dict__[k] += [v]

class Checkpoint(object):
    def __init__(self, run):
        self.run = run
        self._path = f'checkpoints/{self.run}.pt'
        if os.path.exists(self._path):
          self.load()

    def load(self):
        checkpoint = torch.load(self._path)
        for k, v in checkpoint.items():
            self.__dict__[k] = [v]
        return checkpoint

    def save(self):
        checkpoint = {}
        for k, v in self.__dict__.items():
            checkpoint[k] = v
        torch.save(checkpoint, self._path)

    # def load_client(self, clients):
    #     for i, client in enumerate(clients):
    #         self.__dict__["clients"][str(i)]["W"]
        

    def __getitem__(self, key):
        return getattr(self, key)
    def __setitem__(self, key, value):
        setattr(self, key, value)
    def __str__(self):
        return str(self.__dict__)
    def __repr__(self):
        return str(self.__dict__)


def display_train_stats(cfl_stats, eps_1, eps_2, communication_rounds):
    clear_output(wait=True)
    
    plt.figure(figsize=(12,4))
    
    plt.subplot(1,2,1)
    # acc_mean = np.mean(cfl_stats.acc_clients, axis=1)
    # acc_std = np.std(cfl_stats.acc_clients, axis=1)
    # plt.fill_between(cfl_stats.rounds, acc_mean-acc_std, acc_mean+acc_std, alpha=0.5, color="C0")
    # plt.plot(cfl_stats.rounds, acc_mean, color="C0", label='acc')

    metrics = [cfl_stats.acc_clients] # cfl_stats.f1_clients
    colors = ["C1"] # "C0"
    names = ["acc"] # "f1", 

    for metric, color, name in zip(metrics, colors, names):
      metric_mean = np.mean(metric, axis=1)
      metric_std = np.std(metric, axis=1)
      plt.fill_between(cfl_stats.rounds, metric_mean - metric_std, metric_mean + metric_std, alpha=0.5, color=color)
      plt.plot(cfl_stats.rounds, metric_mean, color=color, label=name)
    
    if "split" in cfl_stats.__dict__:
        for s in cfl_stats.split:
            plt.axvline(x=s, linestyle="-", color="k", label=r"Split")
    
    
    plt.text(x=communication_rounds, y=1, ha="right", va="top", 
             s="Clusters: {}".format([x.tolist() for x in cfl_stats.clusters[-1]]))
    
    plt.xlabel("Communication Rounds")
    plt.ylabel("Metric")
    
    plt.xlim(0, communication_rounds)
    plt.ylim(0,1)
    plt.legend()
    
    plt.subplot(1,2,2)
    
    plt.plot(cfl_stats.rounds, cfl_stats.mean_norm, color="C1", label=r"$\|\sum_i\Delta W_i \|$")
    plt.plot(cfl_stats.rounds, cfl_stats.max_norm, color="C2", label=r"$\max_i\|\Delta W_i \|$")
    
    plt.axhline(y=eps_1, linestyle="--", color="k", label=r"$\varepsilon_1$")
    plt.axhline(y=eps_2, linestyle=":", color="k", label=r"$\varepsilon_2$")

    if "split" in cfl_stats.__dict__:
        for s in cfl_stats.split:
            plt.axvline(x=s, linestyle="-", color="k", label=r"Split")

    plt.xlabel("Communication Rounds")
    plt.legend()
    
    plt.xlim(0, communication_rounds)
    #plt.ylim(0, 2)
    
    plt.show()