import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict
from prettytable import PrettyTable

class Profilable(nn.Module):
    """
    Оборачивает модуль из pytorch. Позволяет проводить профилировку
    """
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.gpu_time = [] # хранит время выполнения каждого forward
        
    def forward(self, x):
        start = time.perf_counter()
        x = self.module(x)
        self.gpu_time.append(time.perf_counter() - start)
        return x
    
    def __iter__(self):
        return iter(self.module)
       
    def __len__(self):
        return len(self.module)
    
class Profiler(nn.Module):
    """
    Выполняет профилировку переданной модели
    """
    def __init__(self, model, model_name):
        super().__init__()
        self.data = None # хранит результаты профилировки для каждого модуля в модели
        self.model = model
        self.model_name = model_name 
        self.total_time_mean = None
        
        self.model.eval()
        self._wrap(model)
    
    def _wrap(self, module):
        for k, v in module._modules.items():
            if isinstance(v, nn.ModuleList):
                self._wrap(v)
            else:
                if not isinstance(v, Profilable):
                    module._modules[k] = Profilable(v)
                else:
                    module._modules[k].gpu_time = []
    
    def forward(self, x):
        with torch.no_grad():
            return self.model(x)
    
    def collect_data(self):
        """
        Собирает данные профилировки
        """
        self.data = defaultdict(list)
        for name, module in self.model._modules.items():
            if isinstance(module, nn.ModuleList):
                for k, v in module._modules.items():
                    self.data[name + f" {k}"] = v.gpu_time
            else:
                self.data[name] = module.gpu_time
    
    def _get_stats(self):
        """
        Возвращает статистики, вычисленные по результатам профилировки
        """
        if self.data is None:
            raise RuntimeError("Метод collect_data не был вызван.")
        
        data = np.array(list(self.data.values()))
        
        sums = np.sum(data, 0)
        means = np.mean(data, 1)
        stds = np.std(data, 1)

        total_time_mean, total_time_std = np.mean(sums), np.std(sums)
        self.total_time_mean = total_time_mean
        return means, stds, total_time_mean, total_time_std
    
    def save_table(self, filename):
        """
        Сохраняет таблицу в файл с именем filename.txt
        """
        if self.data is None:
            raise RuntimeError("Метод collect_data не был вызван.")
        means, stds, total_time_mean, total_time_std = self._get_stats()
        
        with open(filename + ".txt", "w") as f:
            f.write(self.model_name)
            for i, k in enumerate(self.data.keys()):
                f.write(f"{k:<15} mean: {means[i] * 1000:<7.2f} ms, \tstd: {stds[i] * 1000:<7.2f} ms, \tpercent: {means[i] / total_time_mean * 100:<7.2f}%")
            f.write(f"Total time: {total_time_mean * 1000:.2f} ms, \tstd: {total_time_std * 1000:.2f} ms")
            
    def show_table(self):
        """
        Показывает таблицу результатов профилировки
        """
        if self.data is None:
            raise RuntimeError("Метод collect_data не был вызван.")
        means, stds, total_time_mean, total_time_std = self._get_stats()
        
        print(self.model_name)
        
        t = PrettyTable(["Name", 'Mean, ms', 'Std, ms', "Percent, %"])
        for i, k in enumerate(self.data.keys()):
            mean_ms = round(means[i] * 1000, 2)
            std_ms = round(stds[i] * 1000, 2)
            percent_ms = round(means[i] / total_time_mean * 100, 2)
            t.add_row([k, mean_ms, std_ms, percent_ms])
        print(t)
        print(f"Total time: {total_time_mean * 1000:.2f} ms, \tstd: {total_time_std * 1000:.2f} ms")
        
        
    def plot_results(self, figsize=(20, 10)):
        """
        Строит график результатов профилировки
        """
        if self.data is None:
            raise RuntimeError("Метод collect_data не был вызван.")

        plt.figure(figsize=figsize)
        for name, gpu_time in self.data.items():
            plt.plot(gpu_time, label = name)
        plt.xlabel("Batch number")
        plt.ylabel("GPU Time")
        plt.legend()
        plt.show()
        
    