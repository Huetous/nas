import time
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict

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

class Profiler(nn.Module):
    """
    Выполняет профилировку переданной модели
    """
    def __init__(self, model, model_name):
        super().__init__()
        self.data = None # хранит результаты профилировки для каждого модуля в модели
        self.model = model
        self.model_name = model_name 
        
        # оборачиваем модули в Profilable класс
        for k, v in self.model._modules.items():
            self.model._modules[k] = Profilable(v)
    
    def forward(self, x):
        return self.model(x)
    
    def collect_data(self):
        """
        Собирает данные профилировки
        """
        self.data = defaultdict(list)
        for name, module in self.model._modules.items():
            self.data[name] = module.gpu_time
        
    def show_table(self):
        """
        Показывает таблицу результатов профилировки
        """
        if self.data is None:
            raise RuntimeError("Метод collect_data не был вызван.")

        means = [np.mean(v) for _, v in self.data.items()]
        stds = [np.std(v) for _, v in self.data.items()]
        total = np.sum(means)
        
        print(self.model_name)
        for i, k in enumerate(self.data.keys()):
            print(f"{k:<15} mean: {means[i] * 1000:<7.2f} ms, \tstd: {stds[i] * 1000:<7.2f} ms, \tpercent: {means[i] / total * 100:<7.2f}%")
        print(f"Total time: {total * 1000:.2f} ms")
        self.total_time = total  # сохраняем общее время выполнения
        
    def plot_results(self):
        """
        Строит график результатов профилировки
        """
        if self.data is None:
            raise RuntimeError("Метод collect_data не был вызван.")

        plt.figure(figsize=(10, 10))
        for name, gpu_time in self.data.items():
            plt.plot(gpu_time, label = name)
        plt.xlabel("Batch number")
        plt.ylabel("GPU Time")
        plt.legend()
        plt.show()