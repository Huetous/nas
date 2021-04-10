import time
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import defaultdict

class Profiler(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.data = defaultdict(list)
        self.model = module.model
    
    def forward(self, x):
        for i, block in enumerate(self.model, start = 1):
            start = time.time()
            x = block(x)
            self.data[i].append(time.time() - start)
        return x
    
    def get_results(self):
        return self.data
    
    def plot_results(self):
        plt.figure(figsize=(10, 10))
        for name, gpu_time in self.data.items():
            plt.plot(gpu_time, label = name)
        plt.xlabel("Batch number")
        plt.ylabel("GPU Time")
        plt.legend()
        plt.show()