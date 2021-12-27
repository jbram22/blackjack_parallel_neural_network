import sys
import os

import numpy
import matplotlib.pyplot as plt

# train-time dictionary
d = {}

# non-parallelized
non_parallel = open("non_parallel_nn.txt", "r")
non = non_parallel.read()
d['not_parallel'] = non

# 2 process nn training time
two_procs = open("nn_train_2.txt", "r")
two = two_procs.read()
d['two_procs'] = two

# 3 process nn training time
three_procs = open("nn_train_3.txt", "r")
three = three_procs.read()
d['three_procs'] = three

# 4 process nn training time
four_procs = open("nn_train_4.txt", "r")
four = four_procs .read()
d['four_procs'] = four

keys = d.keys()
values = d.values()

plt.bar(keys, values)

plt.ylabel('Training Time (seconds)')
plt.title('Neural Network Training Time Per Process')

plt.show()

  
