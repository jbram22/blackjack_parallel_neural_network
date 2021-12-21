#!/bin/zsh

python3 non_parallel_nn.py

mpirun -n 4 python3 -m mpi4py data_parallel_nn.py
mpirun -n 3 python3 -m mpi4py data_parallel_nn.py
mpirun -n 2 python3 -m mpi4py data_parallel_nn.py

python3 plotting.py