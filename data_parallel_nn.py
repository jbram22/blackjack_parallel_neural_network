# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # This is a data-parallelized Neural Network # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

############################################################################################################
########################################### IMPORT PACKAGES ################################################
############################################################################################################

# General
import os
import functools
import time
import numpy as np
import pandas as pd
import random
import math
import warnings

# Parallelization
from mpi4py import MPI


##############################################################################################################
########################################## HELPER FUNCTIONS ##################################################
##############################################################################################################

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x) *(1-sigmoid (x))

def train_network(wh,wo,epochs,train_X,train_Y):
  for epoch in range(epochs):
    
    # slice data 
    sliced_inputs = np.asarray(np.split(train_X, comm.size))
    sliced_labels = np.asarray(np.split(train_Y, comm.size))
    size = int(len(train_X)/comm.size)
    inputs_buf = np.zeros((size,hidden_layer_size))
    labels_buf = np.zeros(len(train_Y),dtype='i')

    # send data to each process
    comm.Scatter(sliced_inputs, inputs_buf, root=0)
    comm.Scatter(sliced_labels, labels_buf, root=0)

    ### neural network iterations ###

    ## feedforward ##

    # hidden layer
    zh = np.dot(train_X, wh)
    ah = sigmoid(zh)

    # output layer
    zo = np.dot(ah, wo)
    ao = sigmoid(zo)

    # error calculation
    error_out = ((1 / (2*len(train_X))) * (np.power((ao - train_Y), 2)))
      
    ## backpropogation ##

    # backpropogation from output layer to hidden layer
    dcost_dao = ao - train_Y
    dao_dzo = sigmoid_der(zo) 
    dzo_dwo = ah
    dcost_wo = np.dot(dzo_dwo.T, (dcost_dao * dao_dzo))

    # backpropogate from hidden layer to input layer
    dcost_dzo = dcost_dao * dao_dzo
    dzo_dah = wo
    dcost_dah = np.dot(dcost_dzo , dzo_dah.T)
    dah_dzh = sigmoid_der(zh) 
    dzh_dwh = train_X
    dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)

    comm.Barrier()

    # average error for all processes
    error_buf = [0] * comm.size

    try:
        error_buf = comm.gather(error_out)
        error_out = sum(error_buf) / len(error_buf)
    except TypeError as e:
        pass

     # if comm.rank == 0:
    #   print(f'error at iteration {epoch}: {error_out.sum()}')

    # gather gradients of weights for hidden layer from all processes
    dcost_wh_buf = np.asarray([np.zeros_like(dcost_wh)] * comm.size)
    comm.Gather(dcost_wh, dcost_wh_buf)
    comm.Barrier()
    dcost_wh = functools.reduce(np.add, dcost_wh_buf) / comm.size # average gradients across all processes

    # gather gradients of weights for output layer
    dcost_wo_buf = np.asarray([np.zeros_like(dcost_wo)] * comm.size)
    comm.Gather(dcost_wo, dcost_wo_buf)
    comm.Barrier()
    dcost_wo = functools.reduce(np.add, dcost_wo_buf) / comm.size # average gradients across all processes

    # update weights
    wh -= lr * dcost_wh
    wo -= lr * dcost_wo

    # send updated weights to processes
    comm.Bcast([wh, MPI.DOUBLE])
    comm.Bcast([wo, MPI.DOUBLE])

  return wh,wo

def predict(theta1,theta2, inputs):
    a2 = np.dot(inputs, theta1)  
    a2 = sigmoid(a2)
    a3 = np.dot(a2, theta2)  
    a3 = pd.Series(sigmoid(a3).reshape(-1))  
    predictions = np.where(a3 >= 0.5,1,-1)
    return pd.Series(predictions)

def accuracy_measures(predictions,actual):
  df = pd.concat([predictions,actual],axis = 1) # concatenate predicitons & actual labels into single dataframe
  df.columns = ['predictions','actual']
  df['correct'] = np.where(df.predictions == df.actual,1,0)
  # true positives
  positives = df.loc[df.actual == 1]
  true_positives = positives.correct.sum()
  # false negatives
  false_negatives = (positives.predictions == -1).sum()
  # tru negatives
  negatives = df.loc[df.actual == -1]
  true_negatives = negatives.correct.sum()
  # false Positives
  false_positives = (negatives.predictions == -1).sum()
  # overall accuracy
  accuracy = (true_positives + true_negatives)/(true_positives + true_negatives + false_positives + false_negatives)
  # precision
  precision = true_positives/(true_positives + false_positives)  
  # recall (sensitivity)
  sensitivity = true_positives/(true_positives+false_negatives)
  # specificity 
  specificity = true_negatives/(true_negatives + false_positives)
  return accuracy,precision, sensitivity, specificity


############################################################################################################
######################################## EXECUTION & PERFORMANCE ###########################################
############################################################################################################

if __name__ == '__main__':

  #suppress warnings
  warnings.filterwarnings('ignore')

####################################################
############ DATA IMPORT & FORMATTING ##############
####################################################

  model_df = pd.read_csv('blackjack.csv')
  X = np.array(model_df[[i for i in model_df.columns if i not in {'correct_action','outcome'}]])
  train_X = np.array(model_df[['player_initial_total', 'has_ace', 'dealer_card','count']])
  train_Y = np.array(model_df['correct_action']).reshape(-1,1) 

####################################################
############### MPI INITIALIZATION #################
####################################################

  # Init MPI
  comm = MPI.COMM_WORLD

  # structure of the 3-layer neural network
  hidden_layer_size = 10
  output_layer_size = 1
  lr = 1 # learning rate
  epochs = 50 # iterations

  # randomly initialize weights
  if comm.rank == 0:
    wh = np.random.rand(train_X.shape[1],hidden_layer_size) # weights for hidden layer
    wo = np.random.rand(hidden_layer_size, 1) # weights for output layer
  else:
    wh = np.random.rand(train_X.shape[1],hidden_layer_size)
    wo = np.random.rand(hidden_layer_size, 1)
  comm.Barrier()

  # communicate weight vectors
  comm.Bcast([wh, MPI.DOUBLE])
  comm.Bcast([wo, MPI.DOUBLE])


  #################################################
  ############ NEURAL NETWORK TRAINING ############
  #################################################

  if comm.rank == 0:
    start = time.time()

  wh,wo = train_network(wh,wo,epochs,train_X,train_Y)

  if comm.rank == 0:
    end = time.time()
    train_time = round(end-start,2)
    print(f'\nEND OF TRAINING, took {train_time} seconds\n')

    # write training time to file for plotting
    out_filename = f'nn_train_{comm.size}.txt'
    outfile = open(out_filename, "w")
    outfile.write(str(train_time))
   
  ################################################
  ############ PREDICTIONS & RESULTS #############
  ################################################

    # generate predictions
    predictions = predict(wh,wo,train_X)
    actual = pd.Series(train_Y.reshape(-1))

    # compute & display results
    accuracy,precision, sensitivity, specificity = accuracy_measures(predictions,actual)
    print('PERFORMANCE RESULTS:')
    print(f'accuracy: {100*round(accuracy,2)}%')
    print(f'precision: {100*round(precision,2)}%')
    print(f'sensitivity: {100*round(sensitivity,2)}%')
    print(f'specificity: {100*round(specificity,2)}%\n')
