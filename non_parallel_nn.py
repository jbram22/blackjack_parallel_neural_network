# # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
# # # # This is a NON-PARALLELIZED Neural Network # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

############################################################################################################
########################################### IMPORT PACKAGES ################################################
############################################################################################################

import time
import numpy as np
import pandas as pd
import random
import warnings


##############################################################################################################
########################################## HELPER FUNCTIONS ##################################################
##############################################################################################################

def sigmoid(x):
  return 1/(1+np.exp(-x))

def sigmoid_der(x):
  return sigmoid(x) *(1-sigmoid (x))

def train_network(wh,wo,epochs,train_X,train_Y):
  """
    @input  | wh: numpy array of weights
            | for the hidden layer
            |
            | wo: numpy array of weights
            | for the output layer
            | 
            | epochs: number of iterations
            |
            | train_X: numpy array consisting of
            | training data for neural net
            |
            | train_Y: numy array (1D) of training
            | labels
    ----------------------------------------------
    @goal   | add 4 of the same card, as specified
            | by the 'card' parameter
    ----------------------------------------------        
    @output | a list (deck of cards), now with 
            | 4 additional cards
    """
  for epoch in range(epochs):

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

    # update weights
    wh -= lr * dcost_wh
    wo -= lr * dcost_wo

  return wh,wo

def predict(theta1,theta2, inputs):
  """
  @input  | theta1: numpy array of weights
          | for the hidden layer
          |
          | theta2: numpy array of weights
          | for the output layer
          |
          | inputs: numpy array of current stats
          | for "our player"
  ----------------------------------------------
  @goal   | predict if "we" should hit or stay
  ----------------------------------------------       
  @output | 1 for "hit" and 0 for "stay"
  """
  a2 = np.dot(inputs, theta1)  
  a2 = sigmoid(a2)
  a3 = np.dot(a2, theta2)  
  a3 = pd.Series(sigmoid(a3).reshape(-1))  
  predictions = np.where(a3 >= 0.5,1,-1)
  return pd.Series(predictions)

def accuracy_measures(predictions,actual):
  """
  @input  | predictions: numpy array (1D) of
          | predictions for each data point
          | in train_X dataframe
          |
          | actual: numpy array (1D) of the actual
          | labels of the datapoints in train_Y
  ----------------------------------------------
  @goal   | compute various accuracy measures to
          | assess model accuracy
  ----------------------------------------------       
  @output | various accuracy measures
  """
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
  warnings.filterwarnings('ignore')
  # data input & formatting
  model_df = pd.read_csv('blackjack.csv')
  X = np.array(model_df[[i for i in model_df.columns if i not in {'correct_action','outcome'}]])
  train_X = np.array(model_df[['player_initial_total', 'has_ace', 'dealer_card','count','same_shoe_games']])
  train_Y = np.array(model_df['correct_action']).reshape(-1,1) 

  # structure of the 3-layer neural network
  hidden_layer_size = 10
  output_layer_size = 1
  lr = 1 # learning rate
  epochs = 50 # iterations

  # RANDOMLY assign weights
  wh = np.random.rand(train_X.shape[1],hidden_layer_size)
  wo = np.random.rand(hidden_layer_size, 1)

  # train network
  start = time.time()
  wh,wo = train_network(wh,wo,epochs,train_X,train_Y)
  end = time.time()
  train_time = round(end-start,2)
  print(f'\nEND OF TRAINING, took {train_time} seconds\n')
  # write training time to file for plotting
  out_filename = 'non_parallel_nn.txt'
  outfile = open(out_filename, "w")
  outfile.write(str(train_time))

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

