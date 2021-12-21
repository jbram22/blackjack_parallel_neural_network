# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # play blackjack but use the neural net to make hit/stay decision # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

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

# Function to add 4 of any specific card to a deck 
def add_complete(deck,card):
  """
    @input  | list (deck of cards
    ----------------------------------------------
    @goal   | add 4 of the same card, as specified
            | by the 'card' parameter
    ----------------------------------------------        
    @output | a list (deck of cards), now with 
            | 4 additional cards
    """
  deck.append(card)
  deck.append(card)
  deck.append(card)
  deck.append(card)
  return deck

def make_shoe():
  """
    @goal   | create blackjack "shoe", which is shuffled
            | collection of 6 decks, all composed of 52 cards 
    ---------------------------------------------------------        
    @output | a list (the blackjack "shoe")
    """
  deck = []
  for i in range(1,7): # make 6 decks (casino standard)
    for i in range(2,11): # add numbered cards
      deck = add_complete(deck,i)
    for face in {'J','Q','K','A'}: # add face cards
      deck = add_complete(deck,face)
  horn = random.shuffle(deck)
  return deck

def hand_value(hand): # may incorporate ability to value aces as 2 
  """
  @input  | a list, representing
          | a player's hand, consisting of
          | two (or more) cards from the horn
  ----------------------------------------------
  @goal   | compute the value of the player's 
          | hand
  ----------------------------------------------       
  @output | an integer representing the value of 
          | a player's hand 
  """
  aces = 0
  face_cards = {'J','Q','K'}

  total = 0
  for card in hand:
    if card in face_cards:
      total += 10
    elif card == 'A':
      aces += 1
    else:
      total += card
  
  # consider possible values for an ace (they can count for 1 or 11)
  if aces >= 1:
    for i in range(aces):
      if total + 11 > 21:
        total += 1
      else:
        total += 11
  
  return total


# Card Counter
def count_cards(cards,current_count,number_of_decks):
  """
  @input  | cards: list of cards to be added to
          | the count
          |
          | current_count (float): current count, 
          | according to omega 2 method
          |
          | number of decks (integer): number of
          | decks used to find the true count
  ----------------------------------------------
  @goal   | compute the card count, according to 
          | omega 2 counting system
  ----------------------------------------------       
  @output | card count, according to omega 2
  """
  count = 0
  for card in cards:
    if (card == 2) or (card == 3) or (card == 7):
      count += 1
    elif (card == 4) or (card == 5) or (card == 6):
      count += 2
    elif card == 9:
      count -= 1
    elif (card == 10) or (card == 'J') or (card == 'Q') or (card == 'K'):
      count -= 2
  return (count/number_of_decks) + current_count

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
  a3 = sigmoid(a3)
  if a3 >= 0.5:
    return 1
  else:
    return 0

def simulate(shoes,players,wh,wo):
  """
  @input  | shoes: number of shoes to use for 
          | simulation
          |
          | theta2: number of players who will
          | play alongside the NN
          |
          | wh: numpy array of weights
          | for the hidden layer
          |
          | wo: numpy array of weights
          | for the output layer
  ----------------------------------------------
  @goal   | run a large simulation of blackjack
          | hands to assess NN performance
  ----------------------------------------------       
  @output | print statement regarding % of correct 
          | decisions
  """
# shoes = 50000
# players = 4 
  blackjack = 21

  # NN Feature List
  dealer_card_feature = []
  player_card_feature = []
  player_results = []
  player_action = []
  actions = []
  aces = []
  games_played = [] # games played with same shoe

  # Card Counting Variables
  count = 0
  cards_dealt = []
  count_list = []

  for shoe in range(shoes):
      count = 0
      cards_dealt = []
      dealer_cards = make_shoe()
      games_played_with_same_shoe = 0
      while len(dealer_cards) > 156:
        actions = [] # new set of actions for all players at start of each game
        games_played_with_same_shoe += 1
        curr_player_results = np.zeros((1,players)) # numpy array for score keeping, win = 1, tie = 0, loss = -1
        dealer_hand = []
        player_hands = [[] for player in range(players)] # list of players hands (which are also lists)

        # Deal FIRST card
        for player, hand in enumerate(player_hands):
            card = dealer_cards.pop(0)
            player_hands[player].append(card)
            cards_dealt.append(card)
        card = dealer_cards.pop(0)
        dealer_hand.append(card)
        cards_dealt.append(card)

        # Deal SECOND card
        for player, hand in enumerate(player_hands):
            card = dealer_cards.pop(0)
            player_hands[player].append(card)
        card = dealer_cards.pop(0)
        dealer_hand.append(card)
        cards_dealt.append(card)

        # Dealer checks for 21
        if hand_value(dealer_hand) == blackjack:
            for player in range(players):
                if hand_value(player_hands[player]) == blackjack:
                    curr_player_results[0,player] = 0 # if player also has blackjack, so they simply get their money back
                else:
                    curr_player_results[0,player] = -1 # player does not have blackjack, and therefore lose
            actions = list(np.zeros(players)) # nobody can act if dealer has blackjack
        else:
            for player in range(players):
                action = 0 # initialize every player's action to 0 (stand)
                # Players check for 21
                if hand_value(player_hands[player]) == blackjack:
                    curr_player_results[0,player] = 1
                else:
                    # Hit randomly, check for busts
                    ''' As described in assumptions, this is to create a 'rich' dataset '''
                    if 'A' in player_hands[player]:
                      has_ace = 1
                    else:
                      has_ace = 0
                    
                    dealer_shows = dealer_hand[0]

                    if dealer_hand[0] in {'J','Q','K'}:
                      dealer_shows = 10
                    elif dealer_hand[0] == 'A':
                      dealer_shows = 11

                    stats = np.array([hand_value(player_hands[player]),has_ace, dealer_shows, count,games_played_with_same_shoe]).reshape(1,-1)

                    if (predict(wh, wo, stats) == 1) and (hand_value(player_hands[player]) != 21):
                      card = dealer_cards.pop(0)
                      player_hands[player].append(card)
                      cards_dealt.append(card)
                      action = 1
                      if hand_value(player_hands[player]) > 21:
                          curr_player_results[0,player] = -1
                actions.append(action)
        
        # Dealer hits based on the rules (dealer hits until they receive 17)
        while hand_value(dealer_hand) < 17:
            card = dealer_cards.pop(0)
            dealer_hand.append(card)
            cards_dealt.append(card)
            
        # check if dealer busted
        if hand_value(dealer_hand) > 21:
            for player in range(players):
                if curr_player_results[0,player] != -1:
                    curr_player_results[0,player] = 1
        
        # compare dealer hand to players hand's
        else:
            for player in range(players):
                if hand_value(player_hands[player]) > hand_value(dealer_hand):
                    if hand_value(player_hands[player]) <= 21:
                        curr_player_results[0,player] = 1 # player wins
                elif hand_value(player_hands[player]) == hand_value(dealer_hand):
                    curr_player_results[0,player] = 0 # player ties dealer
                else:
                    curr_player_results[0,player] = -1 # player loses

        # update count 
        count += count_cards(cards_dealt,count,6)

        # collect features
        player_card_feature.append(player_hands) # list of individual players hands (which are also lists)
        player_results.append(list(curr_player_results[0])) # list of players record (win, loss, tie)
        count_list.append(count) # keep track of count for this shoe
        aces.append(cards_dealt.count('A')) # record number of aces dealt out of shoe
        player_action.append(actions[0]) # list of player actions (hit or stay)
        games_played.append(games_played_with_same_shoe) # list of games played with same shoes

        # Dealer's Face-Up Card
        if dealer_hand[0] == 'A':
          dealer_card_feature.append(11)
        elif dealer_hand[0] in {'J','Q','K'}:
          dealer_card_feature.append(10)
        else:
          dealer_card_feature.append(dealer_hand[0]) # card dealer is showing



  # Transfer data into pandas datadframe
  model_df = pd.DataFrame()
  model_df['dealer_card'] = dealer_card_feature 
  model_df['player_initial_total'] = [hand_value(i[0][0:2]) for i in player_card_feature] # compute our initial total (assuming we are player 1)
  model_df['count'] = count_list
  model_df['aces_dealt'] = aces
  model_df['same_shoe_games'] = games_played

  has_ace = []
  for i in player_card_feature:
      if ('A' in i[0][0:2]):
          has_ace.append(1)
      else:
          has_ace.append(0)
  model_df['has_ace'] = has_ace

  model_df['player_action'] = player_action

  model_df['outcome'] = [i[0] for i in player_results] # 1 is win, -1 is loss

  model_df['correct_action'] = np.where(model_df.outcome != -1, 1,0) # correct action


  print('SIMULATION OVER\n')
  percent_correct = model_df.correct_action.value_counts()[1] / (model_df.correct_action.value_counts()[1] + model_df.correct_action.value_counts()[0])
  print(f'Blackjack NN played {len(model_df)} hands and correctly hit on {100*round(percent_correct,2)}% of the total hands')


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

  # run simulation (print results)
  simulate(50000,4,wh,wo)
