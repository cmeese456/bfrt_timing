from keras.layers import Dense, Dropout, Activation
from keras.layers import GRU, LSTM
from keras.models import Sequential
import sys
import csv
import numpy as np
import pandas as pd
from datetime import datetime
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tabulate import tabulate

# Necessary variables
attr = 'volume'
INPUT_LENGTH = 12
new_sample_size_per_comm_round = INPUT_LENGTH

# Function definitions
def get_scaler(df_whole):
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(df_whole[attr].values.reshape(-1, 1))
    return scaler


def process_train_data_single(df_train, scaler, INPUT_LENGTH):
    flow_train = scaler.transform(df_train[attr].values.reshape(-1, 1)).reshape(1, -1)[0]
    train_set = []
    for i in range(INPUT_LENGTH, len(flow_train)):
        train_set.append(flow_train[i - INPUT_LENGTH: i + 1])
    train = np.array(train_set)
    X_train = train[:, :-1]
    y_train = train[:, -1]

    return X_train, y_train

def process_test_one_step(df_test, scaler, INPUT_LENGTH):
    flow_test = scaler.transform(df_test[attr].values.reshape(-1, 1)).reshape(1, -1)[0]
    test_set = []
    for i in range(INPUT_LENGTH, len(flow_test)):
        test_set.append(flow_test[i - INPUT_LENGTH: i + 1])
    test = np.array(test_set)
    X_test = test[:, :-1]
    y_test = test[:, -1]
    return X_test, y_test

# Load the data
d_19992 = pd.read_csv('./19992_NB.csv', encoding='utf-8').fillna(0)

# Store times in DF
times = []

# Build the models and measure the time
## GRU
t1_s = time.process_time()
model_gru = Sequential()
model_gru.add(GRU(50, input_shape=(12, 1), return_sequences=True))
model_gru.add(GRU(50))
model_gru.add(Dropout(0.2))
model_gru.add(Dense(1, activation='sigmoid'))
model_gru.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
t1_e = time.process_time()
times.append(["build_gru", round(t1_e - t1_s, 3)])

## LSTM
t2_s = time.process_time()
model_lstm = Sequential()
model_lstm.add(LSTM(128, input_shape=(12, 1), return_sequences=True))
model_lstm.add(LSTM(128))
model_lstm.add(Dropout(0.2))
model_lstm.add(Dense(1, activation='sigmoid'))
model_lstm.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
t2_e = time.process_time()
times.append(["build_lstm", round(t2_e - t2_s, 3)])

# Time transformation, training, testing for all MDL
mdl_list = [24, 36, 48, 60, 72]
scaler = get_scaler(d_19992)
ending_index = 83

## Loop through MDL list
for mdl in mdl_list:
  # START Data preparation timing
  t3_s = time.process_time()
  ## Generate starting index
  starting_index = 83 - mdl
  ## Slice training data
  training_data = d_19992[starting_index: ending_index+1]
  ## Create x_train and x_test
  x_train, y_train = process_train_data_single(training_data, scaler, INPUT_LENGTH)
  ## Slice testing data
  test_data_starting_index = ending_index - 12 + 1
  test_data_ending_index = test_data_starting_index + 12 * 2 -1
  test_data = d_19992[test_data_starting_index: test_data_ending_index+1]
  x_test, y_test = process_test_one_step(test_data, scaler, INPUT_LENGTH)
  ## Transform the data
  for train_data in ['x_train', 'x_test']:
    vars()[train_data] = np.reshape(vars()[train_data], (vars()[train_data].shape[0], vars()[train_data].shape[1], 1))
  # END data preparation timing
  t3_e = time.process_time()
  temp_s = 'data_processing_mdl_' + str(mdl)
  times.append([temp_s, round(t3_e - t3_s, 3)])

  # START Training Timing LSTM
  t4_s = time.process_time()
  model_lstm.fit(x_train, y_train, batch_size=1, epochs=5, validation_split=0.0)
  t4_e = time.process_time()
  # END Training Timing LSTM
  temp_s2 = 'training_LSTM_mdl_' + str(mdl)
  times.append([temp_s2, round(t4_e - t4_s, 3)])

  # START Training Timing GRU
  t5_s = time.process_time()
  model_gru.fit(x_train, y_train, batch_size=1, epochs=5, validation_split=0.0)
  t5_e = time.process_time()
  # END Training Timing GRU
  temp_s3 = 'training_GRU_mdl_' + str(mdl)
  times.append([temp_s3, round(t5_e - t5_s, 3)])

  # START Testing Timing LSTM
  t6_s = time.process_time()
  predicts_lstm = model_lstm.predict(x_test)
  t6_e = time.process_time()
  # END Testing Timing LSTM
  temp_s4 = 'testing_LSTM_mdl_' + str(mdl)
  times.append([temp_s4, round(t6_e - t6_s, 3)])

  # START Testing Timing GRU
  t7_s = time.process_time()
  predicts_gru = model_gru.predict(x_test)
  t7_e = time.process_time()
  # END Testing Timing GRU
  temp_s5 = 'testing_GRU_mdl_' + str(mdl)
  times.append([temp_s5, round(t7_e - t7_s, 3)])

# Simulate FEDAVG
## Initialize Client List
participants = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25]
## Loop through client list
for clients in participants:
  ## Initialize empty weight matrix
  local_model_weights = []
  ## Fill the weight matrix according to number of clients
  for i in range (0, clients):
    local_model_weights.append(model_lstm.get_weights())
  ## START TIMING
  t8_s = time.process_time()
  fed_avg = np.mean(local_model_weights, axis=0)
  t8_e = time.process_time()
  temp_s6 = "FedAVG with Participants=" + str(clients)
  times.append([temp_s6, round(t8_e - t8_s, 5)])


# Create the DF
df = pd.DataFrame(times, columns=['Function', 'Time(s)'])
with open(f'./timings.txt', "a+") as file:
  file.write('\nNEW TRIAL\n')
  file.write(tabulate(df, headers='keys', tablefmt='psql'))
  file.write('\n')