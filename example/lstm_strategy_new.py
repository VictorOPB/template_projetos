import pandas as pd
import numpy as np
from modules.load_data import load_data
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error


def initial_analysis():
    
  dict_data = load_data()
  log_returns = np.log(dict_data['prices']).diff().fillna(0)
  
  volatility_window = 63  # Assuming daily data

  # Initialize an empty DataFrame to store momentum scores of each analyzed period, and volatility within the 63-day period:
  momentum_scores_df = pd.DataFrame(index=log_returns.index)
  momentum_scores_21 = pd.DataFrame(index=log_returns.index)
  momentum_scores_63 = pd.DataFrame(index=log_returns.index)
  momentum_scores_126 = pd.DataFrame(index=log_returns.index)
  vol_df = pd.DataFrame(index=log_returns.index)
  volatilities = []
  for column in log_returns:
    # Calculate the rolling standard deviation to measure mean volatility for this stock
    vol_df[f'{column}'] = np.nan_to_num(log_returns[column].rolling(volatility_window).std())
    vol_values = np.nan_to_num(log_returns[column].rolling(volatility_window).std().values)
    mean_vol = np.mean(vol_values)
    volatilities.append(mean_vol)

    # Define multiple momentum periods in terms of trading days
    momentum_periods = [21, 63, 126]  # 1-month, 3-month, and 6-month

    # Calculate momentum scores for each period and store in the DataFrame
    for period in momentum_periods:
      momentum_scores = log_returns[column].rolling(period).sum()
      momentum_scores = momentum_scores.shift(-period)
      if period == 21:
          momentum_scores_21[f'{column}'] = np.nan_to_num(momentum_scores)
      elif period == 63:
          momentum_scores_63[f'{column}'] = np.nan_to_num(momentum_scores)
      else:
          momentum_scores_126[f'{column}'] = np.nan_to_num(momentum_scores)

  weight_21 = .4
  weight_63 = .3
  weight_126 = .3

  momentum_scores_df = weight_21*momentum_scores_21 + weight_63*momentum_scores_63 + weight_126*momentum_scores_126
  momentum_scores_df.fillna(0)
  vol_df.fillna(0)
  column = 'TSLA'
  for period in momentum_periods:
    if period == 21:
      plt.figure()
      sns.histplot(momentum_scores_21[f'{column}'], bins=30, kde=True, color='blue', edgecolor='k')
      plt.title(f"Distribution of '{column}' Stock {period}-Day Momentum Scores (Log Returns)")
      plt.xlabel('Momentum Score')
      plt.ylabel('Frequency')
      plt.grid(True)
    elif period == 63:
      plt.figure()
      sns.histplot(momentum_scores_63[f'{column}'], bins=30, kde=True, color='blue', edgecolor='k')
      plt.title(f"Distribution of '{column}' Stock {period}-Day Momentum Scores (Log Returns)")
      plt.xlabel('Momentum Score')
      plt.ylabel('Frequency')
      plt.grid(True)
    elif period == 126:
      plt.figure()
      sns.histplot(momentum_scores_126[f'{column}'], bins=30, kde=True, color='blue', edgecolor='k')
      plt.title(f"Distribution of '{column}' Stock {period}-Day Momentum Scores (Log Returns)")
      plt.xlabel('Momentum Score')
      plt.ylabel('Frequency')
      plt.grid(True)

  plt.figure()
  sns.histplot(momentum_scores_df[f'{column}'], bins=30, kde=True, color='blue', edgecolor='k')
  plt.title(f"Distribution of '{column}' Stock Weighted (.4, .3, .3) Momentum Scores (Log Returns)")
  plt.xlabel('Momentum Score')
  plt.ylabel('Frequency')
  plt.grid(True)

  # Define thresholds for low volatility and momentum based on volatilities array and momentum_scores_df
  low_volatility_threshold = 0.02
  momentum_threshold = np.log(1.1) # 110% momentum score, in an attempt to guarantee nice stock volume with good upward prospect trend.

  # Create boolean DataFrames for low volatility and momentum
  low_volatility_condition = vol_df < low_volatility_threshold
  momentum_condition = momentum_scores_df > momentum_threshold

  # Combine the conditions to select best 15 assets 
  selected_assets_df = low_volatility_condition & momentum_condition

  best_assets = []

  for column in selected_assets_df.columns.tolist():
      occurrences = np.count_nonzero(np.array(selected_assets_df[column]))
      best_assets.append({ 'name': column, 'occurrences': occurrences })

  best_assets = sorted(best_assets, key=lambda x: x['occurrences'], reverse=True)
  sel_stocks = [i['name'] for i in best_assets[:10]]

  print('Maximum volatility: ', np.max(volatilities))

  return sel_stocks

def prepare_model_data(sel_stocks):
  
  dict_data = load_data()
  df = dict_data['prices']
  df = df[sel_stocks]
  log_df = np.log(df).diff().fillna(0)

  plt.figure(1)
  for stock in sel_stocks:
    plt.plot(df.index, df[stock])
  plt.legend(sel_stocks)

  plt.figure(2)
  for stock in sel_stocks:
    plt.plot(log_df.index, log_df[stock])
  plt.legend(sel_stocks)

  num_of_past_dates = 60
  windowed_dfs = {}
  for stock in sel_stocks:
    windowed_stock_df = df_to_windowed_df(log_df[[stock]],
                            '2017-12-29',
                            '2019-06-28',
                            n=num_of_past_dates)
    windowed_dfs[stock] = windowed_stock_df

  maxes = [i.drop(columns=['Target Date', 'Target of Stock']).max().max() for i in windowed_dfs.values()]
  max_scale = np.max(maxes)

  print('Maximum scale: ', max_scale)

  return {
    'windowed_dfs': windowed_dfs,
    'max_scale': max_scale,
  }

def mount_wallet(sel_stocks, dfs_dict):

  train_model(sel_stocks[0], dfs_dict)

  # for stock in sel_stocks:
  #   train_model(stock, dfs_dict)




# Private Functions

def str_to_datetime(s):
  split = s.split('-')
  year, month, day = int(split[0]), int(split[1]), int(split[2])
  return datetime.datetime(year=year, month=month, day=day)

#

def df_to_windowed_df(dataframe, first_date_str, last_date_str, n=3):
  first_date = str_to_datetime(first_date_str)
  last_date  = str_to_datetime(last_date_str)

  target_date = first_date
  
  dates = []
  X, Y = [], []

  last_time = False
  col_name = list(dataframe.columns)[0]
  while True:
    df_subset = dataframe.loc[:target_date].tail(n+1)
    
    if len(df_subset) != n+1:
      print(f'Error: Window of size {n} is too large for date {target_date}')
      return

    values = df_subset[col_name].to_numpy()
    x, y = values[:-1], values[-1]

    dates.append(target_date)
    X.append(x)
    Y.append(y)

    next_week = dataframe.loc[target_date:target_date+datetime.timedelta(days=7)]
    next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
    next_date_str = next_datetime_str.split('T')[0]
    year_month_day = next_date_str.split('-')
    year, month, day = year_month_day
    next_date = datetime.datetime(day=int(day), month=int(month), year=int(year))
    
    if last_time:
      break
    
    target_date = next_date

    if target_date == last_date:
      last_time = True
    
  ret_df = pd.DataFrame({})
  ret_df['Target Date'] = dates
  
  X = np.array(X)
  for i in range(0, n):
    X[:, i]
    ret_df[f'Target-{n-i}'] = X[:, i]
  
  ret_df['Target of Stock'] = Y

  return ret_df

def windowed_df_to_date_X_y(windowed_dataframe):
  df_as_np = windowed_dataframe.to_numpy()

  dates = df_as_np[:, 0]

  middle_matrix = df_as_np[:, 1:-1]
  X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))

  Y = df_as_np[:, -1]

  return dates, X.astype(np.float32), Y.astype(np.float32)

def train_model(stock, dfs_dict):
  max_scale = dfs_dict['max_scale']
  stocks_df = dfs_dict['windowed_dfs']
  stocks_number = len(stocks_df.keys())
  dates, X1, y1 = windowed_df_to_date_X_y(stocks_df[stock])


  q_80 = int(len(dates) * .8)
  q_90 = int(len(dates) * .9)

  X_matrices = []
  y_matrices = []
  X_matrices_train = []
  X_matrices_val = []
  X_matrices_test = []
  y_matrices_train = []
  y_matrices_val = []
  y_matrices_test = []

  X_matrices.append(X1)
  y_matrices.append(y1)
  X_matrices_train.append(X1[:q_80])
  X_matrices_val.append(X1[q_80:q_90])
  X_matrices_test.append(X1[q_90:])
  y_matrices_train.append(y1[:q_80])
  y_matrices_val.append(y1[q_80:q_90])
  y_matrices_test.append(y1[q_90:])

  del stocks_df[stock]

  for stock in stocks_df:
    _, X, y = windowed_df_to_date_X_y(stocks_df[stock])
    X_matrices.append(X)
    y_matrices.append(y)
    X_matrices_train.append(X[:q_80])
    X_matrices_val.append(X[q_80:q_90])
    X_matrices_test.append(X[q_90:])
    y_matrices_train.append(y[:q_80])
    y_matrices_val.append(y[q_80:q_90])
    y_matrices_test.append(y[q_90:])

  dates_train = dates[:q_80]
  dates_val = dates[q_80:q_90]
  dates_test = dates[q_90:]

  X_train = np.concatenate(tuple(X_matrices_train),axis=2)
  y_train = np.concatenate(tuple(y_matrices_train))

  X_val = np.concatenate(tuple(X_matrices_val),axis=2)
  y_val = np.concatenate(tuple(y_matrices_val))

  X_test = np.concatenate(tuple(X_matrices_test),axis=2)
  y_test = np.concatenate(tuple(y_matrices_test))

  len(X_val)

  scaled_Xtrain = X_train / max_scale
  scaled_y_train = y_train/ max_scale

  scaled_Xval = X_val / max_scale
  scaled_y1val = y_val[0] / max_scale

  scaled_Xtest = X_test / max_scale
  scaled_ytest = y_test / max_scale

  # Parâmetros para experimentação
  num_dense_layers_list = [1, 2, 3]  # Número de camadas Dense
  num_neurons_list = [4, 8, 16, 32]  # Número de neurônios em cada camada

  best_mse = float('inf')  # Melhor MSE inicializado com infinito
  best_combination = None  # Melhor combinação de hiperparâmetros inicializada com None

  for num_dense_layers in num_dense_layers_list:
    for num_neurons in num_neurons_list:
      print(f"Experimentando com {num_dense_layers} camadas Dense e {num_neurons} neurônios por camada")

      # Crie o modelo
      model = Sequential([layers.Input(shape=(60, stocks_number)),
                          layers.LSTM(96)])
      
      for _ in range(num_dense_layers):
          model.add(layers.Dense(num_neurons, activation=activations.elu))

      model.add(layers.Dense(1))

      # Compile o modelo
      optimizer = Adam(learning_rate=0.0001, epsilon=1e-8)
      model.compile(loss=Huber(delta=1.0), optimizer=optimizer, metrics=['mean_squared_error'])

      # Adicione Early Stopping para evitar overfitting
      early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

      # Treine o modelo
      model.fit(scaled_Xtrain, scaled_y_train[0], validation_data=(scaled_Xval, scaled_y1val), epochs=100, callbacks=[early_stopping])

      # Avalie o modelo
      train_predictions = model.predict(scaled_Xtrain).flatten()
      mse = mean_squared_error(scaled_y_train[0], train_predictions)
      print("MSE:", mse)
      print("\n")

      # Atualize a melhor combinação se uma combinação melhor for encontrada
      if mse < best_mse:
          best_mse = mse
          best_combination = (num_dense_layers, num_neurons)

  # Imprima a melhor combinação e seu respectivo MSE
  print(f"Melhor combinação: {best_combination}")
  print(f"Melhor MSE: {best_mse:.8f}")

  # Change params based on best combination

  model = Sequential([layers.Input(shape=(60, stocks_number)),
                      layers.LSTM(96),
                      layers.Dense(8, activation=activations.elu),
                      layers.Dense(8, activation=activations.elu),
                      layers.Dense(1)])

  model.compile(loss=Huber(delta=1.0), 
                optimizer=Adam(learning_rate=0.0001, epsilon=1e-8),
                metrics=['mean_squared_error'])

  # Adicione Early Stopping para evitar overfitting
  early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)


  model.fit(scaled_Xtrain, scaled_y_train[0], validation_data=(scaled_Xval, scaled_y1val), epochs=100, callbacks=[early_stopping])

  train_predictions = model.predict(scaled_Xtrain).flatten()
  mse = mean_squared_error(scaled_y_train[0], train_predictions)
  print("MSE:", mse)

  # Show plots

  np.reshape(train_predictions, np.shape(dates_train))
  plt.plot(dates_train, scaled_y_train[0])
  plt.plot(dates_train, train_predictions)
  plt.legend(['Training Observations', 'Training Predictions'])

  val_predictions = model.predict(scaled_Xval).flatten()

  plt.plot(dates_val, val_predictions)
  plt.plot(dates_val, scaled_y1val)
  plt.legend(['Validation Predictions', 'Validation Observations'])

  test_predictions = model.predict(scaled_Xtest).flatten()

  plt.plot(dates_test, test_predictions)
  plt.plot(dates_test, scaled_ytest[0])
  plt.legend(['Testing Predictions', 'Testing Observations'])

  plt.plot(dates_train, train_predictions)
  plt.plot(dates_train, y_train[0])
  plt.plot(dates_val, val_predictions)
  plt.plot(dates_val, y_val[0])
  plt.plot(dates_test, test_predictions)
  plt.plot(dates_test, y_test[0])
  plt.legend(['Training Predictions', 
              'Training Observations',
              'Validation Predictions', 
              'Validation Observations',
              'Testing Predictions', 
              'Testing Observations'])