import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from kerastuner import RandomSearch

 
df = pd.read_excel('NEW.xlsx')

 
X = df.iloc[:, [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 30, 31, 32, 33, 34, 35]].values
y = df[['aar', 'eeaa', 'peaa', 'geaa', 'dnamtladjage']].values

 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

 
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

 
def build_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units1', min_value=32, max_value=128, step=32), activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(units=hp.Int('units2', min_value=16, max_value=64, step=16), activation='relu'))
    model.add(Dense(units=5))   
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

 
tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=1,
    directory='my_dir',
    project_name='helloworld'
)

 
tuner.search(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))

 
best_model = tuner.get_best_models(num_models=1)[0]

 
best_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), verbose=1)

 
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

 
y_pred_dl = best_model.predict(X_test)
y_pred_lr = lr_model.predict(X_test)

 
plt.figure(figsize=(15, 10))
for i, col in enumerate(['aar', 'eeaa', 'peaa', 'geaa', 'dnamtladjage']):
    plt.subplot(3, 2, i + 1)
    plt.scatter(y_test[:, i], y_pred_dl[:, i], label='Deep Learning', alpha=0.5)
    plt.scatter(y_test[:, i], y_pred_lr[:, i], label='Linear Regression', alpha=0.5)
    plt.plot([y_test[:, i].min(), y_test[:, i].max()], [y_test[:, i].min(), y_test[:, i].max()], 'k--', lw=2)
    plt.title(col)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()

plt.tight_layout()
plt.show()

 
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
 
import sys
print(sys.version)

import os
print(os.getcwd())
