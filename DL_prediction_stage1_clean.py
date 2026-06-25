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

# 加载数据集
df = pd.read_excel('NEW.xlsx')

# 选择特征和目标变量
X = df.iloc[:, [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 30, 31, 32, 33, 34, 35]].values
y = df[['aar', 'eeaa', 'peaa', 'geaa', 'dnamtladjage']].values

# 数据归一化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割数据集：80%训练集，10%验证集，10%测试集
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 定义模型构建函数
def build_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units1', min_value=32, max_value=128, step=32), activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(units=hp.Int('units2', min_value=16, max_value=64, step=16), activation='relu'))
    model.add(Dense(units=5))  # 输出层有5个单位
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 创建 Keras Tuner 实例
tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=1,
    directory='my_dir',
    project_name='helloworld'
)

# 开始超参数调优
tuner.search(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))

# 获取最佳模型
best_model = tuner.get_best_models(num_models=1)[0]

# 训练最佳模型
best_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), verbose=1)

# 训练线性回归模型
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# 预测
y_pred_dl = best_model.predict(X_test)
y_pred_lr = lr_model.predict(X_test)

# 画出预测结果
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

# 打印最佳超参数
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
print("最佳超参数:")
print(f"第一层单元数: {best_hyperparameters.get('units1')}")
print(f"第二层单元数: {best_hyperparameters.get('units2')}")

import sys
print(sys.version)

import os
print(os.getcwd())
