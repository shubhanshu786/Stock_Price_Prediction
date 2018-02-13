
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


dataframe = pd.read_csv('./HistoricalQuotes_APPLE.csv') #Apple stock price fr 10 years, downloaded from nasdaq.com


# In[3]:


dataframe = dataframe.iloc[::-1].reset_index(drop=True)


# In[4]:


dataframe = dataframe.iloc[:, 3:4].values


# In[5]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
scaled_price = sc.fit_transform(dataframe)


# In[6]:


X_train = []
y_train = []
X_test = []
y_test = []
time_units = 128
test_size = 80 # 4 month data
for i in range(time_units, len(scaled_price)-test_size):
    X_train.append(scaled_price[i-time_units:i, 0])
    y_train.append(scaled_price[i,0])
for i in range(len(scaled_price)-test_size-1, len(scaled_price)):
    X_test.append(scaled_price[i-time_units:i, 0])
    y_test.append(scaled_price[i,0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)


# In[7]:


X_train = X_train.reshape((X_train.shape[0], X_train.shape[1],1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1],1))


# In[8]:


from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout


# In[9]:


model = Sequential()
model.add(LSTM(return_sequences=True, units=256, input_shape=(X_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=256, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=256))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='adam')


# In[10]:


model.fit(X_train, y_train,batch_size=32, epochs=100)


# In[14]:


X_test  = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
predicted_price = model.predict(X_test[1:])


# In[15]:


y_test = y_test.reshape((y_test.shape[0],1))


# In[16]:


predicted_price = sc.inverse_transform(predicted_price)
real_price = sc.inverse_transform(y_test)
plt.plot(real_price, color = 'red', label='Real')
plt.plot(predicted_price, color = 'green', label = 'predicted')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Apple Stock Price')
plt.legend()
plt.show()


# In[17]:


from keras.models import load_model


# In[18]:


model.save('apple_stock_price.h5')

