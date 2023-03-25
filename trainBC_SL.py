import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split

# rtds constraints
i_ref = 0.68
v_ref = 0.48
high_d = i_ref + 0.05*i_ref
low_d = i_ref - 0.05*i_ref
high_q = 1e-3
low_q = -1e-3
high_md = 0.60
low_md = 0.36
v_unsafe1 = 0.36
v_unsafe2 = 0.60
v_lb = 0.4657
v_ub = 0.4942
v_fsc = 0.03*v_ref
v_safe1 = 0.4659
v_safe2 = 0.4940

# generates initial configuration
def D1(flag):
    i_d = np.random.uniform(low=low_d, high=high_d)
    i_q = np.random.uniform(low=low_q, high=high_q)
    i_od = np.random.uniform(low=low_d, high=high_d)
    i_oq = np.random.uniform(low=low_q, high=high_q)
    if flag % 2 == 0:
        v_od = np.random.uniform(low=v_unsafe1, high=v_lb)
    else:
        v_od = np.random.uniform(low=v_ub, high=v_unsafe2)
    v_oq = np.random.uniform(low=low_q, high=high_q)
    i_ld = np.random.uniform(low=low_d, high=high_d)
    i_lq = np.random.uniform(low=low_q, high=high_q)
    m_d = np.random.uniform(low=low_md, high=high_md)
    m_q = np.random.uniform(low=low_q, high=high_q)

    state = np.array([i_d, i_q, i_od, i_oq, v_od, v_oq, i_ld, i_lq, m_d, m_q])
    dataset = 1
    return state, dataset

def D2(flag):
    i_d = np.random.uniform(low=low_d, high=high_d)
    i_q = np.random.uniform(low=low_q, high=high_q)
    i_od = np.random.uniform(low=low_d, high=high_d)
    i_oq = np.random.uniform(low=low_q, high=high_q)
    if flag % 2 == 0:
        v_od = np.random.uniform(low=v_safe1, high=v_ref)
    else:
        v_od = np.random.uniform(low=v_ref, high=v_safe2)
    v_oq = np.random.uniform(low=low_q, high=high_q)
    i_ld = np.random.uniform(low=low_d, high=high_d)
    i_lq = np.random.uniform(low=low_q, high=high_q)
    m_d = np.random.uniform(low=low_md, high=high_md)
    m_q = np.random.uniform(low=low_q, high=high_q)

    state = np.array([i_d, i_q, i_od, i_oq, v_od, v_oq, i_ld, i_lq, m_d, m_q])
    dataset = 2
    return state, dataset

def D3(flag):
    i_d = np.random.uniform(low=low_d, high=high_d)
    i_q = np.random.uniform(low=low_q, high=high_q)
    i_od = np.random.uniform(low=low_d, high=high_d)
    i_oq = np.random.uniform(low=low_q, high=high_q)
    if flag % 2 == 0:
        v_od = np.random.uniform(low=v_lb, high=v_safe1)
    else:
        v_od = np.random.uniform(low=v_safe2, high=v_ub)
    v_oq = np.random.uniform(low=low_q, high=high_q)
    i_ld = np.random.uniform(low=low_d, high=high_d)
    i_lq = np.random.uniform(low=low_q, high=high_q)
    m_d = np.random.uniform(low=low_md, high=high_md)
    m_q = np.random.uniform(low=low_q, high=high_q)

    state = np.array([i_d, i_q, i_od, i_oq, v_od, v_oq, i_ld, i_lq, m_d, m_q])
    dataset = 3
    return state, dataset

unsafe1 = []
unsafe2 = []
safe1 = []
safe2 = []
safe3 = []
safe4 = []

for i in range(500):
    state, dataset = D1(0)
    unsafe1.append(state)
    state, dataset = D1(1)
    unsafe2.append(state)

for i in range(500):
    state, dataset = D2(0)
    safe1.append(state)
    state, dataset = D2(1)
    safe2.append(state)

for i in range(500):
    state, dataset = D3(0)
    safe3.append(state)
    state, dataset = D3(1)
    safe4.append(state)

df1 = pd.DataFrame(safe1, columns=['i_d', 'i_q', 'i_od', 'i_oq', 'v_od', 'v_oq', 'i_ld', 'i_lq', 'm_d', 'm_q'])
df1['u'] = pd.Series([np.random.uniform(-0.01,-0.001) for x in range(len(df1.index))])
df2 = pd.DataFrame(safe2, columns=['i_d', 'i_q', 'i_od', 'i_oq', 'v_od', 'v_oq', 'i_ld', 'i_lq', 'm_d', 'm_q'])
df2['u'] = pd.Series([np.random.uniform(-0.01,-0.001) for x in range(len(df2.index))])

df3 = pd.DataFrame(safe3, columns=['i_d', 'i_q', 'i_od', 'i_oq', 'v_od', 'v_oq', 'i_ld', 'i_lq', 'm_d', 'm_q'])
df3['u'] = pd.Series([np.random.uniform(-0.001,0) for x in range(len(df3.index))])
df4 = pd.DataFrame(safe4, columns=['i_d', 'i_q', 'i_od', 'i_oq', 'v_od', 'v_oq', 'i_ld', 'i_lq', 'm_d', 'm_q'])
df4['u'] = pd.Series([np.random.uniform(-0.001,0) for x in range(len(df4.index))])

df5 = pd.DataFrame(unsafe1, columns=['i_d', 'i_q', 'i_od', 'i_oq', 'v_od', 'v_oq', 'i_ld', 'i_lq', 'm_d', 'm_q'])
df5['u'] = pd.Series([np.random.uniform(0.0001,0.01) for x in range(len(df5.index))])
df6 = pd.DataFrame(unsafe2, columns=['i_d', 'i_q', 'i_od', 'i_oq', 'v_od', 'v_oq', 'i_ld', 'i_lq', 'm_d', 'm_q'])
df6['u'] = pd.Series([np.random.uniform(0.0001,0.01) for x in range(len(df6.index))])

frames = [df1, df2, df3, df4, df5, df6]
df = pd.concat(frames, ignore_index=True)
print(len(df))
# df.to_csv('Results/data.csv')

#adjust for NN usage
input = ['i_d', 'i_q', 'i_od', 'i_oq', 'v_od', 'v_oq', 'i_ld', 'i_lq', 'm_d', 'm_q']
output = ['u']
x = df[input]
y = df[output]
X_train = x.values
Y_train = y.values
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.01)

#DNN training
model = Sequential()
n_cols = X_train.shape[1]

#hidden layers
model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
model.add(Dense(10, activation='relu'))

#output layers
model.add(Dense(1))

#optimization
adam = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False)

model.compile(optimizer='adam', loss='mean_squared_error', metrics =['accuracy'])
model.summary()

#training
history = model.fit(X_train, Y_train, batch_size=1000, epochs=1000, verbose=1)

#training error visualization
# plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Error')
plt.xlabel('Epoch')
# plt.legend(['val_loss','train_loss'])
plt.show()

#saving NN model
model.save("Results/candidateBC.h5")
# model.save_weights("Results/candidateBC_weights.h5")