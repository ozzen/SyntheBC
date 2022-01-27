import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split

# rtds constraints
high_d = 0.69
low_d = 0.68
high_q = 1e-5
low_q = -1e-5
high_md = 0.39
low_md = 0.38
high_mq = 0.055
low_mq = 0.052
high_v_oq = 0.008
low_v_oq = 0.007
v_ref = 0.48
v_init = 0.01*v_ref
v_fsc = 0.03*v_ref

# generates initial configuration
def D1(flag):
    i_d = np.random.uniform(low=low_d, high=high_d)
    i_q = np.random.uniform(low=low_q, high=high_q)
    i_od = np.random.uniform(low=low_d, high=high_d)
    i_oq = np.random.uniform(low=low_q, high=high_q)
    if flag % 2 == 0:
        v_od = np.random.uniform(low=v_ref-v_fsc-v_init, high=v_ref-v_fsc)
    else:
        v_od = np.random.uniform(low=v_ref+v_fsc, high=v_ref+v_fsc+v_init)
    v_oq = np.random.uniform(low=low_v_oq, high=high_v_oq)
    i_ld = np.random.uniform(low=low_d, high=high_d)
    i_lq = np.random.uniform(low=0, high=0)
    m_d = np.random.uniform(low=low_md, high=high_md)
    m_q = np.random.uniform(low=low_mq, high=high_mq)

    state = np.array([i_d, i_q, i_od, i_oq, v_od, v_oq, i_ld, i_lq, m_d, m_q])
    dataset = 1
    return state, dataset

def D2():
    i_d = np.random.uniform(low=low_d, high=high_d)
    i_q = np.random.uniform(low=low_q, high=high_q)
    i_od = np.random.uniform(low=low_d, high=high_d)
    i_oq = np.random.uniform(low=low_q, high=high_q)
    v_od = np.random.uniform(low=v_ref-v_init, high=v_ref+v_init)
    v_oq = np.random.uniform(low=low_v_oq, high=high_v_oq)
    i_ld = np.random.uniform(low=low_d, high=high_d)
    i_lq = np.random.uniform(low=0, high=0)
    m_d = np.random.uniform(low=low_md, high=high_md)
    m_q = np.random.uniform(low=low_mq, high=high_mq)

    state = np.array([i_d, i_q, i_od, i_oq, v_od, v_oq, i_ld, i_lq, m_d, m_q])
    dataset = 2
    return state, dataset

def D3():
    i_d = np.random.uniform(low=low_d, high=high_d)
    i_q = np.random.uniform(low=low_q, high=high_q)
    i_od = np.random.uniform(low=low_d, high=high_d)
    i_oq = np.random.uniform(low=low_q, high=high_q)
    v_od = np.random.uniform(low=v_ref-v_fsc, high=v_ref+v_fsc)
    v_oq = np.random.uniform(low=low_v_oq, high=high_v_oq)
    i_ld = np.random.uniform(low=low_d, high=high_d)
    i_lq = np.random.uniform(low=0, high=0)
    m_d = np.random.uniform(low=low_md, high=high_md)
    m_q = np.random.uniform(low=low_mq, high=high_mq)

    state = np.array([i_d, i_q, i_od, i_oq, v_od, v_oq, i_ld, i_lq, m_d, m_q])
    dataset = 3
    return state, dataset

unsafe1 = []
unsafe2 = []
safe1 = []
safe2 = []

for i in range(100000):
    state, dataset = D2()
    safe1.append(state)

for i in range(50000):
    state, dataset = D1(0)
    unsafe1.append(state)
    state, dataset = D1(1)
    unsafe2.append(state)

df1 = pd.DataFrame(safe1, columns=['i_d', 'i_q', 'i_od', 'i_oq', 'v_od', 'v_oq', 'i_ld', 'i_lq', 'm_d', 'm_q'])
df1['u'] = pd.Series([-1 for x in range(len(df1.index))])
df2 = pd.DataFrame(unsafe1, columns=['i_d', 'i_q', 'i_od', 'i_oq', 'v_od', 'v_oq', 'i_ld', 'i_lq', 'm_d', 'm_q'])
df2['u'] = pd.Series([1 for x in range(len(df2.index))])
df3 = pd.DataFrame(unsafe2, columns=['i_d', 'i_q', 'i_od', 'i_oq', 'v_od', 'v_oq', 'i_ld', 'i_lq', 'm_d', 'm_q'])
df3['u'] = pd.Series([1 for x in range(len(df2.index))])
frames = [df1, df2, df3]
df = pd.concat(frames, ignore_index=True)
df.to_csv('Results/data.csv')

#adjust for NN usage
input = ['i_d', 'i_q', 'i_od', 'i_oq', 'v_od', 'v_oq', 'i_ld', 'i_lq', 'm_d', 'm_q']
output = ['u']
x = df[input]
y = df[output]
X = x.values
Y = y.values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

#DNN training
model = Sequential()
n_cols = X.shape[1]

#hidden layers
model.add(Dense(20, activation='relu', input_shape=(n_cols,)))
model.add(Dense(20, activation='relu'))

#output layers
model.add(Dense(1))

#optimization
adam = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False)
model.compile(optimizer='adam', loss='mean_squared_error', metrics =['accuracy'])
model.summary()

#training
history = model.fit(X_train, Y_train, validation_data=(X_test,Y_test), batch_size = 500 , epochs=10000, verbose = 1)

#finding dataset D3
dataset3 = []

for iter in range(10000):
    state, dataset = D3()
    safe2.append(state)
    df4 = pd.DataFrame(safe2, columns=['i_d', 'i_q', 'i_od', 'i_oq', 'v_od', 'v_oq', 'i_ld', 'i_lq', 'm_d', 'm_q'])
    z_test = df4[input]
    Z_test = z_test.values
    action = model.predict(Z_test)
    if iter % 100 == 0:
        print("Processing level: {}".format(iter))
    u = float(action[0])
    if u <= 0 and u >= -0.001:
        dataset3.append(state)

if len(dataset3) > 0:
    df4 = pd.DataFrame(dataset3, columns=['i_d', 'i_q', 'i_od', 'i_oq', 'v_od', 'v_oq', 'i_ld', 'i_lq', 'm_d', 'm_q'])
    df4['u'] = pd.Series([-1 for x in range(len(df1.index))])
    frames = [df1, df2, df3, df4]
    df = pd.concat(frames, ignore_index=True)
    df.to_csv('Results/training_data.csv')
else:
    print("D3 not found")

#training error visualization
plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.legend(['val_loss','train_loss'])
plt.show()

#saving NN model
model.save("Results/candidateBC.h5")
model.save_weights("Results/candidateBC_weights.h5")
