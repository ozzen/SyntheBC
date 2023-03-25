import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import keras.backend as K
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split

# rtds constraints
i_d_ref = 0.68
v_ref = 0.48
high_d = i_d_ref + 0.05*i_d_ref
low_d = i_d_ref - 0.05*i_d_ref
high_q = 1e-3
low_q = -1e-3
high_md = 0.60
low_md = 0.36
v_unsafe1 = 0.36
v_unsafe2 = 0.60
v_lb = 0.4657
v_ub = 0.4943
v_fsc = 0.03*v_ref
v_safe1 = 0.4659
v_safe2 = 0.4941

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

for i in range(5):
    state, dataset = D1(0)
    unsafe1.append(state)
    state, dataset = D1(1)
    unsafe2.append(state)

for i in range(5):
    state, dataset = D2(0)
    safe1.append(state)
    state, dataset = D2(1)
    safe2.append(state)

for i in range(1):
    state, dataset = D3(0)
    safe3.append(state)
    state, dataset = D3(1)
    safe4.append(state)

df1 = pd.DataFrame(safe1, columns=['i_d', 'i_q', 'i_od', 'i_oq', 'v_od', 'v_oq', 'i_ld', 'i_lq', 'm_d', 'm_q'])
df1['u'] = pd.Series([np.random.uniform(-0.01,-0.001) for x in range(len(df1.index))])
df2 = pd.DataFrame(safe2, columns=['i_d', 'i_q', 'i_od', 'i_oq', 'v_od', 'v_oq', 'i_ld', 'i_lq', 'm_d', 'm_q'])
df2['u'] = pd.Series([np.random.uniform(-0.01,-0.001) for x in range(len(df2.index))])

df3 = pd.DataFrame(safe3, columns=['i_d', 'i_q', 'i_od', 'i_oq', 'v_od', 'v_oq', 'i_ld', 'i_lq', 'm_d', 'm_q'])
df3['u'] = pd.Series([np.random.uniform(0,0) for x in range(len(df3.index))])
df4 = pd.DataFrame(safe4, columns=['i_d', 'i_q', 'i_od', 'i_oq', 'v_od', 'v_oq', 'i_ld', 'i_lq', 'm_d', 'm_q'])
df4['u'] = pd.Series([np.random.uniform(0,0) for x in range(len(df4.index))])

df5 = pd.DataFrame(unsafe1, columns=['i_d', 'i_q', 'i_od', 'i_oq', 'v_od', 'v_oq', 'i_ld', 'i_lq', 'm_d', 'm_q'])
df5['u'] = pd.Series([np.random.uniform(0.0001,0.01) for x in range(len(df5.index))])
df6 = pd.DataFrame(unsafe1, columns=['i_d', 'i_q', 'i_od', 'i_oq', 'v_od', 'v_oq', 'i_ld', 'i_lq', 'm_d', 'm_q'])
df6['u'] = pd.Series([np.random.uniform(0.0001,0.01) for x in range(len(df6.index))])

frames = [df1, df2, df5, df6]
df = pd.concat(frames, ignore_index=True)
# print(len(df))
# df.to_csv('Results/data.csv')

#adjust for NN usage
input = ['i_d', 'i_q', 'i_od', 'i_oq', 'v_od', 'v_oq', 'i_ld', 'i_lq', 'm_d', 'm_q']
output = ['u']
x = df[input]
y = df[output]
x_train = x.values
y_train = y.values
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.01)

#DNN training
model = Sequential()
n_cols = x_train.shape[1]

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
history = model.fit(x_train, y_train, batch_size=10, epochs=10, verbose=1)

print("\n Finished 1st phase of training. Initiating 2nd phase of training now..... \n")

# preprocessing for training the NN for the 3rd BaC condition
frames_new = [df3, df4]
df_new = pd.concat(frames_new, ignore_index=True)
df_new.to_csv('Results/data.csv')
input_new = ['i_d', 'i_q', 'i_od', 'i_oq', 'v_od', 'v_oq', 'i_ld', 'i_lq', 'm_d', 'm_q']
output_new = ['u']

x_new = df_new[input_new]
y_new = df_new[output_new]
x_train_new = x_new.values
y_train_new = y_new.values
print(x_new)

# model dynamics
def f(x):
    x1,x2,x3,x4,x5,x6,x7,x8,x9,x10 = x
    p_star = 1
    q_star = 1e-6
    w = 60
    r_n = 1e3
    r_c = 0.0384
    c_f = 2500
    r_f = 2e-3
    l_f = 100e-6
    i_ref_d = 2.08
    i_ref_q = 1e-6
    k_p = 0.5
    v_bd = 0.48
    v_bq = 1e-6
    f1=[]
    f1.append(-p_star*x1 + w*x2 + v_bd)
    f1.append(-q_star*x2 - w*x1 + v_bq)
    f1.append(-r_c*x3 +w*x4 + x5 - v_bd)
    f1.append(-r_c*x4 - w*x3 + x6 - v_bq)
    f1.append(w*x6 + (x7-x3)/c_f)
    f1.append(-w*x5 + (x8-x4)/c_f)
    f1.append(-(r_f/l_f)*x7 + w*x8 + (x9-x5)/l_f)
    f1.append(-(r_f/l_f)*x8 - w*x7 + (x10-x6)/l_f)
    f1.append(-w*x8 + k_p*(i_ref_d-x7))
    f1.append(-w*x7 + k_p*(i_ref_q-x8))
    return f1

x = [0.646, 0.001, 0.646, 0.001, 0.49340257875690746, 0.001, 0.714, 0.001, 0.6, -0.001]

# loss function
def loss_func(y_true, y_pred):
    # grads = tf.GradientTape(model.output, model.input)
    # W1 = model.layers[0].weights[0]
    # W2 = model.layers[1].get_weights()[0]
    # W3 = model.layers[2].get_weights()[0]
    # print(W1)
    sub_loss = K.maximum(y_pred,y_true)
    loss = K.sum(sub_loss)
    return loss

# model.add_loss(loss_func(y_train_new, y_train_new, x_train_new))
model.compile(optimizer='adam', loss=loss_func, metrics=['accuracy'])
history = model.fit(x_train_new, y_train_new, batch_size=2, epochs=10, verbose=1)

extractor = keras.Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
features = extractor(x_train_new)
print(features)

# Z1 = model.layers[1].output
# W1 = model.layers[0].get_weights()[0]
# W2 = model.layers[1].get_weights()[0]
# W3 = model.layers[2].get_weights()[0]
# W = W1@f(x)
# print(W)
# print(W1,Z1)

#training error visualization
# # plt.plot(history.history['val_loss'])
# plt.plot(history.history['loss'])
# plt.title('Model Loss')
# plt.ylabel('Error')
# plt.xlabel('Epoch')
# # plt.legend(['val_loss','train_loss'])
# plt.show()

#saving NN model
# model.save("Results/candidateBC.h5")
# model.save_weights("Results/candidateBC_weights.h5")