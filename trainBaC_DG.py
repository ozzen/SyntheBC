import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split

# rtds constraints
low_Tm = 0.06
high_Tm = 0.80
low_Wpu = 0.967
high_Wpu = 1.033
low_Vcon = -0.0371
high_Vcon = 0.0
low_Efd = 0.5
high_Efd = 3.3
low_PsiD = 1.0
high_PsiD = 1.42
low_PsiQ = 0.14
high_PsiQ = 1.0
m = 1
low_L = 0.10
high_L = 0.65
low_Id = 0.02
high_Id = 0.40
low_Iq = 0.01
high_Iq = 0.30
low_Vd = 11.85
high_Vd = 13.10
low_Vq = 0.01
high_Vq = 1.30

# safety constraints
Wpu_unsafe1 = 0.9
Wpu_unsafe2 = 1.1
Wpu_safe1 = 0.975
Wpu_safe2 = 1.025
Vd_unsafe1 = 10
Vd_unsafe2 = 15
Vd_safe1 = 11.95
Vd_safe2 = 13

# generates initial configuration
def D1(flag):
    Tm = np.random.uniform(low=low_Tm, high=high_Tm)
    Vcon = np.random.uniform(low=low_Vcon, high=high_Vcon)
    Efd = np.random.uniform(low=low_Efd, high=high_Efd)
    PsiD = np.random.uniform(low=low_PsiD, high=high_PsiD)
    PsiQ = np.random.uniform(low=low_PsiQ, high=high_PsiQ)
    # L = 1.0
    L = np.random.uniform(low=low_L, high=high_L)
    Id = np.random.uniform(low=low_Id, high=high_Id)
    Iq = np.random.uniform(low=low_Iq, high=high_Iq)
    Vq = np.random.uniform(low=low_Vq, high=high_Vq)
    if flag == 1:
        Wpu = np.random.uniform(low=Wpu_unsafe1, high=Wpu_safe1)
        Vd = np.random.uniform(low=Vd_safe1, high=Vd_safe2)
    if flag == 2:
        Wpu = np.random.uniform(low=Wpu_safe2, high=Wpu_unsafe2)
        Vd = np.random.uniform(low=Vd_safe1, high=Vd_safe2)
    if flag == 3:
        Wpu = np.random.uniform(low=Wpu_safe1, high=Wpu_safe2)
        Vd = np.random.uniform(low=Vd_unsafe1, high=Vd_safe1)
    if flag == 4:
        Wpu = np.random.uniform(low=Wpu_safe1, high=Wpu_safe2)
        Vd = np.random.uniform(low=Vd_safe2, high=Vd_unsafe2)

    state = np.array([Tm,Wpu,Vcon,Efd,PsiD,PsiQ,m,L,Id,Iq,Vd,Vq])
    return state

def D2():
    Tm = np.random.uniform(low=low_Tm, high=high_Tm)
    Wpu = np.random.uniform(low=Wpu_safe1, high=Wpu_safe2)
    Vcon = np.random.uniform(low=low_Vcon, high=high_Vcon)
    Efd = np.random.uniform(low=low_Efd, high=high_Efd)
    PsiD = np.random.uniform(low=low_PsiD, high=high_PsiD)
    PsiQ = np.random.uniform(low=low_PsiQ, high=high_PsiQ)
    # L = 1.0
    L = np.random.uniform(low=low_L, high=high_L)
    Id = np.random.uniform(low=low_Id, high=high_Id)
    Iq = np.random.uniform(low=low_Iq, high=high_Iq)
    Vd = np.random.uniform(low=Vd_safe1, high=Vd_safe2)
    Vq = np.random.uniform(low=low_Vq, high=high_Vq)

    state = np.array([Tm,Wpu,Vcon,Efd,PsiD,PsiQ,m,L,Id,Iq,Vd,Vq])
    return state

def D3(flag):
    Tm = np.random.uniform(low=low_Tm, high=high_Tm)
    Vcon = np.random.uniform(low=low_Vcon, high=high_Vcon)
    Efd = np.random.uniform(low=low_Efd, high=high_Efd)
    PsiD = np.random.uniform(low=low_PsiD, high=high_PsiD)
    PsiQ = np.random.uniform(low=low_PsiQ, high=high_PsiQ)
    # L = 1.0
    L = np.random.uniform(low=low_L, high=high_L)
    Id = np.random.uniform(low=low_Id, high=high_Id)
    Iq = np.random.uniform(low=low_Iq, high=high_Iq)
    Vq = np.random.uniform(low=low_Vq, high=high_Vq)
    if flag == 1:
        Wpu = np.random.uniform(low=low_Wpu, high=Wpu_safe1)
        Vd = np.random.uniform(low=Vd_safe1, high=Vd_safe2)
    if flag == 2:
        Wpu = np.random.uniform(low=Wpu_safe2, high=high_Wpu)
        Vd = np.random.uniform(low=Vd_safe1, high=Vd_safe2)
    if flag == 3:
        Wpu = np.random.uniform(low=Wpu_safe1, high=Wpu_safe2)
        Vd = np.random.uniform(low=low_Vd, high=Vd_safe1)
    if flag == 4:
        Wpu = np.random.uniform(low=Wpu_safe1, high=Wpu_safe2)
        Vd = np.random.uniform(low=Vd_safe2, high=high_Vd)

    state = np.array([Tm,Wpu,Vcon,Efd,PsiD,PsiQ,m,L,Id,Iq,Vd,Vq])
    return state

unsafe1 = []
unsafe2 = []
unsafe3 = []
unsafe4 = []
initsafe = []
safe1 = []
safe2 = []
safe3 = []
safe4 = []

for i in range(200000):
    state = D1(1)
    unsafe1.append(state)
    state = D1(2)
    unsafe2.append(state)
    state = D1(3)
    unsafe3.append(state)
    state = D1(4)
    unsafe4.append(state)

for i in range(400000):
    state = D2()
    initsafe.append(state)

for i in range(200000):
    state = D3(1)
    safe1.append(state)
    state = D3(2)
    safe2.append(state)
    state = D3(3)
    safe3.append(state)
    state = D3(4)
    safe4.append(state)

df0 = pd.DataFrame(initsafe, columns=['Tm','Wpu','Vcon','Efd','PsiD','PsiQ','m','L','Id','Iq','Vd','Vq'])
df0['u'] = pd.Series([np.random.uniform(-0.01,-0.001) for x in range(len(df0.index))])

df1 = pd.DataFrame(safe1, columns=['Tm','Wpu','Vcon','Efd','PsiD','PsiQ','m','L','Id','Iq','Vd','Vq'])
df1['u'] = pd.Series([np.random.uniform(-0.001,0) for x in range(len(df1.index))])
df2 = pd.DataFrame(safe2, columns=['Tm','Wpu','Vcon','Efd','PsiD','PsiQ','m','L','Id','Iq','Vd','Vq'])
df2['u'] = pd.Series([np.random.uniform(-0.001,0) for x in range(len(df2.index))])
df3 = pd.DataFrame(safe3, columns=['Tm','Wpu','Vcon','Efd','PsiD','PsiQ','m','L','Id','Iq','Vd','Vq'])
df3['u'] = pd.Series([np.random.uniform(-0.001,0) for x in range(len(df3.index))])
df4 = pd.DataFrame(safe4, columns=['Tm','Wpu','Vcon','Efd','PsiD','PsiQ','m','L','Id','Iq','Vd','Vq'])
df4['u'] = pd.Series([np.random.uniform(-0.001,0) for x in range(len(df4.index))])

df5 = pd.DataFrame(unsafe1, columns=['Tm','Wpu','Vcon','Efd','PsiD','PsiQ','m','L','Id','Iq','Vd','Vq'])
df5['u'] = pd.Series([np.random.uniform(0.0001,0.01) for x in range(len(df5.index))])
df6 = pd.DataFrame(unsafe2, columns=['Tm','Wpu','Vcon','Efd','PsiD','PsiQ','m','L','Id','Iq','Vd','Vq'])
df6['u'] = pd.Series([np.random.uniform(0.0001,0.01) for x in range(len(df6.index))])
df7 = pd.DataFrame(unsafe3, columns=['Tm','Wpu','Vcon','Efd','PsiD','PsiQ','m','L','Id','Iq','Vd','Vq'])
df7['u'] = pd.Series([np.random.uniform(0.0001,0.01) for x in range(len(df7.index))])
df8 = pd.DataFrame(unsafe4, columns=['Tm','Wpu','Vcon','Efd','PsiD','PsiQ','m','L','Id','Iq','Vd','Vq'])
df8['u'] = pd.Series([np.random.uniform(0.0001,0.01) for x in range(len(df8.index))])

frames = [df0, df1, df2, df3, df4, df5, df6, df7, df8]
df = pd.concat(frames, ignore_index=True)
print(len(df))
# df.to_csv('data.csv')

# adjust for NN usage
input = ['Tm','Wpu','Vcon','Efd','PsiD','PsiQ','m','L','Id','Iq','Vd','Vq']
output = ['u']
x = df[input]
y = df[output]
X_train = x.values
Y_train = y.values
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.01)

# DNN training
model = Sequential()
n_cols = X_train.shape[1]

# hidden layers
model.add(Dense(20, activation='relu', input_shape=(n_cols,)))
model.add(Dense(20, activation='relu'))

# output layers
model.add(Dense(1))

# optimization
adam = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False)

model.compile(optimizer='adam', loss='mean_squared_error', metrics =['accuracy'])
model.summary()

# training
history = model.fit(X_train, Y_train, batch_size=1000, epochs=2000, verbose=2)

# training error visualization
# plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Error')
plt.xlabel('Epoch')
# plt.legend(['val_loss','train_loss'])
plt.show()

#saving NN model
model.save("Results/candidateBaC_DG_isld.h5")
# model.save_weights("Results/candidateBC_weights.h5")