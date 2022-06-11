import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model

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
v_lb = 0.4656
v_ub = 0.4944
v_fsc = 0.03*v_ref
v_safe1 = 0.4660
v_safe2 = 0.4940

# generates initial configuration
def control(flag):
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
    return state

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

# for i in range(1,1):
state = [0.646, 0.001, 0.646, 0.001, 0.4941, 0.001, 0.646, 0.001, 0.6, 0.001]
func = f(state)
print("f: ",func)

state = np.array([0.646, 0.001, 0.646, 0.001, 0.4941, 0.001, 0.646, 0.001, 0.6, 0.001])
state = [state]
df = pd.DataFrame(state, columns=['i_d', 'i_q', 'i_od', 'i_oq', 'v_od', 'v_oq', 'i_ld', 'i_lq', 'm_d', 'm_q'])
input = ['i_d', 'i_q', 'i_od', 'i_oq', 'v_od', 'v_oq', 'i_ld', 'i_lq', 'm_d', 'm_q']
x = df[input]
x_test = x.values
print("x_test: ",x_test)

model = load_model('Results/candidateBC.h5')
extractor = keras.Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
features = extractor(x_test)
print(features)

w1 = model.layers[0].get_weights()[0]
w2 = model.layers[1].get_weights()[0]
w3 = model.layers[2].get_weights()[0]
W1 = np.transpose(w1)
W2 = np.transpose(w2)
print("W1: ",w1,"\n W2: ",w2,"\n W3: ",w3)
# deriv = W1@Z1
# print(W)
# print(W1,Z1)
