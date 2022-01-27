import numpy as np
import pandas as pd
from keras.models import load_model

model = load_model('Results/candidateBC.h5')

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

for i in range(100):
    data = []
    state, dataset = D2()
    data.append(state)
    df4 = pd.DataFrame(data, columns=['i_d', 'i_q', 'i_od', 'i_oq', 'v_od', 'v_oq', 'i_ld', 'i_lq', 'm_d', 'm_q'])
    input = ['i_d', 'i_q', 'i_od', 'i_oq', 'v_od', 'v_oq', 'i_ld', 'i_lq', 'm_d', 'm_q']
    z_test = df4[input]
    Z_test = z_test.values
    action = model.predict(Z_test)
    u = float(action[0])
    if dataset == 1:
        if u > 0:
            R = True
        else:
            R = False
    if dataset == 2:
        if u <= 0:
            R = True
        else:
            R = False
    if dataset == 3:
        if u <= 0 and u >= -0.001:
            R = True
        else:
            R = False
    print(R)
