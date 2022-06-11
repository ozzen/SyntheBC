import numpy as np
import pandas as pd
from keras.models import load_model

model = load_model('Results/candidateBC.h5')

# rtds constraints
i_d_ref = 0.68
v_ref = 0.48
high_d = i_d_ref + 0.05*i_d_ref
low_d = i_d_ref - 0.05*i_d_ref
high_q = 1e-3
low_q = -1e-3
high_md = 0.40
low_md = 0.30
high_mq = 0.055
low_mq = 0.0
v_unsafe1 = 0.36
v_unsafe2 = 0.60
v_fsc = 0.03*v_ref
v_safe = 0.005*v_ref

# generates initial configuration
def D1(flag):
    i_d = np.random.uniform(low=low_d, high=high_d)
    i_q = np.random.uniform(low=low_q, high=high_q)
    i_od = np.random.uniform(low=low_d, high=high_d)
    i_oq = np.random.uniform(low=low_q, high=high_q)
    if flag % 2 == 0:
        v_od = np.random.uniform(low=v_unsafe1, high=v_ref-v_fsc)
    else:
        v_od = np.random.uniform(low=v_ref+v_fsc, high=v_unsafe2)
    v_oq = np.random.uniform(low=low_q, high=high_q)
    i_ld = np.random.uniform(low=low_d, high=high_d)
    i_lq = np.random.uniform(low=low_q, high=high_q)
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
    v_od = np.random.uniform(low=v_ref-v_fsc+v_safe, high=v_ref+v_fsc-v_safe)
    v_oq = np.random.uniform(low=low_q, high=high_q)
    i_ld = np.random.uniform(low=low_d, high=high_d)
    i_lq = np.random.uniform(low=low_q, high=high_q)
    m_d = np.random.uniform(low=low_md, high=high_md)
    m_q = np.random.uniform(low=low_mq, high=high_mq)

    state = np.array([i_d, i_q, i_od, i_oq, v_od, v_oq, i_ld, i_lq, m_d, m_q])
    dataset = 2
    return state, dataset

def D3(flag):
    i_d = np.random.uniform(low=low_d, high=high_d)
    i_q = np.random.uniform(low=low_q, high=high_q)
    i_od = np.random.uniform(low=low_d, high=high_d)
    i_oq = np.random.uniform(low=low_q, high=high_q)
    if flag % 2 == 0:
        v_od = np.random.uniform(low=v_ref-v_fsc, high=v_ref-v_fsc+v_safe)
    else:
        v_od = np.random.uniform(low=v_ref+v_fsc-v_safe, high=v_ref+v_fsc)
    v_oq = np.random.uniform(low=low_q, high=high_q)
    i_ld = np.random.uniform(low=low_d, high=high_d)
    i_lq = np.random.uniform(low=low_q, high=high_q)
    m_d = np.random.uniform(low=low_md, high=high_md)
    m_q = np.random.uniform(low=low_mq, high=high_mq)

    state = np.array([i_d, i_q, i_od, i_oq, v_od, v_oq, i_ld, i_lq, m_d, m_q])
    dataset = 3
    return state, dataset

# U = -1
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
    print(u)
    # if u > U and u < 0:
    #     print(state)
    #     U = u
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
        if u <= 0: #and u >= -0.001:
            R = True
        else:
            R = False
    print(R)
