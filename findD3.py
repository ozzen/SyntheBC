import numpy as np
import pandas as pd
from keras.models import load_model

model = load_model('Results/candidateBC.h5')

s1 = [6.89628771e-01, -3.61680311e-06 , 6.87630758e-01, 3.89178706e-06, 4.94407329e-01, 7.46948555e-03, 6.85391545e-01, 0.00000000e+00, 3.83142319e-01, 5.38107183e-02]
s2 = [6.80844986e-01, -1.50422015e-06, 6.89671321e-01, -1.60272760e-06, 4.94113030e-01, 7.29961465e-03, 6.85496195e-01, 0.00000000e+00, 3.84238016e-01, 5.47936746e-02]

def dataset():
    i_d = np.random.uniform(low=s1[0], high=s2[0])
    i_q = np.random.uniform(low=s1[1], high=s2[1])
    i_od = np.random.uniform(low=s1[2], high=s2[2])
    i_oq = np.random.uniform(low=s1[3], high=s2[3])
    v_od = np.random.uniform(low=s2[4], high=0.48+0.0144-0.003)
    v_oq = np.random.uniform(low=s1[5], high=s2[5])
    i_ld = np.random.uniform(low=s1[6], high=s2[6])
    i_lq = np.random.uniform(low=s1[7], high=s2[7])
    m_d = np.random.uniform(low=s1[8], high=s2[8])
    m_q = np.random.uniform(low=s1[9], high=s2[9])

    state = np.array([i_d, i_q, i_od, i_oq, v_od, v_oq, i_ld, i_lq, m_d, m_q])
    return state

for i in range(100):
    data = []
    state = dataset()
    data.append(state)
    df4 = pd.DataFrame(data, columns=['i_d', 'i_q', 'i_od', 'i_oq', 'v_od', 'v_oq', 'i_ld', 'i_lq', 'm_d', 'm_q'])
    input = ['i_d', 'i_q', 'i_od', 'i_oq', 'v_od', 'v_oq', 'i_ld', 'i_lq', 'm_d', 'm_q']
    z_test = df4[input]
    Z_test = z_test.values
    action = model.predict(Z_test)
    u = float(action[0])
    print(u)
    if u <= 0 and u >= -0.001:
        R = True
    else:
        R = False
    print(R)