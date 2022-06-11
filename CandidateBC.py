import numpy as np
import gurobipy as gp
from keras.models import load_model

model = load_model('Results/2l/candidateBC_10n_2l.h5')

# load weights and biases
W1 = model.layers[0].get_weights()[0]
b1 = model.layers[0].get_weights()[1]
W2 = model.layers[1].get_weights()[0]
b2 = model.layers[1].get_weights()[1]
W3 = model.layers[2].get_weights()[0]
b3 = model.layers[2].get_weights()[1]
W3_temp = np.transpose(W3)

# admissible regions
# lb = [-1,-1,-1,-1,0.4656,-1,-1,-1,-1,-1]
# ub = [1,1,1,1,0.4944,1,1,1,1,1]

lb = [0.646,-1e-3,0.646,-1e-3,0.4656,-1e-3,0.646,-1e-3,0.3,-1e-3]
ub = [0.714,1e-3,0.714,1e-3,0.4944,1e-3,0.714,1e-3,0.5,1e-3]

# lb = [0.646,-1e-3,0.646,-0.304940,0.4656,-1e-3,0.646,-1e-3,0.3,-1e-3]
# ub = [0.714,1e-3,0.714,0.304940,0.4944,1e-3,0.714,1e-3,0.5,1e-3]
#
# lb = [0.646,-1e-3,0.646,-1e-3,0.4656,-0.250513,0.646,-1e-3,0.3,-1e-3]
# ub = [0.714,1e-3,0.714,1e-3,0.4944,0.250513,0.714,1e-3,0.5,1e-3]

# lb = [0.646,-1e-3,0.646,-1e-3,0.4656,-1e-3,0.646,-0.792198,0.3,-1e-3]
# ub = [0.714,1e-3,0.714,1e-3,0.4944,1e-3,0.714,0.792198,0.5,1e-3]

# big-M bounds
lk = -1e9
uk = 1e9

m = gp.Model('safe')
n1 = W1.shape[0]
n2 = W2.shape[0]

# z_val_constr = []
# relu_constr = []
# zlist = []
# xlist = []
# tlist = []

# for k in range(len(model.layers) - 1):
#     W = model.layers[k].get_weights()[0]
#     b = model.layers[k].get_weights()[1]
#     inp, op = W.shape
#     z = m.addMVar(op, name='z{0}'.format(k + 1))
#     t = m.addMVar(op, vtype=gp.GRB.BINARY, name='t{0}'.format(k + 1))
#     xtmp = m.addMVar(op, vtype=gp.GRB.CONTINUOUS, name='x{0}'.format(k + 1))
#     print(inp, op)
#
#     for i in range(op):
#         z_val_constr.append(m.addConstr(z[i] == gp.quicksum([gp.quicksum([W[j][i] * xlist[-1][j] for j in range(inp)]), b[i]]), 'z{0}val_constr{1}'.format(k + 1, i)))
#
#     relu_constr.append(m.addConstr(xtmp - z + lk <= lk * t, 'relu_cons_{0}a'.format(k + 1)))
#     relu_constr.append(m.addConstr(z - xtmp <= 0, 'relu_cons_{0}b'.format(k + 1)))
#     relu_constr.append(m.addConstr(xtmp - uk * t <= 0, 'relu_cons_{0}c'.format(k + 1)))
#     relu_constr.append(m.addConstr(xtmp >= 0, 'relu_cons_{0}d'.format(k + 1)))
#
#     zlist.append(z)
#     xlist.append(xtmp)
#     tlist.append(t)

x = m.addMVar(n1,name='state',vtype=gp.GRB.CONTINUOUS,lb=lb,ub=ub)

# enforcing  safe/unsafe region constraints
safe_region_constr_ub = m.addConstr(x[4] >= 0.4656, "saferegion1")
safe_region_constr_lb = m.addConstr(x[4] <= 0.4944, "saferegion2")

#Unsafe Region #1
# unsafe_region_constr_ub=m.addConstr(x[4] >= 0.4944,"unsaferegion1")
# unsafe_region_constr_lb=m.addConstr(x[4] <= 0.60,"unsaferegion2")

#Unsafe Region#2
# unsafe_region_constr_ub=m.addConstr(x[4] >= 0.36,"unsaferegion1")
# unsafe_region_constr_lb=m.addConstr(x[4] <= 0.4656,"unsaferegion2")

## Relu constraints Unrolled for each hidden layer

# 1st hidden layer temporary variable
Z1 = m.addMVar(n2,vtype=gp.GRB.CONTINUOUS,name='z1',lb=-gp.GRB.INFINITY,ub=gp.GRB.INFINITY)

# Z1 = W1*x + b1
z_value_constr = m.addConstr(Z1==np.transpose(W1)@x+b1,'Z1Value')

# binary indicator variable
t1 = m.addMVar(n2,vtype=gp.GRB.BINARY, name="t1")

# output of hidden layer 1 i.e. x1 = ReLU(Z1)
x1 = m.addMVar(n2,vtype=gp.GRB.CONTINUOUS,name='x1')

# ReLU constraints same as defined in SynthBC paper
relu_constr_1a = m.addConstr(x1-Z1+lk <= lk*t1,'relu_cons_1a')
relu_constr_1b = m.addConstr(Z1-x1 <= 0,'relu_cons_1b')
relu_constr_1c = m.addConstr(x1-uk*t1 <= 0,'relu_cons_1c')
relu_constr_1d = m.addConstr(x1 >= 0,'relu_cons_1d')

# 2nd hidden layer temporary variables
Z2 = m.addMVar(n2,vtype=gp.GRB.CONTINUOUS,name='z2',lb=-gp.GRB.INFINITY,ub=gp.GRB.INFINITY)

# binary indicator variables
t2 = m.addMVar(n2,vtype=gp.GRB.BINARY, name="t2")

# Z2 = W2*x1 + b2
z_value_constr2 = m.addConstr(Z2==np.transpose(W2)@x1+b2,'Z2Value')

# output of hidden layer 2 i.e. x2 = ReLU(Z2)
x2 = m.addMVar(n2,vtype=gp.GRB.CONTINUOUS,name='x2')

# ReLU constraints
relu_constr_2a=m.addConstr(x2-Z2+lk <= lk*t2,'relu_cons_2a')
relu_constr_2b=m.addConstr(Z2-x2 <= 0,'relu_cons_2b')
relu_constr_2c=m.addConstr(x2-uk*t2 <= 0,'relu_cons_2c')
relu_constr_2d=m.addConstr(x2 >= 0,'relu_cons_2d')

# output layer: X3 = W3*x2 + b3
x3 = m.addMVar(1,vtype=gp.GRB.CONTINUOUS,name='x3',lb=-gp.GRB.INFINITY,ub=gp.GRB.INFINITY)
x3_constraint = m.addConstr(x3==W3_temp@x2+b3,'x3_val')
# W_op = model.layers[-1].get_weights()[0]
# b_op = model.layers[-1].get_weights()[1]
# x3_constraint = m.addConstr(x3==np.transpose(W_op) @ xlist[-1] + b_op,'x3_val')

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

# derivative Condition constraints
v = m.addMVar(1,name='V')
x3_constraint = m.addConstr(W3_temp@x2+b3==0,'x3_val')
v_constr = m.addConstr(v==np.multiply(W3,np.multiply(t2@W2,t1@W1)),'v_val')

# set objective for Gurobi:
# for initial states x3 should be -ve. Obj: p* = max(x3). if p* < 0 then BaC satisifies initial states condition
# for unsafe states x3 should be +ve. Obj: p* = min(x3). if p* > 0 then BaC satisfies unsafe region condition
# m.setObjective(x3, gp.GRB.MAXIMIZE)
# m.setObjective(x3,gp.GRB.MINIMIZE)
m.setObjective(v@f(x),gp.GRB.MAXIMIZE)

# save the model
m.write("grb.lp")

# call optimizer
m.optimize()
