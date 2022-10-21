# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 10:00:15 2022

@author: reghu
"""

import numpy as np
from casadi import *
import pdb
import sys
sys.path.append('../../')
import do_mpc
from casadi import *
import CoolProp.CoolProp as CP
import pandas as pd

from MLPfunc import createMultiLayerPerceptron

w_val = pd.read_csv('p_list scaled.csv')
actfun="tanh"
weight_matrix = w_val['Value'].values

mlp = createMultiLayerPerceptron(3,1,[8,8,8],actfun)

model_type = 'continuous'
model = do_mpc.model.Model(model_type, symvar_type='SX')

nD_Evap = 30

lengthEvap = 2.3 # [m]
dz_E = lengthEvap/nD_Evap # [m]
gammaP = 0.89
dr_ES = 0.0036
r_ES = 0.0212 - dr_ES
vel_param = 0.89;
ratio_WN = 3    
A_z_E = r_ES**2 * np.pi * gammaP
A_z_EP= r_ES**2 * np.pi * (1-gammaP)
A_r_E = r_ES * 2 * np.pi * dz_E
A_z_W = ((r_ES + dr_ES)**2 - r_ES**2) * np.pi
# fluid properties
P = 30*1e5 # Pa 30bar

rhoW = 7.87e6
rhoP = rhoW
ktune = 2
#approximated first order time constant
Tc  = 10*60


rhoL = 1.099282151*CP.PropsSI("D", "Q", 0, "P", P, "Water") * 1000  # g/m³
rhoV = 1.093333333*CP.PropsSI("D", "Q", 1, "P", P, "Water") * 1000  # g/m³

# Saturation temperature from aspen is 202.5 deg C
T_sat = 0.86*CP.PropsSI("T", "Q", 0, "P", P, "Water")-273.15 # °C

hL = CP.PropsSI("H", "Q", 0, "P", P, "Water") / 1000  # j/g
hV = 1.433098337*CP.PropsSI("H", "Q", 1, "P", P, "Water") / 1000  # j/g
hLV = hV - hL
S_L = (rhoL / rhoV)

#k1 = model.set_variable('parameter', 'k1')
k1 = 2
# can tune this overall HT coeff params    
U_L  = 3000
U_V  = 1000

cp_W = 0.5
cp_P = cp_W
S_V  = 1000
D_W  = 15
T_env = 20
kEvap =k1*2#no parameters to be varied for Rev0


# create state variables
a_E  = model.set_variable(var_type='_x', var_name='a_E', shape=(nD_Evap, 1))
T_FE = model.set_variable(var_type='_x', var_name='T_FE', shape=(nD_Evap, 1))
T_WE = model.set_variable(var_type='_x', var_name='T_WE', shape=(nD_Evap, 1))
T_PE = model.set_variable(var_type='_x', var_name='T_PE', shape=(nD_Evap, 1))
V_pseudo = model.set_variable(var_type='_x', var_name='V_pseudo', shape=(1, 1))
# create input variables
Qdot_E = model.set_variable(var_type='_u', var_name='Qdot_E', shape=(1, 1))
mdot_W = model.set_variable(var_type='_u', var_name='mdot_W', shape=(1, 1))
mdot_N = model.set_variable(var_type='_u', var_name='mdot_N', shape=(1, 1))


# inlet conditions
a_In = 1
T_In = 60
# some calculations
v_L = (mdot_N + mdot_W) * 1000 / 3600 / rhoL / A_z_E
WbyN = mdot_W/mdot_N 
# create RHS
#some helper functions
x  = SX.sym('x')
aE = SX.sym('aE')
TF = SX.sym('TF')
r_s = 0.1

#softplus = Function('soft', [x], [(x + sqrt(x**2 + 0.01))/2])
softplus = Function('relu', [x], [fmax(x,0)])
mDotE = Function('mDotE', [aE, TF], [softplus(r_s * rhoL * aE * (TF-T_sat)/T_sat)])
mDotC = Function('mDotC', [aE, TF], [softplus(r_s * rhoV * (1-aE) * (T_sat-TF)/T_sat)])

v_shift = 0.0;
    
#initial input of water by naphtha, liq fraction and temperature
xinit = vertcat(WbyN,aE,TF)

U_F   = Function('U_F', [aE], [aE * (U_L - U_V) + U_V])
cp_F  = Function('cp_F',[aE,TF],[mlp["fun"](xinit,weight_matrix)])
rho_F = Function('U_F', [aE], [aE * (rhoL - rhoV) + rhoV])

# Evap
da_E  = SX.sym('da_E',  nD_Evap, 1)
dT_FE = SX.sym('dT_FE', nD_Evap, 1)
dT_PE = SX.sym('dT_PE', nD_Evap, 1)
dT_WE = SX.sym('dT_WE', nD_Evap, 1)
dV_pseudo = SX.sym('dV_pseudo',1,1)

dV_pseudo = (v_L-V_pseudo)/Tc
model.set_rhs("V_pseudo",dV_pseudo)
#update the vF based on V_pseudo
v_F   = Function('v_F',[aE], [((V_pseudo/1.92)*334.63*aE*aE - (V_pseudo/1.92)*657.25*aE+(V_pseudo/1.92)*328.628)])


# get correct heating power
Qdot_E = (-0.318 * Qdot_E**2 + 64.163 * Qdot_E + 275.54)

# get the heated zone in the evaporator
stoptHeatingEv1 = np.float(3*nD_Evap/10)
startHeatingEv1  = np.float(1*nD_Evap/15)

stoptHeatingEv2 = np.float(5*nD_Evap/6)
startHeatingEv2  = np.float(17*nD_Evap/30)

LengthHeat = (stoptHeatingEv1 -startHeatingEv1)*dz_E


#############################model equations

# define pdes
for i in range(0, nD_Evap):
    if i != 0:
        # get actual heating
        Qdot_E_i = casadi.if_else((i <= stoptHeatingEv1 and i >= startHeatingEv1) or (i <= stoptHeatingEv2 and i >= startHeatingEv2), Qdot_E/LengthHeat*dz_E, 0)
        # volume fraction
        da_E[i] = -V_pseudo * (a_E[i]-a_E[i-1])/dz_E + (mDotC(a_E[i], T_FE[i]) - mDotE(a_E[i], T_FE[i]))/rhoL
        # fluid temperature
        cpF = cp_F(a_E[i],T_FE[i])
        #cpF = cp_F(a_E[i])
        vF  = v_F(a_E[i])
        rhoF = rho_F(a_E[i])
        
        
        
        dT_FE[i] = -vF * (T_FE[i] - T_FE[i-1])/dz_E + A_r_E * U_F(a_E[i]) * (T_WE[i]-T_FE[i]) / (cpF * rhoF * A_z_E * dz_E) \
            - (mDotE(a_E[i], T_FE[i]) - mDotC(a_E[i], T_FE[i])) * hLV / (cpF * rhoF) \
            - S_V * U_F(a_E[i]) * (T_FE[i] - T_PE[i])/ (cpF * rhoF * (gammaP))
        
           
    # packing temperature
    dT_PE[i] = S_V * U_F(a_E[i]) * (T_FE[i] - T_PE[i]) / (cp_P * rhoP * (1-gammaP))
    # wall temperature
    if i != nD_Evap-1 and i != 0:
        dT_WE[i] = Qdot_E_i / (cp_W * rhoW * A_z_W * dz_E) - kEvap * (T_WE[i] - T_env) / (rhoW * cp_W * A_z_W)\
            - A_r_E * U_F(a_E[i]) * (T_WE[i]-T_FE[i]) / (cp_W * rhoW * A_z_W * dz_E) \
            + D_W / (rhoW * cp_W) * (T_WE[i+1] - 2*T_WE[i] + T_WE[i-1])/dz_E**2

     
    
# define boundaries
da_E[0] = -V_pseudo * (a_E[0] - a_In)/dz_E + (mDotC(a_E[0], T_FE[0]) - mDotE(a_E[0], T_FE[0]))/rhoL

dT_FE[0] = -v_F(a_E[0]) * (T_FE[0] - T_In)/dz_E + A_r_E * U_F(a_E[0]) * (T_WE[0]-T_FE[0]) / (cp_F(a_E[0],T_FE[0]) * rho_F(a_E[0]) * A_z_E * dz_E) \
            - (mDotE(a_E[0], T_FE[0]) - mDotC(a_E[0], T_FE[0])) * hLV / (cp_F(a_E[0],T_FE[0]) * rho_F(a_E[0])) \
            - S_V * U_F(a_E[0]) * (T_FE[0] - T_PE[0])/ (cp_F(a_E[0],T_FE[0]) * rho_F(a_E[0]))


dT_WE[0] = 2 * D_W / (rhoW * cp_W) * (T_WE[1]-T_WE[0])/dz_E**2 \
       - A_r_E * U_F(a_E[0]) * (T_WE[0] - T_FE[0]) / (cp_W * rhoW * A_z_W * dz_E)
dT_WE[-1] = 2 * D_W / (rhoW * cp_W) * (T_WE[-2] - T_WE[-1]) / dz_E**2  \
       - A_r_E * U_F(a_E[-1]) * (T_WE[-1] - T_FE[-1]) / (cp_W * rhoW * A_z_W * dz_E)
      
model.set_rhs("a_E",  da_E,process_noise = False)
model.set_rhs("T_FE", dT_FE,process_noise = False)
model.set_rhs("T_WE", dT_WE,process_noise = False)
model.set_rhs("T_PE", dT_PE,process_noise = False)

##########################################################state measurements

TFE_measin = model.set_meas('TFE_measin', T_FE[4],meas_noise=False)# assume noise free
TFE_measout = model.set_meas('TFE_measout', T_FE[28],meas_noise=False)# assume noise free

# Input measurements
QdotE_meas = model.set_meas('QdotE_meas', Qdot_E, meas_noise=False)


# return the complete model
model.setup()

#########################################mhe setup

mhe = do_mpc.estimator.MHE(model,[])

setup_mhe = {
    't_step': 0.5,
    'n_horizon': 2,
    'store_full_solution': True,
    'meas_from_data': True
}
mhe.set_param(**setup_mhe)

P_v = np.eye(0)#dimensions same as the 
#number of measured states that are affected by noise
P_x = np.eye(121)#dimension same as number of states
P_p = np.eye(0)#we arent performing parameter estimation yet

mhe.set_default_objective(P_x,P_p=P_p,P_v=P_v)

p_template_mhe = mhe.get_p_template()
def p_fun_mhe(t_now):
    #p_template_mhe['k1'] = 2.0
    return p_template_mhe

mhe.set_p_fun(p_fun_mhe)

#mhe.bounds['lower','_x', 'T_FE'] = 120
#mhe.bounds['upper','_x', 'T_FE'] = 500

mhe.setup()

#######################################simulator

simulator = do_mpc.simulator.Simulator(model)

simulator.set_param(t_step = 0.5)

p_template_sim = simulator.get_p_template()

def p_fun_sim(t_now):
    #p_template_sim['k1'] = 1.5
    return p_template_sim

simulator.set_p_fun(p_fun_sim)

simulator.setup()

##############################testing with falsified data

dataset = pd.read_excel("Dummy Data.xlsx")


dataset = dataset[dataset["time_min"] > 180]
firstIndex = dataset.index[0]

nD_E = 30

vd_init = 0.0001*np.ones([1,1])

x0 =  vertcat(np.ones([nD_E, 1]),60*np.ones([nD_E,1]),300*np.ones([2*nD_E,1]),vd_init)

simulator.x0 = x0
u0 = vertcat(37.6, 2.55, 0.9)


for i in range(300):
    simulator.make_step(u0)

x0 = simulator.x0
# reset simulator and set the initial value
simulator.reset_history()


x0_mhe = x0
mhe.x0_mhe = x0_mhe
mhe.set_initial_guess()


def random_u(u0):#to generate random changes in the MV for heating step 
#other two inputs of flow have not been changed

    # Hold the current value with 80% chance or switch to new random value.
    u_next = np.random.rand() # New candidate value.
    switch = np.random.rand() >= 0.8 # switching? 0 or 1.
    u0[0] = (1-switch)*u0[0] + switch*u_next # Old or new value.
    return u0

for i in range(500):
    #u0 = random_u(u0)
    y0 = simulator.make_step(u0)
    x0 = mhe.make_step(y0) # MHE estimation step


simvalues = simulator.data._x
estvalues = mhe.data._x
############################plotting

import matplotlib.pyplot as plt
import matplotlib as mpl
# Customizing Matplotlib:
mpl.rcParams['font.size'] = 18
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['axes.grid'] = True

mhe_graphics = do_mpc.graphics.Graphics(mhe.data)
sim_graphics = do_mpc.graphics.Graphics(simulator.data)

fig, ax = plt.subplots(3, sharex=True, figsize=(16,9))
fig.align_ylabels()

for g in [sim_graphics, mhe_graphics]:
    # Plot the temperatures on the first axis:
    g.add_line(var_type='_x', var_name='T_FE', axis=ax[0])
    ax[0].set_prop_cycle(None)
    g.add_line(var_type='_x', var_name='T_WE', axis=ax[1])
    ax[1].set_prop_cycle(None)

    # Plot the input on the second axis:
    g.add_line(var_type='_u', var_name='Qdot_E', axis=ax[2])
    ax[2].set_prop_cycle(None)



