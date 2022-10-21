# -*- coding: utf-8 -*-
"""
Created on Sat May 14 17:20:49 2022

@author: reghu
"""
#changes made -latest change on 11 Sep

# import some stuff
from casadi.tools import *
import matplotlib.pyplot as plt
import pandas as pd
from template_model import template_model
from template_simulator import template_simulator

q_tune = 1;
ktune = 2;
param = 0.86;
# initialize model and simulator object
model = template_model(ktune,param)
simulator = template_simulator(model, 30.0)#change the value for changing scan freq

# load and cut data set
dataset = pd.read_excel("preprocessed_set1.xlsx")


dataset = dataset[dataset["time_min"] > 180]
firstIndex = dataset.index[0]


## simulate the model
nD_E = 30


# get some reasonable initial states and inputs

vd_init = 0.0001*np.ones([1,1])

x0 =  vertcat(np.ones([nD_E, 1]),60*np.ones([nD_E,1]),300*np.ones([2*nD_E,1]),vd_init)
              

u0 = vertcat(37.6, 2.55, 0.9)
# get correct initial values by simulating into steady state
simulator.x0 = x0
for i in range(300):
    simulator.make_step(u0)
x0 = simulator.x0
# reset simulator and set the initial value
simulator.reset_history()
simulator.x0 = x0

# simulate the model according to the data set
for i in dataset.index:
    # TC03 MV for heating in the superheater
    u0 = vertcat(dataset["TC01.MV"][i], dataset["WI01.PV"][i], dataset["FIC14.PV"][i])
    simulator.make_step(u0)

## plot everything
fig, ax = plt.subplots(3, sharex=True)

# plot data
dataset.plot(x="time_min", y="TI04.PV", ax=ax[0])
dataset.plot(x="time_min", y="TI21.PV", ax=ax[1])
#dataset.plot(x="time_min", y="TC01.PV", ax=ax[1])
dataset.plot(x="time_min", y="TC01.MV", ax=ax[2])

# get sensor locations
TI04 = nD_E
TC01 = 2*nD_E-2
TI21 = 2*nD_E-1

# plot inlet temperature
values = simulator.data._x
ax[0].plot(dataset["time_min"], values[:, TI04], '--')
ax[0].set_ylim([50, 80])
ax[0].set_ylabel("T [°C]")
ax[0].grid(True)
ax[0].set_title("Evaporator, (solid:data, dashed:simulation)")

# plot outlet and wall temperature
#ax[1].plot(dataset["time_min"], values[:, TC01], '--')
ax[1].plot(dataset["time_min"], values[:, TI21], "--")
#ax[1].plot(dataset["time_min"], values[:, TI21]-values[:, 120], "--")
ax[1].set_ylim([250, 400])
ax[1].set_ylabel("T [°C]")
ax[1].grid(True)

# plot input value
ax[2].set_ylim([35, 40])
ax[2].set_ylabel("MV [%]")
ax[2].grid(True)
ax[2].set_xlabel("t [min]")

# show plots
plt.show()
