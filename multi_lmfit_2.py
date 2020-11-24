# -*- coding: utf-8 -*-
"""
Created on Mon May 11 12:51:16 2020

@author: srikanth nayak
"""

import numpy as np
from lmfit import Minimizer, Parameters, report_fit
from matplotlib import pyplot as plt
import pandas as pd
# import pdb

def chi(w_ir, w1,a1,t1,w2,a2,t2,a_nr):
    
    phi_nr = np.pi    
    chi_eff = np.divide(a1,(w_ir - w1 - 1j * t1)) + np.divide(a2,(w_ir - w2 - 1j * t2)) + a_nr * np.exp(phi_nr *1j)
    
    return chi_eff

# define objective function: returns the array to be minimized
def fcn2min(p, data):
    w1 = p['w1']
    t1 = p['t1']
    w2 = p['w2']
    t2 = p['t2']
    a_nr = p['a_nr']
    w_ir = data['w_ir']
    resid =[]
    for i in range(len(data)-1):
        chi_eff = chi(w_ir, w1, p['a1_'+str(i+1)], t1, w2, p['a2_'+str(i+1)], t2, a_nr)
        chi_eff_sq = (np.absolute(chi_eff))**2
        # pdb.set_trace()
        resid += [a-b for a, b in zip (chi_eff_sq , data['I'+str(i+1)] )]
    return resid
                                   
  
# get the data to be fitted
#######################################
#######################################
#Add filenames here
filenames=['5000um_corrected.csv', '5000um_corrected.csv']
figname = 'mlm_test.jpeg'
#######################################
#######################################

#Initializing variables and parameters
fit_range=[3400,3800]
data1 = pd.read_csv(filenames[0], names = ['w_ir', 'I','a','b','c'], skiprows = 1)
xdata=data1.w_ir[np.logical_and(data1.w_ir>fit_range[0],data1.w_ir<fit_range[1] )]
data = {'w_ir':xdata}

p = Parameters()
p.add('w1', value = 3550, min = 3540, max = 3680)
p.add('t1', value = 30, min = 1, max = 1000)
p.add('w2', value = 3600, min = 3590, max = 3630)
p.add('t2', value = 10, min = 1, max = 500)
p.add('a_nr', value = 0.5, min = -10, max  = 100)

for i,value in enumerate(filenames,1):
    dummy = pd.read_csv(value, names = ['w_ir', 'I','a','b','c'], skiprows = 1)
    data['I'+str(i)] = dummy.I[np.logical_and(data1.w_ir>fit_range[0],data1.w_ir<fit_range[1] )]
    p.add('a1_'+ str(i),  value = -100, min = -1000, max = 0)
    p.add('a2_'+ str(i),  value = 0, min = -1000, max = 0, vary = False)
    

# do fit, here with the default leastsq algorithm
minner = Minimizer(fcn2min, p, fcn_args = [data])
result = minner.minimize()

# write error report
report_fit(result)


w1f = result.params['w1'].value
t1f = result.params['t1'].value
w2f = result.params['w2'].value
t2f = result.params['t2'].value
a_nrf = result.params['a_nr'].value

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20
[fig,ax] = plt.subplots(1,2,figsize = [18,9])
marker_list = ['+', 'v', 's', 'o', '<', 'D', '>', '^']
color_list = ['k','m','g','c','r','b']

for i, name in enumerate(filenames,1): 
    k = len(xdata)
    final = data['I'+str(i)] + result.residual[(i-1)*k:i*k]
    a1f = result.params['a1_'+str(i)].value
    a2f = result.params['a2_'+str(i)].value
    imfit = np.imag(chi(xdata, w1f, a1f,t1f,w2f,a2f,t2f,a_nrf))
    ax[0].plot(xdata, data['I'+str(i)], color = color_list[i-1], marker = marker_list[i-1], linestyle = '', label = name[:-4]) 
    ax[0].plot(xdata, final, color = color_list[i-1], label = '_no_legend_', linewidth = 3)
    ax[1].plot(xdata,imfit, color = color_list[i-1], label = name[:-4] )
    
ax[0].legend(loc = 'upper right')
ax[0].set_xlabel('Wavenumber, cm$^{-1}$')
ax[0].set_ylabel('$I_{SFG}$, arb. units')
ax[1].legend(loc = 'upper right')
ax[1].set_xlabel('Wavenumber, cm$^{-1}$')
ax[1].set_title('Imaginary part')

# fig.savefig(figname, bbox_inches ='tight')
