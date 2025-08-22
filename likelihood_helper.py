# Copyright (c) 2024, Markus Gaug
# All rights reserved.
# 
# This source code is licensed under the GNU General Public License v3.0 found in the
# LICENSE file in the root directory of this source tree. 
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
import numpy as np
import pandas as pd
from scipy.optimize import fsolve, curve_fit
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from multiprocessing import Pool
from functools import partial

from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.tsa.seasonal import STL
#from plot_helper import mjd_corrector, mjd_start_2003, number_days_year
#from coverage_helper import expected_data_per_day

mjd_corrector  = 55000   # used to bring back all mjd to precision required with a float data member
mjd_start_2003 = 52640   # MJD of 1/1/2003
number_days_year = 365.2422 # average number of days per calendar year
number_of_CPUs_available = 45 # for pooling

def chi2_fmt(x):    # custom formatter for contour label of chi2 plot
    x = np.sqrt(x)
    return rf"{x:.0f}$\sigma$"


def quad_func(x, mu, sigma):
    return (x-mu)**2/sigma**2

def mu_Haslebacher(p,m,verb=False):
    '''
    Monthly expectation value as used in Haslebacher et al. (2022) for temperature

    Note the re-definition of 'b', which is now the temperature increase per decade

    Note that 'm' is not bound to be a monthly aveage, but can have finer time binning
    '''
    a     = p[0]
    # initialize b as increase per decade
    b     = p[1]/12/10.
    C     = p[2]
    phimu = p[3]
    omega = np.pi/6

    if (verb):
        #print (m)
        #print ('phimu:',phimu)
        print ('m-phimu:',m-phimu)
        print ('m-phimu shape:',(m-phimu).shape)
        print ('cos:',np.cos(m-phimu))
        
    return a + b * m + C*np.sin(omega*(m-phimu))
    
def mu(p,m,verb=False):
    '''
    Montly temperature expectation value 
    Modified version of Eq. (19) of Haslebacher et al. (2022)

    Note the re-definition of 'b', which is now the temperature increase per decade

    Note that 'm' is not bound to be a monthly aveage, but can have finer time binning,
    albeit it has units of months. 
    '''
    a     = p[0]
    # initialize b as increase per decade
    b     = p[1]/12/10.
    C     = p[2]
    phimu = p[3]
    omega = np.pi/6

    if (verb):
        #print (m)
        #print ('phimu:',phimu)
        print ('m-phimu:',m-phimu)
        print ('m-phimu shape:',(m-phimu).shape)
        print ('cos:',np.cos(m-phimu))
        

    # subtract the mean of the function (1+cos(x))^2 to not alter the
    # significance of a as mean temperature, see also
    # https://www.wolframalpha.com/input?i=calculate+the+mean+value+of+%281%2Bcos%28x%29%29%5E2+from+0+to+2pi
    return a + b * m + C*(0.5*((np.cos(omega*(m-phimu))+1)**2) - 0.75)
    
def mu_sq(p,m,verb=False):
    '''
    Montly temperature expectation value 
    Modified version of Eq. (19) of Haslebacher et al. (2022) by an additional quadratic term

    Note the re-definition of 'b', which is now the temperature increase per decade

    Note that 'm' is not bound to be a monthly aveage, but can have finer time binning,
    albeit it has units of months. 
    '''
    a     = p[0]
    # initialize b as increase per decade
    b     = p[1]/12/10.
    # initialize q as increase per decade^2    
    q     = p[2]/12/10./12/10.
    C     = p[3]
    phimu = p[4]
    omega = np.pi/6

    if (verb):
        #print (m)
        #print ('phimu:',phimu)
        print ('m-phimu:',m-phimu)
        print ('m-phimu shape:',(m-phimu).shape)
        print ('cos:',np.cos(m-phimu))
        

    # subtract the mean of the function (1+cos(x))^2 to not alter the
    # significance of a as mean temperature, see also
    # https://www.wolframalpha.com/input?i=calculate+the+mean+value+of+%281%2Bcos%28x%29%29%5E2+from+0+to+2pi
    return a + b * m + q * m**2 + C*(0.5*((np.cos(omega*(m-phimu))+1)**2) - 0.75)
    
def mu2(p,m,verb=False):
    '''
    Montly temperature expectation value 
    Modified version of Eq. (19) of Haslebacher et al. (2022)

    Note the re-definition of 'b', which is now the temperature increase per decade

    Note that 'm' is not bound to be a monthly aveage, but can have finer time binning,
    albeit it has units of months. 
    '''
    a     = p[0]
    # initialize b as increase per decade
    b     = p[1]/12/10.
    C     = p[2]
    phimu = p[3]
    omega = np.pi/6

    if (verb):
        #print (m)
        #print ('phimu:',phimu)
        print ('m-phimu:',m-phimu)
        print ('m-phimu shape:',(m-phimu).shape)
        print ('cos:',np.cos(m-phimu))
        

    # subtract the mean of the function (1+cos(x))^2 to not alter the
    # significance of a as mean temperature, see also
    # https://www.wolframalpha.com/input?i=calculate+the+mean+value+of+%281%2Bcos%28x%29%29%5E2+from+0+to+2pi
    return a + b * m + C*(0.5*((np.cos(omega*(m-phimu))+1)**2) - 0.75) + p[4]*np.sin(omega*(m-p[5]))
    
def mu_nob(p,m,verb=False):
    '''
    Montly temperature expectation value 
    Modified version of Eq. (19) of Haslebacher et al. (2022)

    Case of no temperature increase ('b' parameter)

    Note that 'm' is not bound to be a monthly aveage, but can have finer time binning,
    albeit it has units of months. 
    '''
    a     = p[0]
    C     = p[1]
    phimu = p[2]
    omega = np.pi/6

    if (verb):
        #print (m)
        #print ('phimu:',phimu)
        print ('m-phimu:',m-phimu)
        print ('m-phimu shape:',(m-phimu).shape)
        print ('cos:',np.cos(m-phimu))
        

    # subtract the mean of the function (1+cos(x))^2 to not alter the
    # significance of a as mean temperature, see also
    # https://www.wolframalpha.com/input?i=calculate+the+mean+value+of+%281%2Bcos%28x%29%29%5E2+from+0+to+2pi
    return a + C*(0.5*((np.cos(omega*(m-phimu))+1)**2) - 0.75)

def hum_dev(x,*p):
    hum_dev_start = 80
    return -p[0]*(x-hum_dev_start)-p[1]*(x-hum_dev_start)**2

def mu_hum(p,m,h,p0,p1,verb=False):
    '''
    Montly temperature expectation value 
    Modified version of Eq. (19) of Haslebacher et al. (2022)

    Note the re-definition of 'b', which is now the temperature increase per decade

    Note that 'm' is not bound to be a monthly aveage, but can have finer time binning,
    albeit it has units of months. 
    '''
    a     = p[0]
    # initialize b as increase per decade
    b     = p[1]/12/10.
    C     = p[2]
    phimu = p[3]
    omega = np.pi/6

    if (verb):
        #print (m)
        #print ('phimu:',phimu)
        print ('m-phimu:',m-phimu)
        print ('m-phimu shape:',(m-phimu).shape)
        print ('cos:',np.cos(m-phimu))
        
    # humidity correction
    #dHm = np.where(h>80,-0.029*(h-80.)-0.0064*(h-80)**2,0.)
    dHm = np.where(h>80,hum_dev(h,p0,p1),0.)    
    
    # subtract the mean of the function (1+cos(x))^2 to not alter the
    # significance of a as mean temperature, see also
    # https://www.wolframalpha.com/input?i=calculate+the+mean+value+of+%281%2Bcos%28x%29%29%5E2+from+0+to+2pi
    return a + b * m + C*(0.5*((np.cos(omega*(m-phimu))+1)**2) - 0.75) + dHm
    
def mu_dphim(p,m,verb=False):
    '''
    Montly temperature expectation value 
    Modified version of Eq. (19) of Haslebacher et al. (2022)

    Note the re-definition of 'b', which is now the temperature increase per decade

    Note that 'm' is not bound to be a monthly aveage, but can have finer time binning

    Here, we have introduced an additional parameter quantifying the decadal phase shift
    '''
    a     = p[0]
    # parameter b has original units of month^{-1}, initialize b as increase per decade
    b     = p[1]/12/10.
    C     = p[2]
    phimu = p[3]
    # dphim has original units of months/month, initialize dphim as increase of days per decade
    dphim = p[4]/30/12/10.
    omega = np.pi/6/(1-dphim)

    if (verb):
        #print (m)
        #print ('phimu:',phimu)
        print ('m-phimu:',m-phimu)
        print ('m-phimu shape:',(m-phimu).shape)
        print ('cos:',np.cos(m-phimu))
        
    return a + b * m + C*(0.5*((np.cos(omega*(m-phimu-dphim*m))+1)**2) - 0.75)
    
def mu_dCm(p,m,verb=False):
    '''
    Monthly temperature expectation value 
    Modified version of Eq. (19) of Haslebacher et al. (2022)

    Note the re-definition of 'b', which is now the temperature increase per decade

    Note that 'm' is not bound to be a monthly aveage, but can have finer time binning

    Here, we have introduced an additional parameter quantifying the decadal increase of amplitude
    '''
    a     = p[0]
    # parameter b has original units of month^{-1}, initialize b as increase per decade
    b     = p[1]/12/10.
    C     = p[2]
    phimu = p[3]
    # dCm has original units of months^{-1}, initialize dCm as increase per decade
    dCm   = p[4]/12/10.
    omega = np.pi/6

    # do not allow negative amplitudes
    if (np.any(dCm*m<-1)):
        dCm = -1./np.max(m)
    
    if (verb):
        #print (m)
        #print ('phimu:',phimu)
        print ('m-phimu:',m-phimu)
        print ('m-phimu shape:',(m-phimu).shape)
        print ('cos:',np.cos(m-phimu))
    
    return a + b * m + C*(1+dCm*m)*(0.5*((np.cos(omega*(m-phimu))+1)**2) - 0.75)
    
def mu_dCm_hum(p,m,h,p0,p1,verb=False):
    '''
    Monthly temperature expectation value 
    Modified version of Eq. (19) of Haslebacher et al. (2022)

    Note the re-definition of 'b', which is now the temperature increase per decade

    Note that 'm' is not bound to be a monthly aveage, but can have finer time binning

    Here, we have introduced an additional parameter quantifying the decadal increase of amplitude
    '''
    a     = p[0]
    # parameter b has original units of month^{-1}, initialize b as increase per decade
    b     = p[1]/12/10.
    C     = p[2]
    phimu = p[3]
    # dCm has original units of months^{-1}, initialize dCm as increase per decade
    dCm   = p[4]/12/10.
    omega = np.pi/6

    # do not allow negative amplitudes
    if (np.any(dCm*m<-1)):
        dCm = -1./np.max(m)

    # humidity correction
    #dHm = np.where(h>80,-0.029*(h-80.)-0.0064*(h-80)**2,0.)
    dHm = np.where(h>80,hum_dev(h,p0,p1),0.)
    
    if (verb):
        #print (m)
        #print ('phimu:',phimu)
        print ('m-phimu:',m-phimu)
        print ('m-phimu shape:',(m-phimu).shape)
        print ('cos:',np.cos(m-phimu))
    
    return a + b * m + C*(1+dCm*m)*(0.5*((np.cos(omega*(m-phimu))+1)**2) - 0.75) + dHm
    
    
def sig2(p,m):
    '''
    Montly expectation value standard deviation 
    '''
    sigma0= p[0]
    D     = p[1]
    phisig= p[2]
    omega = np.pi/6

    return sigma0 + D*np.sin(omega*(m-phisig))/(1-0.5*np.cos(omega*(m-phisig)))

    
def logpdf_Haslebacher(params,T,m):
    '''
    Product of Gaussian PDFs for temperature measurements
    '''

    sigma0 = params[4] #4.3 #p[2]
    #D      = params[5] #0.5 #p[3]
    #phisig = params[6] # 2.5 #p[4]
    
    mu_i = mu_Haslebacher(params[0:4],m)
    sigma_i = sigma0

    if (np.any(sigma_i[sigma_i < 0])):
        print ('Sigma < 0 found for ',sigma0,', ',D,' , ',phisig)
    
    #x = (T-mu_i)/sigma_i 
    #return -0.5*x**2 - np.log(sigma_i)
    return norm.logpdf(T,mu_i,sigma_i)

def logpdf(params,T,m):
    '''
    Product of Gaussian PDFs for temperature measurements
    '''

    sigma0 = params[4] #4.3 #p[2]
    #D      = params[5] #0.5 #p[3]
    #phisig = params[6] # 2.5 #p[4]
    
    mu_i = mu(params[0:4],m)
    sigma_i = sigma0

    if (np.any(sigma_i[sigma_i <= 0])):
        print ('Sigma < 0 found for ',sigma0)

    #print ('TEST ',params,' x=',(T-mu_i)/sigma_i)
    #x = (T-mu_i)/sigma_i 
    #return -0.5*x**2 - np.log(sigma_i)
    return norm.logpdf(T,mu_i,sigma_i)

def logpdf_sq(params,T,m):
    '''
    Product of Gaussian PDFs for temperature measurements
    '''

    sigma0 = params[5] #4.3 #p[2]
    #D      = params[5] #0.5 #p[3]
    #phisig = params[6] # 2.5 #p[4]
    
    mu_i = mu_sq(params[0:5],m)
    sigma_i = sigma0

    if (np.any(sigma_i[sigma_i <= 0])):
        print ('Sigma < 0 found for ',sigma0)

    #print ('TEST ',params,' x=',(T-mu_i)/sigma_i)
    #x = (T-mu_i)/sigma_i 
    #return -0.5*x**2 - np.log(sigma_i)
    return norm.logpdf(T,mu_i,sigma_i)

def logpdf2(params,T,m):
    '''
    Product of Gaussian PDFs for temperature measurements
    '''

    sigma0 = params[6] #4.3 #p[2]
    #D      = params[5] #0.5 #p[3]
    #phisig = params[6] # 2.5 #p[4]
    
    mu_i = mu2(params[0:6],m)
    sigma_i = sigma0

    if (np.any(sigma_i[sigma_i <= 0])):
        print ('Sigma < 0 found for ',sigma0)

    #print ('TEST ',params,' x=',(T-mu_i)/sigma_i)
    #x = (T-mu_i)/sigma_i 
    #return -0.5*x**2 - np.log(sigma_i)
    return norm.logpdf(T,mu_i,sigma_i)

def logpdf4(params,T,m):
    '''
    Product of Gaussian PDFs for temperature measurements
    '''

    #sigma0 = params[6] #4.3 #p[2]
    #D      = params[5] #0.5 #p[3]
    #phisig = params[6] # 2.5 #p[4]
    
    mu_i = mu2(params[0:6],m)
    sigma_i = sig2(params[6:9],m)

    if (np.any(sigma_i[sigma_i <= 0])):
        print ('Sigma < 0 found for ',sigma_i[sigma_i <= 0])

    #print ('TEST ',params,' x=',(T-mu_i)/sigma_i)
    #x = (T-mu_i)/sigma_i 
    #return -0.5*x**2 - np.log(sigma_i)
    return norm.logpdf(T,mu_i,sigma_i)

def logpdf_offset(params,T,m):
    '''
    Product of Gaussian PDFs for temperature measurements
    '''

    sigma0 = params[4] 
    mu_i = mu(params[0:4],m) + params[-1]
    sigma_i = sigma0

    if (np.any(sigma_i[sigma_i <= 0])):
        print ('Sigma < 0 found for ',sigma0)

    return norm.logpdf(T,mu_i,sigma_i)

def logpdf2_offset(params,T,m):
    '''
    Product of Gaussian PDFs for temperature measurements
    '''

    sigma0 = params[6] 
    mu_i = mu2(params[0:6],m) + params[-1]
    sigma_i = sigma0

    if (np.any(sigma_i[sigma_i <= 0])):
        print ('Sigma < 0 found for ',sigma0)

    return norm.logpdf(T,mu_i,sigma_i)

def logpdf4_offset(params,T,m):
    '''
    Product of Gaussian PDFs for temperature measurements
    '''

    mu_i = mu2(params[0:6],m) + params[-1]
    sigma_i = sig2(params[6:9],m)

    if (np.any(sigma_i[sigma_i <= 0])):
        print ('Sigma < 0 found for ',sigma_i[sigma_i <= 0])

    return norm.logpdf(T,mu_i,sigma_i)

def logpdf_nob(params,T,m):
    '''
    Product of Gaussian PDFs for temperature measurements
    '''

    sigma0 = params[3] #4.3 #p[2]
    #D      = params[5] #0.5 #p[3]
    #phisig = params[6] # 2.5 #p[4]
    
    mu_i = mu_nob(params[0:3],m)
    sigma_i = sigma0

    if (np.any(sigma_i[sigma_i <= 0])):
        print ('Sigma < 0 found for ',sigma0)

    #print ('TEST ',params,' x=',(T-mu_i)/sigma_i)
    #x = (T-mu_i)/sigma_i 
    #return -0.5*x**2 - np.log(sigma_i)
    return norm.logpdf(T,mu_i,sigma_i)

def logpdf_hum(params,T,m,h,p0,p1):
    '''
    Product of Gaussian PDFs for temperature measurements
    '''

    sigma0 = params[4] #4.3 #p[2]
    #D      = params[5] #0.5 #p[3]
    #phisig = params[6] # 2.5 #p[4]
    
    mu_i = mu_hum(params[0:4],m,h,p0,p1)
    sigma_i = sigma0

    if (np.any(sigma_i[sigma_i <= 0])):
        print ('Sigma < 0 found for ',sigma0)

    #print ('TEST ',params,' x=',(T-mu_i)/sigma_i)
    #x = (T-mu_i)/sigma_i 
    #return -0.5*x**2 - np.log(sigma_i)
    return norm.logpdf(T,mu_i,sigma_i)

def logpdf_hum_offset(params,T,m,h,p0,p1):
    '''
    Product of Gaussian PDFs for temperature measurements
    '''

    sigma0 = params[4] #4.3 #p[2]
    #D      = params[5] #0.5 #p[3]
    #phisig = params[6] # 2.5 #p[4]
    
    mu_i = mu_hum(params[0:4],m,h,p0,p1) + params[-1]
    sigma_i = sigma0

    if (np.any(sigma_i[sigma_i <= 0])):
        print ('Sigma < 0 found for ',sigma0)

    #print ('TEST ',params,' x=',(T-mu_i)/sigma_i)
    #x = (T-mu_i)/sigma_i 
    #return -0.5*x**2 - np.log(sigma_i)
    return norm.logpdf(T,mu_i,sigma_i)

def logpdf_dphim(params,T,m):
    '''
    Product of Gaussian PDFs for temperature measurements
    '''

    sigma0 = params[5] #4.3 #p[2]
    #D      = params[5] #0.5 #p[3]
    #phisig = params[6] # 2.5 #p[4]
    
    mu_i = mu_dphim(params[0:5],m)
    sigma_i = sigma0

    if (np.any(sigma_i[sigma_i < 0])):
        print ('Sigma < 0 found for ',sigma0,', ',D,' , ',phisig)
    
    #x = (T-mu_i)/sigma_i 
    #return -0.5*x**2 - np.log(sigma_i)
    return norm.logpdf(T,mu_i,sigma_i)

def logpdf_dCm(params,T,m):
    '''
    Product of Gaussian PDFs for temperature measurements
    '''

    sigma0 = params[5] #4.3 #p[2]
    #D      = params[5] #0.5 #p[3]
    #phisig = params[6] # 2.5 #p[4]
    
    mu_i = mu_dCm(params[0:5],m)
    sigma_i = sigma0

    if (np.any(sigma_i[sigma_i < 0])):
        print ('Sigma < 0 found for ',sigma0)
    
    #print ('TEST',(T-mu_i)/sigma_i)
    #return -0.5*x**2 - np.log(sigma_i)
    return norm.logpdf(T,mu_i,sigma_i)

def logpdf_dCm_hum(params,T,m,h,p0,p1):
    '''
    Product of Gaussian PDFs for temperature measurements
    '''

    sigma0 = params[5] #4.3 #p[2]
    #D      = params[5] #0.5 #p[3]
    #phisig = params[6] # 2.5 #p[4]
    
    mu_i = mu_dCm_hum(params[0:5],m,h,p0,p1)
    sigma_i = sigma0

    if (np.any(sigma_i[sigma_i < 0])):
        print ('Sigma < 0 found for ',sigma0)
    
    #print ('TEST',(T-mu_i)/sigma_i)
    #return -0.5*x**2 - np.log(sigma_i)
    return norm.logpdf(T,mu_i,sigma_i)

def loglike(params,T,m):
    return -1.* logpdf(params,T,m).sum()

def loglike_sq(params,T,m):
    return -1.* logpdf_sq(params,T,m).sum()

def loglike2(params,T,m):
    return -1.* logpdf2(params,T,m).sum()

def loglike4(params,T,m):
    return -1.* logpdf4(params,T,m).sum()

def loglike_2sets(params,T1,m1,T2,m2):
    return -1.* logpdf(params,T1,m1).sum() - logpdf_offset(params,T2,m2).sum()

def loglike2_2sets(params,T1,m1,T2,m2):
    return -1.* logpdf2(params,T1,m1).sum() - logpdf2_offset(params,T2,m2).sum()

def loglike4_2sets(params,T1,m1,T2,m2):
    return -1.* logpdf4(params,T1,m1).sum() - logpdf4_offset(params,T2,m2).sum()

def loglike_nob(params,T,m):
    return -1.* logpdf_nob(params,T,m).sum()

def loglike_hum(params,T,m,h,p0,p1):
    return -1.* logpdf_hum(params,T,m,h,p0,p1).sum()

def loglike_hum_2sets(params,T1,m1,h,T2,m2,h2,p0,p1):
    return -1.* logpdf_hum(params,T1,m1,h,p0,p1).sum() - logpdf_hum_offset(params,T2,m2,h2,p0,p1).sum()

def loglike_dphim(params,T,m):
    return -1.* logpdf_dphim(params,T,m).sum()

def loglike_dCm(params,T,m):
    return -1.* logpdf_dCm(params,T,m).sum()

def loglike_dCm_hum(params,T,m,h,p0,p1):
    return -1.* logpdf_dCm_hum(params,T,m,h,p0,p1).sum()

def loglike_Haslebacher(params,T,m):
    return -1.* logpdf_Haslebacher(params,T,m).sum()


class Likelihood_Wrapper:
    """ Modified from https://stackoverflow.com/questions/16739065/how-to-display-progress-of-scipy-optimize-function """
    def __init__(self,function,mufunction,names,inits,bounds,name):

        self.f = function # actual objective function
        self.mu_func = mufunction
        self.names = names # parameter names
        self.inits = inits # parameter inits
        self.bounds = bounds # parameter bounds
        self.name = name
        self.is_offset = False
        self.num_calls = 0 # how many times f has been called
        self.callback_count = 0 # number of times callback has been called, also measures iteration count
        self.list_calls_inp = [] # input of all calls
        self.list_calls_res = [] # result of all calls
        self.decreasing_list_calls_inp = [] # input of calls that resulted in decrease
        self.decreasing_list_calls_res = [] # result of calls that resulted in decrease
        self.list_callback_inp = [] # only appends inputs on callback, as such they correspond to the iterations
        self.list_callback_res = [] # only appends results on callback, as such they correspond to the iterations

    def setargs_df(self,df,arg,mask,is_daily,is_median,day_coverage=85,data_expected=None,np_mask=None):

        self.is_daily = is_daily
        if (is_daily):
            mask_daily = (df.loc[mask,'mjd'].resample('D').count().dropna() > day_coverage/100*data_expected)        

            self.X = df.loc[mask,'mjd'].resample('D').mean().dropna()
            self.X = self.X[mask_daily]
            if (is_median):
                self.Y = df.loc[mask,arg].resample('D').median().dropna()
            else:
                self.Y = df.loc[mask,arg].resample('D').mean().dropna()
            self.Y = self.Y[mask_daily]
        else:   # monthly
            self.X = df.loc[mask,'mjd'].resample('M').mean().dropna()            
            if (is_median):
                self.Y = df.loc[mask,arg].resample('M').median().dropna()
            else:
                self.Y = df.loc[mask,arg].resample('M').mean().dropna()

        self.X_arr = np.array((self.X.values+mjd_corrector-mjd_start_2003)/number_days_year*12)
        self.Y_arr = np.array(self.Y.values)
        if (np_mask):
            self.X_arr = self.X_arr[np_mask]
            self.Y_arr = self.Y_arr[np_mask]

        if (self.X_arr.shape != self.Y_arr.shape):
            print ('ERROR!', self.name, ' X_arr and Y_arr do not have the same shape: ',
                   self.X_arr.shape, ' vs. ',self.Y_arr.shape)
            self.args = (None,None)

        self.H_arr = None
        self.X2_arr = None
        self.Y2_arr = None
        self.is_offset = False        
            
        self.args = (self.Y_arr,self.X_arr)   # further arguments of function

    def setargs_df2(self,df,arg,mask1,mask2,is_daily,is_median,day_coverage=85,np_mask=None,data_expected=None):

        self.is_daily = is_daily
        if (is_daily):
            mask_daily = (df.loc[mask1,'mjd'].resample('D').count().dropna() > day_coverage/100*data_expected)        

            self.X = df.loc[mask1,'mjd'].resample('D').mean().dropna()
            self.X = self.X[mask_daily]
            if (is_median):
                self.Y = df.loc[mask1,arg].resample('D').median().dropna()
            else:
                self.Y = df.loc[mask1,arg].resample('D').mean().dropna()
            self.Y = self.Y[mask_daily]

            mask_daily = (df.loc[mask2,'mjd'].resample('D').count().dropna() > day_coverage/100*data_expected)        

            self.X2 = df.loc[mask2,'mjd'].resample('D').mean().dropna()
            self.X2 = self.X2[mask_daily]
            if (is_median):
                self.Y2 = df.loc[mask2,arg].resample('D').median().dropna()
            else:
                self.Y2 = df.loc[mask2,arg].resample('D').mean().dropna()
            self.Y2 = self.Y2[mask_daily]

        else:   # monthly
            self.X = df.loc[mask1,'mjd'].resample('M').mean().dropna()            
            if (is_median):
                self.Y = df.loc[mask1,arg].resample('M').median().dropna()
            else:
                self.Y = df.loc[mask1,arg].resample('M').mean().dropna()

            self.X2 = df.loc[mask2,'mjd'].resample('M').mean().dropna()            
            if (is_median):
                self.Y2 = df.loc[mask2,arg].resample('M').median().dropna()
            else:
                self.Y2 = df.loc[mask2,arg].resample('M').mean().dropna()
                
        self.X_arr = np.array((self.X.values+mjd_corrector-mjd_start_2003)/number_days_year*12)
        self.Y_arr = np.array(self.Y.values)
        if (np_mask):
            self.X_arr = self.X_arr[np_mask]
            self.Y_arr = self.Y_arr[np_mask]

        self.X2_arr = np.array((self.X2.values+mjd_corrector-mjd_start_2003)/number_days_year*12)
        self.Y2_arr = np.array(self.Y2.values)
        if (np_mask):
            self.X2_arr = self.X2_arr[np_mask]
            self.Y2_arr = self.Y2_arr[np_mask]

        if (self.X_arr.shape != self.Y_arr.shape):
            print ('ERROR!', self.name, ' X_arr and Y_arr do not have the same shape: ',
                   self.X_arr.shape, ' vs. ',self.Y_arr.shape)
            self.args = (None,None,None,None)

        if (self.X2_arr.shape != self.Y2_arr.shape):
            print ('ERROR!', self.name, ' X2_arr and Y2_arr do not have the same shape: ',
                   self.X2_arr.shape, ' vs. ',self.Y2_arr.shape)
            self.args = (None,None,None,None)

        self.H_arr = None
        self.is_offset = True
        
        self.args = (self.Y_arr,self.X_arr,self.Y2_arr,self.X2_arr)   # further arguments of function

    def setargs_seq(self,X,Y,np_mask=None):

        self.is_daily = False
        self.X = X
        self.Y = Y
        self.X_arr = np.array((self.X.values+mjd_corrector-mjd_start_2003)/number_days_year*12)
        self.Y_arr = np.array(self.Y.values)
        if (np_mask):
            self.X_arr = self.X_arr[np_mask]
            self.Y_arr = self.Y_arr[np_mask]

        if (self.X_arr.shape != self.Y_arr.shape):
            print ('ERROR!', self.name, ' X_arr and Y_arr do not have the same shape: ',
                   self.X_arr.shape, ' vs. ',self.Y_arr.shape)
            self.args = (None,None)

        #print ('X_arr:',self.X_arr)
        #print ('Y_arr:',self.Y_arr)
        self.H_arr = None
            
        self.args = (self.Y_arr,self.X_arr)   # further arguments of function


    def setargs_seq_hum(self,X,Y,H,p0,p1,np_mask=None):

        self.is_daily = False
        self.X = X
        self.Y = Y
        self.H = H
        self.X_arr = np.array((self.X.values+mjd_corrector-mjd_start_2003)/number_days_year*12)
        self.Y_arr = np.array(self.Y.values)
        self.H_arr = np.array(self.H.values)
        if (np_mask):
            self.X_arr = self.X_arr[np_mask]
            self.Y_arr = self.Y_arr[np_mask]
            self.H_arr = self.H_arr[np_mask]

        if (self.X_arr.shape != self.Y_arr.shape):
            print ('ERROR!', self.name, ' X_arr and Y_arr do not have the same shape: ',
                   self.X_arr.shape, ' vs. ',self.Y_arr.shape)
            self.args = (None,None)

        if (self.X_arr.shape != self.H_arr.shape):
            print ('ERROR!', self.name, ' X_arr and H_arr do not have the same shape: ',
                   self.X_arr.shape, ' vs. ',self.H_arr.shape)
            self.args = (None,None)

        #print ('X_arr:',self.X_arr)
        #print ('Y_arr:',self.Y_arr)

        self.H_p0 = p0
        self.H_p1 = p1
        
        self.args = (self.Y_arr,self.X_arr,self.H_arr,self.H_p0,self.H_p1)   # further arguments of function

    def setargs_seq_hum2(self,X,Y,H,mask1,mask2,p0,p1,np_mask=None):

        self.is_daily = False
        self.X = X[mask1]
        self.Y = Y[mask1]
        self.H = H[mask1]
        self.X_arr = np.array((self.X.values+mjd_corrector-mjd_start_2003)/number_days_year*12)
        self.Y_arr = np.array(self.Y.values)
        self.H_arr = np.array(self.H.values)
        if (np_mask):
            self.X_arr = self.X_arr[np_mask]
            self.Y_arr = self.Y_arr[np_mask]
            self.H_arr = self.H_arr[np_mask]
            
        if (self.X_arr.shape != self.Y_arr.shape):
            print ('ERROR!', self.name, ' X_arr and Y_arr do not have the same shape: ',
                   self.X_arr.shape, ' vs. ',self.Y_arr.shape)
            self.args = (None,None)

        if (self.X_arr.shape != self.H_arr.shape):
            print ('ERROR!', self.name, ' X_arr and H_arr do not have the same shape: ',
                   self.X_arr.shape, ' vs. ',self.H_arr.shape)
            self.args = (None,None)

        self.X2 = X[mask2]
        self.Y2 = Y[mask2]
        self.H2 = H[mask2]
        self.X2_arr = np.array((self.X2.values+mjd_corrector-mjd_start_2003)/number_days_year*12)
        self.Y2_arr = np.array(self.Y2.values)
        self.H2_arr = np.array(self.H2.values)
        if (np_mask):
            self.X2_arr = self.X2_arr[np_mask]
            self.Y2_arr = self.Y2_arr[np_mask]
            self.H2_arr = self.H2_arr[np_mask]
            
        if (self.X2_arr.shape != self.Y2_arr.shape):
            print ('ERROR!', self.name, ' X2_arr and Y2_arr do not have the same shape: ',
                   self.X2_arr.shape, ' vs. ',self.Y2_arr.shape)
            self.args = (None,None,None,None)

        if (self.X2_arr.shape != self.H2_arr.shape):
            print ('ERROR!', self.name, ' X2_arr and H2_arr do not have the same shape: ',
                   self.X2_arr.shape, ' vs. ',self.H2_arr.shape)
            self.args = (None,None,None,None)

        #print ('X_arr:',self.X_arr)
        #print ('Y_arr:',self.Y_arr)
        self.H_p0 = p0
        self.H_p1 = p1
        
        self.args = (self.Y_arr,self.X_arr,self.H_arr,
                     self.Y2_arr,self.X2_arr,self.H2_arr,self.H_p0,self.H_p1)   # further arguments of function

        
    def objective(self, x, *args):
        """Executes the actual objective function and returns the result, while
        updating the lists too. Pass to optimizer without arguments or
        parentheses."""
        result = self.f(x, *args) # the actual evaluation of the function
        #print ('x',x)
        #print ('RESULT', result)
        #print ('ARGS', *args)        
        if not self.num_calls: # first call is stored in all lists
            self.decreasing_list_calls_inp.append(x)
            self.decreasing_list_calls_res.append(result)
            self.list_callback_inp.append(x)
            self.list_callback_res.append(result)
        elif result < self.decreasing_list_calls_res[-1]:
            self.decreasing_list_calls_inp.append(x)
            self.decreasing_list_calls_res.append(result)
        self.list_calls_inp.append(x)
        self.list_calls_res.append(result)
        self.num_calls += 1
        return result

    def callback(self, xk, *_):
        """Callback function that can be used by optimizers of scipy.optimize.
        The third argument "*_" makes sure that it still works when the
        optimizer calls the callback function with more than one argument. Pass
        to optimizer without arguments or parentheses."""
        s1 = ""
        xk = np.atleast_1d(xk)
        # search backwards in input list for input corresponding to xk
        for i, x in reversed(list(enumerate(self.list_calls_inp))):
            x = np.atleast_1d(x)
            if np.allclose(x, xk):
                break
    
        for comp in xk:
            s1 += f"{comp:10.5e}\t"
        s1 += f"{self.list_calls_res[i]:10.5e}"

        self.list_callback_inp.append(xk)
        self.list_callback_res.append(self.list_calls_res[i])

        if not self.callback_count:
            s0 = ""
            for j, _ in enumerate(xk):
                #tmp = f"{self.names[j]:10s}"
                s0 += f"{self.names[j]:10s}\t"
            s0 += "Objective"
            print(s0)
        print(s1)
        self.callback_count += 1

    def like_minimize(self,method='L-BFGS-B',tol=1e-9):

        self.res = minimize(self.objective,
                            self.inits,
                            method=method,                         
                            args=self.args,
                            tol=tol,
                            bounds=self.bounds,
                            callback=self.callback)
        print(f"Number of calls to function instance {self.num_calls}")
        print(f"Number of calls to callback instance {self.callback_count}")
        #print (lik_model)
        print (self.res.message)
        print ('Result for ',self.name,':', self.res.x, ' function value=',self.res.fun)
        #print ('Result for mu: ',self.mu_func(self.res.x[0:-1],self.X_arr))
        print ('Result for loglike: ',self.f(self.res.x,*self.args))

        return self.res.success
        #print ('H_inv: ', lik_model.hess_inv)
        

    def approx_hessian(self,eps=1e-5, only_diag=False):
        """
        Computes an approximation of the Hessian matrix of a function using the finite difference method.
        
        Parameters:
        x (array-like): The point at which to compute the Hessian matrix.
        f (callable): The function to compute the Hessian of.
                  The function should take an array-like object as input and return a scalar.
        epsilon (float, optional): The step size for the finite difference method.
        
        Returns:
        hessian (ndarray): An approximation of the Hessian matrix of func at x.
        """
        p = self.res.x
        n = p.size
        self.hessian = np.zeros((n, n))

        # Compute the off-diagonal elements
        if not only_diag:
            for i in range(n):
                for j in range(n):
                    if i == j or i < j:
                        continue # the hessian is symmetrical, so we can compute only half of it
                    p1 = p.copy()
                    p1[i] += eps
                    p1[j] += eps
                    
                    p2 = p.copy()
                    p2[i] += eps
                    p2[j] -= eps
                    
                    p3 = p.copy()
                    p3[i] -= eps
                    p3[j] += eps
                    
                    p4 = p.copy()
                    p4[i] -= eps
                    p4[j] -= eps

                    self.hessian[i][j] = (self.f(p1,*self.args) - self.f(p2,*self.args) - self.f(p3,*self.args) + self.f(p4,*self.args)) / (4 * eps ** 2)
        self.hessian = self.hessian + self.hessian.transpose()  # fill the element under the diagonal with the same numbers

        # compute diagonal elements
        for i in range(n):
            p_forward = p.copy()
            p_forward[i] += eps

            p_backward = p.copy()
            p_backward[i] -= eps

            self.hessian[i][i] = (self.f(p_forward,*self.args) - 2 * self.f(p,*self.args) + self.f(p_backward,*self.args)) / (eps ** 2)

        print ('Approx. Hessian: eps=',eps,self.hessian)
        self.H_inv = np.linalg.inv(self.hessian)
        print ('Hessian Inverted: ',self.H_inv)        
        self.p_err = np.sqrt(np.diag(self.H_inv))
            
        return self.hessian

    def objective_for_profiling_2d(self,x,*args):

        x[self._i] = self._prof_vi
        x[self._j] = self._prof_vj
        return self.objective(x,*args)

    def objective_for_profiling(self,x,*args):

        x[self._i] = self._prof_v
        return self.objective(x,*args)

    def _like_minimize_for_profiling(self,method='L-BFGS-B',tol=1e-9):

        res = minimize(self.objective_for_profiling,
                       self.res.x,  # initialize with result from global minimum                       
                       method=method,                         
                       args=self.args,
                       tol=tol,
                       bounds=self.bounds,
                       callback=self.callback)
        print(f"Number of calls to objective function {self.num_calls}")
        print(f"Number of calls to callback instance {self.callback_count}")
        #print (lik_model)
        print (res.message)
        print ('Result for ',self.name,':', res.x, ' function value=',res.fun)
        #print ('H_inv: ', lik_model.hess_inv)
        #print ('Result for mu: ',self.mu_func(res.x[0:-1],self.X_arr))
        print ('Result for loglike: ',self.f(res.x,*self.args))

        return res if res.success else None

    def _like_minimize_for_profiling_2d(self,method='L-BFGS-B',tol=1e-9):

        res = minimize(self.objective_for_profiling_2d,
                       self.res.x,  # initialize with result from global minimum
                        method=method,                         
                       args=self.args,
                       tol=tol,
                       bounds=self.bounds,
                       callback=self.callback)
        #print(f"Number of calls to objective function {self.num_calls}")
        #print(f"Number of calls to callback instance {self.callback_count}")
        #print (lik_model)
        print (res.message)
        print ('Result for ',self.name,':', res.x, ' function value=',res.fun)
        #print ('H_inv: ', lik_model.hess_inv)
        #print ('Result for mu: ',self.mu_func(res.x[0:-1],self.X_arr))
        print ('Result for loglike: ',self.f(res.x,*self.args))

        return res if res.success else None
        
    def _search_pmax(self,p_min,perr,L_min,NN,chi2,method,tol):

        Delta = np.abs(np.sqrt(chi2)*perr)  # first try, most probably too small 
        p_max = 0.
        if (p_min > 0):
            p_max = p_min+NN*Delta/2.
        else:
            p_max = p_min-NN*Delta/2.

        print ('Profile likelihood, calculate maximum borders at ',p_max)
        self._prof_v = p_max
        res = self._like_minimize_for_profiling(method,tol)

        if (res is None):
            return None, None
        
        D_max  = 2.*(res.fun-L_min) 
        # re-scale Delta to achieved maximum of D close to chi2
        Delta = Delta * np.sqrt(chi2/D_max) 
        print ('Found D_max=',D_max, ' need to achieve:', chi2, ' new Delta: ',Delta)
        while (np.sqrt(chi2/D_max) < 0.8 or np.sqrt(chi2/D_max) > 1.25):
            # too high risk for deviation from parabaola, re-scale again with a test
            if (p_min > 0):
                p_max = p_min+NN*Delta/2.
            else:
                p_max = p_min-NN*Delta/2.
            
            print ('Profile likelihood, check new maximum borders at ',p_max)
            self._prof_v = p_max            
            res = self._like_minimize_for_profiling(method,tol)
            if (res is None):
                return None, None
            D_max  = 2.*(res.fun-L_min) 
            # re-scale Delta to achieved maximum of D close to chi2
            Delta = Delta * np.sqrt(chi2/D_max) 
            print ('Found new D_max=',D_max, ' need to achieve:', chi2,' new Delta: ',Delta)

        return Delta, p_max

    def _profile(self,p,method,tol):
        self._prof_v = p
        print ('Profile likelihood, test parameter=',p)                    
        res = self._like_minimize_for_profiling(method,tol)
        
        if (res is None):
            return np.nan

        return res.fun
    
    def _profile2d(self,p,method,tol):
        self._prof_vj = p
        #print ('Profile likelihood, test parameter=',p)                    
        res = self._like_minimize_for_profiling_2d(method,tol)
        
        if (res is None):
            return np.nan

        return res.fun
    
    def profile_likelihood(self,i,chi2=4,NN=20,method='L-BFGS-B',tol=1e-9,col='b',alpha=None,add_sigma=None):

        # First calculate the absolute minimum:
        print ('Profile likelihood, calculate global minimum with method', method, ' and tol',tol)
        success = self.like_minimize(method,tol)

        if (not success):
            return None, None, None

        self.approx_hessian(only_diag=True)
        
        self.p_min = self.res.x[i]
        L_min = self.res.fun

        print ('Profile likelihood, found minimum at: ',self.p_min,' with result',L_min)
        
        ps = []
        Ds = []

        self._i = i

        Delta, p_max = self._search_pmax(self.p_min,self.p_err[i],L_min,NN,chi2,method,tol)
        
        self.nus = np.zeros((len(self.res.x),1))

        self.ps = np.arange(self.p_min-NN*Delta/2., self.p_min + NN*Delta/2., Delta)
        p = Pool(number_of_CPUs_available)
        Ls = p.map(partial(self._profile,method=method,tol=tol),self.ps)
        self.Ds = np.array(2.*(Ls-L_min))
        
        #for p in self.ps:
        #
        #    self._prof_v = p
        #    print ('Profile likelihood, test parameter=',p)                    
        #    res = self._like_minimize_for_profiling(method,tol)
        #
        #    if (res is None):
        #        return None, None, None
        #            
        #    self.nus = np.c_[self.nus,np.array(res.x)]
        #    ps.append(p)
        #    Ds.append(2.*(res.fun-L_min))

        #self.ps = np.array(ps)
        #self.Ds = np.array(Ds)


        
        print ('Found ps:',self.ps, 'p_min=',self.p_min)
        ps_pos = self.ps[self.ps > self.p_min]
        ps_neg = self.ps[self.ps < self.p_min]

        Ds_pos = self.Ds[self.ps > self.p_min]
        Ds_neg = self.Ds[self.ps < self.p_min]

        ps_pos_fit = ps_pos[Ds_pos < 1]
        ps_neg_fit = ps_neg[Ds_neg < 1]

        Ds_pos_fit = Ds_pos[Ds_pos < 1]
        Ds_neg_fit = Ds_neg[Ds_neg < 1]

        self.ps_fit = np.append(ps_neg_fit, ps_pos_fit)
        self.Ds_fit = np.append(Ds_neg_fit, Ds_pos_fit)

        if add_sigma is not None:
            # search first the minimum
            popt, pcov = curve_fit(quad_func, self.ps_fit, self.Ds_fit, p0=[self.p_min,self.p_err[i]])
            fitted_min = popt[0]

            sigma_func = quad_func(self.ps,fitted_min,add_sigma)

            print ('self.Ds:', self.Ds)
            print ('sigma_func:', sigma_func)            

            self.Ds = np.where(self.Ds>sigma_func, self.Ds - sigma_func, self.Ds)
            print ('new self.Ds:', self.Ds)            

            Ds_pos = self.Ds[self.ps > self.p_min]
            Ds_neg = self.Ds[self.ps < self.p_min]

            ps_pos_fit = ps_pos[Ds_pos < 2]
            ps_neg_fit = ps_neg[Ds_neg < 2]
            
            Ds_pos_fit = Ds_pos[Ds_pos < 2]
            Ds_neg_fit = Ds_neg[Ds_neg < 2]
            
            self.ps_fit = np.append(ps_neg_fit, ps_pos_fit)
            self.Ds_fit = np.append(Ds_neg_fit, Ds_pos_fit)
        
            print ('HERE',self.ps_fit, self.Ds_fit)
        
        self.ps = self.ps[self.Ds<chi2*1.1]
        self.Ds = self.Ds[self.Ds<chi2*1.1]        
        
        plt.scatter(self.ps, self.Ds, s = 10, color=col, alpha=alpha)

        if (len(self.ps_fit) > 3):
            popt, pcov = curve_fit(quad_func, self.ps_fit, self.Ds_fit, p0=[self.p_min,self.p_err[i]])
            self.Nexp = quad_func(self.ps_fit, *popt)
            self.Nexp_all = quad_func(self.ps, *popt)

            ps = self.ps[self.Nexp_all <= chi2]
            self.Nexp_all = self.Nexp_all[self.Nexp_all <= chi2]            
            
            plt.plot(self.ps_fit,self.Nexp,'-',lw=3,label=f'{self.name}: '+r'$\mu$=%.2f, $\sigma$=%.2f' % tuple(popt),color=col,alpha=alpha)
            plt.plot(ps,self.Nexp_all,'--',color=col,alpha=alpha)

        ax = plt.gca()
        ax.yaxis.set_tick_params(labelsize=20)
        ax.xaxis.set_tick_params(labelsize=20)
        plt.legend(loc='upper right', fontsize=12)

        #confidence_intervals(chi2=4):
        xvals   = np.linspace(self.ps[0], self.ps[-1], 500)
        yinterp = np.interp(xvals, self.ps, self.Ds)

        xvals_pos = xvals[xvals > self.p_min]
        xvals_neg = xvals[xvals < self.p_min]

        yinterp_pos = yinterp[xvals > self.p_min]
        yinterp_neg = yinterp[xvals < self.p_min]

        # y should be sorted for both of these methods
        order_pos = yinterp_pos.argsort()
        y_pos = yinterp_pos[order_pos]
        x_pos = xvals_pos[order_pos]

        order_neg = yinterp_neg.argsort()
        y_neg = yinterp_neg[order_neg]
        x_neg = xvals_neg[order_neg]

        intervals_1 = []
        intervals_4 = []
        intervals_9 = []        
    
        print ('x_pos: ', x_pos, ' y_pos: ' , y_pos , ' chi2: ', 1)
        yidx = y_pos.searchsorted(1, 'left')
        if (yidx >= len(y_pos)):
            intervals_1.append(x_pos[yidx-1])
        else:
            intervals_1.append(x_pos[yidx])
        intervals_1.append(x_neg[y_neg.searchsorted(1, 'left')])
    
        yidx = y_pos.searchsorted(4, 'left')
        if (yidx >= len(y_pos)):
            intervals_4.append(x_pos[yidx-1])
        else:
            intervals_4.append(x_pos[yidx])
        
        intervals_4.append(x_neg[y_neg.searchsorted(4, 'left')])

        print (f'{self.name}:')
        print ('Found intervals for chi2=',1,': ',intervals_1)
        print ('Found intervals for chi2=',4,': ',intervals_4)
        if (chi2 >= 9):
            yidx = y_pos.searchsorted(9, 'left')
            if (yidx >= len(y_pos)):
                intervals_9.append(x_pos[yidx-1])
            else:
                intervals_9.append(x_pos[yidx])

            idx9 = y_neg.searchsorted(9, 'left')
            if (idx9 >=0) and (idx9<len(x_neg)):
                intervals_9.append(x_neg[y_neg.searchsorted(9, 'left')])
            else:
                intervals_9.append(-999.)
            print ('Found intervals for chi2=',9,': ',intervals_9)

        return np.array(intervals_1), np.array(intervals_4), np.array(intervals_9)


    def profile_likelihood_2d(self,i,j,chi2=4,NN=20,method='L-BFGS-B',tol=1e-9,clabel=''):

        self._i = i
        self._j = j        

        # First calculate the absolute minimum:
        print ('Profile likelihood, calculate global minimum with method', method, ' and tol',tol)
        success = self.like_minimize(method,tol)

        if (not success):
            return

        self.approx_hessian(eps=1e-1,only_diag=True)
        
        self.p_min_i = self.res.x[i]
        self.p_min_j = self.res.x[j]
        L_min = self.res.fun

        print ('Profile likelihood, found minimum at: ',self.p_min_i,' and ',self.p_min_j,' with result',L_min)
        
        pis = []
        pjs = []
        Ds = []

        self._i = i
        Delta_i, p_max_i = self._search_pmax(self.p_min_i,self.p_err[i],L_min,NN,chi2,method,tol)
        
        self._i = j
        Delta_j, p_max_j = self._search_pmax(self.p_min_j,self.p_err[j],L_min,NN,chi2,method,tol)
        
        self._i = i
        self._j = j

        #self.nus = np.zeros((len(self.res.x),1))

        pis = np.linspace(self.p_min_i-NN*Delta_i/2.,self.p_min_i+NN*Delta_i/2,NN)
        pjs = np.linspace(self.p_min_j-NN*Delta_j/2.,self.p_min_j+NN*Delta_j/2,NN)        

        print ('pis for ',self.name,': ',pis)
        print ('pjs for ',self.name,': ',pjs)                
        
        Ds  = np.zeros((NN,NN))
        
        for ix, p_i in np.ndenumerate(pis):

            self._prof_vi = p_i

            p = Pool(number_of_CPUs_available)
            Ls = p.map(partial(self._profile2d,method=method,tol=tol),pjs)
            Ds[ix,:] = np.array(2.*(Ls-L_min))
            
        #print ('DDD for ',self.name,': ',Ds)
            
        ax = plt.gca()
        PI, PJ = np.meshgrid(pis,pjs)
        CS = ax.contour(PI, PJ, Ds.T, levels=[1,4,9], cmap='jet')
        ax.clabel(CS, fontsize=20, inline=True,fmt=chi2_fmt)
        ax.yaxis.set_tick_params(labelsize=18)
        ax.xaxis.set_tick_params(labelsize=18)
        #cbar = plt.colorbar(CS)
        #cbar.ax.set_ylabel(clabel, fontsize = 25)

    def red_residuals(self, is_sigma2=False):

        idxlast = -1
        if is_sigma2:
            idxlast = -3
        if self.is_offset:
            idxlast = idxlast-1
        
        if self.H_arr is None:
            return (self.Y_arr - self.mu_func(self.res.x[0:idxlast],self.X_arr))/self.res.x[idxlast]
        else:
            return (self.Y_arr - self.mu_func(self.res.x[0:idxlast],self.X_arr,self.H_arr,self.H_p0,self.H_p1))/self.res.x[idxlast]            

        if self.is_offset is False:
            return res1
        
        return pd.concat([(self.Y2 - self.mu_func(self.res.x[0:idxlast],self.X2_arr) - self.res.x[-1])/self.res.x[idxlast],res1])
    
    def full_residuals(self, is_sigma2=False):

        idxlast = -1
        if is_sigma2:
            idxlast = -3
        if self.is_offset:
            idxlast = idxlast-1
        
        if self.H_arr is None:
            res1 = self.Y - self.mu_func(self.res.x[0:idxlast],self.X_arr)
        else:
            res1 = self.Y - self.mu_func(self.res.x[0:idxlast],self.X_arr,self.H_arr,self.H_p0,self.H_p1)

        if self.is_offset is False:
            return res1

        return pd.concat([self.Y2 - self.mu_func(self.res.x[0:idxlast],self.X2_arr) - self.res.x[-1],res1])
                
    def chi_square_ndf(self,is_sigma2=False):

        residuals = self.red_residuals(is_sigma2=is_sigma2)
        self.chi2 = (residuals**2).sum()/(len(residuals)-len(self.res.x)-1)
        print('CHI2/NDF for ',self.name,' = ',self.chi2)
        return self.chi2

    def plot_residuals(self,color='r',is_sigma2=False):

        idxlast = -1
        if is_sigma2:
            idxlast = -3
        if self.X2_arr is not None:
            idxlast = idxlast-1
        
        # see discussion of splitting df into consecutive indices on
        # https://stackoverflow.com/questions/56257329/how-to-split-a-dataframe-based-on-consecutive-index
    
        #print ('DIFF TMP:', self.Y.index.to_series().diff(1))
        if (self.is_daily):
            list_of_X = np.split(self.X,np.flatnonzero(self.X.index.to_series().diff(1) != '1 days'))
            list_of_Y = np.split(self.Y,np.flatnonzero(self.Y.index.to_series().diff(1) != '1 days'))
        else:
            # welcome to the fun of pandas...
            list_of_X = np.split(self.X,np.flatnonzero(((self.X.index.to_series().diff(1) != '28 days')
                                                         & (self.X.index.to_series().diff(1) != '29 days')
                                                         & (self.X.index.to_series().diff(1) != '30 days')
                                                         & (self.X.index.to_series().diff(1) != '31 days'))))        
            list_of_Y = np.split(self.Y,np.flatnonzero(((self.Y.index.to_series().diff(1) != '28 days')
                                                         & (self.Y.index.to_series().diff(1) != '29 days')
                                                         & (self.Y.index.to_series().diff(1) != '30 days')
                                                         & (self.Y.index.to_series().diff(1) != '31 days'))))        
        #print ('list: ',list_of_df)

        for ii, df_y in np.ndenumerate(list_of_Y):            
            #print ('II: ',ii[0])
            df_x = list_of_X[ii[0]].add(mjd_corrector-mjd_start_2003).mul(12/number_days_year)
            #print ('INDEX:', df_x.index)
            #print ('MONTH:', df_x.values)
            #print ('TEMP:', df_y.values)
            plt.plot(df_x.index,
                     df_y.values-self.mu_func(self.res.x[0:idxlast],np.array(df_x.values.astype(float)),verb=False),
                     color=color)
            
        if self.X2_arr is not None: 
            if (self.is_daily):
                list_of_X = np.split(self.X2,np.flatnonzero(self.X2.index.to_series().diff(1) != '1 days'))
                list_of_Y = np.split(self.Y2,np.flatnonzero(self.Y2.index.to_series().diff(1) != '1 days'))
            else:
                # welcome to the fun of pandas...
                list_of_X = np.split(self.X2,np.flatnonzero(((self.X2.index.to_series().diff(1) != '28 days')
                                                             & (self.X2.index.to_series().diff(1) != '29 days')
                                                             & (self.X2.index.to_series().diff(1) != '30 days')
                                                             & (self.X2.index.to_series().diff(1) != '31 days'))))        
                list_of_Y = np.split(self.Y2,np.flatnonzero(((self.Y2.index.to_series().diff(1) != '28 days')
                                                             & (self.Y2.index.to_series().diff(1) != '29 days')
                                                             & (self.Y2.index.to_series().diff(1) != '30 days')
                                                             & (self.Y2.index.to_series().diff(1) != '31 days'))))        

        for ii, df_y in np.ndenumerate(list_of_Y):            
            #print ('II: ',ii[0])
            df_x = list_of_X[ii[0]].add(mjd_corrector-mjd_start_2003).mul(12/number_days_year)
            #print ('INDEX:', df_x.index)
            #print ('MONTH:', df_x.values)
            #print ('TEMP:', df_y.values)
            plt.plot(df_x.index,
                     df_y.values-self.mu_func(self.res.x[0:idxlast],np.array(df_x.values.astype(float)),verb=False)-self.res.x[-1],
                     color=color)
            
        
def temperature_STL(df):
    print (df)
    print (df.head())
    stl = STL(df, period=12,seasonal=7,robust=True)
    res = stl.fit()
    fig = res.plot()



class TemperatureRegression:

    def __init__(self, y, X, beta):
        self.X = (X+55000-52640.)/365.2422*12   # months w.r.t. 1/1/2003      
        self.n = X.shape
        # Reshape y as a n_by_1 column vector
        self.y =  y   #.reshape(self.n,1)
        # Reshape β as a k_by_1 column vector
        self.beta = beta

    def mu(self,beta):
        return beta[0] + beta[1]/12*self.X + beta[2]*(0.5*((np.cos(np.pi/6*(self.X-beta[3]))+1)**2) - 0.75)

    def sigma(self):
        sigma0 = 4.8
        D      = 3.
        phisig = 2.5
        return sigma0 + D*np.sin(np.pi/6*(self.X-phisig))/(1-0.5*np.cos(np.pi/6*(self.X-phisig)))

    def logL(self,beta):
        mu = self.mu(beta)
        sigma = self.sigma()
        return -1.*norm.logpdf(self.y,mu,sigma).sum()

    def G(self):
        y = self.y
        mu = self.mu()
        X = self.X
        print ('X.T @ (y-mu)',X.T @ (y-mu))        
        return X.T @ (y - mu)

    def H(self):
        X = self.X
        mu = self.mu()
        #print ('X=',X,' mu=',mu)
        #print ('mu*X=',mu*X)
        print ('X.T @ mu*X=',X.T @ (mu*X))
        return -(X.T @ (mu * X))


def newton_raphson(model, tol=1e-3, max_iter=1000, display=True):

    i = 0
    error = 100  # Initial error value

    # Print header of output
    if display:
        header = f'{"Iteration_k":<13}{"Log-likelihood":<16}{"θ":<60}'
        print(header)
        print("-" * len(header))

    # While loop runs while any value in error is greater
    # than the tolerance until max iterations are reached
    while np.any(error > tol) and i < max_iter:
        H, G = model.H(), model.G()
        beta_new = model.beta - G/H
        #beta_new = model.beta - (np.linalg.inv(H) @ G)
        error = np.abs(beta_new - model.beta)
        model.beta = beta_new

        # Print iterations
        if display:
            beta_list = [f'{t:.3}' for t in list(model.beta.flatten())]
            update = f'{i:<13}{model.logL():<16.8}{beta_list}'
            print(update)

        i += 1

    print(f'Number of iterations: {i}')
    print(f'beta_hat = {model.beta.flatten()}')

    # Return a flat array for β (instead of a k_by_1 column vector)
    return model.beta.flatten()



class Temperature_Likelihood(GenericLikelihoodModel):
    def __init__(self, endog, exog, **kwds):
        super(Temperature_Likelihood, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
        #beta = params[-1]
        b = params[0]
        phimu = params[1]
        p = [ 11.2, 6.1, 8.0, 4.8, 2.5 ]
        ll = logpdf(self.endog, self.exog, p=p, b=b, phimu=phimu)
        return -ll

    def fit(self, start_params=None, maxiter=1000, maxfun=500, **kwds):
        # we have one additional parameter and we need to add it for summary
        self.exog_names.append('beta')
        self.exog_names.append('phimu')
        if start_params == None:
            # Reasonable starting values
            start_params = np.append(np.zeros(self.exog.shape[1]), 0.)
            start_params = np.append(start_params, 4.5)
            #start_params = [ 0., 4.5]
            # intercept
            #start_params[-2] = np.log(self.endog.mean())
        return super(Temperature_Likelihood, self).fit(start_params=start_params,
                                                       maxiter=maxiter, maxfun=maxfun,
                                                       **kwds)




def likelihood_solver(x, k, weights=None):

    p0    = x[0]
    alpha = x[1]

    N = len(k)
    i_arr = np.arange(N)-(N-1)/2
    p_i = p0 + alpha*i_arr

    p_i[np.where(p_i<=0.)] = 1e-19   # check whether necessary! 
    
    if weights is None:
        eq1 = -N  + np.sum(   k       / p_i )
        eq2 =       np.sum(   k*i_arr / p_i )
    else:
        eq1 = -np.sum(weights)       + np.sum(   k       / p_i )
        eq2 = -np.sum(i_arr*weights) + np.sum(   k*i_arr / p_i )
    
    return [ eq1, eq2 ]

def likelihood_solver_profile(x, k, alpha, weights=None):

    p0    = x[0]
    
    N = len(k)
    i_arr = np.arange(N)-(N-1)/2
    p_i = p0 + alpha*i_arr

    p_i[np.where(p_i<=0.)] = 1e-19   # check whether necessary! 
    
    if weights is None:
        eq1 = -N + np.sum(k / p_i )
    else:
        eq1 = -np.sum(weights) + np.sum(k / p_i )

    return [ eq1 ]

def calc_loglikelihood(k, alpha, p0, weights=None):

    # Calculate the log-likelihood, Eq. A2
    N = len(k)
    i_arr = np.arange(N)-(N-1)/2

    p_i = p0 + alpha*i_arr

    print ('p_i (before weights)=',p_i)
    print ('w_i=',weights)    
    if weights is not None:
        p_i = p_i * weights

    print ('p_i (after weights)=',p_i)        
    
    #arg[np.where(arg<0.)] = 0.01
    #print ('p_i=',p_i)
    if weights is not None:
        ids = np.where(weights > 0.)
        return -np.sum(p_i[ids]) + np.sum(np.where(k[ids]>0,k[ids]*np.log(p_i[ids]),0.))

    return -np.sum(p_i) + np.sum(np.where(k>0,k*np.log(p_i),0))

def p0_solver(k, alpha, i_arr, weights=None):

    N = len(k)

    p0_start = np.sum(k)/N
    if weights is not None:
        p0_start = np.sum(k)/np.sum(weights)

    print ('P0 solver: p0_start=', p0_start)
    p0 = fsolve(lambda x: likelihood_solver_profile(x, k, alpha, weights), p0_start)

    p_i = p0 + alpha*i_arr

    print ('P0 solver: p0=',p0,' pi=',p_i)
    
    fac = 1.00000000001
    if weights is not None:
        #    p_i = (p0 + alpha_i_arr) * weights
        if weights[0] < weights[-1]:
            fac = fac/weights[0]
        else:
            fac = fac/weights[-1]

    print ('P0 solver: fac=',fac,' p_i=', p_i)

    if np.any(p_i < 0):   # unphysical (negative) Poissonian probability detected
        #print ('negative p_i detected in: ',p_i,' will change p_0 from: ',p0,' to: ', np.abs(alpha) * (N-1)/2 * fac)

        p0_test = np.abs(alpha) * (N-1)/2 * fac # apply physical limit 

        #if weights is not None:
        #    print ('new p_is (weights):', (p0+alpha*i_arr)*weights)
        #else:
        #    print ('new p_is (no weights():', p0+alpha*i_arr)

        print ('P0 solver: final p0=',p0_test)
        
    return p0

def profile_likelihood(k, chi2=4, weights=None):

    N = len(k)
    i_arr = np.arange(N)-(N-1)/2

    p0_start = np.sum(k)/N
    if weights is not None:
        p0_start = np.sum(k)/np.sum(weights)

    print ('Calc p0min,alphamin: p0_start=',p0_start)
    p0_min, alpha_min = fsolve(lambda x: likelihood_solver(x, k, weights), (p0_start, 0))

    # check for unphysical occurrence probabilities
    p_i_test = p0_min + alpha_min*i_arr    
    if np.any(p_i_test <= 0):
        if (alpha_min < 0):
            alpha_min = -p0_min / i_arr[-1]
        if (alpha_min > 0):
            alpha_min = -p0_min / i_arr[0]
        
    print ('Calc Lmin k=',k,' alpha_min=',alpha_min,' p0_min=',p0_min,' weights=',weights)    
    L_min = calc_loglikelihood(k, alpha_min, p0_min,weights)

    p0s = []
    alphas = []
    Ds = []
    NN = 200

    Delta = 0.001   # first try, most probably too small 
    alpha_max = 0.
    if (alpha_min > 0):
        alpha_max = alpha_min+NN*Delta/2.
    else:
        alpha_max = alpha_min-NN*Delta/2.
    
    p0 = p0_solver(k, alpha_max,i_arr,weights)
    print ('Found p0=',p0)
    print ('Calc Loglike with k=',k,' alpha_max=',alpha_max,' weights=',weights,' L_min=',L_min)
    D_max  = -2.*(calc_loglikelihood(k, alpha_max, p0, weights)-L_min) 
    # re-scale Delta to achieved maximum of D close to chi2
    print ('Found D_max=',D_max, ' need to achieve:', chi2)
    Delta = Delta * np.sqrt(chi2/D_max) * 1.1
    while (np.sqrt(chi2/D_max) < 0.8 or np.sqrt(chi2/D_max) > 1.25):
        # too high risk for deviation from parabaola, re-scale again with a test
        if (alpha_min > 0):
            alpha_max = alpha_min+NN*Delta/2.
        else:
            alpha_max = alpha_min-NN*Delta/2.
            
        p0 = p0_solver(k, alpha_max,i_arr,weights)
        D_max  = -2.*(calc_loglikelihood(k, alpha_max, p0,weights)-L_min) 
        # re-scale Delta to achieved maximum of D close to chi2
        Delta = Delta * np.sqrt(chi2/D_max) 
        print ('Found new D_max=',D_max,' alpha_max=',alpha_max, ' need to achieve:', chi2,' new Delta: ',Delta)

    Delta = Delta * 1.2
    print ('PROFILE: alpha_min=', alpha_min, ' Delta=',Delta, ' L_min=',L_min)
    for alpha in np.arange(alpha_min-NN*Delta/2., alpha_min + NN*Delta/2., Delta):

        p0 = p0_solver(k, alpha, i_arr,weights)
        D = -2.*(calc_loglikelihood(k, alpha, p0, weights)-L_min)
        alphas.append(alpha)
        p0s.append(p0)
        Ds.append(D)

        print ('PROFILE TEST: ',alpha,' p0: ',p0,' D=',D)

    return alphas, Ds, alpha_min

def TS(k, weights=None):

    N = len(k)
    i_arr = np.arange(N)-(N-1)/2

    p0_start = np.sum(k)/N
    if weights is not None:
        p0_start = np.sum(k)/np.sum(weights)

    p0_min, alpha_min = fsolve(lambda x: likelihood_solver(x, k, weights), (p0_start, 0))

    if weights is not None:
        N = np.sum(weights)

    f = p0_min + alpha_min*i_arr
    sumlog = np.sum(k*(1+np.log(N*f/np.sum(k))))
    if weights is None: 
        ts = 2*(-N * p0_min + sumlog)
    else:
        ts = 2*(-np.sum(f*weights) + sumlog)

    return np.sqrt(ts), p0_min, alpha_min

def valors_propers_a(array, chi2): 

    if array.size != 0:
        array = np.asarray(array) 
        index = (np.abs(array - chi2)).argmin() 
    
    return index

def confidence_intervals(k, chi2=4, weights=None):

    array_x, array_y, alpha_min = profile_likelihood(k, chi2, weights)

    print ('array_x: ', array_x, ' array_y: ',array_y, ' alpha_min: ', alpha_min)
    
    xvals   = np.linspace(array_x[0], array_x[-1], 500)
    yinterp = np.interp(xvals, array_x, array_y)

    xvals_pos = xvals[xvals > alpha_min]
    xvals_neg = xvals[xvals < alpha_min]

    yinterp_pos = yinterp[xvals > alpha_min]
    yinterp_neg = yinterp[xvals < alpha_min]

    # y should be sorted for both of these methods
    order_pos = yinterp_pos.argsort()
    y_pos = yinterp_pos[order_pos]
    x_pos = xvals_pos[order_pos]

    order_neg = yinterp_neg.argsort()
    y_neg = yinterp_neg[order_neg]
    x_neg = xvals_neg[order_neg]

    intervals = []
    
    print ('x_pos: ', x_pos, ' y_pos: ' , y_pos , ' chi2: ', chi2)

    yidx = y_pos.searchsorted(chi2, 'left')
    if (yidx >= len(y_pos)):
        intervals.append(x_pos[yidx-1])
    else:
        intervals.append(x_pos[yidx])
        
    intervals.append(x_neg[y_neg.searchsorted(chi2, 'left')])
    
    #print ('Found intervals for chi2=',chi2,': ',intervals)

    return np.array(intervals)


def years_pos(x, pos):
    """The two arguments are the value and tick position."""
    return f'{x:.0f}        '

def plot_extremes(k, year_start, year_end, weights=None):

    midyear = 0.5*(year_start + year_end-1)

    N = len(k)
    i_arr = np.arange(N)-(N-1)/2

    plt.clf()
    plt.plot(i_arr+midyear, k, '.', color = 'steelblue', markersize = 10)

    sqrtTS, p0, alpha = TS(k, None)
    plt.plot(i_arr+midyear, p0+alpha*i_arr, color = 'tomato', linestyle =(0, (5, 10)), label='no weights')

    if weights is not None:
        sqrtTS, p0, alpha = TS(k, weights)
        plt.plot(i_arr+midyear, p0+alpha*i_arr, color = 'g', linestyle =(0, (5, 10)), label='with weights')
        #p0 = p0 * weights
        #plt.plot(i_arr+midyear, p0+alpha*i_arr, color = 'gold', linestyle =(0, (5, 10)), label='with weights, p0 corrected')
    plt.legend(loc='best')
    plt.xlabel('Year', fontsize = 25)
    ax = plt.gca()
    start, end = ax.get_xlim()
    years = np.arange(year_start, year_end+1)
    ax.xaxis.set_major_formatter(years_pos)
    #ax.xaxis.set_major_formatter(ticker.NullFormatter()) #set_major_formatter("  %")
    #ax.xaxis.set_minor_formatter(years)
    #plt.xticks(ha='left',fontsize=12)
    ax.xaxis.set_ticks(years) 
    plt.xticks(ha='center')
    starty, endy = ax.get_ylim()
    ax.set_ylim(-0.5,endy)

    step = ((int)(np.ceil(endy)))//5
    if (step < 1):
        step = 1
    #print ('endy: ', np.ceil(endy), ' step: ', step)

    nums = np.arange(0, np.ceil(endy), step)
    ax.yaxis.set_ticks(nums) 

    ax.yaxis.set_tick_params(labelsize=20)
    ax.xaxis.set_tick_params(labelsize=15)


def plot_profile_likelihood(k, weights=None, NN=200, chi2_max=9):

    N = len(k)
    i_arr = np.arange(N)-(N-1)/2

    p0_start = np.sum(k)/N
    if weights is not None:
        p0_start = np.sum(k)/np.sum(weights)

    p0_min, alpha_min = fsolve(lambda x: likelihood_solver(x, k, weights), (p0_start, 0))
    L_min = calc_loglikelihood(k, alpha_min, p0_min, weights)
    print ('Found Lmin=',L_min, ' at: ', alpha_min, ' and ', p0_min, ' p0_start: ',p0_start)
    p0sL = []
    alphasL = []
    DsL = []
    alpha_mod = []
    DsL_mod = []

    Delta = 0.001
    alpha_max = 0.

    if (alpha_min > 0):
        alpha_max = alpha_min+NN*Delta/2.
    else:
        alpha_max = alpha_min-NN*Delta/2.

    p0 = p0_solver(k, alpha_max,i_arr, weights)
    print ('Found p0=',p0)
    D_max = -2.*(calc_loglikelihood(k, alpha_max, p0, weights)-L_min) 

    print ('Found Ds=',D_max, ' need to achieve:', chi2_max)
    Delta = Delta * np.sqrt(chi2_max/D_max) * 1.02

    if (np.sqrt(chi2_max/D_max) < 0.8 or np.sqrt(chi2_max/D_max) > 1.2):
        # too high risk for deviation from parabaola, re-scale again with a test
        if (alpha_min > 0):
            alpha_max = alpha_min+NN*Delta/2.
        else:
            alpha_max = alpha_min-NN*Delta/2.
            
        p0 = p0_solver(k, alpha_max,i_arr, weights)
        D_max  = -2.*(calc_loglikelihood(k, alpha_max, p0, weights)-L_min) 
        # re-scale Delta to achieved maximum of D close to chi2
        print ('Found new Ds_max=',D_max, ' need to achieve:', chi2_max)
        Delta = Delta * np.sqrt(chi2_max/D_max) * 1.02

    
    for alpha in np.arange(alpha_min-NN*Delta/2., alpha_min + NN*Delta/2., Delta):

        alphasL.append(alpha)

        p0_start = np.sum(k)/N
        if weights is not None:
            p0_start = np.sum(k)/np.sum(weights)

        p0 = fsolve(lambda x: likelihood_solver_profile(x, k, alpha, weights), p0_start)

        L = calc_loglikelihood(k, alpha, p0, weights)
        print ('alpha: ', alpha, ' p0: ',p0, ' Ds: ',-2.*(L-L_min))

        fac = 1.00000000001
        if weights is not None:
            #    p_i = (p0 + alpha_i_arr) * weights
            if weights[0] > weights[-1]:
                fac = fac/weights[0]
            else:
                fac = fac/weights[-1]

        if (np.any((p0 + alpha*i_arr) < 0.)):
            p0 = (N-1)/2 * np.abs(alpha) * fac
            alpha_mod.append(alpha)
            L = calc_loglikelihood(k, alpha, p0, weights)            
            DsL_mod.append(-2.*(L-L_min))
            print ('new p0: ',p0, 'new Ds: ',-2.*(L-L_min))

        p0sL.append(p0)
        DsL.append(-2.*(L-L_min))

    alphas = np.array(alphasL)
    Ds     = np.array(DsL)

    alphas_pos = alphas[alphas > alpha_min]
    alphas_neg = alphas[alphas < alpha_min]

    Ds_pos = Ds[alphas > alpha_min]
    Ds_neg = Ds[alphas < alpha_min]

    alphas_pos_fit = alphas_pos[Ds_pos < 1]
    alphas_neg_fit = alphas_neg[Ds_neg < 1]

    Ds_pos_fit = Ds_pos[Ds_pos < 1]
    Ds_neg_fit = Ds_neg[Ds_neg < 1]

    alphas_fit = np.append(alphas_neg_fit, alphas_pos_fit)
    Ds_fit = np.append(Ds_neg_fit, Ds_pos_fit)

    #print ('HERE',alphas_fit, Ds_fit)

    popt, pcov = curve_fit(quad_func, alphas_fit, Ds_fit, p0=[alpha_min,0.01])
    Nexp = quad_func(alphas_fit, *popt)
    Nexp_all = quad_func(alphas, *popt)

    plt.clf()
    plt.scatter(alphasL, Ds, s = 12)
    plt.scatter(alpha_mod, DsL_mod, s = 12)
    plt.plot(alphas_fit,Nexp,'r-',lw=4,label=r'local parabola fit: $\mu$=%.3f/y, $\sigma$=%.3f/y' % tuple(popt))
    plt.plot(alphas,Nexp_all,'r--')
    plt.xlabel(r'$\alpha$ (yearly occurrence increase)', fontsize = 25)
    plt.ylabel(r'$D(\alpha)$', fontsize = 25)
    ax = plt.gca()
    ax.yaxis.set_tick_params(labelsize=25)
    ax.xaxis.set_tick_params(labelsize=25)
    plt.legend(loc='best', fontsize=25)
