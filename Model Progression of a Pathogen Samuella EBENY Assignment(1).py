import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.optimize import fmin


## ----- Progression of a Pathogen
# (a) Evolution equation
# dP/dt = rho * P - mu* P
# We can solve explicitly: P(t) = P0 exp( (rho-mu)*t)
# Exponential dynamics, increasing if rho > mu and decreasing otherwise
rho = 0.5
mu = 0.1
P0 = 1
t = np.linspace(0, 10, 100)
P = P0 * np.exp((rho-mu)*t)
plt.figure()
plt.plot(t, P,label = 'question (a)')
plt.xlabel('time')
plt.ylabel('pathogen population')
plt.legend()

# (b) Addition of a substrate S
# dP/dt = rho * P - mu*P/S
# dS/dt = r - d*P*S

# (c) Equilibrium
# S = mu/rho
# P = r*rho / (d * mu)

def model_patho(X, t, param): ## Pathogen growth is no longer exponential
    P,S = X
    rho, mu, r, d = param

    dP = rho*P - mu*P/S
    dS = r - d*P*S
    
    return [dP, dS]

# Test parameters
X0 = [1, 1]
rho = 0.05
mu = 0.01
r = 2.0
d = 1.5

parameters = (rho, mu, r, d)
Tf = 200
t = np.linspace(0, Tf, Tf*10)

X = odeint(model_patho, X0, t, args=(parameters,))

Seq = mu/rho
Peq = r*rho/(d*mu)

plt.figure()
plt.subplot(211)
plt.plot(t, X[:,0], 'r', label='pathogen')
plt.plot(t, Peq*np.ones(len(t)), '--')
plt.legend()

plt.subplot(212)
plt.plot(t, X[:,1], 'b', label='substrate')
plt.plot(t, Seq*np.ones(len(t)), '--')

plt.legend()

# (d) With equilibrium calculation, 1/S is proportional to P
# We can rewrite the model with a single equation for P and a death term proportional to P**2
def model_patho2(X, t, rho, K) :
    dX = rho*X*(1 - X/K)
    return dX

K = r*rho/(d*mu) # Value of K to have the same equilibrium for both models
X20 = 1.
X2 = odeint(model_patho2, X20, t, args=(rho, K, ))

plt.figure()
plt.plot(t, X2, label='model2')
plt.plot(t, X[:,0], '--', label='model1')
plt.legend()

# (f) Fitting on the American population in millions
data = np.zeros((21, 2))
data[:,0] = np.arange(1790, 1991, 10)
data[:,1] =[ 3.929,
             5.308,
             7.240,
             9.638,
             12.866,
             17.069,
             23.192,
             31.443,
             38.558,
             50.156,
             62.948,
             75.996,
             91.972,
             105.711,
             122.775,
             131.669,
             150.697,
             179.323,
             203.185,
             226.546,
             248.710]

# Plot the data
plt.figure()
plt.plot(data[:,0], data[:, 1], 'o', label='American population')

# Searching rho and K to best fit the data
def dist(param, data):
    rho, K = param
    x0 = data[0,1] # Taking x0 as the first data point
    t = data[:, 0] 
    model = odeint(model_patho2, x0, t, args=(rho, K, ))
    dist = np.sum( (model[:,0] - data[:,1])**2 )
    return dist

# Initial guess
rho0 = 0.1
K0 = 300
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.optimize import fmin


## ----- Progression of a Pathogen
# (a) Evolution equation
# dP/dt = rho * P - mu* P
# We can solve explicitly: P(t) = P0 exp( (rho-mu)*t)
# Exponential dynamics, increasing if rho > mu and decreasing otherwise
rho = 0.5
mu = 0.1
P0 = 1
t = np.linspace(0, 10, 100)
P = P0 * np.exp((rho-mu)*t)
plt.figure()
plt.plot(t, P,label = 'question (a)')
plt.xlabel('time')
plt.ylabel('pathogen population')
plt.legend()

# (b) Addition of a substrate S
# dP/dt = rho * P - mu*P/S
# dS/dt = r - d*P*S

# (c) Equilibrium
# S = mu/rho
# P = r*rho / (d * mu)

def model_patho(X, t, param): ## Pathogen growth is no longer exponential
    P,S = X
    rho, mu, r, d = param

    dP = rho*P - mu*P/S
    dS = r - d*P*S
    
    return [dP, dS]

# Test parameters
X0 = [1, 1]
rho = 0.05
mu = 0.01
r = 2.0
d = 1.5

parameters = (rho, mu, r, d)
Tf = 200
t = np.linspace(0, Tf, Tf*10)

X = odeint(model_patho, X0, t, args=(parameters,))

Seq = mu/rho
Peq = r*rho/(d*mu)

plt.figure()
plt.subplot(211)
plt.plot(t, X[:,0], 'r', label='pathogen')
plt.plot(t, Peq*np.ones(len(t)), '--')
plt.legend()

plt.subplot(212)
plt.plot(t, X[:,1], 'b', label='substrate')
plt.plot(t, Seq*np.ones(len(t)), '--')

plt.legend()

# (d) With equilibrium calculation, 1/S is proportional to P
# We can rewrite the model with a single equation for P and a death term proportional to P**2
def model_patho2(X, t, rho, K) :
    dX = rho*X*(1 - X/K)
    return dX

K = r*rho/(d*mu) # Value of K to have the same equilibrium for both models
X20 = 1.
X2 = odeint(model_patho2, X20, t, args=(rho, K, ))

plt.figure()
plt.plot(t, X2, label='model2')
plt.plot(t, X[:,0], '--', label='model1')
plt.legend()

# (f) Fitting on the American population in millions
data = np.zeros((21, 2))
data[:,0] = np.arange(1790, 1991, 10)
data[:,1] =[ 3.929,
             5.308,
             7.240,
             9.638,
             12.866,
             17.069,
             23.192,
             31.443,
             38.558,
             50.156,
             62.948,
             75.996,
             91.972,
             105.711,
             122.775,
             131.669,
             150.697,
             179.323,
             203.185,
             226.546,
             248.710]

# Plot the data
plt.figure()
plt.plot(data[:,0], data[:, 1], 'o', label='American population')

# Searching rho and K to best fit the data
def dist(param, data):
    rho, K = param
    x0 = data[0,1] # Taking x0 as the first data point
    t = data[:, 0] 
    model = odeint(model_patho2, x0, t, args=(rho, K, ))
    dist = np.sum( (model[:,0] - data[:,1])**2 )
    return dist

# Initial guess
rho0 = 0.1
K0 = 300
param_opt = fmin(dist, [rho0, K0], args=(data,)) # Minimization
print('optimal parameters = ', param_opt)

# Comparing the model with the data
rho_op = param_opt[0]
K_op =  param_opt[1]
x0 =  data[0,1]
t = np.arange(1790, 2021, 1)
sol = odeint(model_patho2, x0, t, args=(rho_op, K_op, ))

plt.plot(t, sol, label='model')
plt.legend()

## Inflammation
# (a) Plotting the function f
# We observe that theta corresponds to the infection point and w changes the slope, the steepness
x = np.linspace(0, 10, 100)
plt.figure()
plt.plot(x, func_f(x, 1.0, 0.5), label='theta = 1.0, w = 0.5')
plt.plot(x, func_f(x, 1.5, 0.5), label='theta = 1.5, w = 0.5')
plt.plot(x, func_f(x, 0.5, 0.5), label='theta = 0.5, w = 0.5')
plt.plot(x, func_f(x, 1.0, 1.5), label='theta = 1, w = 1.5')
plt.plot(x, func_f(x, 1.0, 0.1), label='theta = 1, w = 0.1')
plt.legend()

plt.figure()
plt.plot(x, func_f(x, 1.0, 0.5), label='f')
plt.plot(x, x**2/(1+x**2), label='x^2 / (1+x^2)')
plt.plot(x, x**3/(1+x**3), label='x^3 / (1+x^3)')
plt.plot(x, x**4/(1+x**4), label='x^4 / (1+x^4)')
plt.legend()

# (c) Solving the system
# Choosing parameters
kp = 20
kpm = 2
kmp = 20
klm = 10
kl =  10
param = (kp, kpm, kmp, klm, kl)

x0 = [0.01, 0.001, 0]
t = np.linspace(0, 20, 200)
kumar = odeint(model_infla, x0, t, args=(param,))

plt.figure()
plt.plot(t, kumar[:,0], label='pathogen')
plt.plot(t, kumar[:,1], label='pro-inflammation')
plt.plot(t, kumar[:,2], label='delayed response')
plt.legend()
plt.xlabel('time')

## Healthy Case
# Initially, we observe a periodic solution, there is pathogen control with inflammatory responses, but it is never eliminated, so once the inflammatory response passes, the pathogen reappears.
# If we wait longer, the inflammatory response "jumps" to a positive equilibrium and the pathogen is eliminated.
t = np.linspace(0, 100, 1000)

paramS = (3., 30.0, 25., 15., 1.)

x0 = [0.01, 0.05, 0.539]
casS = odeint(model_infla, x0, t, args=(paramS,))

plt.figure()
plt.plot(t, casS[:,0], label='pathogen')
plt.plot(t, casS[:,1], label='pro-inflammation')
plt.plot(t, casS[:,2], label='delayed response')
plt.legend()
plt.title('Healthy Case')
plt.xlabel('time')

## Persistent Infectious Inflammation Case
# A constant equilibrium state is reached, the pathogen is controlled at a very low level, but inflammation persists and is very high.
paramIPI = (3., 15.0, 25., 15., 1.)

casIPI = odeint(model_infla, x0, t, args=(paramIPI,))

plt.figure()
plt.plot(t, casIPI[:,0], label='pathogen')
plt.plot(t, casIPI[:,1], label='pro-inflammation')
plt.plot(t, casIPI[:,2], label='delayed response')
plt.legend()
plt.title('Persistent Infectious Inflammation Case')
plt.xlabel('time')

## Persistent Non-Infectious Inflammation Case
# A constant equilibrium state is reached, the pathogen is controlled very quickly, but the system spirals out of control and the inflammatory response is very large and persistent.
x1_0 = [0.1, 0.05, 0.539] # Increase in initial pathogen levels
x2_0 = [0.2, 0.05, 0.539] # Increase in initial pathogen levels
x3_0 = [1.0, 0.05, 0.539] # Increase in initial pathogen levels
x1 = odeint(model_infla, x1_0, t, args=(paramS,)) # Keeping parameters of the healthy case
x2 = odeint(model_infla, x2_0, t, args=(paramS,)) # Keeping parameters of the healthy case
x3 = odeint(model_infla, x3_0, t, args=(paramS,)) # Keeping parameters of the healthy case

plt.figure(figsize=(11,7))
plt.title('Persistent Non-Infectious Inflammation Case')
plt.subplot(311)
plt.title('pathogen')
plt.plot(t, x1[:,0], label='p0 = 0.1')
plt.plot(t, x2[:,0], label='p0 = 0.2')
plt.plot(t, x3[:,0], label='p0 = 1.0')
plt.legend()
plt.subplot(312)
plt.title('pro-inflammation')
plt.plot(t, x1[:,1])
plt.plot(t, x2[:,1])
plt.plot(t, x3[:,1])

plt.subplot(313)
plt.title('delayed response')
plt.plot(t, x1[:,2])
plt.plot(t, x2[:,2])
plt.plot(t, x3[:,2])

plt.xlabel('time')
plt.tight_layout()

## Immune-Deficiency Case
# We observe that the lower kmp is, the higher the pathogen is maintained, and the pro-inflammatory agents of both responses are weak. We can observe a constant equilibrium state.
t = np.linspace(0, 50, 500)
paramID = (3., 30.0, 0.1, 15., 1.) # Reduction in the capacity of inflammatory agents to destroy the pathogen

x0 = [0.01, 0.05, 0.539]
casID = odeint(model_infla, x0, t, args=(paramID,))

plt.figure()
plt.plot(t, casID[:,0], label='pathogen')
plt.plot(t, casID[:,1], label='pro-inflammation')
plt.plot(t, casID[:,2], label='delayed response')
plt.legend()
plt.title('Immune-Deficiency Case')
plt.xlabel('time')

plt.show()
= fmin(dist, [rho0, K0], args=(data,)) # Minimization
print('optimal parameters = ', param_opt)

# Comparing the model with the data
rho_op = param_opt[0]
K_op =  param_opt[1]
x0 =  data[0,1]
t = np.arange(1790, 2021, 1)
sol = odeint(model_patho2, x0, t, args=(rho_op, K_op, ))

plt.plot(t, sol, label='model')
plt.legend()

## Inflammation
# (a) Plotting the function f
# We observe that theta corresponds to the infection point and w changes the slope, the steepness
x = np.linspace(0, 10, 100)
plt.figure()
plt.plot(x, func_f(x, 1.0, 0.5), label='theta = 1.0, w = 0.5')
plt.plot(x, func_f(x, 1.5, 0.5), label='theta = 1.5, w = 0.5')
plt.plot(x, func_f(x, 0.5, 0.5), label='theta = 0.5, w = 0.5')
plt.plot(x, func_f(x, 1.0, 1.5), label='theta = 1, w = 1.5')
plt.plot(x, func_f(x, 1.0, 0.1), label='theta = 1, w = 0.1')
plt.legend()

plt.figure()
plt.plot(x, func_f(x, 1.0, 0.5), label='f')
plt.plot(x, x**2/(1+x**2), label='x^2 / (1+x^2)')
plt.plot(x, x**3/(1+x**3), label='x^3 / (1+x^3)')
plt.plot(x, x**4/(1+x**4), label='x^4 / (1+x^4)')
plt.legend()

# (c) Solving the system
# Choosing parameters
kp = 20
kpm = 2
kmp = 20
klm = 10
kl =  10
param = (kp, kpm, kmp, klm, kl)

x0 = [0.01, 0.001, 0]
t = np.linspace(0, 20, 200)
kumar = odeint(model_infla, x0, t, args=(param,))

plt.figure()
plt.plot(t, kumar[:,0], label='pathogen')
plt.plot(t, kumar[:,1], label='pro-inflammation')
plt.plot(t, kumar[:,2], label='delayed response')
plt.legend()
plt.xlabel('time')

## Healthy Case
# Initially, we observe a periodic solution, there is pathogen control with inflammatory responses, but it is never eliminated, so once the inflammatory response passes, the pathogen reappears.
# If we wait longer, the inflammatory response "jumps" to a positive equilibrium and the pathogen is eliminated.
t = np.linspace(0, 100, 1000)

paramS = (3., 30.0, 25., 15., 1.)

x0 = [0.01, 0.05, 0.539]
casS = odeint(model_infla, x0, t, args=(paramS,))

plt.figure()
plt.plot(t, casS[:,0], label='pathogen')
plt.plot(t, casS[:,1], label='pro-inflammation')
plt.plot(t, casS[:,2], label='delayed response')
plt.legend()
plt.title('Healthy Case')
plt.xlabel('time')

## Persistent Infectious Inflammation Case
# A constant equilibrium state is reached, the pathogen is controlled at a very low level, but inflammation persists and is very high.
paramIPI = (3., 15.0, 25., 15., 1.)

casIPI = odeint(model_infla, x0, t, args=(paramIPI,))

plt.figure()
plt.plot(t, casIPI[:,0], label='pathogen')
plt.plot(t, casIPI[:,1], label='pro-inflammation')
plt.plot(t, casIPI[:,2], label='delayed response')
plt.legend()
plt.title('Persistent Infectious Inflammation Case')
plt.xlabel('time')

## Persistent Non-Infectious Inflammation Case
# A constant equilibrium state is reached, the pathogen is controlled very quickly, but the system spirals out of control and the inflammatory response is very large and persistent.
x1_0 = [0.1, 0.05, 0.539] # Increase in initial pathogen levels
x2_0 = [0.2, 0.05, 0.539] # Increase in initial pathogen levels
x3_0 = [1.0, 0.05, 0.539] # Increase in initial pathogen levels
x1 = odeint(model_infla, x1_0, t, args=(paramS,)) # Keeping parameters of the healthy case
x2 = odeint(model_infla, x2_0, t, args=(paramS,)) # Keeping parameters of the healthy case
x3 = odeint(model_infla, x3_0, t, args=(paramS,)) # Keeping parameters of the healthy case

plt.figure(figsize=(11,7))
plt.title('Persistent Non-Infectious Inflammation Case')
plt.subplot(311)
plt.title('pathogen')
plt.plot(t, x1[:,0], label='p0 = 0.1')
plt.plot(t, x2[:,0], label='p0 = 0.2')
plt.plot(t, x3[:,0], label='p0 = 1.0')
plt.legend()
plt.subplot(312)
plt.title('pro-inflammation')
plt.plot(t, x1[:,1])
plt.plot(t, x2[:,1])
plt.plot(t, x3[:,1])

plt.subplot(313)
plt.title('delayed response')
plt.plot(t, x1[:,2])
plt.plot(t, x2[:,2])
plt.plot(t, x3[:,2])

plt.xlabel('time')
plt.tight_layout()

## Immune-Deficiency Case
# We observe that the lower kmp is, the higher the pathogen is maintained, and the pro-inflammatory agents of both responses are weak. We can observe a constant equilibrium state.
t = np.linspace(0, 50, 500)
paramID = (3., 30.0, 0.1, 15., 1.) # Reduction in the capacity of inflammatory agents to destroy the pathogen

x0 = [0.01, 0.05, 0.539]
casID = odeint(model_infla, x0, t, args=(paramID,))

plt.figure()
plt.plot(t, casID[:,0], label='pathogen')
plt.plot(t, casID[:,1], label='pro-inflammation')
plt.plot(t, casID[:,2], label='delayed response')
plt.legend()
plt.title('Immune-Deficiency Case')
plt.xlabel('time')

plt.show()
