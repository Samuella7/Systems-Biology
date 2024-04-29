import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.optimize import fmin


## ----- progression d'un pathogene
# (a) equation d'evo
# dP/dt = rho * P - mu* P
# on peut resoudre explicitement : P(t) = P0 exp( (rho-mu)*t)
# dynamique exponentielle, croissante si rho > mu et décroissante sinon
rho = 0.5
mu = 0.1
P0 = 1
t = np.linspace(0, 10, 100)
P = P0 * np.exp((rho-mu)*t)
plt.figure()
plt.plot(t, P,label = 'question (a)')
plt.xlabel('temps')
plt.ylabel('pop pathogene')
plt.legend()

#(b) ajout d'un substrant S
# dP/dt = rho * P - mu*P/S
# dS/dt = r - d*P*S

#(c) equilibre
# S = mu/rho
# P = r*rho / (d * mu)

def model_patho(X, t, param): ## la croissance du pathogene n'est plus exponentielle
    P,S = X
    rho, mu, r, d = param

    dP = rho*P - mu*P/S
    dS = r - d*P*S
    
    return [dP, dS]

# parametres tests
X0 = [1, 1]
rho = 0.05
mu = 0.01
r = 2.0
d = 1.5

parametres= (rho, mu, r, d)
Tf = 200
t = np.linspace(0, Tf, Tf*10)

X = odeint(model_patho, X0,t, args = (parametres,))

Seq = mu/rho
Peq = r*rho/(d*mu)

plt.figure()
plt.subplot(211)
plt.plot(t, X[:,0], 'r', label = 'patho')
plt.plot(t, Peq*np.ones(len(t)), '--')
plt.legend()

plt.subplot(212)
plt.plot(t, X[:,1], 'b', label = 'substrat')
plt.plot(t, Seq*np.ones(len(t)), '--')

plt.legend()

# (d) avec le calcul de l'equilibre on a que 1/S est proportionnel à P
# on peut re-ecrire le model avec une unique equation sur P et un terme de mort prop à P**2
def model_patho2(X, t, rho, K) :
    dX = rho*X*(1 - X/K)
    return dX

K = r*rho/(d*mu) # valeur de K pour avoir le meme equilibre pour les 2 modeles
X20 = 1.
X2 = odeint(model_patho2, X20,t, args = (rho, K, ))

plt.figure()
plt.plot(t, X2, label = 'model2')
plt.plot(t, X[:,0], '--', label = 'model1')
plt.legend()

#(f) ajustement sur la pop americaine en million
data = np.zeros((21, 2))
data[:,0] = np.arange(1790,1991, 10)
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

#plot les donnees
plt.figure()
plt.plot(data[:,0], data[:, 1], 'o', label = 'pop americaine')

# on cherche rho et K pour approcher au mieux les donnees
# approcher au mieux signifie rendre la distance suivante la plus petite
def dist(param, data):
    rho, K = param
    x0 =  data[0,1] # je prend x0 = le premier point de donnee
    t = data[:, 0] 
    model = odeint(model_patho2, x0, t, args = (rho, K, ))
    dist = np.sum( (model[:,0] - data[:,1])**2 )
    return dist

# initial guess
rho0 = 0.1
K0 = 300
paramopt = fmin(dist, [rho0, K0], args = (data,)) # minimisation
print('param opt = ', paramopt)

# on compare le model aux donnees
rho_op = paramopt[0]
K_op =  paramopt[1]
x0 =  data[0,1]
t = np.arange(1790, 2021, 1)
sol = odeint(model_patho2, x0, t, args = (rho_op, K_op, ))

plt.plot(t, sol, label = 'model')
plt.legend()

## En 2020, la population américaine est de 331.449 milions
print('en 2020 : 331.449 milions, et le modèle prédit : ', sol[-1])

## ----- Inflammation
def func_f(m, theta, w) :
    return 1 + np.tanh((m-theta)/w)
    

def model_infla(X, t, param) : #question (a)
    p, m, l = X
    kp, kpm, kmp,klm, kl = param
    theta = 1.0
    w = 0.5
    dp = kp*p*(1-p)-kpm*p*m #croissance logistique + destruction par m
    dm = (kmp*p + l)*m*(1-m) - m # recrutement (croissance) qui depend des pathogenes p + logistique, degradation normale
    dl = klm*func_f(m, theta, w) - kl*l # croissance stimullée par m avec un seuil : fonction f, degradation normale
    return [dp, dm, dl]

#question (b) : on trace la fonction f
# on remarque que theta correspond au point d'infection et w change la pente, la raideur
x = np.linspace(0, 10, 100)
plt.figure()
plt.plot(x, func_f(x, 1.0, 0.5), label = 'theta = 1.0, w = 0.5')
plt.plot(x, func_f(x, 1.5, 0.5), label = 'theta = 1.5, w = 0.5')
plt.plot(x, func_f(x, 0.5, 0.5), label = 'theta = 0.5, w = 0.5')
plt.plot(x, func_f(x, 1.0, 1.5), label = 'theta = 1, w = 1.5')
plt.plot(x, func_f(x, 1.0, 0.1), label = 'theta = 1, w = 0.1')
plt.legend()

plt.figure() # on remarque que les deux courbes on une allure similaire, la fonction f à pour max 2, alors que les fonction x^n / (1+x^n) le max est 1. si on regarde la fonction 2*x^n / (1+x^n) avec n grand, on s'approche de la fonction f
plt.plot(x, func_f(x, 1.0, 0.5), label = 'f')
plt.plot(x, x**2/(1+x**2), label = 'x^2 / (1+x^2)')
plt.plot(x, x**3/(1+x**3), label = 'x^3 / (1+x^3)')
plt.plot(x, x**4/(1+x**4), label = 'x^4 / (1+x^4)')
plt.legend()

#question (c) - resoudre le system
#choix de paramètres
kp = 20
kpm = 2
kmp = 20
klm = 10
kl =  10
param = (kp, kpm, kmp, klm, kl)

x0 = [0.01, 0.001, 0]
t = np.linspace(0, 20, 200)
kumar = odeint(model_infla, x0, t, args = (param,))

plt.figure()
plt.plot(t, kumar[:,0], label = 'pathogene')
plt.plot(t, kumar[:,1], label = 'pro-inflammation')
plt.plot(t, kumar[:,2], label = 'reponse retardée')
plt.legend()
plt.xlabel('temps')


## - cas sain
# dans un premier temps on observe une solution periodique, il y a un controle du pathogene avec les réponses inflammatoire, mais celui ci n'est jamais éliminé, donc une fois la réponse inflammatoire passée, le pathogène reapparait.
# Si on attend plus longtemps, la réponse inflammatoire "saute" vers un equilibre positif et le pathogène est eliminé.
t = np.linspace(0, 100, 1000)

paramS = (3., 30.0, 25., 15., 1.)

x0 = [0.01, 0.05, 0.539]
casS = odeint(model_infla, x0, t, args = (paramS,))

plt.figure()
plt.plot(t, casS[:,0], label = 'pathogene')
plt.plot(t, casS[:,1], label = 'pro-inflammation')
plt.plot(t, casS[:,2], label = 'reponse retardée')
plt.legend()
plt.title('cas sain')
plt.xlabel('temps')


## - cas inflammation persistante infectieuse
# un etat d'equilibre constant est atteint, le pathogene est controlé à un niveau très faible, mais l'inflammation persiste et est très elevée.

paramIPI = (3., 15.0, 25., 15., 1.)

casIPI = odeint(model_infla, x0, t, args = (paramIPI,))

plt.figure()
plt.plot(t, casIPI[:,0], label = 'pathogene')
plt.plot(t, casIPI[:,1], label = 'pro-inflammation')
plt.plot(t, casIPI[:,2], label = 'reponse retardée')
plt.legend()
plt.title('cas inflammation persistante infectieuse')
plt.xlabel('temps')


## - cas inflammation persistante non-infectieuse.
# un étitat d'équilible constant est atteint, le pathogene est controlé très rapidement, mais le systeme s'emballe et la réponse inflammatoire est très grande et persistante

x1_0 = [0.1, 0.05, 0.539] # on augmente les pathogenes present au debut
x2_0 = [0.2, 0.05, 0.539] # on augmente les pathogenes present au debut
x3_0 = [1.0, 0.05, 0.539] # on augmente les pathogenes present au debut
x1 = odeint(model_infla, x1_0, t, args = (paramS,))# on garde les paramètres du cas sain
x2 = odeint(model_infla, x2_0, t, args = (paramS,))# on garde les paramètres du cas sain
x3 = odeint(model_infla, x3_0, t, args = (paramS,))# on garde les paramètres du cas sain

plt.figure(figsize=(11,7))
plt.title('cas inflammation persistante non-infectieuse')
plt.subplot(311)
plt.title('pathogene')
plt.plot(t, x1[:,0], label = 'p0 = 0.1')
plt.plot(t, x2[:,0], label = 'p0 = 0.2')
plt.plot(t, x3[:,0], label = 'p0 = 1.0')
plt.legend()
plt.subplot(312)
plt.title('pro-inflammation')
plt.plot(t, x1[:,1])
plt.plot(t, x2[:,1])
plt.plot(t, x3[:,1])

plt.subplot(313)
plt.title('reponse retardée')
plt.plot(t, x1[:,2])
plt.plot(t, x2[:,2])
plt.plot(t, x3[:,2])

plt.xlabel('temps')
plt.tight_layout()
## - cas Immuno-deficience
# on observe que plus on reduit kmp, plus le pathogene se maintient à un niveau élevé et les agents pro-inflammation des deux réponses sont faible. On peut observer un état d'équilibre constant.
t = np.linspace(0, 50, 500)
paramID = (3., 30.0, 0.1, 15., 1.) # reduction de la capacité de destruction du pathogene par les agent inflammatoire

x0 = [0.01, 0.05, 0.539]
casID = odeint(model_infla, x0, t, args = (paramID,))

plt.figure()
plt.plot(t, casID[:,0], label = 'pathogene')
plt.plot(t, casID[:,1], label = 'pro-inflammation')
plt.plot(t, casID[:,2], label = 'reponse retardée')
plt.legend()
plt.title('cas immuno-deficience')
plt.xlabel('temps')

plt.show()
