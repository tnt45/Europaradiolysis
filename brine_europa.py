### OXIDANTS PRODUCTION EUROPA'S SALTY BRINES ###

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from math import pi 
matplotlib.rcParams['text.usetex'] = True

R = 8.314

# apply to the earth's ocean
ml_kearth = 0.4                                 # molality of potassium in earth's ocean (0.4 g/kg)
muy_o2 = 32e-3                                  # molecular mass of oxygen in kg/mol
rho_w = 1000                                    # density of water (kg/m3)


# parameters for potassium
f40k_c = 0.0117e-2                              # mass fraction of 40K per K today
f40k_past = f40k_c * 12                          # mass fraction of 40K 4.5 Gyr
lda_40k = 5.54e-10                              # the decay parameter lambda in yr-1 
Go2_beta = 0.22                                  # molecules/eV
Ea_40k = 0.566e+6                               # converting MeV to eV
muy_40k = 40                                    # g/mol

f_ar = 0.11                                     # fraction of 40K decay to 40Ar

#Q40k_past = ml_kearth * f40k_past * moc_earth   # mass of 40K in Earth's ocean 3.8 Gyr ago
r_o2 = 10**-6.17                                 # production rate of o2 production from 40k (Draganic+1991)
r_h2o2 = 10**-7.63                              # production rate of h2o2 production from 40k (Draganic+1991)

# parameters for uranium-238
ml_238u_earth = 3e-6                            # molality of 238U in earth's ocean today (g/kg) (Choppin, G., Liljenzin, J-O., Rydberg, J. Radiochemistry and nuclear chemistry. Butterworth-Heinemann, (2002))
ml_238u_earthpast = 1.8 * ml_238u_earth         # molality of 238U in earth's ocean 3.8 Gyr ago (1.8 times higher than today)
#Q238u_past = ml_238u_earthpast * moc_earth      # mass of 238U in Earth's ocean 3.8 Gyr ago
muy_238u = 238                                  # g/mol 
lda_238u = 1.55e-10                             # decay parameter lambda in yr-1 for 238u

Go2_alpha = 0.7                                   # yield of o2 by alpha particle 
Ealpha_238u = 45e+6                             # converting MeV to eV
Ebeta_238u = 7.2e+6

# parameters for uranium-235
ml_235u_earth = 0.022e-6                          # molality of 235U in earth's ocean today (g/kg) (Chen, J. H. & Wasserburg, G. J. Precise isotopic analysis of uranium in picomole and subpicomole quantities. Anal. Chem. 53.13, 2060-2067 (1981).)
ml_235u_earthpast = 43 * ml_235u_earth          # molality of 235U in earth's ocean 3.8 gyr ago (g/kg, 43 times higher than today)
#Q235u_past = ml_235u_earthpast * moc_earth      # mass of 238U in Earth's ocean 3.8 Gyr ago
#Q235u_past = 43*3.3e+10
muy_235u = 235
lda_235u = 9.72e-10                             # decay parameter for uranium-235 (yr-1)

Ealpha_235u = 43e+6
Ebeta_235u = 3.2e+6

# parameters for throium-232
muy_232th = 232
ml_232th_earth = 0.05e-6                        # 232Th in earth's ocean today (g/kg)
ml_232th_earthpast = 1.21 * ml_232th_earth
lda_232th = 4.693e-11                           # decay parameter for 232 thorium
Ealpha_232th = 37e+6
Ebeta_232th = 6.0e+6


# oxidants production in water reservoir (kg) by Ershov: 
# G_o2 (radiolytic yield per 100 eV), Ea: average energy release per decay
# ml: the initial molality of radioactive element (g/kg), for 40K: needs to multiply K per abundance with fraction of 40K/K
# m_brine: mass of the reservoir (kg)
# muy_iz: molecular mass of each radioactive element (g/mol)
# lda: the lambda decay constant

def years_to_100000yr(x, pos):
    return f"{x/1e5:g}"


def prod_o2(G_o2,Ea, ml, V_brine, muy_iz,lda,t):
    V_brinem3 = V_brine * 1e+9                                                  # converting km^3 to m^3
    m_brine = rho_w * V_brinem3 
    Qiz = ml * m_brine                                                          
    m_o2 = G_o2 * (Ea/100) * (Qiz/muy_iz) * muy_o2 * (1 - np.exp(-lda*t))
    n_o2 = G_o2 * (Ea/100) * (Qiz/muy_iz) * (1 - np.exp(-lda*t))                # total accumulation of O2 production (mol)
    r_o2 = G_o2 * (Ea/100) * (Qiz/muy_iz) * lda * np.exp(-lda*t)                # rate of O2 production
    n_h2 = n_o2*2
    c_o2 = n_o2 / m_brine
    c_h2 = n_h2 /m_brine
    return c_o2,c_h2,r_o2

def argon(ml, V_brine, muy_iz,lda,t):
    V_brinem3 = V_brine * 1e+9                                                  # converting km^3 to m^3
    m_brine = rho_w * V_brinem3 
    Qiz = ml * m_brine  
    n_40ar = (Qiz/muy_iz) * (np.exp(lda*t)-1) * f_ar                         # f_ar: fraction of 40K decay to 40Ar
    return n_40ar

# radiolytic production in water reservoir (Draganic+1991)

def prod91_o2(ro2,rh2o2,ml,V_brine,muy_iz,lda,t):
    V_brinem3 = V_brine * 1e+9                                                  # converting km^3 to m^3
    m_brine = rho_w * V_brinem3 
    Qiz = ml * m_brine
    m_o2 = muy_o2 * ro2 * (Qiz/muy_iz) * (1 - np.exp(-lda*t)) / lda
    n_o2 = ro2 * (Qiz/muy_iz) * (1 - np.exp(-lda*t)) / lda
    n_h2 = n_o2*1.995
    n_h2o2 = rh2o2 * (Qiz/muy_iz) * (1 - np.exp(-lda*t)) / lda
    c_o2 = n_o2 / m_brine
    c_h2o2 = n_h2o2 / m_brine
    c_h2 = n_h2 /m_brine
    return c_o2,c_h2,c_h2o2

'''
# the dependence of o2 production rate & molality of potassium in the reservoir
def rate(ml, t):
    r_o2 = Go2_beta * (Ea_40k/100) * (ml/muy_40k) * lda_40k * np.exp(-lda_40k*t)
    return r_o2 

tpast = 4.5e+9
tau = 2e+6
ml_k = np.arange(0,1,0.001)
rateo2_40know_lake = rate(f40k_c * ml_k,1e+6)
rateo2_40kpast_ocean = rate(f40k_past * ml_k,0)
rateo2_40know_ocean = rate(f40k_past * ml_k,tpast)


fig1a, ax1a = plt.subplots()
font = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 14,
        }

ax1a.semilogy(ml_k,rateo2_40know_lake,color='blue',label='Lake (Modern)',)
ax1a.semilogy(ml_k,rateo2_40know_ocean,color='orange',label='Ocean (Modern)',linestyle='-.')
ax1a.semilogy(ml_k,rateo2_40kpast_ocean,color='orange',label='Ocean (4.5 Gyrs ago)')
ax1a.set_xlabel("Molality of Potassium ($g.kg^{-1}$)",size=25)
ax1a.set_ylabel("$O_2$ production ($mol.yr^{-1}.kg^{-1}$)",size=25)
plt.axvspan(0.04, 0.4, color='gray', alpha=0.3, label='This study')
plt.legend()
'''

tau = 2e+6
t = np.arange(0,tau,5e+4)           # time in years

''' Oxygen, and hydron production from different radioactive decay for a nominal case lake (20000 km^3) '''

### Earth's modern seawater salinity ###
o2n_40kearth,h2n_40kearth,ro2_40kearth = prod_o2(Go2_beta,Ea_40k,f40k_c * ml_kearth,20000,muy_40k,lda_40k,t)
o2n_40k_d91,h2_40k_d91,h2o2_40k_d91 = prod91_o2(r_o2,r_h2o2,f40k_c * ml_kearth,20000,muy_40k,lda_40k,t)
n40ar_earth = argon(f40k_c * ml_kearth,20000,muy_40k,lda_40k,t)

o2n_u238alpha,h2_u238alpha,ro2_u238alpha = prod_o2(Go2_alpha,Ealpha_238u,ml_238u_earth,20000,muy_238u,lda_238u,t)                    # o2,h2 production from alpha particles
o2n_u235alpha,h2_u235alpha,ro2_u235alpha = prod_o2(Go2_alpha,Ealpha_235u,ml_235u_earth,20000,muy_235u,lda_235u,t)

o2n_u238beta,h2_u238beta,ro2_u238beta = prod_o2(Go2_beta,Ebeta_238u,ml_238u_earth,20000,muy_238u,lda_238u,t)                       # o2,h2 production from beta particles
o2n_u235beta,h2_u235beta,ro2_u235beta = prod_o2(Go2_beta,Ebeta_235u,ml_235u_earth,20000,muy_235u,lda_235u,t) 

o2n_th232alpha,h2_th232alpha,ro2_th232alpha = prod_o2(Go2_alpha,Ealpha_232th,ml_232th_earth,20000,muy_232th,lda_232th,t)
o2n_th232beta,h2_th232beta,ro2_th232beta = prod_o2(Go2_beta,Ebeta_232th,ml_232th_earth,20000,muy_232th,lda_232th,t) 

o2n_u238_earth = o2n_u238alpha + o2n_u238beta
o2n_u235_earth = o2n_u235alpha + o2n_u235beta
o2n_th232_earth = o2n_th232alpha + o2n_th232beta

h2n_u238_earth = h2_u238alpha + h2_u238beta
h2n_u235_earth = h2_u235beta + h2_u235alpha
h2n_th232_earth = h2_th232beta + h2_th232alpha

o2total_earth = o2n_40kearth + o2n_u238alpha + o2n_u235alpha + o2n_u238beta + o2n_u235beta  + o2n_th232alpha +  o2n_th232beta                          # total contribution from all radioactive elements
h2total_earth = h2n_40kearth + h2_u238alpha + h2_u235alpha + h2_u238beta + h2_u235beta + h2_th232beta  + h2_th232alpha
ro2_u238 = ro2_u238alpha + ro2_u238beta
ro2_u235 = ro2_u235alpha + ro2_u235beta
ro2_th232 = ro2_th232alpha + ro2_th232beta
ro2_earth = ro2_40kearth + ro2_u238alpha + ro2_u235alpha + ro2_u238beta + ro2_u235beta + ro2_th232alpha + ro2_th232beta

### Enceladus's case ###
o2n_40kendus,h2n_40kendus,ro2_40kendus = prod_o2(Go2_beta,Ea_40k,1.8 * f40k_c * ml_kearth,20000,muy_40k,lda_40k,t)                # Enceladus's salinity
n40ar_endus = argon(1.8 * f40k_c * ml_kearth,20000,muy_40k,lda_40k,t)

### Lower limit case ###
o2n_40kmin,h2n_40kmin,ro2_40kmin = prod_o2(Go2_beta,Ea_40k,f40k_c * ml_kearth * 1e-2,20000,muy_40k,lda_40k,t)                  # 100 times lower salinity than seawater
n40ar_min = argon(f40k_c * ml_kearth * 1e-2,20000,muy_40k,lda_40k,t)

### Plot for O2 production ###
fig1, ax1 = plt.subplots()
font = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 14,
        }




ax1.semilogy(t,o2total_earth,color='b',label=r'$^{40}\mathrm{K}$ + $^{238}\mathrm{U}$ + $^{235}\mathrm{U}$ + $^{232}\mathrm{Th}$ (seawater)')
ax1.semilogy(t,o2n_40kendus,color='red',linestyle='-.',label=r'$^{40}\mathrm{K}$ (Enceladus)')
ax1.semilogy(t,o2n_40kearth,color='orange',label=r'$^{40}\mathrm{K}$ (seawater)')
#ax1.semilogy(t,o2n_40k_d91,color='orange',linestyle='dashed',label='40K (seawater,Draganic+91)')

ax1.semilogy(t,o2n_40kmin,color='orange',linestyle=':',label = r'$^{40}\mathrm{K}$ (1\% seawater)')
ax1.semilogy(t,o2n_u238_earth,color='green',label=r'$^{238}\mathrm{U}$ (seawater)')
ax1.semilogy(t,o2n_u235_earth,color='k',label=r'$^{235}\mathrm{U}$ (seawater)')
ax1.semilogy(t,o2n_th232_earth,color='tab:purple',label=r'$^{232}\mathrm{Th}$ (seawater)')

plt.axvline(x=1e+5,color='k',linestyle='--')
plt.axvline(x=1e+6,color='k',linestyle='--')
plt.ylim(1e-13,5e-3)

plt.axhline(2e-6,color='gray',linestyle='dashed',label='Classical $O_2$ Pasteur point')
plt.axhspan(1e-9, 1e-8, color='yellow', alpha=0.3, label='$O_2$ minimum for $Ecoli$ growth')



# Custom formatter to divide x-axis values by 100,000
formatter = ticker.FuncFormatter(lambda t, _: f'{t / 100000:.1f}')
plt.gca().xaxis.set_major_formatter(formatter)


ax1.annotate("", xy=(1e+5, 0.6e-4), xytext=(1e+6, 0.6e-4),
            arrowprops=dict(arrowstyle='<->', color='k', lw=2))

plt.text(3.5e+5, 1.3e-4, "$\\tau = 0.1-1 Myr$",
         fontsize=15, color='red',
         weight='bold', style='italic',
         bbox=dict(facecolor='yellow', alpha=0.5))

ax1.set_xlabel("Time (x 100,000 yrs)",size=25)

ax1.set_ylabel("Cumulative $O_2$ ($mol.l^{-1}$)",size=25)
#ax1.set_title("Perched lakes: V = 20000 $km^3$ ",size=25)

plt.legend(loc = "lower right")


### Plot for H2 production ###
### Plot ###
fig1b, ax1b = plt.subplots()
font = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 14,
        }


ax1b.semilogy(t,h2total_earth,color='b',label=r'$^{40}\mathrm{K}$ + $^{238}\mathrm{U}$ + $^{235}\mathrm{U}$ + $^{232}\mathrm{Th}$ (seawater)')
ax1b.semilogy(t,h2n_40kendus,color='red',linestyle='-.',label=r'$^{40}\mathrm{K}$ (Enceladus)')
ax1b.semilogy(t,h2n_40kearth,color='orange',label=r'$^{40}\mathrm{K}$ (seawater)')
#ax1.semilogy(t,o2n_40k_d91,color='orange',linestyle='dashed',label='40K (seawater,Draganic+91)')

ax1b.semilogy(t,h2n_40kmin,color='orange',linestyle=':',label = r'$^{40}\mathrm{K}$ (1\% seawater)')
ax1b.semilogy(t,h2n_u238_earth,color='green',label=r'$^{238}\mathrm{U}$ (seawater)')
ax1b.semilogy(t,h2n_u235_earth,color='k',label=r'$^{235}\mathrm{U}$ (seawater)')
ax1b.semilogy(t,h2n_th232_earth,color='tab:purple',label=r'$^{232}\mathrm{Th}$ (seawater)')

plt.axvline(x=1e+5,color='k',linestyle='--')
plt.axvline(x=1e+6,color='k',linestyle='--')
plt.ylim(1e-13,10e-3)

#plt.axhspan(1e-9, 1e-8, color='yellow', alpha=0.3, label='$O_2$ minimum for $Ecoli$ growth')
#plt.axhline(2e-6,color='red',linestyle='dashed',label='Typical $O_2$ minimum for modern organisms')


# Custom formatter to divide x-axis values by 100,000
formatter = ticker.FuncFormatter(lambda t, _: f'{t / 100000:.1f}')
plt.gca().xaxis.set_major_formatter(formatter)


ax1b.annotate("", xy=(1e+5, 1.2e-4), xytext=(1e+6, 1.2e-4),
            arrowprops=dict(arrowstyle='<->', color='k', lw=2))

plt.text(3.5e+5, 2.6e-4, "$\\tau = 0.1-1 Myr$",
         fontsize=15, color='red',
         weight='bold', style='italic',
         bbox=dict(facecolor='yellow', alpha=0.5))

ax1b.set_xlabel("Time (x 100,000 yrs)",size=25)

ax1b.set_ylabel("Cumulative $H_2$ ($mol.l^{-1}$)",size=25)
#ax1b.set_title("Perched lakes: V = 20000 $km^3$ ",size=25)

plt.legend(loc = "lower right")

### H2O2 production compared to O2, H2 production: Earth's ocean salinity ###
fig2, ax2 = plt.subplots()
font = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 14,
        }

#h2o2_40kearth = o2n_40kearth * 0.04

#ax2.semilogy(t,o2total_earth,color='r',label='$O_2$ production (40K + 238U + 235U decay)')

ax2.semilogy(t,h2n_40kearth,color='b',label='$H_2$ production (Ershov2020)')
ax2.semilogy(t,h2_40k_d91,color='b',linestyle='dashed',label='$H_2$ production (Draganic+91)',marker='o',markersize=4)

ax2.semilogy(t,o2n_40kearth,color='orange',label='$O_2$ production (Ershov2020)')
ax2.semilogy(t,o2n_40k_d91,color='orange',linestyle='dashed',label='$O_2$ production (Draganic+91)',marker='o',markersize=4)


ax2.semilogy(t,h2o2_40k_d91,color='green',linestyle='dashed', label='$H_2O_2$ production (Draganic+91)',marker='o',markersize=4)
#ax2.semilogy(t,h2o2_40kearth,color='b',label='$H_2O_2$ production (Ershov2020)')

plt.axvline(x=1e+5,color='k',linestyle='--')
plt.axvline(x=1e+6,color='k',linestyle='--')
plt.ylim(1e-13,1e-3)

#plt.axhspan(2e7, 2e8, color='gray', alpha=0.3, label='Survival limit ($O_2$ = 1-10 nM)')


# Custom formatter to divide x-axis values by 100,000
formatter = ticker.FuncFormatter(lambda t, _: f'{t / 100000:.1f}')
plt.gca().xaxis.set_major_formatter(formatter)


ax2.annotate("", xy=(1e+5, 0.6e-5), xytext=(1e+6, 0.6e-5),
            arrowprops=dict(arrowstyle='<->', color='k', lw=2))

plt.text(3.5e+5, 1.3e-5, "$\\tau = 0.1-1 Myr$",
         fontsize=15, color='red',
         weight='bold', style='italic',
         bbox=dict(facecolor='yellow', alpha=0.5))
ax2.set_xlabel("Time (x 100,000 yrs)",size=25)

ax2.set_ylabel("Radiolytic production ($mol.l^{-1}$)",size=25)
ax2.set_title("V = 20000 $km^3$, seawater's salinity ",size=25)

plt.legend(loc='lower right')

''' Calculating the production of 40Ar 

### Plot ###
fig3, ax3 = plt.subplots()
font = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 14,
        }

ax3.semilogy(t,n40ar_endus,color='g',linestyle='-.',label='40K decay (Enceladus)')
ax3.semilogy(t,n40ar_earth,color='g',label='40K decay (seawater)')
ax3.semilogy(t,n40ar_min,color='g',linestyle=':',label='40K decay (1$\\%$ seawater)')


plt.axvline(x=1e+5,color='k',linestyle='--')
plt.axvline(x=1e+6,color='k',linestyle='--')
#plt.ylim(1e6,5e11)

plt.axhspan(2e4, 2e5, color='gray', alpha=0.3, label='Detection limit (1-10 ppt)')


# Custom formatter to divide x-axis values by 100,000
formatter = ticker.FuncFormatter(lambda t, _: f'{t / 100000:.1f}')
plt.gca().xaxis.set_major_formatter(formatter)


ax3.annotate("", xy=(1e+5, 7e10), xytext=(1e+6, 7e10),
            arrowprops=dict(arrowstyle='<->', color='k', lw=2))

plt.text(3.5e+5, 1.2e11, "$\\tau = 0.1-1 Myr$",
         fontsize=15, color='red',
         weight='bold', style='italic',
         bbox=dict(facecolor='yellow', alpha=0.5))

ax3.set_xlabel("time (x 100,000 yrs)",size=25)

ax3.set_ylabel("$40_Ar$ production (mol)",size=25)
ax3.set_title("Perched lakes: V = 20000 $km^3$ ",size=25)

plt.legend()
'''
'''Calculating the chemical affinity of 2H2 + O2 = 2H2O for a lake at 20,000 km3 volume'''

def affin(c_o2,c_h2,Vlake,logK):
    Q = 1 / (c_h2**2 * c_o2)
    A = np.log(10)*R*273*(logK - np.log10(Q))             # affinity (J/mol), logK: equillibrium constant at 273K, at 1-5 km (10-50bar)
    A_kJ = A * 1e-3
    return A_kJ

affin_40kearth = affin(o2n_40kearth,h2n_40kearth,20000,100.96)
affin_40endus = affin(o2n_40kendus,h2n_40kendus,20000,100.96)
affin_40kmin = affin(o2n_40kmin,h2n_40kmin,20000,100.96)

affin_earth = affin(o2total_earth,h2total_earth,20000,100.96)
affin_u238earth = affin(o2n_u238_earth,h2n_u238_earth,20000,100.96)
affin_u235earth = affin(o2n_u235_earth,h2n_u235_earth,20000,100.96)
affin_th232earth = affin(o2n_th232_earth,h2n_th232_earth,20000,100.96)



fig4, ax4 = plt.subplots()
font = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 14,
        }

ax4.annotate("", xy=(1e+5, 367), xytext=(1e+6, 367),
            arrowprops=dict(arrowstyle='<->', color='k', lw=2))

plt.text(3.5e+5, 370, "$\\tau = 0.1-1 Myr$",
         fontsize=15, color='red',
         weight='bold', style='italic',
         bbox=dict(facecolor='yellow', alpha=0.5))


ax4.plot(t,affin_earth,label=r'$^{40}\mathrm{K}$ + $^{238}\mathrm{U}$ + $^{235}\mathrm{U}$ + $^{232}\mathrm{Th}$ (seawater)',color='blue')
ax4.plot(t,affin_40endus,label=r'$^{40}\mathrm{K}$ (Enceladus)',color='red',linestyle='-.')
ax4.plot(t,affin_40kearth,label=r'$^{40}\mathrm{K}$ (seawater)',color='orange')

ax4.plot(t,affin_40kmin,label = r'$^{40}\mathrm{K}$ (1\% seawater)',color='orange',linestyle=':')

ax4.plot(t,affin_u238earth,color='green',label=r'$^{238}\mathrm{U}$ (seawater)')
ax4.plot(t,affin_u235earth,color='k',label=r'$^{235}\mathrm{U}$ (seawater)')
ax4.plot(t,affin_th232earth,color='tab:purple',label=r'$^{232}\mathrm{Th}$ (seawater)')

#plt.axhspan(10, 20, color='gray', alpha=0.3, label=r'$\Delta G_{min}$')

# Custom formatter to divide x-axis values by 100,000
formatter = ticker.FuncFormatter(lambda t, _: f'{t / 100000:.1f}')
plt.gca().xaxis.set_major_formatter(formatter)

plt.axvline(x=1e+5,color='k',linestyle='--')
plt.axvline(x=1e+6,color='k',linestyle='--')

ax4.set_xlabel("Time (x 100,000 yrs)",size=25)
ax4.set_ylabel("Affinity (kJ/mol)",size=25)
#ax4.set_title("$2H_2 + O_2 = 2H_2O$",size=25)


plt.legend()



''' Calculating the cell density that could be supported'''

Emin_aerobic = 3e-14                             # kJ/mol/cell (Hoelher 2016?)
m_cell = 1e-15                                   # asusming the average mass of wet Ecoli ~ 1 picogram


def cell(affi,ro2,E_base,Vlake):
    E = affi * ro2
    Vlake_l = Vlake*1e12                      # converting lake volume (km3) to ml
    cell_density = (E/E_base)/Vlake_l
    return cell_density


ncell_earth = cell(affin_earth,ro2_earth,Emin_aerobic,20000)
ncell_40kearth = cell(affin_40kearth,ro2_40kearth,Emin_aerobic,20000)
ncell_40kendus = cell(affin_40endus,ro2_40kendus,Emin_aerobic,20000)
ncell_40kmin = cell(affin_40kmin,ro2_40kmin,Emin_aerobic,20000)

ncell_u238earth = cell(affin_u238earth,ro2_u238,Emin_aerobic,20000)
ncell_u235earth = cell(affin_u235earth,ro2_u235,Emin_aerobic,20000)
ncell_th232earth = cell(affin_th232earth,ro2_th232,Emin_aerobic,20000)


fig5, ax5a = plt.subplots()
font = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 30,
        }

ax5a.semilogy(t,ncell_40kendus,label=r'$^{40}\mathrm{K}$ (Enceladus)',color='red',linestyle='dashed')
ax5a.semilogy(t,ncell_earth,label=r'$^{40}\mathrm{K}$ + $^{238}\mathrm{U}$ + $^{235}\mathrm{U}$ + $^{232}\mathrm{Th}$ (seawater)',color='blue')
ax5a.semilogy(t,ncell_40kearth,label=r'$^{40}\mathrm{K}$ (seawater)',color='orange')
ax5a.semilogy(t,ncell_40kmin,label = r'$^{40}\mathrm{K}$ (1\% seawater)',color='orange',linestyle=':')


ax5a.semilogy(t,ncell_u238earth,color='green',label=r'$^{238}\mathrm{U}$ (seawater)')
ax5a.semilogy(t,ncell_u235earth,color='k',label=r'$^{235}\mathrm{U}$ (seawater)')
ax5a.semilogy(t,ncell_th232earth,color='tab:purple',label=r'$^{232}\mathrm{Th}$ (seawater)')

#plt.axhspan(10, 20, color='gray', alpha=0.3, label=r'$\Delta G_{min}$')

plt.axvline(x=1e+5,color='k',linestyle='--')
plt.axvline(x=1e+6,color='k',linestyle='--')

# Custom formatter to divide x-axis values by 100,000
formatter = ticker.FuncFormatter(lambda t, _: f'{t / 100000:.1f}')
plt.gca().xaxis.set_major_formatter(formatter)

ax5a.annotate("", xy=(1e+5, 3e4), xytext=(1e+6, 3e4),
            arrowprops=dict(arrowstyle='<->', color='k', lw=2))

plt.text(3.5e+5, 4e4, "$\\tau = 0.1-1 Myr$",
         fontsize=15, color='red',
         weight='bold', style='italic',
         bbox=dict(facecolor='yellow', alpha=0.5))

ax5a.set_xlabel("Time (x 100,000 yrs)",size=25)
ax5a.set_ylabel("Number of cells per liter",size=25)

'''
ax5a.text(1.5e9, 6e2, "Blue whale biomass (global) ",
         fontsize=17, color='k',
         weight='bold', style='italic',
         bbox=dict(facecolor='white', alpha=0.5))'''

ax5a.set_ylim(0,1e5)
ax5b = ax5a.twinx()
ax5b.set_ylabel("Total biomass (kg) ",size=25)

ymin,ymax = ax5a.get_ylim()
ax5b.set_ylim(ymin * 1e12 * m_cell, ymax * 1e12 * m_cell)


ax5b.set_yscale('log')

ax5a.legend(loc='lower right')
#plt.yticks(range(0, 26, 2))





plt.show()