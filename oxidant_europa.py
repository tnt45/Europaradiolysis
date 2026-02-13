### OXIDANTS PRODUCTION EUROPA'S OCEAN ###

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from math import pi 
matplotlib.rcParams['text.usetex'] = True

R = 8.314

'''Parameters'''

### Europa: assume 30 km ice shell, ocean thickness ~ 100 km, the rest is mantle
rho_w = 1000                                    # density of water (kg/m3)
m_Eu = 4.79884e+22                              # mass of Europa in kg 
rho_w = 1000                                    # water density, assuming 1000 kg/m3
Voc = 2.754e+18                                  # volume of Europa's ocean in m3 (average ocean depth ~ 100 km)
m_oc = rho_w * Voc
m_r = 4.434e+22
m_ice = 8.999e+20

# apply to the earth's ocean
ml_kearth = 0.4                                 # molality of potassium in earth's ocean (0.4 g/kg)
muy_o2 = 32e-3                                  # molecular mass of oxygen in kg/mol


# parameters for potassium
f40k_c = 0.0117e-2                              # mass fraction of 40K per K today
f40k_past = f40k_c * 12                          # mass fraction of 40K, 4.5 Gyr
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
ml_238u_earthpast = 2.0 * ml_238u_earth         # molality of 238U in earth's ocean 4.5 Gyr ago (1.8 times higher than today)
#Q238u_past = ml_238u_earthpast * moc_earth      # mass of 238U in Earth's ocean 3.8 Gyr ago
muy_238u = 238                                  # g/mol 
lda_238u = 1.55e-10                             # decay parameter lambda in yr-1 for 238u

Go2_alpha = 0.7                                   # yield of o2 by alpha particle 
Ealpha_238u = 45e+6                             # converting MeV to eV
Ebeta_238u = 7.2e+6

# parameters for uranium-235
ml_235u_earth = 0.022e-6                          # molality of 235U in earth's ocean today (g/kg) (Chen, J. H. & Wasserburg, G. J. Precise isotopic analysis of uranium in picomole and subpicomole quantities. Anal. Chem. 53.13, 2060-2067 (1981).)
ml_235u_earthpast = 79.3 * ml_235u_earth          # molality of 235U in earth's ocean 4.5 gyr ago (g/kg, 43 times higher than today)
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

tau = 4.5e+9
t = np.arange(0,tau,1e+7)           # time in years
# radiolytic production in Europa's ocean (Draganic+1991)

def prod91_o2(ro2,ml,muy_iz,lda,t):
    ml_40k = ml * f40k_past 
    Qiz = ml_40k * m_oc
    n_o2 = ro2 * (Qiz/muy_iz) * (1 - np.exp(-lda*t)) / lda
    dndt_o2 = ro2 * (Qiz/muy_iz) * np.exp(-lda*t)
    n_h2 = n_o2*2
    c_o2 = n_o2 / m_oc
    c_h2 = n_h2 /m_oc
    return c_o2,c_h2,dndt_o2

o2n_40k_d91, h2n_40k_d91,dndt_o2_40k_d91 = prod91_o2(r_o2,ml_kearth,muy_40k,lda_40k,t)
 
# radiolyctic production in Europa's ocean (Ersov 2020 formulation)

def prod_o2(G_o2,Ea, ml, muy_iz,lda,t):
    Qiz = ml * m_oc                                                       
    n_o2 = G_o2 * (Ea/100) * (Qiz/muy_iz) * (1 - np.exp(-lda*t))                # total accumulation of O2 production (mol)
    dndt_o2 = G_o2 * (Ea/100) * (Qiz/muy_iz) * lda * np.exp(-lda*t)
    n_h2 = n_o2*2
    c_o2 = n_o2 / m_oc                                                          # molality of oxygen production (mol/kg)
    c_h2 =  n_h2 / m_oc
    return c_o2, c_h2, dndt_o2

  
o2n_40kearth,h2n_40kearth,ro2_40kearth = prod_o2(Go2_beta,Ea_40k,f40k_past * ml_kearth,muy_40k,lda_40k,t)
o2n_40kendus,h2n_40kendus,ro2_40kendus = prod_o2(Go2_beta,Ea_40k,f40k_past * ml_kearth * 1.8,muy_40k,lda_40k,t)             # Glein & Truong 2025: 1.8 times K concentration compared to seawater
o2n_40kmin,h2n_40kmin,ro2_40kmin = prod_o2(Go2_beta,Ea_40k,f40k_past * ml_kearth * 1e-2,muy_40k,lda_40k,t)

o2n_u238alpha,h2n_u238alpha,ro2_u238alpha = prod_o2(Go2_alpha,Ealpha_238u,ml_238u_earthpast,muy_238u,lda_238u,t)                                     #o2,h2 production from alpha particles (mol/kg)
o2n_u235alpha,h2n_u235alpha,ro2_u235alpha = prod_o2(Go2_alpha,Ealpha_235u,ml_235u_earthpast,muy_235u,lda_235u,t)
o2n_th232alpha,h2n_th232alpha,ro2_th232alpha = prod_o2(Go2_alpha,Ealpha_232th,ml_232th_earthpast,muy_232th,lda_232th,t)

o2n_u238beta,h2n_u238beta,ro2_u238beta = prod_o2(Go2_beta,Ebeta_238u,ml_238u_earthpast,muy_238u,lda_238u,t)                                          # o2,h2 production from beta particles (mol/kg)
o2n_u235beta,h2n_u235beta,ro2_u235beta = prod_o2(Go2_beta,Ebeta_235u,ml_235u_earthpast,muy_235u,lda_235u,t)
o2n_th232beta,h2n_th232beta,ro2_th232beta = prod_o2(Go2_beta,Ebeta_232th,ml_232th_earthpast,muy_232th,lda_232th,t) 

o2ntotal_earth = o2n_40kearth + o2n_u238alpha + o2n_u235alpha + o2n_u238beta + o2n_u235beta  + o2n_th232alpha  +  o2n_th232beta                     # total contribution from all radioactive elements (mol/kg)
h2ntotal_earth = h2n_40kearth + h2n_u238alpha + h2n_u235alpha + h2n_u238beta + h2n_u235beta + h2n_th232beta + h2n_th232alpha

o2n_earth_238u = o2n_u238alpha + o2n_u238beta
o2n_earth_235u = o2n_u235alpha + o2n_u235beta  
o2n_earth_232th =  o2n_th232beta + o2n_th232alpha

h2n_earth_238u = h2n_u238alpha + h2n_u238beta
h2n_earth_235u = h2n_u235alpha + h2n_u235beta
h2n_earth_232th = h2n_th232alpha + h2n_th232beta


# rate of production (mol/yr)
ro2_earth_total = ro2_40kearth + ro2_u238alpha + ro2_u235alpha + ro2_u238beta + ro2_u235beta
ro2_earth_238u = ro2_u238alpha + ro2_u238beta
ro2_earth_235u = ro2_u235alpha + ro2_u235beta
ro2_earth_232th = ro2_th232alpha + ro2_th232beta
### cumulative mole production plot ###

def years_to_gyr(x, pos):
    return f"{x/1e9:g}"


# Plottting the cumulative O2 concentration ### 
fig1, ax1 = plt.subplots() 
font = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 14,
        }

#ax1.semilogy(t,o2n_40k_d91,color='g',linestyle='dashed',label='40K (seawater,Draganic+91)')

ax1.semilogy(t,o2ntotal_earth,color='b',label=r'$^{40}\mathrm{K}$ + $^{238}\mathrm{U}$ + $^{235}\mathrm{U}$ + $^{232}\mathrm{Th}$ (seawater)')
ax1.semilogy(t,o2n_40kendus,color='r',linestyle='-.',label=r'$^{40}\mathrm{K}$ (Enceladus)')
ax1.semilogy(t,o2n_40kearth, color='orange', label=r'$^{40}\mathrm{K}$ (seawater)')

ax1.semilogy(t,o2n_40kmin,color='orange',linestyle=':',label = r'$^{40}\mathrm{K}$ (1\% seawater)')

ax1.semilogy(t,o2n_earth_238u,color='g',label=r'$^{238}\mathrm{U}$ (seawater)')
ax1.semilogy(t,o2n_earth_235u,color='k',label=r'$^{235}\mathrm{U}$ (seawater)')
ax1.semilogy(t,o2n_earth_232th,color='tab:purple',label=r'$^{232}\mathrm{Th}$ (seawater)')


ax1.xaxis.set_major_formatter(FuncFormatter(years_to_gyr))
ax1.set_xlabel("Time (Gyr)",size=25)
ax1.set_ylabel("Cumulative $O_2$ ($mol.l^{-1}$)",size=25)
plt.axhline(2e-6,color='gray',linestyle='dashed',label='Classical $O_2$ Pasteur point')
plt.axhspan(1e-9, 1e-8, color='yellow', alpha=0.3, label='$O_2$ minimum for $Ecoli$ growth')
#plt.axhspan(2e-6, 2e-4, color='yellow', alpha=0.5, label='$O_2$ zones on Earth')

plt.legend()

### Plotting the cumulative H2 concentration ###
fig1a, ax1a = plt.subplots() 
font = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 14,
        }

#ax1.semilogy(t,o2n_40k_d91,color='g',linestyle='dashed',label='40K (seawater,Draganic+91)')

ax1a.semilogy(t,h2ntotal_earth,color='b',label=r'$^{40}\mathrm{K}$ + $^{238}\mathrm{U}$ + $^{235}\mathrm{U}$ + $^{232}\mathrm{Th}$ (seawater)')
ax1a.semilogy(t,h2n_40kendus,color='r',linestyle='-.',label=r'$^{40}\mathrm{K}$ (Enceladus)')
ax1a.semilogy(t,h2n_40kearth, color='orange', label=r'$^{40}\mathrm{K}$ (seawater)')

ax1a.semilogy(t,h2n_40kmin,color='orange',linestyle=':',label = r'$^{40}\mathrm{K}$ (1\% seawater)')

ax1a.semilogy(t,h2n_earth_238u,color='g',label=r'$^{238}\mathrm{U}$ (seawater)')
ax1a.semilogy(t,h2n_earth_235u,color='k',label=r'$^{235}\mathrm{U}$ (seawater)')
ax1a.semilogy(t,h2n_earth_232th,color='tab:purple',label=r'$^{232}\mathrm{Th}$ (seawater)')


ax1a.xaxis.set_major_formatter(FuncFormatter(years_to_gyr))
ax1a.set_xlabel("Time (Gyr)",size=25)
ax1a.set_ylabel("Cumulative $H_2$ ($mol.l^{-1}$)",size=25)


plt.legend()

### Production rate ###
fig2, ax2 = plt.subplots()
font = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 14,
        }

#ax2.semilogy(t,dndt_o2_40k_d91,color='g',linestyle='dashed',label='40K decay (seawater,Draganic+91)')

ax2.semilogy(t,ro2_earth_total,color='b',label=r'$^{40}\mathrm{K}$ + $^{238}\mathrm{U}$ + $^{235}\mathrm{U}$ + $^{232}\mathrm{Th}$ (seawater)')
ax2.semilogy(t,ro2_40kendus,color='r',linestyle='-.',label=r'$^{40}\mathrm{K}$ (Enceladus)')



ax2.semilogy(t, ro2_40kearth, color='orange', label=r'$^{40}\mathrm{K}$ (seawater)')

ax2.semilogy(t,ro2_40kmin,color='orange',linestyle=':',label = r'$^{40}\mathrm{K}$ (1\% seawater)')

ax2.semilogy(t,ro2_earth_238u,color='g',label=r'$^{238}\mathrm{U}$ (seawater)')
ax2.semilogy(t,ro2_earth_235u,color='k',label=r'$^{235}\mathrm{U}$ (seawater)')
ax2.semilogy(t,ro2_earth_232th,color='tab:purple',label=r'$^{232}\mathrm{Th}$ (seawater)')


plt.axhspan(5.91e+9, 1.77e+10, color='gray', alpha=0.3, label='Exospheric $O_2$ (Juno)')
plt.axhline(2.0e+6, color='gray', linestyle='--', linewidth=1.5, label="Minimumn $O_2$ surface delivery ")

ax2.xaxis.set_major_formatter(FuncFormatter(years_to_gyr))
ax2.set_xlabel("Time (Gyr)",size=25)
ax2.set_ylabel("$O_2$ production rate (mol/yr)",size=25)

plt.legend()

'''Calculating the chemical affinity of 2H2 + O2 = 2H2O for europa's ocean'''

V_l = Voc  * 1e+3                             # converting ocean volume from m3 to liter 
V_ml = (m_oc/rho_w) * 1e+6                            # converting ocean volume from m3 to milileter

def affin(n_o2,n_h2,logK):
    #o2_molar = n_o2 / V_l
    #h2_molar = n_h2 / V_l
    Q = 1 / (n_h2**2 * n_o2)
    A = np.log(10)*R*273*(logK - np.log10(Q))             # affinity (J/mol), logK: equillibrium constant at 273K, at 1-5 km (10-50bar)
    A_kJ = A * 1e-3
    return A_kJ
affin_40kendus = affin(o2n_40kendus,h2n_40kendus,100.96)
affin_40kearth = affin(o2n_40kearth,h2n_40kearth,100.96)

affin_40kmin = affin(o2n_40kmin,h2n_40kmin,100.96)

affin_238uearth = affin(o2n_earth_238u,h2n_earth_238u,100.96)
affin_235uearth= affin(o2n_earth_235u,h2n_earth_235u,100.96)
affin_232thearth = affin(o2n_earth_232th,h2n_earth_232th,100.96)


affin_earth = affin(o2ntotal_earth,h2ntotal_earth,100.96)

fig4, ax4 = plt.subplots()
font = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 14,
        }


ax4.xaxis.set_major_formatter(FuncFormatter(years_to_gyr))

ax4.plot(t,affin_earth,label=r'$^{40}\mathrm{K}$ + $^{238}\mathrm{U}$ + $^{235}\mathrm{U}$ + $^{232}\mathrm{Th}$ (seawater)',color='b')
ax4.plot(t,affin_40kendus,label=r'$^{40}\mathrm{K}$ (Enceladus)',color='red',linestyle='dashed')

ax4.plot(t,affin_40kearth,label=r'$^{40}\mathrm{K}$ (seawater)',color='orange')
ax4.plot(t,affin_40kmin,label = r'$^{40}\mathrm{K}$ (1\% seawater)',color='orange',linestyle=':')

ax4.plot(t,affin_238uearth,label=r'$^{238}\mathrm{U}$ (seawater)',color='g')
ax4.plot(t,affin_235uearth,color='k',label=r'$^{235}\mathrm{U}$ (seawater)')
ax4.plot(t,affin_232thearth,color='tab:purple',label=r'$^{232}\mathrm{Th}$ (seawater)')

ax4.set_xlabel("Time (Gyr)",size=25)
ax4.set_ylabel("Affinity (kJ/mol)",size=25)
#ax4.set_title("$2H_2 + O_2 = 2H_2O$",size=25)

plt.legend()

''' Calculating the cell density that could be supported'''

Emin_aerobic = 3e-14                             # kJ/mol/cell (Hoelher 2016?)
m_cell = 1e-15                                   # asusming the average mass of wet Ecoli ~ 1 picogram

def cell(affi,ro2,E_base):
    E = affi * ro2 
    cell_number = E/E_base                    
    cell_conc = (E/E_base)/V_l                      # number of cells per liter
    return cell_number,cell_conc

ncell_40kendus,conc_40kendus = cell(affin_40kendus,ro2_40kendus,Emin_aerobic)
ncell_40kearth,conc_40kearth = cell(affin_40kearth,ro2_40kearth,Emin_aerobic)
ncell_40kmin,conc_40kmin = cell(affin_40kmin,ro2_40kmin,Emin_aerobic)

ncell_238uearth,conc_238uearth = cell(affin_238uearth,ro2_earth_238u,Emin_aerobic)
ncell_235uearth,conc_235uearth= cell(affin_235uearth,ro2_earth_235u,Emin_aerobic)
ncell_232thearth,conc_232thearth = cell(affin_232thearth,ro2_earth_232th,Emin_aerobic)

ncell_totalearth,conc_totalearth = cell(affin_earth,ro2_earth_total,Emin_aerobic)

fig5, ax5a = plt.subplots()
font = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 14,
        }

ax5a.xaxis.set_major_formatter(FuncFormatter(years_to_gyr))

ax5a.semilogy(t,conc_40kendus,label=r'${}^{40}\mathrm{K}$ (Enceladus)',color='red',linestyle='dashed')
ax5a.semilogy(t,conc_totalearth,label=r'${}^{40}\mathrm{K}$ + $^{238}\mathrm{U}$ + $^{235}\mathrm{U}$ + $^{232}\mathrm{Th}$ (seawater)',color='b')



ax5a.semilogy(t,conc_40kearth,label=r'${}^{40}\mathrm{K}$ (seawater)',color='orange')

ax5a.semilogy(t,conc_40kmin,label = r'${}^{40}\mathrm{K}$ (1\% seawater)',color='orange',linestyle=':')

ax5a.semilogy(t,conc_238uearth,label=r'$^{238}\mathrm{U}$ (seawater)',color='g')
ax5a.semilogy(t,conc_235uearth,color='k',label=r'$^{235}\mathrm{U}$ (seawater)')
ax5a.semilogy(t,conc_232thearth,color='tab:purple',label=r'$^{232}\mathrm{Th}$ (seawater)')


ax5a.set_xlabel("Time (Gyr)",size=25)
ax5a.set_ylabel("Number of cells per liter",size=30)


ax5a.text(1.2e9, 6e2, "Blue whale biomass (2015) ",
         fontsize=20, color='k',
         weight='bold', style='italic',
         bbox=dict(facecolor='white', alpha=0.5))
         


ax5a.text(1.2e9, 1e4, "Blue whale biomass (1900s) ",
         fontsize=20, color='k',
         weight='bold', style='italic',
         bbox=dict(facecolor='white', alpha=0.5))


ax5b = ax5a.twinx()
ax5b.set_ylabel("Total biomass (kg) ",size=30)

ymin,ymax = ax5a.get_ylim()
ax5b.set_ylim(ymin * V_l * m_cell, ymax * V_l * m_cell)

ax5b.xaxis.set_major_formatter(FuncFormatter(years_to_gyr))
ax5b.set_yscale('log')
'''
ax5b.text(0.5e10, 6e2, "Blue whale biomass (1900s) ",
         fontsize=17, color='k',
         weight='bold', style='italic',
         bbox=dict(facecolor='white', alpha=0.5))
         '''
ax5a.legend(loc='lower left')



plt.axhspan(1e9, 4e9, color='purple', alpha=0.35, label='Blue whale biomass (Cureent)')
plt.axhspan(2e10, 5e10, color='yellow', alpha=0.35, label='Blue whale biomass (1900s)')

plt.tight_layout()

# Argon production

def argon(ml, muy_iz,lda,t):
    Qiz = ml * m_oc
    Voc_l = Voc * 1e+3
    n_40ar = (Qiz/muy_iz) * (np.exp(lda*t)-1) * f_ar                         # f_ar: fraction of 40K decay to 40Ar     
    c_40ar = n_40ar/Voc_l                                                    # converting mol to concentration (mol/L)
    return c_40ar                                                            

c40ar_earth = argon(f40k_c * ml_kearth,muy_40k,lda_40k,t)
c40ar_endus = argon(f40k_c * ml_kearth * 1.8,muy_40k,lda_40k,t)
c40ar_min = argon(f40k_c * ml_kearth * 1e-2,muy_40k,lda_40k,t)

def years_to_gyr(x, pos):
    return f"{x/1e9:g}"

fig6, ax6 = plt.subplots()
font = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 14,
        }

ax6.semilogy(t,c40ar_endus,color='k',linestyle='-.',label=r'${}^{40}\mathrm{K}$ (Enceladus)')
ax6.semilogy(t,c40ar_earth ,color='k',label=r'${}^{40}\mathrm{K}$ (seawater)')

ax6.semilogy(t,c40ar_min,color='k',linestyle=':',label=r'${}^{40}\mathrm{K}$ (1\% seawater)')

plt.axhspan(1e-6, 1e-5, color='purple', alpha=0.3, label='MASPEX detection ($>$1 ppm)')
plt.axhspan(1e-7, 1e-6, color='yellow', alpha=0.4, label='MASPEX lower threshold (sub-ppm)')
#plt.axhline(y=1e-7,color='r',alpha=0.3,linestyle='--')
#lt.axhline(y=1e-6,color='r',alpha=0.3,linestyle='--')
plt.legend(loc='lower right')

ax6.xaxis.set_major_formatter(FuncFormatter(years_to_gyr))
ax6.set_xlabel("Time (Gyr)",size=25)

ax6.set_yscale('log')
ax6.set_ylabel(r'$^{40}\mathrm{Ar}$ concentration ($\mathrm{mol\ L^{-1}}$)',size=25)
ax6.set_ylim(1e-12, 1e-5)

ax6b = ax6.twinx()
ax6b.set_yscale('log')
ax6b.set_ylabel(r"$^{40}\mathrm{Ar}/\mathrm{H_2O}$ mixing ratio",size=25)
ax6b.set_ylim(1e-12, 1e-5)

'''
# Plotting 36Ar on Europa depending on different sources of water 

cases = ["CI-like", "Comet-like", "Mixed"]
values = [0.5, 4.9, 0.94]       # means
errors = [0.0, 1.9, 0.19]       # ± for comet-like; 0 for chondrite-like


fig7, ax7 = plt.subplots()
font = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 14,
        }
x = np.arange(len(cases))
colors = ['blue','green',"red"]

for i in range(len(cases)):
    plt.errorbar(
        x[i], values[i], yerr=errors[i],
        fmt='o', capsize=5, markersize=8,
        color=colors[i], ecolor=colors[i]
    )

#plt.axhspan(1, 100, color='blue', alpha=0.3, label='MASPEX nominal detection ($>$1 ppm)')
#plt.axhspan(0.1, 1, color='yellow', alpha=0.3, label='MASPEX lower threshold (sub-ppm)')

plt.axhline(1.0, color='k', linestyle='--', linewidth=1.5, label="MASPEX detection")
plt.axhline(0.1, color='k', linestyle=':', linewidth=1.5, label="MASPEX lower threshold")


plt.xticks(x, cases, rotation=0, fontsize=25)
plt.ylabel(r"$^{36}\mathrm{Ar}/\mathrm{H_2O}$ (ppm)", fontsize=25)
plt.yscale('log')
plt.legend()

plt.tight_layout()


# Plotting the 40Ar/36Ar 

# Data
bodies = ["Earth", "Mars", "Venus", "Titan", "Europa"]
values = [298.56, 1900, 1.03, 165, (0.02 + 2.0) / 2.0]  # Europa midpoint for plotting
errors_lower = [0.31, 300, 0.04, 67, ((0.02 + 2.0) / 2.0) - 0.02]
errors_upper = [0.31, 300, 0.04, 67, 2.0-((0.02 + 2.0) / 2.0)]

# Assign a unique color to each body
colors = ["tab:blue", "tab:red", "green", "tab:orange", "tab:purple"]

x = np.arange(len(bodies))


fig8, ax8 = plt.subplots()
font = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 14,
        }
for i in range(len(bodies)):
    plt.errorbar(
        x[i], values[i],
        yerr=[[errors_lower[i]], [errors_upper[i]]],
        fmt='o', capsize=5, markersize=10, color=colors[i], label=bodies[i]
    )

plt.xticks(x, bodies,size=18)

plt.ylabel(r"$^{40}\mathrm{Ar}/^{36}\mathrm{Ar}$",size=25)

plt.grid(True, linestyle='--', alpha=0.6)
plt.yscale('log')
plt.legend(fontsize=12)
plt.tight_layout()

'''
plt.show()