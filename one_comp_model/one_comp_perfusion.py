import numpy as np
import functions as fcts
import matplotlib.pyplot as plt
from copy import deepcopy

K1_high = 0.9
K1_low = 0.6

Vt = 1.0
fbv = 0.05

k2_high = K1_high/Vt
k2_low = K1_low/Vt


g11 = fcts.ExpDecayFunction(4)
g11.scale = 50.0
g12 = fcts.ExpDecayFunction(8)
g12.scale = -53.0
g13 = fcts.ExpDecayFunction(0.5)
g13.scale = 2.0
p1 = fcts.PlateauFunction()
p1.scale = 1.0

# generate an arterial input function as sum of 3 exponentials + a plateau
C_A1 = fcts.ExpConvSumFunction([g11,g12,g13,p1])

# generate a second input function where the "peak/plateau" ratio is lower
g21 = deepcopy(g11)
g21.scale = 37.0
g22 = deepcopy(g12)
g22.scale = -49.0
g23 = deepcopy(g13)
g23.scale = 2.0
p2 = deepcopy(p1)
p2.scale = 10.0

C_A2 = fcts.ExpConvSumFunction([g21,g22,g23,p2])

# constant infuction IF 
C_A3 = fcts.ExpConvSumFunction([p2])

# tissue response = K1 * convolution(C_A, exp(-k2*t))
C_t1_high = C_A1.expconv(k2_high)
C_t1_high.scale = K1_high
C_t1_low = C_A1.expconv(k2_low)
C_t1_low.scale = K1_low

C_t2_high = C_A2.expconv(k2_high)
C_t2_high.scale = K1_high
C_t2_low = C_A2.expconv(k2_low)
C_t2_low.scale = K1_low

C_t3_high = C_A3.expconv(k2_high)
C_t3_high.scale = K1_high
C_t3_low = C_A3.expconv(k2_low)
C_t3_low.scale = K1_low

# calculate PET concentrations including fractional blood volume
scaled_CA1 = deepcopy(C_A1)
scaled_CA1.scale *= 0.05
tmp1_high = deepcopy(C_t1_high)
tmp1_high.scale *= 1-fbv
tmp1_low = deepcopy(C_t1_low)
tmp1_low.scale *= 1-fbv

C_PET1_high = fcts.IntegrableSumFunction([scaled_CA1, tmp1_high])
C_PET1_low = fcts.IntegrableSumFunction([scaled_CA1, tmp1_low])

scaled_CA2 = deepcopy(C_A2)
scaled_CA2.scale *= 0.05
tmp2_high = deepcopy(C_t2_high)
tmp2_high.scale *= 1-fbv
tmp2_low = deepcopy(C_t2_low)
tmp2_low.scale *= 1-fbv

C_PET2_high = fcts.IntegrableSumFunction([scaled_CA2, tmp2_high])
C_PET2_low = fcts.IntegrableSumFunction([scaled_CA2, tmp2_low])

scaled_CA3 = deepcopy(C_A3)
scaled_CA3.scale *= 0.05
tmp3_high = deepcopy(C_t3_high)
tmp3_high.scale *= 1-fbv
tmp3_low = deepcopy(C_t3_low)
tmp3_low.scale *= 1-fbv

C_PET3_high = fcts.IntegrableSumFunction([scaled_CA3, tmp3_high])
C_PET3_low = fcts.IntegrableSumFunction([scaled_CA3, tmp3_low])

# discrete time array for plots
t = np.linspace(0.001, 8, 1000)

fig, ax = plt.subplots(3, 4, figsize=(12,9), sharex = True)

ax[0,0].plot(t, C_A1(t), 'k', label = 'C_A(t)')
ax[0,0].plot(t, C_t1_high(t), 'r', label = f'C_t1(t,K1={K1_high})')
ax[0,0].plot(t, C_t1_low(t), 'b', label = f'C_t2(t,K1={K1_low})')
ax[0,0].plot(t, C_PET1_high(t), 'r:', label = f'C_PET1(t,K1={K1_high})')
ax[0,0].plot(t, C_PET1_low(t), 'b:', label = f'C_PET2(t,K1={K1_low})')

ax[0,1].plot(t, C_t1_high(t)/C_t1_low(t), 'k', label = 'C_t1(t) / C_t2(t)')
ax[0,1].plot(t, C_PET1_high(t)/C_PET1_low(t), 'k:', label = 'C_PET1(t) / C_PET2(t)')

ax[0,2].plot(t, C_t1_high.indefinite_integral(t) - C_t1_high.indefinite_integral(0), 'r', label = r'$\int_0^t$ C_t1($\tau$) d$\tau$')
ax[0,2].plot(t, C_t1_low.indefinite_integral(t) - C_t1_low.indefinite_integral(0), 'b', label = r'$\int_0^t$ C_t2($\tau$) d$\tau$')
ax[0,2].plot(t, C_PET1_high.indefinite_integral(t) - C_PET1_high.indefinite_integral(0), 'r:', label = r'$\int_0^t$ C_PET1($\tau$) d$\tau$')
ax[0,2].plot(t, C_PET1_low.indefinite_integral(t) - C_PET1_low.indefinite_integral(0), 'b:', label = r'$\int_0^t$ C_PET2($\tau$) d$\tau$')

ax[0,3].plot(t, (C_t1_high.indefinite_integral(t) - C_t1_high.indefinite_integral(0)) / 
                (C_t1_low.indefinite_integral(t) - C_t1_low.indefinite_integral(0)), 'k', 
                label = r'$\int_0^t$ C_t1($\tau$) d$\tau$ / $\int_0^t$ C_t2($\tau$) d$\tau$')
ax[0,3].plot(t, (C_PET1_high.indefinite_integral(t) - C_PET1_high.indefinite_integral(0)) / 
                (C_PET1_low.indefinite_integral(t) - C_PET1_low.indefinite_integral(0)), 'k:',
                label = r'$\int_0^t$ C_PET1($\tau$) d$\tau$ / $\int_0^t$ C_PET2($\tau$) d$\tau$')

ax[1,0].plot(t, C_A2(t), 'k')
ax[1,0].plot(t, C_t2_high(t), 'r')
ax[1,0].plot(t, C_t2_low(t), 'b')
ax[1,0].plot(t, C_PET2_high(t), 'r:')
ax[1,0].plot(t, C_PET2_low(t), 'b:')

ax[1,1].plot(t, C_t2_high(t)/C_t2_low(t), 'k')
ax[1,1].plot(t, C_PET2_high(t)/C_PET2_low(t), 'k:')

ax[1,2].plot(t, C_t2_high.indefinite_integral(t) - C_t2_high.indefinite_integral(0), 'r')
ax[1,2].plot(t, C_t2_low.indefinite_integral(t) - C_t2_low.indefinite_integral(0), 'b')
ax[1,2].plot(t, C_PET2_high.indefinite_integral(t) - C_PET2_high.indefinite_integral(0), 'r:')
ax[1,2].plot(t, C_PET2_low.indefinite_integral(t) - C_PET2_low.indefinite_integral(0), 'b:')

ax[1,3].plot(t, (C_t2_high.indefinite_integral(t) - C_t2_high.indefinite_integral(0)) / 
                (C_t2_low.indefinite_integral(t) - C_t2_low.indefinite_integral(0)), 'k')
ax[1,3].plot(t, (C_PET2_high.indefinite_integral(t) - C_PET2_high.indefinite_integral(0)) / 
                (C_PET2_low.indefinite_integral(t) - C_PET2_low.indefinite_integral(0)), 'k:')

ax[2,0].plot(t, C_A3(t), 'k')
ax[2,0].plot(t, C_t3_high(t), 'r')
ax[2,0].plot(t, C_t3_low(t), 'b')
ax[2,0].plot(t, C_PET3_high(t), 'r:')
ax[2,0].plot(t, C_PET3_low(t), 'b:')

ax[2,1].plot(t, C_t3_high(t)/C_t3_low(t), 'k')
ax[2,1].plot(t, C_PET3_high(t)/C_PET3_low(t), 'k:')

ax[2,2].plot(t, C_t3_high.indefinite_integral(t) - C_t3_high.indefinite_integral(0), 'r')
ax[2,2].plot(t, C_t3_low.indefinite_integral(t) - C_t3_low.indefinite_integral(0), 'b')
ax[2,2].plot(t, C_PET3_high.indefinite_integral(t) - C_PET3_high.indefinite_integral(0), 'r:')
ax[2,2].plot(t, C_PET3_low.indefinite_integral(t) - C_PET3_low.indefinite_integral(0), 'b:')

ax[2,3].plot(t, (C_t3_high.indefinite_integral(t) - C_t3_high.indefinite_integral(0)) / 
                (C_t3_low.indefinite_integral(t) - C_t3_low.indefinite_integral(0)), 'k')
ax[2,3].plot(t, (C_PET3_high.indefinite_integral(t) - C_PET3_high.indefinite_integral(0)) / 
                (C_PET3_low.indefinite_integral(t) - C_PET3_low.indefinite_integral(0)), 'k:')

for axx in ax.ravel():
    axx.grid(ls = ':')    
    axx.set_xlabel('t (min)')

for axx in ax[:,1::2].ravel():
    axx.set_ylim(0, 1.05*K1_high/K1_low)

fkwargs = dict(fontsize = 'small')
for axx in ax[0,:]:
    axx.legend(**fkwargs)

ax[0,0].set_title('concentrations', **fkwargs) 
ax[0,1].set_title('ratio of concentrations', **fkwargs) 
ax[0,2].set_title('time integral of concentrations', **fkwargs) 
ax[0,3].set_title('ratio of time integrals of concentrations', **fkwargs) 

fig.tight_layout()
fig.show()