# this Python script uses cosimulation to optimize capacitor values for an elliptical birdcage
# the requirements.txt file lists the packages needed to run it (or run "pip install -r requirements.txt")
# for more details see <https://github.com/dezanche/>
# Copyright ©2024 Nicola De Zanche

# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

import cosimpy
import skrf as rf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np
import re
import pandas as pd
import pint
from pint import UnitRegistry

# some constants
mu_0 = 4*np.pi*1E-7
f0 = 128E6 #frequency at which fields are calculated
f0_target = 128E6 # target frequency for tuning
Z0 = 1000 # [ohm] impedance for renormalized S parameters

# boolean to enable optimization
perform_optimization = True#False#

# weighting of resonance frequency objective in optimization
# invert this number if switching from costfn defined based on+ Y to Z and vice versa
frequency_weighting = 100

# read in S parameters
S_full = cosimpy.S_Matrix.importTouchstone('Elliptical_Birdcage_Coil_36ports.s36p')
freqs = S_full.frequencies

# read in H field (converting to B)
EM_field = cosimpy.EM_Field.importFields_hfss('Coil_fields', freqs=[128e6], nPorts=36, b_multCoeff = mu_0, imp_efield=False)

# define coil with fields
rf_coil = cosimpy.RF_Coil(S_full, EM_field)

# hopefully robust way to get some information automatically about the grid points on which the fields are defined
first_file = open('Coil_fields/bfield_128000000.0_port1.fld', 'r')
first_line = first_file.readline()
brackets_text = re.findall(r'\[.*?\]', first_line)

ureg = UnitRegistry()
pattern = '{mm}mm'
grid_min = ureg.parse_pattern(brackets_text[0][1:-1], pattern, many=True)
grid_max = ureg.parse_pattern(brackets_text[1][1:-1], pattern, many=True)
grid_size = ureg.parse_pattern(brackets_text[2][1:-1], pattern, many=True)

x_points = np.arange(pint.Quantity.from_list(np.stack(grid_min[0])).magnitude,pint.Quantity.from_list(np.stack(grid_max[0])+np.stack(grid_size[0])).magnitude,pint.Quantity.from_list(np.stack(grid_size[0])).magnitude)
y_points = np.arange(pint.Quantity.from_list(np.stack(grid_min[1])).magnitude,pint.Quantity.from_list(np.stack(grid_max[1])+np.stack(grid_size[1])).magnitude,pint.Quantity.from_list(np.stack(grid_size[1])).magnitude)
field_grids = np.meshgrid(x_points,y_points)

# initialize vector with number of elements of grid in each dimension
no_points = pd.DataFrame([['nx', 1], ['ny', 1], ['nz', 1]])#, columns=['axis', 'points']
# exclude zeros in denominator to avoid 'ZeroDivisionError: float division by zero'
for n in range(3):
    if np.stack(grid_size[n]) != 0:
        no_points[1][n] = int((np.stack(grid_max[n])-np.stack(grid_min[n]))/np.stack(grid_size[n]))+1  #assumes division results in integer but doesn't check


# initial capacitor values for optimization
C_1 = 23E-12 # [F]
C_2 = 22E-12 # [F]
C_3 = 17E-12 # [F]
C_4 = 15E-12 # [F]
C_leg1 = 16E-12 # [F] effective (single) leg capacitance
C_leg2 = 14E-12 # [F]
C_leg3 = 11E-12 # [F]

# equivalent series resistance of capacitors
ESR = 0.1 # [ohm]

# S parameters of capacitors
S_cleg1 = cosimpy.S_Matrix.sMatrixRCseries(ESR, C_leg1, freqs)
S_cleg2 = cosimpy.S_Matrix.sMatrixRCseries(ESR, C_leg2, freqs)
S_cleg3 = cosimpy.S_Matrix.sMatrixRCseries(ESR, C_leg3, freqs)
S_c1 = cosimpy.S_Matrix.sMatrixRCseries(ESR, C_1, freqs)
S_c2 = cosimpy.S_Matrix.sMatrixRCseries(ESR, C_2, freqs)
S_c3 = cosimpy.S_Matrix.sMatrixRCseries(ESR, C_3, freqs)
S_c4 = cosimpy.S_Matrix.sMatrixRCseries(ESR, C_4, freqs)

# S parameters of a series short
#S_short = cosimpy.S_Matrix.sMatrixRLseries(0, 0, freqs)
S_short = cosimpy.S_Matrix.sMatrixShort(freqs)

# S parameters for ports across capacitors 1 and 4 (without matching components, gives singular matrix?)
S_P1 = cosimpy.S_Matrix.sMatrixPInetwork(S_c1, None, S_short)
S_P4 = cosimpy.S_Matrix.sMatrixPInetwork(S_c4, None, S_short)

# lists containing lumped port terminations (external circuits)
# ports across both C1 and C4
ext_circ_2port = [S_P1, S_c2, S_c3, S_P4, S_c3, S_c2, S_c1, S_c2, S_c3, S_c4, S_c3, S_c2, S_c1, S_c2, S_c3, S_c4, S_c3, S_c2, S_c1, S_c2, S_c3, S_c4, S_c3, S_c2, S_cleg1, S_cleg2, S_cleg3, S_cleg3, S_cleg2, S_cleg1, S_cleg1, S_cleg2, S_cleg3, S_cleg3, S_cleg2, S_cleg1]
# 1 port across C1
ext_circ_C1 = [S_P1, S_c2, S_c3, S_c4, S_c3, S_c2, S_c1, S_c2, S_c3, S_c4, S_c3, S_c2, S_c1, S_c2, S_c3, S_c4, S_c3, S_c2, S_c1, S_c2, S_c3, S_c4, S_c3, S_c2, S_cleg1, S_cleg2, S_cleg3, S_cleg3, S_cleg2, S_cleg1, S_cleg1, S_cleg2, S_cleg3, S_cleg3, S_cleg2, S_cleg1]
# 1 port across C4
ext_circ_C4 = [S_c1, S_c2, S_c3, S_P4, S_c3, S_c2, S_c1, S_c2, S_c3, S_c4, S_c3, S_c2, S_c1, S_c2, S_c3, S_c4, S_c3, S_c2, S_c1, S_c2, S_c3, S_c4, S_c3, S_c2, S_cleg1, S_cleg2, S_cleg3, S_cleg3, S_cleg2, S_cleg1, S_cleg1, S_cleg2, S_cleg3, S_cleg3, S_cleg2, S_cleg1]

## RF coil with ports connected to terminations
rf_coil_conn = rf_coil.singlePortConnRFcoil(ext_circ_2port, comp_Pinc = True)
Y_matrix = rf.s2y(rf_coil_conn.s_matrix.S)
B_field_conn = rf_coil_conn.em_field.b_field

# B field with quadrature excitation of ports (equal magnitude)
quad_field = B_field_conn[0][0] + 1j * B_field_conn[0][1]

# from Hoult CMR12(4)-173
B1_plus = np.reshape(np.absolute((quad_field[0] + 1j * quad_field[1])/2),[no_points[1][1].astype(int), no_points[1][0].astype(int)])  # Eq. 14 (magnitude only)
#B1_minus = np.absolute((quad_field[0] - 1j * quad_field[1])/2)  # Eq. 15 (magnitude only)

## calculate cost function
# elliptical inscribed region
mask = (field_grids[0] / pint.Quantity.from_list(np.stack(grid_max[0])).magnitude) ** 2 + (field_grids[1] / pint.Quantity.from_list(np.stack(grid_max[1])).magnitude) ** 2 <= 1
initial_SD = np.std(B1_plus[np.where(mask)])
#initial_cost = initial_SD + np.sqrt(np.imag(Z_matrix[np.where(freqs == f0)][0, 0, 0]) ** 2 + np.imag(Z_matrix[np.where(freqs == f0)][0, 1, 1]) ** 2) * initial_SD * frequency_weighting
initial_cost = initial_SD + np.sqrt(np.imag(Y_matrix[np.where(freqs == f0_target)][0, 0, 0]) ** 2 + np.imag(Y_matrix[np.where(freqs == f0_target)][0, 1, 1]) ** 2) * initial_SD * frequency_weighting
#np.mean(B1_plus)
#np.std(B1_minus)
#np.mean(B1_minus)

## plots before optimization

EM_field.plotEMField('b_field', 'mag' ,f0 , np.arange(36)+1,'xy',0)

plt.figure()
#rf_coil_conn.em_field.plotEMField("b_field", comp="b1+", freq=f0, ports=[1,2], plane='xy', sliceIdx=0)
plt.imshow(B1_plus*mask, origin = 'lower',  extent = [x_points[0],x_points[-1],y_points[0],y_points[-1]])
#plt.clim(0, B1_plus.max())
plt.title('B1+ before optimization')
plt.xlabel('cost function ' + initial_cost.astype(str))
plt.colorbar()

plt.figure()
plt.imshow(mask, origin = 'lower',  extent = [x_points[0],x_points[-1],y_points[0],y_points[-1]])
plt.title('field optimization mask')
plt.colorbar()

# display 36x36 S matrix at 128MHz
plt.figure()
plt.imshow(np.abs(S_full.S[np.where(freqs == f0)])[0,:,:])
plt.title('full |S| matrix at ' + str(f0) + 'Hz')
plt.colorbar()

plt.show()



### cost function
def costfn(capacitances):
    print('.', end='',flush=True) # print one dot at each iteration to show progress
    S_c1 = cosimpy.S_Matrix.sMatrixRCseries(ESR, capacitances[0], freqs)
    S_c2 = cosimpy.S_Matrix.sMatrixRCseries(ESR, capacitances[1], freqs)
    S_c3 = cosimpy.S_Matrix.sMatrixRCseries(ESR, capacitances[2], freqs)
    S_c4 = cosimpy.S_Matrix.sMatrixRCseries(ESR, capacitances[3], freqs)
    S_cleg1 = cosimpy.S_Matrix.sMatrixRCseries(ESR, capacitances[4], freqs)
    S_cleg2 = cosimpy.S_Matrix.sMatrixRCseries(ESR, capacitances[5], freqs)
    # S_cleg3 is kept constant otherwise solution is not unique
    S_cleg3 = cosimpy.S_Matrix.sMatrixRCseries(ESR, capacitances[6], freqs)
    S_P1 = cosimpy.S_Matrix.sMatrixPInetwork(S_c1, None, S_short)
    S_P4 = cosimpy.S_Matrix.sMatrixPInetwork(S_c4, None, S_short)
    ext_circ = [S_P1, S_c2, S_c3, S_P4, S_c3, S_c2, S_c1, S_c2, S_c3, S_c4, S_c3, S_c2, S_c1, S_c2, S_c3, S_c4, S_c3, S_c2, S_c1, S_c2, S_c3, S_c4, S_c3, S_c2, S_cleg1, S_cleg2, S_cleg3, S_cleg3, S_cleg2, S_cleg1, S_cleg1, S_cleg2, S_cleg3, S_cleg3, S_cleg2, S_cleg1]
    rf_coil_conn = rf_coil.singlePortConnRFcoil(ext_circ, comp_Pinc = True)
    B_field_conn = rf_coil_conn.em_field.b_field
    quad_field = B_field_conn[0][0] + 1j * B_field_conn[0][1]
    B1_plus = np.reshape(np.absolute((quad_field[0] + 1j * quad_field[1])/2),[no_points[1][1].astype(int), no_points[1][0].astype(int)])  # Eq. 14 (magnitude only)
    # minimize SD of B1+ and make |Im{Z11}| + |Im{Z22}| small at 128MHz to obtain 2 degenerate modes (resonance frequency objective)
    # Z_matrix = rf.s2z(rf_coil_conn.s_matrix.S)
    # cost = np.std(B1_plus[np.where(mask)]) + np.sqrt(np.imag(Z_matrix[np.where(freqs == f0)][0, 0, 0]) ** 2 + np.imag(Z_matrix[np.where(freqs == f0)][0, 1, 1]) ** 2) * initial_SD * frequency_weighting
    # alternative with |Im{Y11}| + |Im{Y22}| small
    Y_matrix = rf.s2y(rf_coil_conn.s_matrix.S)
    cost = np.std(B1_plus[np.where(mask)]) + np.sqrt(np.imag(Y_matrix[np.where(freqs == f0_target)][0, 0, 0]) ** 2 + np.imag(Y_matrix[np.where(freqs == f0_target)][0, 1, 1]) ** 2) * initial_SD * frequency_weighting
    return cost
    


### run optimization
#costfn([C_1, C_2, C_3, C_4, C_leg1, C_leg2])
if perform_optimization:
    print('optimizing capacitances ...')
    result = minimize(costfn, [C_1, C_2, C_3, C_4, C_leg1, C_leg2, C_leg3], method='nelder-mead', options={'fatol': initial_cost/100, 'disp': True})
    optimized_capacitances = result['x']
    final_cost = result['fun']
else:
    print('skipping optimization ...')
    optimized_capacitances = [C_1, C_2, C_3, C_4, C_leg1, C_leg2, C_leg3]
    final_cost = initial_cost

# update model with optimized values
S_c1 = cosimpy.S_Matrix.sMatrixRCseries(ESR, optimized_capacitances[0], freqs)
S_c2 = cosimpy.S_Matrix.sMatrixRCseries(ESR, optimized_capacitances[1], freqs)
S_c3 = cosimpy.S_Matrix.sMatrixRCseries(ESR, optimized_capacitances[2], freqs)
S_c4 = cosimpy.S_Matrix.sMatrixRCseries(ESR, optimized_capacitances[3], freqs)
S_cleg1 = cosimpy.S_Matrix.sMatrixRCseries(ESR, optimized_capacitances[4], freqs)
S_cleg2 = cosimpy.S_Matrix.sMatrixRCseries(ESR, optimized_capacitances[5], freqs)
S_cleg3 = cosimpy.S_Matrix.sMatrixRCseries(ESR, optimized_capacitances[6], freqs)
S_P1 = cosimpy.S_Matrix.sMatrixPInetwork(S_c1, None, S_short)
S_P4 = cosimpy.S_Matrix.sMatrixPInetwork(S_c4, None, S_short)

# lists containing lumped port terminations (external circuits)
# ports across both C1 and C4
ext_circ_2port = [S_P1, S_c2, S_c3, S_P4, S_c3, S_c2, S_c1, S_c2, S_c3, S_c4, S_c3, S_c2, S_c1, S_c2, S_c3, S_c4, S_c3, S_c2, S_c1, S_c2, S_c3, S_c4, S_c3, S_c2, S_cleg1, S_cleg2, S_cleg3, S_cleg3, S_cleg2, S_cleg1, S_cleg1, S_cleg2, S_cleg3, S_cleg3, S_cleg2, S_cleg1]
# 1 port across C1
ext_circ_C1 = [S_P1, S_c2, S_c3, S_c4, S_c3, S_c2, S_c1, S_c2, S_c3, S_c4, S_c3, S_c2, S_c1, S_c2, S_c3, S_c4, S_c3, S_c2, S_c1, S_c2, S_c3, S_c4, S_c3, S_c2, S_cleg1, S_cleg2, S_cleg3, S_cleg3, S_cleg2, S_cleg1, S_cleg1, S_cleg2, S_cleg3, S_cleg3, S_cleg2, S_cleg1]
# 1 port across C4
ext_circ_C4 = [S_c1, S_c2, S_c3, S_P4, S_c3, S_c2, S_c1, S_c2, S_c3, S_c4, S_c3, S_c2, S_c1, S_c2, S_c3, S_c4, S_c3, S_c2, S_c1, S_c2, S_c3, S_c4, S_c3, S_c2, S_cleg1, S_cleg2, S_cleg3, S_cleg3, S_cleg2, S_cleg1, S_cleg1, S_cleg2, S_cleg3, S_cleg3, S_cleg2, S_cleg1]

## calculate currents at all coil ports
rf_coil_portC1 = rf_coil.singlePortConnRFcoil(ext_circ_C1, comp_Pinc = True)
rf_coil_portC4 = rf_coil.singlePortConnRFcoil(ext_circ_C4, comp_Pinc = True)
VI_portC1 = rf_coil_portC1.s_matrix.compVI()
VI_portC4 = rf_coil_portC4.s_matrix.compVI()

## calculate quadrature field
rf_coil_conn = rf_coil.singlePortConnRFcoil(ext_circ_2port, comp_Pinc = True)

B_field_conn = rf_coil_conn.em_field.b_field

# equal excitation into both ports
quad_field = B_field_conn[0][0] + 1j * B_field_conn[0][1]

B1_plus = np.reshape(np.absolute((quad_field[0] + 1j * quad_field[1])/2),[no_points[1][1].astype(int), no_points[1][0].astype(int)])


### PLOT RESULTS

fig = rf_coil_conn.s_matrix.plotS(["S1-1", "S2-2", "S1-2"], smooth = False)

# create scikit-rf network object
ntwk = rf.Network(frequency=freqs, s=rf_coil_conn.s_matrix.S , name='connected coil')
ntwk.frequency.unit = 'mhz'
ntwk.renormalize(Z0)

# plot additional figures for comparison
plt.figure()
plt1 = ntwk.plot_s_db()
plt.ylabel('Magnitude (dB)')
plt.title('renormalized S parameters')

#plt.figure()
#plt2 = ntwk.plot_s_smith()

plt.figure()
ntwk.plot_z_re()
plt.axis((freqs[0], freqs[-1], -50, 1000))
plt.ylabel('Real impedance R (ohm)')

plt.figure()
ntwk.plot_z_im()
plt.axis((freqs[0], freqs[-1], -300, 300))
plt.ylabel('Imaginary impedance X (ohm)')

plt.figure()
ntwk.plot_y_re()
plt.axis((freqs[0], freqs[-1], -0.03, 0.15))
plt.ylabel('Real admittance G (ohm)')

plt.figure()
ntwk.plot_y_im()
plt.axis((freqs[0], freqs[-1], -0.1, 0.1))
plt.ylabel('Imaginary admittance B (ohm)')

plt.figure()
plt.imshow(B1_plus*mask, origin = 'lower', extent = [x_points[0],x_points[-1],y_points[0],y_points[-1]])
#plt.gca().set_xticks(x_points)
#plt.gca().set_yticks(y_points)
plt.title('B1+ after optimization')
plt.xlabel('cost function = ' + final_cost.astype(str))
plt.colorbar()

plt.figure()
#plt.plot(freqs,np.squeeze(VI_portC1[0])) # voltage at driven port as a function of frequency
#plt.plot(np.linspace(1,36,36),np.transpose(np.squeeze(VI_portC1[3]))) # overlapping currents at all ports
ax = plt.axes(projection='3d')
X, Y = np.meshgrid(np.linspace(1,12,12),freqs/1E6)
#ax.plot_wireframe(X, Y, np.squeeze(np.abs(VI_portC1[3]))[:,0:12],rstride=1, cstride=0)
# NB phase factor to make ~ real at 128MHz
ax.plot_wireframe(X, Y, np.squeeze(VI_portC1[3]*np.exp(1j*0.2))[:,0:12],rstride=1, cstride=0)
plt.title('currents on driven end ring capacitors: port C1')

plt.figure()
#plt.plot(freqs,np.squeeze(VI_portC1[0])) # voltage at driven port as a function of frequency
#plt.plot(np.linspace(1,36,36),np.transpose(np.squeeze(VI_portC1[3]))) # overlapping currents at all ports
ax = plt.axes(projection='3d')
X, Y = np.meshgrid(np.linspace(1,12,12),freqs/1E6)
#ax.plot_wireframe(X, Y, np.squeeze(np.abs(VI_portC1[3]))[:,0:12],rstride=1, cstride=0)
# NB phase factor to make ~ real at 128MHz
ax.plot_wireframe(X, Y, np.squeeze(VI_portC4[3]*np.exp(1j*0.95))[:,0:12],rstride=1, cstride=0)
plt.title('currents on driven end ring capacitors: port C4')

plt.figure()
#plt.plot(np.real(VI_portC4[3])[1,0:12],np.imag(VI_portC4[3])[1,0:12])
plt.plot(np.linspace(1,12,12), np.squeeze(np.squeeze(VI_portC4[3]*np.exp(1j*0.95))[np.where(freqs == f0_target),0:12]))
plt.plot(np.linspace(1,12,12), np.squeeze(np.squeeze(VI_portC1[3]*np.exp(1j*0.2))[np.where(freqs == f0_target),0:12]))
plt.title('currents on end ring capacitors at 128 MHz (driven end ring)')

plt.show()
