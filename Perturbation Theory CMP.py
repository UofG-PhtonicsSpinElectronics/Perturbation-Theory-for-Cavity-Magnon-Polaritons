"""

*******************************************************************************

 A python program for using the main principle of perturbation theory to predic 
 the coupling streght between cavity magnon polaritons detailed in:

 Macêdo, Rair, Rory C. Holland, Paul G. Baity, Karen L. Livesey, 
 Robert L. Stamps, Martin P. Weides, and Dmytro A. Bozhko. "An electromagnetic 
 approach to cavity spintronics" Phys. Rev. Applied (In press)/arXiv:2007.11483

 While this program can be easily modified for different systems, we recommend 
 to start using the it in combination with the article above in order to 
 understand the different features presented as well as to better undestand 
 how to relate the quatities calculated and parameters given to other systems. 
 
 Here you'll find:

    * Calculation of the coupling using analytical solutions for the field 
      profiles in a rectangular microwave cavity. This include plots of (a) the 
      field profile (b) the eigenfrequencies of the system, (c) the coupling as 
      a function of x,y positions and (d) the coupling as a fuction of y 
      position at x = 27 mm (anti-node of the cavity's magnetic field). 

    * Eigenfrequencies for a 2D resonator (coplanar waveguide) coupled to a Py 
      film with field and energies inputs from COMSOL.
      
    * S11 parameter calculations.
    
 A few additional points worthy of note: 
 
    * Note that throughout this file, parameters are given for the relevant 
      sections and these might be updated with different names later in the 
      file for different sytems. 

    * This code is written sequentially so that sections can be deleted 
      bottom-up. Therefore, if sections are deleted top-down or from the middle
      the code might not run properly--or at all.
      
 If you use this for any work you might publish, could you please send an email 
 to: Rair.Macedo@Glasgow.ac.uk with an appropriate reference to your work.

*******************************************************************************

 Las updated on Jan 28, 2021
 
*******************************************************************************
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#%%
"""
*******************************************************************************
    Coupling in a 3D rectangular cavity
    
    This section reproduces the data given in Figure 2 of the article.

******************************************************************************* 
"""

""" We start by calculating the oscillating magnetic field, hc, profile inside 
    the cavity, using the Equations provided in Appendix B. The cavity 
    dimensions are:
"""
a = 54.0 * 1e-3 # m
b = 36.0 * 1e-3 # m
c = 7.50 * 1e-3 # m

""" You can change a, b, and, c to obtain other cavity aspect ratios. Since we
    are interested in a TM_{110} mode, we make: 
"""
m = n = 1.0 

kappa_0x = (m * np.pi) / a
    
kappa_0y = (n * np.pi) / b
    
mu_0 = (4.0 * np.pi) * 1e-7        # Vacuum permeability in H/m
    
def h_cx(omega, X, Y):

    h_x = (1j * kappa_0y / (omega * mu_0)) *\
            np.sin(kappa_0x * X) * np.cos(kappa_0y * Y)

    return h_x
    
def h_cy(omega, X, Y):    
    
    h_y = - (1j * kappa_0x / (omega * mu_0)) * \
            np.cos(kappa_0x * X) * np.sin(kappa_0y * Y)
            
    return h_y            

""" We can use the following parameters to plot the h_c field profile at 
    resonance as follows: 
"""

omega_c = 4.98 * 2 * np.pi * 1e9       # Angular cavity resonance frequency Hz

x = np.linspace(0, a, 1000)             # Create x, y grid for positions inside
y = np.linspace(0, b, 1000)             # the cavity
X, Y = np.meshgrid(x, y)
   

h_cx_plot = h_cx(omega_c, X, Y)         # evaluat h_cx at the x,y positions
h_cy_plot = h_cy(omega_c, X, Y)         # evaluat h_cy at the x,y positions

h_c_amplitude = np.abs(h_cx_plot) ** 2 + np.abs(h_cy_plot) ** 2 

""" Plot the amplitude of h_c as given in Figure 2(b)
""" 
plt.figure(1, figsize=(10, 10))
plt.rcParams['font.size'] = 25
plt.title('Figure 2(b) -- |h$_c$|$^2$')
extent = [ Y.min() * 1e3, Y.max() * 1e3, X.min() * 1e3, X.max() * 1e3]
plt.imshow(h_c_amplitude.transpose(), extent=extent, aspect = 'auto')
plt.xlabel('y (mm)')
plt.ylabel('x (mm)')
cbar = plt.colorbar()
cbar.set_ticks([])
plt.show()
#%%
""" Now that we have calculated the fields inside the resonator, we can find 
    the eignefrequencies for the system. This can be done using Eq. (12) as
    follows:
"""

def omega_a(omega_c, omega_0, omega_m, Wp, Wc):
    
    omega_a =  0.5 * (omega_c + omega_0 + \
                      np.sqrt((omega_c - omega_0) ** 2 + \
                              2 * omega_c * omega_m * Wp / Wc))
    return omega_a

def omega_b(omega_c, omega_0, omega_m, Wp, Wc):
    
    omega_b =  0.5 * (omega_c + omega_0 - \
                      np.sqrt((omega_c - omega_0) ** 2 + \
                              2 * omega_c * omega_m * Wp / Wc))
    return omega_b

""" We can define the stored energy in the empty cavity (Wc) and the energy at 
    the sample position/perturbation (Wp) as defined in page 6. We do
"""
eps_0 = 8.85e-12                         # Vacuum permittivity in F/m, 
gamma = 28 * 2.0 * np.pi * 1e9           # Gyromagnetic ratio in Hz/T,
Ms = 0.1758 / mu_0                       # Saturation Magnetisation in T, and
Rs = 0.25                                # Sample radius in mm -- change this  
                                         # for different sample sizes.
omega_m = gamma * mu_0 * Ms
V_s = 4.0/3.0 * np.pi * (Rs * 1e-3) ** 3 # Volume of a YIG sphere in m^3.

x_anode = a / 2.0                        # x and y positions at the anti-node 
y_anode = 0.25 * 1e-3                    # of the oscillating magnetic field h_c

W_c = 0.5 * eps_0 * a * b * c            # Energy stored in an empty cavity

def W_p(h_cx):                           # Energy at sample position allowing
                                         # h_cx to change with position
    W_p = V_s * mu_0 * np.abs(h_cx) ** 2
    
    return W_p
    
    
H_0 = np.linspace(0.172, 0.185, 1000)   # Define a range for the bias field H_0
w0 = gamma * H_0                        # Ferromagnetic resonance is sphere

""" Evaluate the eigenfrequencies omega_a and omega_b at positions x_anode and 
    y-anode, and plot them as given in Figures 2(c)-(e).
""" 

wa_plot = omega_a(omega_c,w0,omega_m, W_p(h_cx(omega_c, x_anode, y_anode)),W_c)
wb_plot = omega_b(omega_c,w0,omega_m, W_p(h_cx(omega_c, x_anode, y_anode)),W_c)

#W_p_new = 6.875 * 1e-15          # In units of Joules
#W_c_new = 2 * 5.55 * 1e-10       # In units of Joules
#
#wa_plot = omega_a(omega_c,w0,omega_m, W_p_new, W_c_new)
#wb_plot = omega_b(omega_c,w0,omega_m, W_p_new, W_c_new)

""" One could also comment lines 173 and 174 and uncomment lines 176 to 180. 
    This would allow using values for Wp and Wc from another source, such as
    comsol as opposed to those just calculated here.
""" 

plt.figure(2, figsize=(8, 10))
plt.rcParams['font.size'] = 25
plt.title('Figure 2(c)')
plt.plot(w0/(2 * np.pi * 1e9 ),wa_plot/(2 * np.pi * 1e9 ), 'r-', linewidth=2.0)
plt.plot(w0/(2 * np.pi * 1e9 ),wb_plot/(2 * np.pi * 1e9 ), 'r-', linewidth=2.0)
plt.xlim(4.95, 5.01)
plt.ylim(4.95, 5.01)
plt.xticks(np.arange(4.96, 5, 0.01))
plt.yticks(np.arange(4.96, 5, 0.01))
plt.xlabel("$\omega_0/2\pi$ (GHz)",)
plt.ylabel("$\omega/2\pi$ (GHz)",)
plt.tight_layout() 
#%%
""" With the equation above, we can then calculate the coupling constant in a 
    few different ways. For this we can use Eq. (13). In order to obtain 
    the coupling at any position x,y within the resonator, as in Fig. 2(f), 
    we also need to use Eq. (C3) for a generalised W_p as follows:
"""

def W_p_xy(omega, X, Y):                          
    """ Eq. (C3) """
    Wp_xy = V_s * mu_0 * \
            (np.abs(h_cx(omega, X, Y)) ** 2 + np.abs(h_cy(omega, X, Y)) ** 2 +\
            1j * (h_cy(omega, X, Y) * np.conj(h_cx(omega, X, Y)) - \
            h_cx(omega, X, Y) * np.conj(h_cy(omega, X, Y))))
    
    return Wp_xy                  

def omega_gap(omega_c, omega_m, X, Y):                          
    """ Eq. (13) """
    w_gap = np.sqrt(2.0 * omega_c * omega_m * W_p_xy(omega_c, X, Y) / W_c)
    
    return w_gap  

wgap_map = 1000 * omega_gap(omega_c, omega_m, X, Y)/(2 * np.pi * 1e9) # in MHz

""" As a function of all x-y positions this gives: 
"""
plt.figure(3, figsize=(10, 10))
plt.rcParams['font.size'] = 25
plt.title('Figure 2(f) -- $\omega_{gap}/2\pi$ (MHz)')
extent = [ Y.min() * 1e3, Y.max() * 1e3, X.min() * 1e3, X.max() * 1e3]
plt.imshow(wgap_map.real.transpose(),extent=extent,cmap=plt.get_cmap('CMRmap'))
plt.xlabel('y (mm)')
plt.ylabel('x (mm)')
cbar = plt.colorbar(ticks=[0, 5, 10, 15])
cbar.ax.set_yticks(['0', '5', '10', '15'])
plt.show()

""" We can also plot omega_gap as a function of y, for a fixed x (a/2). So from
    the anti-node to the node of h_c. Much like Fig. 2(g). This is given below.
""" 
wgap_line = 1000 * omega_gap(omega_c, omega_m, a/2, Y)/(2 * np.pi * 1e9) # MHz

plt.figure(4, figsize=(8, 10))
plt.rcParams['font.size'] = 25
plt.title('Figure 2(g)')
plt.plot(Y * 1e3, wgap_line, 'g-', linewidth=2.0)
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.xticks(np.arange(0, 20, 5))
plt.yticks(np.arange(0, 20, 5))
plt.xlabel('sample y position (mm)',)
plt.ylabel("$\omega_{gap}/2\pi$ (MHz)",)
plt.tight_layout()
#%%

""" If you'd like to simply print out a value of omega_gap (2g) at a particular 
    position, here you have it. We have slightly modified Eq. (13) as the above
    equation gave lists. This now is defined as follows
""" 

def omega_gap_single(omega_c, omega_m, W_p, W_c):                          
    """ Eq. (13) """
    w_gap_value = np.sqrt(2.0 * omega_c * omega_m * W_p / W_c)
    
    return w_gap_value  

""" With this we can then print our a couple of different scenarios. 
    The first being a YIG sphere, same as above, placed at the anti- node of 
    the magnetic field and using the analytic expressions: 
"""
    
gap_1 = omega_gap_single(omega_c,omega_m, W_p_xy(omega_c, a/2.0,0.25*1e-3),W_c) 

print 'omega_gap = ' + str(1000 * np.real(gap_1) / (2.0 * np.pi * 1e9)) + \
      ' MHz (analytics expression for the cavity fields)'

""" And for comparission we take the same sample, and the same position, but
    this time we use field and energies (Wc and Wp) from COMSOL 5.5:
"""
W_p_new = 6.875 * 1e-15          # In units of Joules
W_c_new = 2 * 5.55 * 1e-10       # In units of Joules

gap_2 = omega_gap_single(omega_c, omega_m, W_p_new, W_c_new) # in Hz * 2pi

print 'omega_gap = ' + str(1000 * np.real(gap_2) / (2.0 * np.pi * 1e9)) + \
      ' MHz (fields and energies from COMSOL 5.5)'
 
#%%
"""
*******************************************************************************
    Hybridisation in a 2D microwave resonator
    
    This section reproduces the data given in Figure 3(a) of the article which
    shows a magnetic thin-film stripe coupled to a transmission line resonator.
    For this, we need to start by calculating the appropriat susceptibility
    tensor componnents including demagnetising factors as discussed in Sec. IIA.
    
******************************************************************************* 
"""
def chi_a(Ms, H0, demag_x, demag_z):
    """ Eq. (5) """
    chi_a = Ms / (H0 + (demag_x - demag_z) * Ms) 
    
    return chi_a

def omega_0_demag(gamma, mu_0, demag_x, demag_y, demag_z, H0, Ms):
    """ Eq. (2) """
    w_0_demag = np.sqrt(gamma ** 2 * mu_0 ** 2 * (H0 + (demag_y - demag_z ) \
                        * Ms) * (H0 + (demag_x - demag_z) * Ms))
    
    return w_0_demag

""" With these we can calculate the eigenfrequencies using Eq. (16) as follows.
""" 

def omega_a_demag(omega_c, omega_0, chi_a, wp, wc):

    omega_a_plate =  0.5 * (omega_c + omega_0 + \
                      np.sqrt((omega_c - omega_0) ** 2 + \
                              2 * chi_a * omega_c * omega_0 * wp / wc))
    return omega_a_plate

def omega_b_demag(omega_c, omega_0, chi_a, wp, wc):
    
    omega_b_plate =  0.5 * (omega_c + omega_0 - \
                      np.sqrt((omega_c - omega_0) ** 2 + \
                              2 * chi_a * omega_c * omega_0 * wp / wc))
    return omega_b_plate

""" The parameters for this figure were:
""" 

H0_Py = np.linspace(0.0001, 0.050, 1000) / mu_0 # Static field
Ms_Py = 1 / mu_0                                     # Ms for permalloy
omega_m_Py = gamma * mu_0 * Ms_Py

omega_c_Py = 5.05 * (2 * np.pi) * 1e9                # Resonance freqeuncy (Hz)

demag_Py_x = 0.00520                                 # Demagnetising along x
demag_Py_y = 0.99470                                 # Demagnetising along y
demag_Py_z = 0.00008                                 # Demagnetising along z

omega_0_Py=omega_0_demag(gamma,mu_0,demag_Py_x,demag_Py_y,demag_Py_z,H0_Py,Ms_Py)
                    
chi_a_Py = chi_a(Ms_Py, H0_Py, demag_Py_x, demag_Py_z)

""" In this case, we can't find analytic expressions for the fields, so we 
    shall use values obtained from COMSOL 5.5 for Wp and Wc. These are:
""" 

Wp_Py = 2.868 * 1e-15                                  #In units on Joules
Wc_Py = 3.600 * 1e-11                                  #In units on Joules

""" Now we can evaluate and plot the eigenfrequencies for the system as follows
""" 

wa_rabi_plate = omega_a_demag(omega_c_Py, omega_0_Py, chi_a_Py, Wp_Py, Wc_Py)
wb_rabi_plate = omega_b_demag(omega_c_Py, omega_0_Py, chi_a_Py, Wp_Py, Wc_Py)

plt.figure(5, figsize=(8, 10))
plt.rcParams['font.size'] = 25
plt.title('Figure 3(c)')
plt.plot(H0_Py * mu_0 *1000, wa_rabi_plate/(2 *np.pi *1e9), 'r-',linewidth=2.0)
plt.plot(H0_Py * mu_0 *1000, wb_rabi_plate/(2 *np.pi *1e9), 'r-',linewidth=2.0)
plt.xlim(1, 50)
plt.ylim(4.85, 5.25)
plt.xticks(np.arange(10, 51, 10))
plt.yticks(np.arange(4.9, 5.3, 0.1))
plt.xlabel('$\mu_0$H$_0$ (mT)',)
plt.ylabel("$\omega/2\pi$ (GHz)",)
plt.tight_layout()  
#%%
""" 
*******************************************************************************
    S-parameter calulcation
    
    In this final section we move to the S11 parameter calculation. 
    
    We start by calculating the quality factor for each of the eigenmodes as 
    given in Eq. (18), then we move to the S11 parameter itself for both a 
    single static field value then we move to a case where both field and 
    frequency vary -- both these cases are for a rectangular cavity.
    The last case shows the S11 parameter for a 2D resonator coupled to a YIG
    rectangular prism. 
    
    All these cases are reproducing data from Figure 4.
    
*******************************************************************************    
"""
 
def Qa(omega_a):
    
    Qa = omega_a.real/omega_a.imag
    
    return Qa

def Qb(omega_b):
    
    Qb = omega_b.real/omega_b.imag
    
    return Qb

""" The S11 parameter can then be estimated from Eq. (17) as follows
"""

def Sa(beta, qa, omega, wa):
    Sa = (beta - 1 - 1j * qa * (omega / wa - wa / omega)) / \
        (beta + 1 + 1j * qa * (omega / wa - wa / omega))
    
    return Sa

def Sb(beta, qb, omega, wb):
    Sb = (beta - 1 - 1j * qb * (omega / wb - wb / omega)) / \
        (beta + 1 + 1j * qb * (omega / wb - wb / omega))
    
    return Sb

""" Let us start with a simple case of an experiment where frequency (often 
    from a VNA) is swept and the applied field is fixed at a single value. 
    Taking the externally applied field to be 0.178 T, this corresponds to the
    case given in Fig. 4(b). To reproduce that we do
"""
freq = (np.linspace(4.9, 5.1, 1000)) # Set the frequency range in GHz
omega = freq * 2.0 * np.pi * 1e9     # Converts it into angular frequency in Hz
beta = 1.05                          # Propagation  constant

omega_0_Im = 0.0001 *2 * np.pi *1e9  # Imaginary part of the magnon resonance 
omega_c_Im = 0.0013 *2 * np.pi *1e9  # Imaginary part of the cavity resonance   

omega_0_Re = gamma * 0.178           # Ferromagnetic resonance frequency in Hz
omega_c_Re = 4.98 * 2 * np.pi * 1e9  # Cavity resonance frequency in Hz    

""" The complex resonances for both cavity and magnet, to reproduce both 
    linewidths are given by:
"""
omega_0_complex = omega_0_Re + 1j * omega_0_Im
omega_c_complex = omega_c_Re + 1j * omega_c_Im

""" With these we can evaluate the eigenfrequecies, as done above for Eq. (12)
"""
wa4 = omega_a(omega_c_complex, omega_0_complex, \
      omega_m, W_p(h_cx(omega_c, x_anode, y_anode)), W_c)
wb4 = omega_b(omega_c_complex, omega_0_complex, \
      omega_m, W_p(h_cx(omega_c, x_anode, y_anode)), W_c)
      
#W_p_new = 6.875 * 1e-15          # In units of Joules
#W_c_new = 2 * 5.55 * 1e-10       # In units of Joules
#    
#wa4 = omega_a(omega_c_complex, omega_0_complex, omega_m, W_p_new, W_c_new)
#wb4 = omega_b(omega_c_complex, omega_0_complex, omega_m, W_p_new, W_c_new)      

""" Note that in the eqautions above we have taken W_p and W_c from the 
    analytic expressions calculated at the begining of this code. If you would
    like to plug in values from eswhere, say COMSOL or HFSS, just comment lines
    439 to 442 and uncomment lines 444 to 448. 
    You can then replace our COMSOL values for Wp and Wc for yours as 
    appropriate. For the definitions of Wp and Wc, check the main paper.
    
    Now, with the eigenfrequecies we can evaluate the quality factors:
"""
qa = Qa(wa4)
qb = Qb(wb4)

""" And finally we can evaluate then plot the S11 parameter at a single field:
"""
s11a = Sa(beta, qa, omega, wa4)
s11b = Sb(beta, qb, omega, wb4)
s11 = s11a * s11b

plt.figure(6, figsize=(8, 10))
plt.rcParams['font.size'] = 25
plt.title('Figure 4(b) -- |S$_{11}$|$^2$')
plt.plot(freq, s11.real, 'r-', linewidth=2.0)
plt.xlim(4.95, 5.0)
plt.ylim(0.4, 1.1)
plt.xticks(np.arange(4.95, 5.01, 0.02))
plt.yticks(np.arange(0.4, 1.1, 0.2))
plt.xlabel('$\omega/2\pi$ (GHz)',)
plt.ylabel("|S$_{11}$|",)
plt.tight_layout()  


#%% 
""" Having reproduced a simple case, we can now look at another example: a heat 
    map varying both static field and frequency. This corresponds to an 
    experiment where the field is set and frequency swept. Then, the process is 
    repeated for various different fields. This is the example discussed in 
    Fig. 2(c)-(e), and now we reproduced its theory equivalent in Fig. 4(a).
    
    Here, the frequency range remains the same as our previous plot, but we
    need to reset the external field to vary as follows
"""

Happ = np.linspace(0.172, 0.185, 1000)   # Set a field range in units of T

omega_0_range = gamma * Happ + 1j * omega_0_Im # range of ferromagnetic 
                                               # resonance frequencies for Happ 

XX, YY = np.meshgrid(omega_0_range , omega)

""" Then we evaluate, again, the eigenfrequencies, the quality factors and
    the S11 parameter as follows:
"""
omeag_a_map = omega_a(omega_c_complex, XX, \
       omega_m, W_p(h_cx(omega_c, x_anode, y_anode)), W_c)

omeag_b_map = omega_b(omega_c_complex, XX, \
       omega_m, W_p(h_cx(omega_c, x_anode, y_anode)), W_c)

#W_p_new = 6.875 * 1e-15          # In units of Joules
#W_c_new = 2 * 5.55 * 1e-10       # In units of Joules
#
#omeag_a_map = omega_a(omega_c_complex, XX, omega_m, W_p_new, W_c_new)
#
#omeag_b_map = omega_b(omega_c_complex, XX, omega_m, W_p_new, W_c_new)

""" As explained in the previous plot, comment lines 502-506 and uncomment 
    lines 508-513 in order to use values from elsewhere for Wc and Wp as 
    opposed to analytic results.
"""

Q_a_map = Qa(omeag_a_map)
Q_b_map = Qb(omeag_b_map)

s11a_map = Sa(beta, Q_a_map, YY, omeag_a_map)
s11b_map = Sb(beta, Q_b_map, YY, omeag_b_map)

S11_map = s11a_map * s11b_map

""" Now we can plot this. Before however, we recale the x abd y axis from 
    angular frequency in Hz into frequency in GHz: 
"""

x_map = np.real(XX)/(2.0 * np.pi * 1e9)
y_map = YY/(2.0 * np.pi * 1e9)

plt.figure(7, figsize=(10, 10))
plt.rcParams['font.size'] = 25
plt.title('Figure 4(a) -- |S$_{11}$|$^2$')
plt.contourf(x_map, y_map, np.abs(S11_map), 20, cmap='CMRmap_r')
plt.xlim([4.94, 5.02])
plt.ylim([4.94, 5.02])
plt.xlabel(r'$\omega_0/2\pi$ (GHz)')
plt.ylabel(r'$\omega/2\pi$ (GHz)')
cbar = plt.colorbar(ticks=[0.4, 0.6, 0.8, 1.0])
plt.xticks(np.arange(4.94, 5.01, 0.02))
plt.yticks(np.arange(4.94, 5.01, 0.02))
plt.show()









