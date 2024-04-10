# physical constants
C = 299792458.0                     # speed of light [m/s]
Q = 1.602176634e-19                 # electron Charge [C] 1C = 1A*1s
M0 = 9.1095e-31                     # electron rest mass [kg]
HREV = 6.582119569e-16              # Reduced Plank constant [eV.s]
HRJ = 1.054571817e-34               # Reduced Plank constant [J.s]
HEV = 4.135667696e-15               # Plank constant [eV.s]
HJ = 6.62607015e-34                 # Plank constant [J.s]
KBEV = 8.617333262145e-5            # Boltzmann constant [eV/K]
KBJ = 1.380649e-23                  # Boltzmann constant [eV/K]
RH = 13.6                           # Rydberg constant for hydrogen [eV]
EPS0 = 8.8541878176e-12             # vacuum permittivity [F/m]
EPSRGAAS = 12.9                     # GaAs relative permittivity [1]
MU0 = 1.25663706212e-6              # vacuum permeability
T = 300.                            # temperature [K]


# optical constants
N0 = 1.                             # air refractive index
NGAAS = 3.642                       # GaAs refractive index
NAL = 2.5702                        # Al refractive index at 852.1nm
NALOX = 1.6                         # oxided Al refractive index


# refractive indices at 550°C
# loads al_array, wavelength_array, n_array (real part) and k_array (imaginary part)
#REFRA_DATA_550 = np.load('data/550C_AlGaAs_refractive_indices/indices_arrays.npz', allow_pickle=True)
# loads n and k coefficients to use with polyval in polynomial_fit
#REFRA_MODEL_550 = np.load('data/550C_AlGaAs_refractive_indices/model.npz', allow_pickle=True)


# distributed Bragg Reflectors constants @ 850nm
L_15_AL_DBR = 6.07310545767979e-08
L_90_AL_DBR = 6.995041869479308e-08


# conversion constants
J_TO_EV = 6.24e18                   # [eV] amount of eV in 1J
EV_TO_J = 1./J_TO_EV                # [J]


# QW GaAs
EG_GAAS = 1.424                     # [eV]
ME_GAAS = 0.067*M0                  # e- effective mass p.40 GaAs
MZHH_GAAS = 0.48*M0                 # heavy hole effective mass p.298 GaAs
ME_QW = ME_GAAS                     # e- effective mass in well
MZHH_QW = MZHH_GAAS                 # heavy hole effective mass in well


# matplotlib color cycler
COLOR_CYCLE_5 = ['#405FB5', '#64ABCE', '#BAE166', '#FDAA35', '#E54F35']
COLOR_CYCLE_10 = ['#405FB5', '#5C91C2', '#64ABCE', '#7EC3DD', '#BAE166', '#FEE048', '#FDAA35', '#F88A4F', '#E54F35', '#D71D2A']



