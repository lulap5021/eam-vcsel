import matplotlib.pyplot as plt
import numpy as np


from globals import EG_GAAS, HJ, HREV, HRJ, KBEV, KBJ, ME_QW, MZHH_QW, M0, Q, T
from globals import COLOR_CYCLE_5


MPL_SIZE = 16
plt.rc('font', size=MPL_SIZE)
plt.rc('axes', titlesize=MPL_SIZE)




def algaas_electron_mass(al):
    # http://www.ioffe.ru/SVA/NSM/Semicond/AlGaAs/basic.html

    if al>0.45:
        return 0.063 +0.083*al*M0
    else:
        return 0.85 -0.14*al*M0


def algaas_heavy_hole_mass(al):
    # http://www.ioffe.ru/SVA/NSM/Semicond/AlGaAs/basic.html

    return 0.51 +0.25*al*M0




def hamiltonian(sl, particle):
    if particle == 'electron':
        barrier = 'electron_potential_barrier'
        mass = algaas_electron_mass
    else:
        barrier = 'heavy_hole_potential_barrier'
        mass = algaas_heavy_hole_mass

    dim = sl.shape[0]
    h = np.zeros((dim, dim))

    for i in range(dim -1):
        lz = sl.at[i, 'thickness']
        al = sl.at[i, 'al']
        m = mass(al)


        a = -HREV**2 /(2 * m * lz**2)
        b = HREV**2 /(m * lz**2) +sl.at[i, barrier]
        c = a

        h[i +1  , i     ] = a
        h[i     , i     ] = b
        h[i     , i +1  ] = c

    al = sl.at[dim -1, 'al']
    m = mass(al)

    h[dim -1, dim -1] = HREV**2 /(m * sl.at[dim -1, 'thickness']**2) -sl.at[dim -1, barrier]


    return h




def algaas_potential_barrier(al):
    valence_band_offset = 0.33                      # G.Ji, D.Huang, U.K.Reddy, H.Unlu, T.S.Henderson, and H.Morko9 1987
    conduction_band_offset = 1. -valence_band_offset

    # http://www.ioffe.ru/SVA/NSM/Semicond/AlGaAs/bandstr.html
    if al<0.45:
        delta_eg = 1.247*al
    else:
        delta_eg = 1.9 -EG_GAAS +0.125*al +0.143*al**2

    v_hh = valence_band_offset * delta_eg
    v_e = conduction_band_offset * delta_eg


    return v_e, v_hh


def algaas_super_lattice_potential_barrier(sl):
    # check wether the electron potential column exist
    if not 'electron_potential_barrier' in sl:
        sl.insert(sl.shape[1], "electron_potential_barrier", value=np.nan)
    # check wether the heavy hole potential column exist
    if not 'heavy_hole_potential_barrier' in sl:
        sl.insert(sl.shape[1], "heavy_hole_potential_barrier", value=np.nan)

    for i in range(sl.shape[0]):
        v_e, v_hh = algaas_potential_barrier(sl.at[i, 'al'])

        sl.at[i, 'electron_potential_barrier'] = v_e
        sl.at[i, 'heavy_hole_potential_barrier'] = v_hh


    return sl




def solve_schrodinger(sl):
    # compute the potential barrier array
    sl = algaas_super_lattice_potential_barrier(sl)

    # compute the hamiltonian for electrons
    h_e = hamiltonian(sl, 'electron')
    # solve the schrodinger equation
    eig_e, psi_e = np.linalg.eig(h_e)
    # delete the hamiltonian in order to free RAM
    np.savez_compressed('h_e', h_e)
    del h_e

    # compute the hamiltonian for heavy holes
    h_hh = hamiltonian(sl, 'hole')
    # solve the schrodinger equation
    eig_hh, psi_hh = np.linalg.eig(h_hh)
    # delete the hamiltonian in order to free RAM
    del h_hh

    # remove duplicate states
    eig_e = np.unique(eig_e)
    eig_hh = np.unique(eig_hh)

    # sort array
    eig_e = np.sort(eig_e)
    eig_hh = np.sort(eig_hh)

    # remove the unbound states and their respective wavefunction
    eig_e = np.delete(eig_e, np.where(eig_e > sl['electron_potential_barrier'].max()))
    eig_hh = np.delete(eig_hh, np.where(eig_hh > sl['heavy_hole_potential_barrier'].max()))

    np.savez_compressed('psi_e_ttc', psi_e)

    psi_e = psi_e[:eig_e.shape[0], :]
    psi_hh = psi_hh[:eig_hh.shape[0], :]

    # get the barrier potential arrays
    v_e = sl['electron_potential_barrier'].to_numpy()
    v_hh = sl['heavy_hole_potential_barrier'].to_numpy()


    return v_e, v_hh, eig_e, psi_e, eig_hh, psi_hh




def escape_time(al, qw_width, cb_width, e_field):
    # al :          al content [1]
    # qw_width :    quanrum well width [m]
    # cb_width :    confinement barrier width [m]
    # e_field :     electrical field [kV/cm]


    # parameters dependent variables
    e_field = e_field * 1e3 * 1e2                           # [kV/cm] -> [V/m]
    eg_algaas = EG_GAAS +1.427 * al +0.041 * al **2         # Giugni et al (1992)
    me_qb = (0.067 +0.083 * al) * M0                        # e - effective mass in barrier
    mzhh_qb = (0.48 +0.31 * al) * M0                        # heavy hole mass in barrier
    delta_ev = 0.83 * al                                    # valence band discontinuity [eV] - Yu et al (1989)
    delta_ec = eg_algaas -EG_GAAS -delta_ev                 # conduction band discontinuity[eV]


    # ground state energy
    e_el = (HJ ** 2) / (8 * Q * ME_QW * qw_width ** 2)
    e_hh = (HJ ** 2) / (8 * Q * MZHH_QW * qw_width ** 2)

    # field dependant barrier height for qw
    h_el_qw = delta_ev -e_el -e_field * qw_width / 2
    h_hh_qw = delta_ec -e_hh -e_field * qw_width / 2

    # field dependant barrier height for qb
    h_el_qb = abs(Q) * (delta_ev -e_el -e_field * (qw_width +cb_width) / 2)
    h_hh_qb = abs(Q) * (delta_ec -e_hh -e_field * (qw_width +cb_width) / 2)

    # thermionic emission
    tau_th_el = 1 / (np.sqrt(KBJ * T / (2 * np.pi * ME_QW * qw_width ** 2)) * np.exp(-h_el_qw / (KBEV * T)))
    tau_th_hh = 1 / (np.sqrt(KBJ * T / (2 * np.pi * MZHH_QW * qw_width ** 2)) * np.exp(-h_hh_qw / (KBEV * T)))

    # tunneling
    tau_tu_el = 1 / (HRJ * np.pi / (2 * ME_QW * qw_width ** 2) * np.exp(-2 * cb_width * np.sqrt(2 * me_qb * h_el_qb) / HRJ))
    tau_tu_hh = 1 / (HRJ * np.pi / (2 * MZHH_QW * qw_width ** 2) * np.exp(-2 * cb_width * np.sqrt(2 * mzhh_qb * h_hh_qb) / HRJ))

    # escape time
    tau_el = 1 / ((1 / tau_th_el) + (1 / tau_tu_el))
    tau_hh = 1 / ((1 / tau_th_hh) + (1 / tau_tu_hh))


    return tau_el, tau_hh


def plot_esc_time():
    reso = 200
    cb_width = 10.3e-9
    al_x =          np.linspace(0.15, 0.4, num=reso)
    qw_width =      np.linspace(8, 10, num=reso) * 1e-9


    # aluminium variable
    fig = plt.figure(1)
    ax = fig.add_subplot()
    ax.set_ylabel('Escape time (s)')
    ax.set_xlabel('Aluminium content in confinement barriers (%)')
    ax.set_prop_cycle(color=COLOR_CYCLE_5)
    # escape time calculation
    tau_el, tau_hh = escape_time(al_x, 8.27e-9, cb_width, 0.)

    # plot
    ax.semilogy(al_x * 100, tau_el, label='electrons')
    ax.semilogy(al_x * 100, tau_hh, label='heavy holes')
    ax.legend()
    plt.tight_layout()


    # quantum well width variable
    fig = plt.figure(2)
    ax = fig.add_subplot()
    ax.set_ylabel('Escape time (s)')
    ax.set_xlabel('Quantum well width (nm)')
    ax.set_prop_cycle(color=COLOR_CYCLE_5)
    # escape time calculation
    tau_el, tau_hh = escape_time(0.22, qw_width, cb_width, 0.)

    # plot
    ax.semilogy(qw_width * 1e9, tau_el, label='electrons')
    ax.semilogy(qw_width * 1e9, tau_hh, label='heavy holes')
    ax.legend()
    plt.tight_layout()


    plt.show()


def esc_time_contour_al_qw():
    # al :          al content [1]
    # qw_width :    quanrum well width [m]
    # cb_width :    confinement barrier width [m]
    # e_field :     electrical field [kV/cm]
    n_al = 500
    n_qw = n_al - 10
    al = np.linspace(0.2, 0.5, num=n_al)
    qw_width = np.linspace(6e-9, 12e-9, num = n_qw)
    cb_width = 20e-9
    e_field = 50.
    tau_hh = np.zeros([n_qw, n_al])


    # escape time calculation
    for i in range(len(al)):
        for j in range(len(qw_width)):
            tau_el, tau_hh[j, i] = escape_time(al[i], qw_width[j], cb_width, e_field)


    # logarithm to get the exponent
    tau_hh = np.log10(tau_hh)


    # plot
    fig, ax = plt.subplots()

    # labels
    ax.set_xlabel('Aluminium content [%]')
    ax.set_ylabel('Quantum well width [nm]')

    img = ax.contourf(al * 100, qw_width * 1e9, tau_hh, levels=100, cmap='PuBu')
    fig.colorbar(img, ax=ax, label='log10( Escape time [s] )')

    plt.tight_layout()
    plt.show()


