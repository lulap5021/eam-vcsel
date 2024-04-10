import matplotlib.pyplot as mplt
import numpy as np
from tqdm import tqdm


from globals import COLOR_CYCLE_5
from model import optic as op
from model import transfer_matrix_method as tmm
from model import sqw_lengyel_absorption as sla
from tool import structure_macro as stm
from model import super_lattice_structure as sls




MPL_SIZE = 18
mplt.rc('font', size=MPL_SIZE)
mplt.rc('axes', titlesize=MPL_SIZE)
mplt.rcParams['font.family'] = 'Calibri'




def sqw_absorption():
    # al_x in [1]
    # qw_width in [m]
    # electric field in [V/m]
    n_curves = 5
    al_x =          np.linspace(0.15, 0.4, num=n_curves)
    qw_width =      np.linspace(8e-9, 10e-9, num=n_curves)
    e_field =       np.linspace(0., 140., num=n_curves) * 1e3 * 1e2


    # aluminium variable
    fig = mplt.figure(1)
    ax = fig.add_subplot()
    ax.set_xlim([835, 860])
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Absorption $(10^{4} cm^{-1})$')
    ax.set_prop_cycle(color=COLOR_CYCLE_5)
    for i in range(n_curves):
        # absorption calculation
        wavelength, alpha, psie, psih, zz = sla.gaas_sqw_absorption(al_x[i], 8.5e-9, 6000)

        # plot
        ax.plot(wavelength * 1e3, alpha * 1e-4, label=str(int(al_x[i]*1e2)) +'%')
    ax.legend()


    # quantum well width variable
    fig = mplt.figure(2)
    ax = fig.add_subplot()
    ax.set_xlim([835, 860])
    ax.set_xlabel('Wavelength (nm)')
    ax.set_prop_cycle(color=COLOR_CYCLE_5)
    for i in range(n_curves):
        # absorption calculation
        wavelength, alpha, psie, psih, zz = sla.gaas_sqw_absorption(0.3, qw_width[i], 6000)

        # plot
        ax.plot(wavelength * 1e3, alpha * 1e-4, label=str(int(qw_width[i]*1e9)) +'nm')
    ax.legend()


    # electric field variable
    fig = mplt.figure(3)
    ax = fig.add_subplot()
    ax.set_xlim([835, 860])
    ax.set_ylabel('Absorption $(10^{4} cm^{-1})$')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_prop_cycle(color=COLOR_CYCLE_5)
    for i in range(n_curves):
        # absorption calculation
        wavelength, alpha, psie, psih, zz = sla.gaas_sqw_absorption(0.3, 8.5e-9, e_field[i])

        # plot
        ax.plot(wavelength * 1e3, alpha * 1e-4, label=str(int(e_field[i]*1e-5)) +'kV/cm')
    ax.legend(loc='lower left')


    mplt.tight_layout()
    mplt.show()




def reflectivity(start_wavelength=800e-9,
                 stop_wavelength=900e-9,
                 electric_field=0.,
                 n_points=500):
    sl = stm.eam_bypass_structure()

    # calculate reflectivity using transfer matrix method
    wavelength = np.linspace(start_wavelength, stop_wavelength, num=n_points)
    r = np.zeros(len(wavelength))

    # wavelength in [m]
    # wavelength must be a numpy array
    for i in tqdm(range(len(wavelength))):
        sl_r = op.algaas_super_lattice_refractive_index(sl, electric_field, wavelength[i], lengyel=True)
        #sl_r = op.oxidation(sl_r)

        n = sl_r['refractive_index'].to_numpy(dtype=np.complex128)
        d = sl_r['thickness'].to_numpy(dtype=float)

        r[i] = tmm.reflection(n, d, wavelength[i])


    # plot
    fig, ax = mplt.subplots()
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Reflectivity (1)')
    ax.plot(wavelength *1e9, r, color='slateblue')
    #ax.vlines([846.8, 848.9], 0, 1, transform=ax.get_xaxis_transform(), colors='lavender')


    mplt.show()


def reflectivity_both_struct(start_wavelength=800e-9,
                            stop_wavelength=900e-9,
                            electric_field=0.,
                            n_points=500):
    # calculate reflectivity using transfer matrix method
    wavelength = np.linspace(start_wavelength, stop_wavelength, num=n_points)
    r_e = np.zeros(len(wavelength))
    r_v = np.zeros(len(wavelength))

    sl_eam = stm.eam_bypass_structure()
    sl_vcsel = stm.vcsel_structure()


    # wavelength in [m]
    # wavelength must be a numpy array
    for i in tqdm(range(len(wavelength))):
        sl_eam = op.algaas_super_lattice_refractive_index(sl_eam, electric_field, wavelength[i], lengyel=True)
        sl_vcsel = op.algaas_super_lattice_refractive_index(sl_vcsel, 0., wavelength[i], lengyel=False)

        n_e = sl_eam['refractive_index'].to_numpy(dtype=np.complex128)
        d_e = sl_eam['thickness'].to_numpy(dtype=float)

        n_v = sl_vcsel['refractive_index'].to_numpy(dtype=np.complex128)
        d_v = sl_vcsel['thickness'].to_numpy(dtype=float)

        r_e[i] = tmm.reflection(n_e, d_e, wavelength[i])
        r_v[i] = tmm.reflection(n_v, d_v, wavelength[i])

    # plot
    fig, ax = mplt.subplots()
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Reflectivity (1)')
    ax.plot(wavelength *1e9, r_e, color='cornflowerblue', label='EAM')
    ax.plot(wavelength * 1e9, r_v, color='palevioletred', label='VCSEL')
    ax.vlines([846.8, 848.9], 0, 1, transform=ax.get_xaxis_transform(), colors='lavender')
    ax.legend()


    mplt.show()




def lengyel_test():
    electric_field = 120 * 1e5
    qw_width = 9.5e-9
    al_x = 0.4


    # electric field variable
    fig = mplt.figure(1)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim([835, 860])
    ax.set_xlabel('Wavelength [nm]')
    ax.set_ylabel('Absorption $[x10^{4} cm^{-1}]$')
    # absorption calculation
    wavelength, alpha, psie, psih, zz = sla.gaas_sqw_absorption(al_x, qw_width, electric_field)

    # plot
    ax.plot(wavelength * 1e3, alpha * 1e-4)




def reflectivity_comp(start_wavelength=800e-9,
                        stop_wavelength=900e-9,
                        electric_field=0.,
                        n_points=2000):
    sl_4_6 = sls.structure_eam_vcsel(eam_only=True,
                                      top_eam_dbr_period = 4,
                                      shared_dbr_period = 6)
    sl_6_8 = sls.structure_eam_vcsel(eam_only=True,
                                      top_eam_dbr_period = 6,
                                      shared_dbr_period = 8)
    sl_8_10 = sls.structure_eam_vcsel(eam_only=True,
                                      top_eam_dbr_period = 8,
                                      shared_dbr_period = 10)
    sl_10_12 = sls.structure_eam_vcsel(eam_only=True,
                                       top_eam_dbr_period = 10,
                                       shared_dbr_period = 12)


    # calculate reflectivity using transfer matrix method
    wavelength = np.linspace(start_wavelength, stop_wavelength, num=n_points)
    r_4_6 = np.zeros(len(wavelength))
    r_6_8 = np.zeros(len(wavelength))
    r_8_10 = np.zeros(len(wavelength))
    r_10_12 = np.zeros(len(wavelength))


    # wavelength in [m]
    # wavelength must be a numpy array
    for i in tqdm(range(len(wavelength))):
        sl_4_6 = op.algaas_super_lattice_refractive_index(sl_4_6, electric_field, wavelength[i], lengyel=False)
        sl_6_8 = op.algaas_super_lattice_refractive_index(sl_6_8, electric_field, wavelength[i], lengyel=False)
        sl_8_10 = op.algaas_super_lattice_refractive_index(sl_8_10, electric_field, wavelength[i], lengyel=False)
        sl_10_12 = op.algaas_super_lattice_refractive_index(sl_10_12, electric_field, wavelength[i], lengyel=False)


        n_4_6 = sl_4_6['refractive_index'].to_numpy(dtype=np.complex128)
        d_4_6 = sl_4_6['thickness'].to_numpy(dtype=float)
        n_6_8 = sl_6_8['refractive_index'].to_numpy(dtype=np.complex128)
        d_6_8 = sl_6_8['thickness'].to_numpy(dtype=float)
        n_8_10 = sl_8_10['refractive_index'].to_numpy(dtype=np.complex128)
        d_8_10 = sl_8_10['thickness'].to_numpy(dtype=float)
        n_10_12 = sl_10_12['refractive_index'].to_numpy(dtype=np.complex128)
        d_10_12 = sl_10_12['thickness'].to_numpy(dtype=float)


        r_4_6[i] = tmm.reflection(n_4_6, d_4_6, wavelength[i])
        r_6_8[i] = tmm.reflection(n_6_8, d_6_8, wavelength[i])
        r_8_10[i] = tmm.reflection(n_8_10, d_8_10, wavelength[i])
        r_10_12[i] = tmm.reflection(n_10_12, d_10_12, wavelength[i])


    # plot
    fig, ax = mplt.subplots()
    ax.set_prop_cycle(color=COLOR_CYCLE_5)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Reflectivity (1)')
    ax.plot(wavelength * 1e9, r_4_6, label='4 & 6')
    ax.plot(wavelength * 1e9, r_6_8, label='6 & 8')
    ax.plot(wavelength * 1e9, r_8_10, label='8 & 10')
    ax.plot(wavelength * 1e9, r_10_12, label='10 & 12')

    ax.legend(loc='lower right')

    mplt.tight_layout()
    mplt.show()
