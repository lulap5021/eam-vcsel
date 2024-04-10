import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from tqdm import tqdm


from device import ftir as ft
from model import super_lattice_structure as st, optic as op, transfer_matrix_method as tmm
from tool import pandas_tools as pt




MPL_SIZE = 18
plt.rc('font', size=MPL_SIZE)
plt.rc('axes', titlesize=MPL_SIZE)




def mode_asfunc_clad(start_wavelength=840e-9,
                     stop_wavelength=860e-9,
                     n_points=100):
    # Dual-wavelength laser emission from a coupled semiconductor microcavity
    # Pellandini et al.
    # doi: 10.1063/1.119671
    # figure 2.a


    l_eam_clad = np.linspace(6e-9, 18e-9, num=n_points)

    # calculate reflectivity using transfer matrix method
    wavelength = np.linspace(start_wavelength, stop_wavelength, num=n_points)
    r = np.zeros([n_points, n_points])


    # wavelength in [m]
    # wavelength must be a numpy array
    for i in tqdm(range(len(l_eam_clad))):
        for j in range(len(wavelength)):
            # create eam vcsel structure
            sl = st.structure_eam_vcsel(bypass_dbr=True, grading_type='linear digital', mqw_alloy_type='digital', l_eam_clad = l_eam_clad[i])

            sl_r = op.algaas_super_lattice_refractive_index(sl, 0., wavelength[j], lengyel=True)

            n = sl_r['refractive_index'].to_numpy(dtype=np.complex128)
            d = sl_r['thickness'].to_numpy(dtype=float)

            r[j, i] = tmm.reflection(n, d, wavelength[j])


    # save data
    np.savez_compressed('data/r_mode_clad_no_lengy_' + str(n_points) + '_pts', r)


    # plot
    fig, ax = plt.subplots()

    # labels
    ax.set_xlabel('EAM Cladding thickness [nm]')
    ax.set_ylabel('Wavelength [nm]')
    #ax.set_title('Modes of the coupled system as a function of the detuning by adjusting EAM cladding layers thicknesses')

    img = ax.contourf(l_eam_clad * 1e9, wavelength * 1e9, r, levels=100, cmap='PuBu')
    fig.colorbar(img, ax=ax, label='Reflectivity')

    plt.tight_layout()
    plt.show()


def plot_modes_clad(start_wavelength=840e-9,
                    stop_wavelength=860e-9,
                    n_points=300):
    # Dual-wavelength laser emission from a coupled semiconductor microcavity
    # Pellandini et al.
    # doi: 10.1063/1.119671
    # figure 2.a
    l_eam_clad = np.linspace(6e-9, 18e-9, num=n_points)

    # calculate reflectivity using transfer matrix method
    wavelength = np.linspace(start_wavelength, stop_wavelength, num=n_points)

    # save data
    r = np.load('data/r_mode_clad' + str(n_points) + '_pts.npz')['arr_0']


    # plot
    fig, ax = plt.subplots()

    # labels
    ax.set_xlabel('EAM Cladding thickness [nm]')
    ax.set_ylabel('Wavelength [nm]')
    #ax.set_title('Modes of the coupled system as a function of the detuning by adjusting EAM cladding layers thicknesses')

    img = ax.contourf(l_eam_clad * 1e9, wavelength * 1e9, r, levels=100, cmap='PuBu')
    fig.colorbar(img, ax=ax, label='Reflectivity')

    plt.tight_layout()
    plt.show()




def eam_vcsel_coupling(n_points=500):
    # we will calculate for each cladding the ratio of the eam max electromagnetic amplitude
    # divided by the max amplitude within the vcsel
    # when there is no absorption (real(refractive index))
    # at the vcsel's FP resonnance

    # maximum coupling : ratio_em_amp = 1
    # no coupling : ratio_em_amp = 0


    # arrays of max amplitude in each cavity
    vcsel_em_amp = np.zeros(n_points)
    eam_em_amp = np.zeros(n_points)

    # array of eam cladding thickness
    l_eam_clad = np.linspace(6e-9, 12e-9, num=n_points)


    for i in tqdm(range(len(l_eam_clad))):
        # create eam vcsel structure
        sl = st.structure_eam_vcsel(grading_type='linear digital', mqw_alloy_type='digital', l_eam_clad = l_eam_clad[i], l_vcsel_clad=15e-9)

        # calculate the vcsel's FP resonnance
        vcsel_mode_wavelength = get_vcsel_mode_wavelength(sl)

        # calculate the refractive index
        sl = op.algaas_super_lattice_refractive_index(sl, 0., vcsel_mode_wavelength, lengyel=False)

        # remove the imaginary part of the refractive index
        sl['refractive_index'] = sl['refractive_index'].apply(np.real).astype(float)

        # decrease step size
        sl = pt.cut_in_equal_layers_thickness(sl, 1e-8)

        # calculate the em amplitudes at vcsel mode
        em = tmm.em_amplitude_scattering_matrix(sl, vcsel_mode_wavelength, normalize=False)

        # extract the max amplitude of each cavity
        # index of middle contact is around layer 800 on 3000 layers
        mid = int(8/30 * len(em))
        eam_em_amp[i] = np.max(em[0:mid])
        vcsel_em_amp[i] = np.max(em[mid:])


    # ratio
    ratio_em_amp = eam_em_amp / vcsel_em_amp


    # save data
    np.savez_compressed('data/ratio_em_amp' +str(n_points) +'_pts', ratio_em_amp)


    # plot
    fig, ax = plt.subplots()
    ax.set_xlabel('EAM Cladding thickness (nm)')
    ax.set_ylabel('Coupling (1)')
    ax.plot(l_eam_clad *1e9, ratio_em_amp, color='tab:blue')


    plt.show()


def plot_coupling(n_points=1000):
    # we will calculate for each cladding the ratio of the eam max electromagnetic amplitude
    # divided by the max amplitude within the vcsel
    # when there is no absorption (real(refractive index))
    # at the vcsel's FP resonnance

    # maximum coupling : ratio_em_amp = 1
    # no coupling : ratio_em_amp = 0

    # array of eam cladding thickness
    l_eam_clad = np.linspace(6e-9, 12e-9, num=n_points) * 1e9

    # load data
    ratio_em_amp = np.load('data/ratio_em_amp' +str(n_points) +'_pts.npz')['arr_0']

    fit = np.polyfit(l_eam_clad, np.log(ratio_em_amp), 1)

    r_fit = np.exp(fit[1])*np.exp(fit[0]) ** l_eam_clad

    # plot
    fig, ax = plt.subplots()
    ax.set_xlabel('EAM Cladding thickness (nm)')
    ax.set_ylabel('Coupling (1)')
    ax.plot(l_eam_clad, r_fit, color='tab:blue')

    plt.tight_layout()
    plt.show()



def eam_vcsel_em_amplitude():
    # create eam vcsel structure
    sl = st.structure_eam_vcsel(grading_type='linear digital', mqw_alloy_type='digital')

    # calculate the vcsel's FP resonnance
    vcsel_mode_wavelength = get_vcsel_mode_wavelength(sl)

    # calculate the refractive index
    sl = op.algaas_super_lattice_refractive_index(sl, 0., vcsel_mode_wavelength, lengyel=False)

    # remove the imaginary part of the refractive index
    sl['refractive_index'] = sl['refractive_index'].apply(np.real).astype(float)

    # decrease step size
    sl = pt.cut_in_equal_layers_thickness(sl, 1e-8)


    # calculate the em amplitudes at vcsel mode
    em = tmm.em_amplitude_scattering_matrix(sl, vcsel_mode_wavelength, normalize=True)


    # drop substrate
    sl = sl[:-1]
    sl = pt.add_depth_column(sl)
    z = sl['depth'].to_numpy()


    # plot
    fig, ax = plt.subplots()
    ax.set_xlabel('Depth (nm)')
    ax.set_ylabel('Electromagnetic Amplitude (1)')
    ax.plot(z *1e9, em[:len(z)], color='tab:blue')

    plt.tight_layout()
    plt.show()




def eam_vcsel_temp_coupling(start_wavelength=840e-9,
                            stop_wavelength=860e-9):
    n_temp = 13
    temperature = np.linspace(30, 90, n_temp, dtype=int)

    wavelength, r = ft.import_vertex_70_data_dpt('data/epitaxy/A1398/ftir/30.dpt',
                                                 document_folder=False, normalize=True)
    diff_arr = np.absolute(wavelength - start_wavelength)
    idx_start = diff_arr.argmin()
    diff_arr = np.absolute(wavelength - stop_wavelength)
    idx_stop = diff_arr.argmin()

    wavelength = wavelength[idx_start:idx_stop]
    n_points = len(wavelength)
    z = np.zeros((n_points, n_temp))


    for i in range(n_temp):
        w, r = ft.import_vertex_70_data_dpt('data/epitaxy/A1398/ftir/' +str(temperature[i]) +'.dpt', document_folder=False, normalize=True)
        r = savgol_filter(r[idx_start:idx_stop], 150, 6)
        z[:, i] = r


    # plot
    fig, ax = plt.subplots()

    # labels
    ax.set_xlabel('Temperature [°C]')
    ax.set_ylabel('Wavelength [nm]')

    img = ax.contourf(temperature, wavelength * 1e9, z, levels=100, cmap='PuBu')
    fig.colorbar(img, ax=ax, label='Reflectivity')

    plt.tight_layout()
    plt.show()





def get_vcsel_mode_wavelength(sl,
                              fp_resonnance_frontier=848.7e-9,
                              stop_wavelength=851e-9,
                              n_points=1000):
    # return wavelength [m]
    # will get the wavelength of the vcsel mode
    # fp_resonnance_frontier is a wavelength between the two cavities of high reflection
    # calculate reflectivity using transfer matrix method
    wavelength = np.linspace(fp_resonnance_frontier, stop_wavelength, num=n_points)
    r = np.zeros(n_points)


    # wavelength in [m]
    # wavelength must be a numpy array
    for i in range(len(wavelength)):
        sl_r = op.algaas_super_lattice_refractive_index(sl, 0., wavelength[i], lengyel=True)

        n = sl_r['refractive_index'].to_numpy(dtype=np.complex128)
        d = sl_r['thickness'].to_numpy(dtype=float)

        r[i] = tmm.reflection(n, d, wavelength[i])


    # get index of minimum value (fp resonnance)
    idx = np.argmin(r)


    # return wavelength [m]
    return wavelength[idx]
