from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as mplt
import numpy as np
import scipy.optimize
from tqdm import tqdm


from device import ftir
from model import optic as op
from model import transfer_matrix_method as tmm
from tool import pandas_tools as pt
from model import super_lattice_structure as sls




MPL_SIZE = 24
mplt.rc('font', size=MPL_SIZE)
mplt.rc('axes', titlesize=MPL_SIZE)
mplt.rcParams['font.family'] = 'Calibri'


# position where the ftir measurement was taken, distance away from center of the vcsel 2 wafer [m]
vcsel_2_dist_mm_from_center = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35]) * 1e-3
# wavelength of FP resonnance of the vcsel and eam on the vcsel 2 structure [m]
vcsel_2_lambda_vcsel_raw = np.array([853.4, 853.3, 853.24, 853.09, 852.87, 852.59, 852.27, 851.93, 851.59, 851.17, 850.62, 850.1, 849.67]) * 1e-9
vcsel_2_lambda_eam_raw = np.array([851.5, 851.4, 851.24, 851.09, 850.88, 850.6, 850.34, 850.02, 849.65, 849.24, 848.75, 848.16, 847.67]) * 1e-9




def fp_wavelength_asfunc_pos():
    # alias vectors
    x = vcsel_2_dist_mm_from_center
    lambda_vcsel = vcsel_2_lambda_vcsel_raw
    lambda_eam = vcsel_2_lambda_eam_raw


    # smooth vectors with least square method
    # y = mx + c
    # x being ftir measurement position
    # y being a FP resonance wavelength
    a_vcsel = np.vstack([x, np.ones(len(x))]).T
    m_v, c_v = np.linalg.lstsq(a_vcsel, lambda_vcsel, rcond=None)[0]

    a_eam = np.vstack([x, np.ones(len(x))]).T
    m_e, c_e = np.linalg.lstsq(a_eam, lambda_eam, rcond=None)[0]

    # polynomial polyfit
    v_ply_fit = np.polyfit(x, lambda_vcsel, 2)
    e_ply_fit = np.polyfit(x, lambda_eam, 2)

    # delta FP resonance between the two cavities
    delta_fp = m_v * x + c_v - m_e * x - c_e


    # plot
    # define subplots
    fig, ax = mplt.subplots()

    # add first line to plot
    ax.plot(x * 1e3, delta_fp * 1e9, color='slateblue')
    ax.plot(x * 1e3, (lambda_vcsel - lambda_eam) * 1e9, '2', color='slateblue')

    # add x-axis and y-axis label
    ax.set_xlabel('Distance away from the center of the wafer (mm)')
    ax.set_ylabel('Delta FP resonnance (nm)', color='slateblue')
    ax.set_ylim([0, 4])

    # define second y-axis that shares x-axis with current plot
    ax2 = ax.twinx()

    # add second line to plot
    ax2.plot(x * 1e3, (v_ply_fit[0] * x *x + v_ply_fit[1] * x + v_ply_fit[2]) * 1e9, color='mediumseagreen')
    ax2.plot(x * 1e3, lambda_vcsel * 1e9, '+', color='mediumseagreen', label='VCSEL')
    ax2.plot(x * 1e3, (e_ply_fit[0] * x *x + e_ply_fit[1] * x + e_ply_fit[2]) * 1e9, color='olive')
    ax2.plot(x * 1e3, lambda_eam * 1e9, 'x', color='olive', label='EAM')
    ax2.legend()

    # add second y-axis label
    ax2.set_ylabel('Cavity FP resonnance (nm)', color='olivedrab')




    mplt.tight_layout()
    mplt.show()




def reflectivity(start_wavelength=840e-9,
                 stop_wavelength=860e-9,
                 electric_field=0.,
                 n_points=150):
    # weak coupling
    sl_w = sls.structure_eam_vcsel(bypass_dbr=True, l_eam_clad=11.2e-9, l_vcsel_clad = 17.5e-9)
    # strong coupling
    sl_s = sls.structure_eam_vcsel(bypass_dbr=True, l_eam_clad=12.5e-9, l_vcsel_clad = 17.5e-9)
    # out of tune
    sl_o = sls.structure_eam_vcsel(bypass_dbr=True, l_eam_clad=20e-9, l_vcsel_clad = 17.5e-9)


    # calculate reflectivity using transfer matrix method
    wavelength = np.linspace(start_wavelength, stop_wavelength, num=n_points)
    r_w = np.zeros(len(wavelength))
    r_s = np.zeros(len(wavelength))
    r_o = np.zeros(len(wavelength))


    # wavelength in [m]
    # wavelength must be a numpy array
    for i in tqdm(range(len(wavelength))):
        sl_r_w = op.algaas_super_lattice_refractive_index(sl_w, electric_field, wavelength[i], lengyel=True)
        sl_r_s = op.algaas_super_lattice_refractive_index(sl_s, electric_field, wavelength[i], lengyel=True)
        sl_r_o = op.algaas_super_lattice_refractive_index(sl_o, electric_field, wavelength[i], lengyel=True)

        n_w = sl_r_w['refractive_index'].to_numpy(dtype=np.complex128)
        n_s = sl_r_s['refractive_index'].to_numpy(dtype=np.complex128)
        n_o = sl_r_o['refractive_index'].to_numpy(dtype=np.complex128)

        d_w = sl_r_w['thickness'].to_numpy(dtype=float)
        d_s = sl_r_s['thickness'].to_numpy(dtype=float)
        d_o = sl_r_o['thickness'].to_numpy(dtype=float)


        r_w[i] = tmm.reflection(n_w, d_w, wavelength[i])
        r_s[i] = tmm.reflection(n_s, d_s, wavelength[i])
        r_o[i] = tmm.reflection(n_o, d_o, wavelength[i])


    #wftir, r_sol = ftir.import_vertex_70_data_dpt('24-06/vcsel2_21mm.0.dpt')
    wftir, r_sol = ftir.import_vertex_70_data_dpt('24-06\\vcsel2_21mm.0.dpt', linux=False)
    # find index at 800 and 900 nm
    idx_min = np.absolute(wftir - start_wavelength).argmin()
    idx_max = np.absolute(wftir - stop_wavelength).argmin()
    # reduce vectors range
    wftir = wftir[idx_min:idx_max]
    r_sol = r_sol[idx_min:idx_max]


    # plot
    fig, ax = mplt.subplots()
    mplt.plot(wavelength * 1e9, r_w, color='dodgerblue', label='Weak coupling')
    mplt.plot(wavelength * 1e9, r_o, color='orange', label='Out of tune')
    mplt.plot(wavelength * 1e9, r_s, color='deeppink', label='Strong coupling')
    mplt.plot(wftir * 1e9, r_sol, color='black', label='FTIR measurement')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Reflectivity')
    ax.legend(loc=2)


    mplt.show()


def reflectivity_intro(start_wavelength=700e-9,
                 stop_wavelength=1000e-9,
                 electric_field=0.,
                 n_points=500):
    # weak coupling
    sl_w = sls.structure_eam_vcsel(bypass_dbr=True, l_eam_clad=11.2e-9, l_vcsel_clad = 17.5e-9)
    # strong coupling
    sl_s = sls.structure_eam_vcsel(bypass_dbr=True, l_eam_clad=12.5e-9, l_vcsel_clad = 17.5e-9)

    # calculate reflectivity using transfer matrix method
    wavelength = np.linspace(start_wavelength, stop_wavelength, num=n_points)
    r_w = np.zeros(len(wavelength))
    r_s = np.zeros(len(wavelength))




    # wavelength in [m]
    # wavelength must be a numpy array
    for i in tqdm(range(len(wavelength))):
        sl_r_w = op.algaas_super_lattice_refractive_index(sl_w, electric_field, wavelength[i], lengyel=True)
        sl_r_s = op.algaas_super_lattice_refractive_index(sl_s, electric_field, wavelength[i], lengyel=True)

        n_w = sl_r_w['refractive_index'].to_numpy(dtype=np.complex128)
        n_s = sl_r_s['refractive_index'].to_numpy(dtype=np.complex128)

        d_w = sl_r_w['thickness'].to_numpy(dtype=float)
        d_s = sl_r_s['thickness'].to_numpy(dtype=float)

        r_w[i] = tmm.reflection(n_w, d_w, wavelength[i])
        r_s[i] = tmm.reflection(n_s, d_s, wavelength[i])


    # plot
    fig, ax = mplt.subplots()
    mplt.plot(wavelength * 1e9, r_s, color='deeppink')
    mplt.plot(wavelength * 1e9, r_w, color='dodgerblue')


    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Reflectivity')
    #ax.legend(loc=2)


    mplt.tight_layout()
    mplt.show()




def electromagnetic_amplitude():
    lamba_fp_w = 852e-9
    lamba_fp_s = 852.2e-9


    # weak coupling
    sl_w = sls.structure_eam_vcsel(bypass_dbr=True, l_eam_clad=11.2e-9, l_vcsel_clad = 17.5e-9)
    # strong coupling
    sl_s = sls.structure_eam_vcsel(bypass_dbr=True, l_eam_clad=12.5e-9, l_vcsel_clad = 17.5e-9)

    sl_w = op.algaas_super_lattice_refractive_index(sl_w, 0., lamba_fp_w)
    sl_w = pt.cut_in_equal_layers_thickness(sl_w, 1e-8)
    sl_s = op.algaas_super_lattice_refractive_index(sl_s, 0., lamba_fp_s)
    sl_s = pt.cut_in_equal_layers_thickness(sl_s, 1e-8)

    # scattering matrix or transmission matrix for transfer method
    em_w = tmm.em_amplitude_scattering_matrix(sl_w, lamba_fp_w)
    em_s = tmm.em_amplitude_scattering_matrix(sl_s, lamba_fp_s)

    # insert the electromagnetic amplitude within the super lattice
    sl_w.insert(sl_w.shape[1], 'electromagnetic_amplitude', value=em_w)
    sl_s.insert(sl_s.shape[1], 'electromagnetic_amplitude', value=em_s)

    sl_w = clean_sl(sl_w)
    sl_s = clean_sl(sl_s)
    d_w = sl_w['depth'].to_numpy(dtype=float)
    d_s = sl_s['depth'].to_numpy(dtype=float)
    e_w = sl_w['electromagnetic_amplitude'].to_numpy(dtype=float)
    e_s = sl_s['electromagnetic_amplitude'].to_numpy(dtype=float)


    # dummy structure for refractive index plot
    sl_dummy = dummify(sl_w)
    refra = sl_dummy['refractive_index'].to_numpy(dtype=complex).real


    # flip everything to be in the same orientation than reflectometry
    refra = refra[::-1]
    d_w = d_w.max() - d_w[::-1]
    d_s = d_s.max() - d_s[::-1]
    e_w = e_w[::-1]
    e_s = e_s[::-1]


    # plot
    # setup
    fig = mplt.figure(constrained_layout=True)
    gs = GridSpec(3, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    ax3 = fig.add_subplot(gs[2, :])

    # traces
    # structure
    ax1.plot(d_w * 1e6, refra, linewidth=0.8, color='slateblue')
    # strong coupling
    ax2.plot(d_s * 1e6, e_s, color='deeppink')
    # weak coupling
    ax3.plot(d_w * 1e6, e_w, color='dodgerblue')
    ax3.set_xlabel('Z (nm)')
    ax1.set_ylabel('Refractive index')
    ax2.set_ylabel('$|E(z)|^{2}$')
    ax3.set_ylabel('$|E(z)|^{2}$')

    # add annotation
    at = AnchoredText(
        'a)', prop=dict(size=18), frameon=False, loc='upper left')
    ax1.add_artist(at)
    at = AnchoredText(
        'b)', prop=dict(size=18), frameon=False, loc='upper left')
    ax2.add_artist(at)
    at = AnchoredText(
        'c)', prop=dict(size=18), frameon=False, loc='upper left')
    ax3.add_artist(at)

    # display
    mplt.tight_layout()
    mplt.show()




def clean_sl(sl):
    # remove the air layer
    sl = sl[sl.name != 'air']
    # remove the substrate layer
    sl = sl[sl.name != 'substrate']


    # reset indices
    sl = sl.reset_index(drop=True)


    # check wether the depth column exist
    if not 'depth' in sl:
        sl.insert(sl.shape[1], 'depth', value=np.nan)


    # calculate the depth of every layer
    for i in range(sl.shape[0]):
        sl.at[i, 'depth'] = sl.loc[0:i, 'thickness'].sum()


    return sl


def dummify(sl):
    for i in range(len(sl)):
        name = sl.at[i, 'name']

        if 'grading' in name:
            sl.at[i, 'al'] = 0.5
            sl.at[i, 'refractive_index'] = 3.25

        if 'confinement barrier' in name:
            sl.at[i, 'al'] = 0.2
            sl.at[i, 'refractive_index'] = 3.45


    return sl






def ftir_fit():
    #wavelength, r_sol = ftir.import_vertex_70_data_dpt('24-06/vcsel2_21mm.0.dpt')
    wavelength, r_sol = ftir.import_vertex_70_data_dpt('24-06\\vcsel2_21mm.0.dpt', linux=False)
    # find index at 800 and 900 nm
    idx_800 = np.absolute(wavelength - 820e-9).argmin()
    idx_900 = np.absolute(wavelength - 880e-9).argmin()
    # reduce vectors range
    wavelength = wavelength[idx_800:idx_900]
    r_sol = r_sol[idx_800:idx_900]


    def reflec(wavelength, v_ga6, v_ga11, v_al5, v_al12):
        sl = sls.structure_eam_vcsel(bypass_dbr=True, l_eam_clad=8e-9,
                                       v_ga6=v_ga6,
                                       v_ga11=v_ga11,
                                       v_al5=v_al5,
                                       v_al12=v_al12)

        r = np.zeros(len(wavelength))

        for i in tqdm(range(len(wavelength))):
            sl_r = op.algaas_super_lattice_refractive_index(sl, 0., wavelength[i], lengyel=True)

            n = sl_r['refractive_index'].to_numpy(dtype=np.complex128)
            d = sl_r['thickness'].to_numpy(dtype=float)

            r[i] = tmm.reflection(n, d, wavelength[i])


        return r


    factmi = 0.95
    factma = 1.05
    bound_min = (100. * factmi, 850. * factmi, 900. * factmi, 150. * factmi)
    bound_max = (100. * factma, 850. * factma, 900. * factma, 150. * factma)
    param_bounds = (bound_min, bound_max)
    params, pcov = scipy.optimize.curve_fit(reflec, wavelength, r_sol, p0=[100., 850., 900., 150.], bounds=param_bounds, method='trf')


    print(params, pcov)