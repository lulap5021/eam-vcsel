# importing sys
import sys

# adding tool to the system path
sys.path.insert(0, '/home/llaplanche/PycharmProjects/eam-vcsel/tool')


import matplotlib.pyplot as mplt
import numpy as np
import scipy.io
from scipy.interpolate import splrep, splev
from tqdm import tqdm


from calculation import epitaxy as epx
from device import oes as oe, ftir as vtx
from globals import COLOR_CYCLE_5
from tool import pandas_tools as pt, plot as plt, structure_macro as stm
from model import super_lattice_structure as st, optic as op, transfer_matrix_method as tmm, quantum as qt


MPL_SIZE = 24
mplt.rc('font', size=MPL_SIZE)
mplt.rc('axes', titlesize=MPL_SIZE)




def refra_doping(bypass_dbr=True, electric_field=0., wavelength=850e-9):
    sl = st.structure_eam_vcsel(bypass_dbr=bypass_dbr, vcsel_only=False, grading_type='none', mqw_alloy_type='none',
                                l_eam_clad=15e-9, l_vcsel_clad=15e-9)

    sl = op.algaas_super_lattice_refractive_index(sl, electric_field, wavelength, lengyel=True)
    plt.plot_refra_doping(sl)


def refra(bypass_dbr=True, electric_field=0., wavelength=850e-9):
    sl = st.structure_eam_vcsel(bypass_dbr=bypass_dbr, vcsel_only=False, eam_only=False, grading_type='linear digital', mqw_alloy_type='digital',
                                l_eam_clad=15e-9, l_vcsel_clad=15e-9)

    sl = op.algaas_super_lattice_refractive_index(sl, electric_field, wavelength, lengyel=True)
    plt.plot_refra(sl)


def al_doping(bypass_dbr=True):
    sl = st.structure_eam_vcsel(eam_only=True, bypass_dbr=bypass_dbr)
    plt.plot_al_doping(sl)


def al_mean(sl):
    # check wether the depth column exist
    if not 'depth' in sl:
        sl.insert(sl.shape[1], 'depth', value=np.nan)

    # duplicate each layer with a 0 thickness layer
    # in order to have two points for each layer
    # at the start and the end of each layer
    for i in range(sl.shape[0]):
        j = 2 * i
        row_to_insert = sl.loc[j]
        row_to_insert['thickness'] = 0.
        sl = pt.insert_row(j, sl, row_to_insert)

    # calculate the depth of every layer
    for i in range(sl.shape[0]):
        sl.at[i, 'depth'] = sl.loc[0:i, 'thickness'].sum()

    # convert dataframe to numpy array
    al = sl['al'].to_numpy()
    depth = sl['depth'].to_numpy()
    n = 50
    al_slope = np.linspace(15, 90, n)
    depth_slope = np.linspace(0, 20, n)

    # create a figure
    fig = mplt.figure(1)
    ax = fig.add_subplot()
    ax.set_xlabel('Depth (nm)')
    ax.set_ylabel('Aluminium content (%)')
    ax.set_prop_cycle(color=COLOR_CYCLE_5)

    # plot
    ax.plot(depth * 1e9, al * 1e2, label='Digital alloy')
    ax.plot(depth_slope, al_slope, label='Mean')
    ax.legend(loc='lower right')

    mplt.tight_layout()
    mplt.show()


def reflectivity_depth(sl, file_name, wavelength=670e-9, step=0.5e-9):
    # calculate the reflectivity as a function of depth
    # like an in-situ measurement of reflectivity while etching
    # will plot reflectivity and al content as a function of depth


    # decrease substrate thickness
    sl.at[sl.shape[0] -1, 'thickness'] = 5e-6
    num = int(sl['thickness'].sum() / step)
    r = np.zeros(num)
    al = np.zeros(num)
    depth = np.linspace(0., num*step, num=num)

    sl = op.algaas_super_lattice_refractive_index(sl, 0., wavelength, lengyel=False, only_real=True)


    for i in tqdm(range(num)):
        # etch the super lattice
        sl_etched = pt.etch_super_lattice_from_top(sl, step*i)

        # get the al content of the first layer of the super lattice ([0] is air)
        al[i] = sl_etched.iloc[1]['al']

        n = sl_etched['refractive_index'].to_numpy(dtype=float)
        d = sl_etched['thickness'].to_numpy(dtype=float)

        r[i] = tmm.reflection(n, d, wavelength)


    np.savez_compressed(file_name +'_reflectivity', r)
    np.savez_compressed(file_name +'_al', al)
    plt.plot_reflectivity_depth(al, r, depth)


def reflectivity_heatmap(bypass_dbr=True,
                         eam_only=False,
                         start_wavelength=700e-9,
                         stop_wavelength=1000e-9,
                         electric_field=0.,
                         n_wavelength=90,
                         n_time=160,
                         v_ga6=100,
                         v_ga11=850,
                         v_al5=900,
                         v_al12=150,
                         r_file_name='heatmap_r'):
    time, wavelength, r = epx.reflectivity_heatmap(bypass_dbr=bypass_dbr,
                                                   eam_only=eam_only,
                                                   start_wavelength=start_wavelength,
                                                   stop_wavelength=stop_wavelength,
                                                   electric_field=electric_field,
                                                   n_wavelength=n_wavelength,
                                                   n_time=n_time,
                                                   v_ga6=v_ga6,
                                                   v_ga11=v_ga11,
                                                   v_al5=v_al5,
                                                   v_al12=v_al12)
    np.savez_compressed(r_file_name, r)
    np.savez_compressed('heatmap_time', time)
    np.savez_compressed('heatmap_wavelength', wavelength)
    plt.plot_reflectivity_heatmap(time, wavelength, r)




def electromagnetic_amplitude(bypass_dbr=False, electric_field=0., wavelength=850e-9):
    #sl = st.structure_eam_vcsel(bypass_dbr=bypass_dbr, vcsel_only=False, grading_type='linear digital', mqw_alloy_type='digital')
    sl = stm.eam_vcsel_classic_structure()
    sl = op.algaas_super_lattice_refractive_index(sl, electric_field, wavelength)
    sl = pt.cut_in_equal_layers_thickness(sl, 1e-9)

    # scattering matrix or transmission matrix for transfer method

    em = tmm.em_amplitude_scattering_matrix(sl, wavelength)

    # insert the electromagnetic amplitude within the super lattice
    sl.insert(sl.shape[1], 'electromagnetic_amplitude', value=em)
    plt.plot_refra_em(sl, 'blabla')




def vcsel_electromagnetic_resonnance(electric_field=0.,
                                     start_wavelength=845e-9,
                                     stop_wavelength=855e-9,
                                     n_points=100):
    wavelength = np.linspace(start_wavelength, stop_wavelength, num=n_points)
    max_em_amp = np.zeros(n_points)

    for i in tqdm(range(len(wavelength))):
        sl = st.structure_eam_vcsel(grading_type='linear digital', mqw_alloy_type='digital', l_eam_clad = 8e-9, l_vcsel_clad=15e-9)
        sl = op.algaas_super_lattice_refractive_index(sl, electric_field, wavelength[i], lengyel=False)
        sl['refractive_index'] = sl['refractive_index'].apply(np.real).astype(float)
        sl = pt.cut_in_equal_layers_thickness(sl, 1e-8)
        em = tmm.em_amplitude_scattering_matrix(sl, wavelength[i], normalize=False)
        max_em_amp[i] = np.max(em)

    # normalize
    max_em_amp = max_em_amp / np.max(max_em_amp)

    np.savez_compressed('max_vcsel_em_amp_' +str(n_points) +'_pts', max_em_amp)
    plt.plot_xy(wavelength, max_em_amp)


def clad_coupling_electromagnetic_amplitude(bypass_dbr=False, electric_field=0., wavelength=850e-9):
    sl1 = st.structure_eam_vcsel(bypass_dbr=bypass_dbr, grading_type='linear slope', mqw_alloy_type='mean', l_eam_clad = 1e-9, l_vcsel_clad=8.5e-9)
    sl2 = st.structure_eam_vcsel(bypass_dbr=bypass_dbr, grading_type='linear slope', mqw_alloy_type='mean', l_eam_clad = 5e-9, l_vcsel_clad=8.5e-9)
    sl3 = st.structure_eam_vcsel(bypass_dbr=bypass_dbr, grading_type='linear slope', mqw_alloy_type='mean', l_eam_clad = 10e-9, l_vcsel_clad=8.5e-9)

    sl1 = op.algaas_super_lattice_refractive_index(sl1, electric_field, wavelength)
    sl2 = op.algaas_super_lattice_refractive_index(sl2, electric_field, wavelength)
    sl3 = op.algaas_super_lattice_refractive_index(sl3, electric_field, wavelength)

    sl1 = pt.cut_in_equal_layers_thickness(sl1, 1e-8)
    sl2 = pt.cut_in_equal_layers_thickness(sl2, 1e-8)
    sl3 = pt.cut_in_equal_layers_thickness(sl3, 1e-8)

    #sl1 = sl1[::-1].reset_index(drop=True)
    #sl2 = sl2[::-1].reset_index(drop=True)
    #sl3 = sl3[::-1].reset_index(drop=True)

    em1 = tmm.em_amplitude_scattering_matrix(sl1, wavelength)
    em2 = tmm.em_amplitude_scattering_matrix(sl2, wavelength)
    em3 = tmm.em_amplitude_scattering_matrix(sl3, wavelength)

    sl1.insert(sl1.shape[1], 'electromagnetic_amplitude', value=em1)
    sl2.insert(sl2.shape[1], 'electromagnetic_amplitude', value=em2)
    sl3.insert(sl3.shape[1], 'electromagnetic_amplitude', value=em3)

    #sl1 = sl1[::-1].reset_index(drop=True)
    #sl2 = sl2[::-1].reset_index(drop=True)
    #sl3 = sl3[::-1].reset_index(drop=True)

    plt.plot_refra_clad_coupling_em(sl1, sl2, sl3)




def zandberg_electromagnetic_amplitude(electric_field=0., wavelength=1e-6):
    sl = st.structure_zandbergen()
    sl = op.algaas_super_lattice_refractive_index(sl, electric_field, wavelength)
    sl = pt.cut_in_equal_layers_thickness(sl, 1e-9)
    em = tmm.em_amplitude_scattering_matrix(sl, wavelength)
    sl.insert(sl.shape[1], 'electromagnetic_amplitude', value=em)
    plt.plot_refra_em(sl)




def ftir_reflectivity(filename='eam_2', document_folder=True, linux=True):
    # import data
    wavelength, r = vtx.import_vertex_70_data(filename, document_folder=document_folder, linux=linux)

    # smooth the curve (I know it is cheating but ftir currently is out of service, so data is disgusting)
    bspl = splrep(wavelength, r, s=4.5)
    r = splev(wavelength, bspl)

    #  correct data
    r_max = np.max(r)
    r = r * 0.96 / r_max

    # plot
    plt.plot_reflectivity(wavelength, r)


def ftir_theory_comparison_reflectivity(filename='eam_2', electric_field=0., document_folder=True, linux=True):
    # import data
    wavelength, r = vtx.import_vertex_70_data(filename, document_folder=document_folder, linux=linux)

    # smooth the curve (I know it is cheating but ftir currently is out of service, so data is disgusting)
    bspl = splrep(wavelength, r, s=5)
    r = splev(wavelength, bspl)

    #  correct data
    r_max = np.max(r)
    r = r * 0.99 / r_max

    # get index of 750nm and 950nm
    idx_750 = np.abs(wavelength -750e-9).argmin()
    idx_950 = np.abs(wavelength -950e-9).argmin()

    # slice the arrays
    r = r[idx_750:idx_950]
    wavelength = wavelength[idx_750:idx_950]

    # calculate the theorical reflectivity according
    wavelength_theory = np.linspace(750e-9, 950e-9, num=500)

    if 'eam_2' in filename:
        r_theory = tmm.reflectivity(True, electric_field, wavelength_theory)
    else:
        r_theory = tmm.reflectivity(False, electric_field, wavelength_theory)

    # plot
    plt.plot_mult_reflectivity(wavelength, r, wavelength_theory, r_theory)




def mqw_psi(lz=0.01e-9):
    sl = st.structure_eam_vcsel(amount_eam_qw=5)
    sl = pt.cut_in_equal_layers_thickness(sl, lz)
    v_e, v_hh, eig_e, psi_e, eig_hh, psi_hh = qt.solve_schrodinger(sl)
    np.savez_compressed('v_e', v_e)
    np.savez_compressed('v_hh', v_hh)
    np.savez_compressed('eig_e', eig_e)
    np.savez_compressed('psi_e', psi_e[0, :])
    np.savez_compressed('eig_hh', eig_hh)
    np.savez_compressed('psi_hh', psi_hh[0, :])
    plt.plot_psie()


def benchmark_refractive_index():
    al = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    wavelength = np.arange(500e-9, 1500e-9, 1e-9)
    indices_array = np.empty([al.size, wavelength.size])

    for x in range(al.size):
        n = np.array([])

        for w in wavelength:
            n = np.append(n, op.afromovitz_varshni_real_algaas_refractive_index(al[x], w))

        indices_array[x] = n
        name = 'indices_array_' +str(x) +'.mat'
        scipy.io.savemat(name, mdict={'arr': n})

    name = 'al_array.mat'
    scipy.io.savemat(name, mdict={'arr': al})
    name = 'wavelength_array.mat'
    scipy.io.savemat(name, mdict={'arr': wavelength})

    plt.plot_refra_bench(al, wavelength, indices_array)




def surface_fit_550_test():
    # import data
    data = np.load('../data/550C_AlGaAs_refractive_indices/indices_arrays.npz')

    al = data['al_array']
    wavelength = data['wavelength_array']
    n = data['n_array']

    n_calc = np.zeros([wavelength.shape[0], al.shape[0]])
    for i in range(wavelength.shape[0]):
        for j in range(al.shape[0]):
            n_calc[i, j] = np.real(op.almuneau_complex_550C_algaas_refractive_index(al[j], wavelength[i]))


    plt.plot_2_std_heatmaps(al, wavelength, n, al, wavelength, n_calc)






