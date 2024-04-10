import numpy as np
from tqdm import tqdm


import calculation as cl
import plot as plt
import scipy.io
import structure_macro as stm
from model import super_lattice_structure as st, optic as op, transfer_matrix_method as tmm


def eam_vcsels_heatmaps():
    cl.reflectivity_heatmap(bypass_dbr=False,
                            n_wavelength=1080,
                            n_time=1920,
                            r_file_name='heatmap_r_eam_vcsel_1')
    cl.reflectivity_heatmap(bypass_dbr=True,
                            n_wavelength=1080,
                            n_time=1920,
                            r_file_name='heatmap_r_eam_vcsel_2')

def eam_heatmaps():
    cl.reflectivity_heatmap(bypass_dbr=False,
                            eam_only=True,
                            n_wavelength=900,
                            n_time=1600,
                            r_file_name='heatmap_r_eam_1')
    cl.reflectivity_heatmap(bypass_dbr=True,
                            eam_only=True,
                            n_wavelength=900,
                            n_time=1600,
                            r_file_name='heatmap_r_eam_2')




def reflectivity_eam_vcsel_clad_heatmap():
    n_pts = 30
    l_eam_clad =np.linspace(15, 20, num=n_pts)
    l_vcsel_clad =np.linspace(7, 11, num=n_pts)
    r = np.zeros([n_pts, n_pts])

    for i in tqdm(range(n_pts)):
        for j in range(n_pts):
            # wavelength in [m]
            # wavelength must be a numpy array
            sl = st.structure_eam_vcsel(l_eam_clad=l_eam_clad[i],
                                        l_vcsel_clad=l_vcsel_clad[j],
                                        grading_type='linear slope',
                                        mqw_alloy_type='mean')
            sl = op.algaas_super_lattice_refractive_index(sl, 0., 850e-9, lengyel=True)

            n = sl['refractive_index'].to_numpy(dtype=np.complex128)
            d = sl['thickness'].to_numpy(dtype=float)

            r[j, i] = tmm.reflection(n, d, 850e-9)

    plt.plot_std_heatmap(l_eam_clad, l_vcsel_clad, r)


def reflectivity_vcsel_eam_eam_vcesl():
    cl.reflectivity(stm.vcsel_structure(),
                    n_points=300)
    cl.reflectivity(stm.eam_classic_structure(),
                    n_points=300)
    cl.reflectivity(stm.eam_vcsel_classic_structure(),
                    n_points=500)




def reflectivity_for_etch():
    cl.reflectivity_depth(stm.eam_classic_structure(), 'eam_1_classic_etch')
    cl.reflectivity_depth(stm.eam_bypass_structure(), 'eam_2_bypass_etch')




def structure_for_matlab(vcsel_only=False, bypass_dbr=False, wavelength=850.5e-9):
    l_eam_clad = 5e-9
    l_vcsel_clad = 15e-9
    sl = st.structure_eam_vcsel(bypass_dbr=bypass_dbr, vcsel_only=vcsel_only, grading_type='linear digital', mqw_alloy_type='digital',
                                l_eam_clad=l_eam_clad, l_vcsel_clad=l_vcsel_clad)
    sl = op.algaas_super_lattice_refractive_index(sl, 0., wavelength)

    lz = sl['thickness'].to_numpy(dtype=float)
    n = sl['refractive_index'].apply(np.real).to_numpy(dtype=float)

    scipy.io.savemat('eam_vcsel_n_lz.mat', dict(n=n, lz=lz))



