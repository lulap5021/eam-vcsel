import numpy as np


from globals import C, HEV, KBEV, NALOX, N0, T
from model import sqw_lengyel_absorption as sla




def algaas_super_lattice_refractive_index(super_lattice, electric_field, wavelength, temperature=T, lengyel=True, only_real=False):
    # calculate the refractive index of every layer of a superlattice in pandas dataframe format
    # super_lattice is a dataframe
    # wavelength in [m]
    # temperature in [K]
    # electric field in [V/m]
    # lengyel : activate the Lengyel model for the calculation of QW absorption


    if only_real:
        lengyel = False


    if temperature == 550+273.15 :
        refractive_index_function = almuneau_complex_550C_algaas_refractive_index
    else :
        refractive_index_function = afromovitz_varshni_real_algaas_refractive_index


    for i in range(len(super_lattice)):
        name = super_lattice.at[i, 'name']
        al = super_lattice.at[i, 'al']


        if 'air' in name:
            super_lattice.at[i, 'refractive_index'] = N0


        # layer is a quantum well -> Lengyel model
        elif 'quantum well' in name:
            # get a dataframe with only quantum well layers that have a refractive_index value
            calculated_qw_layers = super_lattice[
                super_lattice['name'].str.contains('eam quantum well')
                & ~super_lattice['refractive_index'].isnull()]


            # if refractive index has never been calculated
            if calculated_qw_layers.empty:
                # refractive index value
                n = refractive_index_function(al, wavelength, temperature=temperature) + 0j

                # confinement barrier aluminium content
                al = confinement_barrier_mean_al_content(super_lattice, i)              # [1]

                # qw thickness
                thickness = super_lattice.at[i, 'thickness']

                if lengyel :
                    alpha = sla.gaas_sqw_absorption_at_wavelength(al, thickness, electric_field, wavelength)
                else:
                    alpha = 0.
                k = -100 * alpha * wavelength / (4 * np.pi)
                #k = alpha * np.real(n) / (4 * np.pi)
                #k = k * 0.05

                n += 1j * k

            else:
                # reset the indices so we cant get the line 0 value
                calculated_qw_layers = calculated_qw_layers.reset_index(drop=True)

                # get the already computed quantum well refractive index value
                n = calculated_qw_layers.at[0, 'refractive_index']

            if only_real:
                super_lattice.at[i, 'refractive_index'] = np.real(n)
            else:
                super_lattice.at[i, 'refractive_index'] = n


        # layer is NOT a quantum well
        else:
            # refractive index value
            n = refractive_index_function(al, wavelength, temperature=temperature) + 0j

            if only_real:
                super_lattice.at[i, 'refractive_index'] = np.real(n)
            else:
                na = super_lattice.at[i, 'na']
                nd = super_lattice.at[i, 'nd']

                # N doped
                if nd > na:
                    n += -1j * 5e-18 * nd * 1e-4 * wavelength / (4 * np.pi)

                # P doped
                elif na > nd:
                    n += -1j * 11.5e-18 * na * 1e-4 * wavelength / (4 * np.pi)

                super_lattice.at[i, 'refractive_index'] = n


    return super_lattice




def afromovitz_varshni_real_algaas_refractive_index(al, wavelength, temperature=T):
    # al [1]
    # wavelength [m]
    # temperature [K]
    # for temperatures in the range [0 ; 70] +273.15 [K]

    t_c = temperature -273.15
    e = HEV*C/wavelength

    e_1 = -0.000402347826086942*t_c +3.65714869565217
    e_1 += (-0.00270520000000000*t_c +1.44506000000000)*al
    e_1 += (0.00278276521739130*t_c -0.251959130434783)*al**2

    e_d = (-0.00799306086956534*t_c +36.3120565217391)
    e_d += (-0.0265130956521739*t_c +2.25562739130435)*al
    e_d += (0.0306061391304348*t_c -3.62955347826087)*al**2

    # Vegard
    e_g = e_vegard(al)

    # Varshni
    e_var = e_varshni(al, e_g, temperature)

    # Afromovitz
    energy = e +1j*1e-2

    n = 1. +e_d/e_1
    n += (e_d/(e_1**3))*energy**2
    n_l = (e_d/(2.*e_1**3*(e_1**2 -e_var**2)))*energy**4
    n_l *= np.log((2.*e_1**2 -e_var**2 -energy**2)/(e_var**2 -energy**2))

    n += n_l
    n = np.real(np.sqrt(n))


    return n


def almuneau_complex_550C_algaas_refractive_index(al, wavelength, temperature=550+273.15):
    # al [1]
    # wavelength [m]
    # temperature [K]
    # for temperatures in the range [70 ; 700] +273.15 [K]

    t_c = temperature -273.15                       # [°C]
    c = 2.565574613e-8
    n = afromovitz_varshni_real_algaas_refractive_index(al, wavelength, temperature=temperature)

    # Vegard
    e_g = e_vegard(al)

    # Varshni
    e_var = e_varshni(al, e_g, temperature)

    n *= (1. +((1. -al) * 0.5405 +al*0.885) * 1e-3 * (t_c -25.) / (4. * e_var))
    k_j, e_g_j = johnson_abs_algaas(al, wavelength, temperature)
    lambda_g = 1.239852066e-6 / e_g_j

    if lambda_g < wavelength :
        abs_f = 2. / np.pi * 2000. * np.arctan(np.exp((1.239852066e-6 / wavelength -e_g_j) / 6e-3))
    else :
        abs_f = c * c_absorbtion(e_g_j) * (
                    np.sqrt(np.max(1.239852066e-6 / wavelength - e_g_j, 0.)) / (1.239852066e-6 / wavelength))

    k = (wavelength * 1e-6 * 1e-4 / (4 * np.pi)) * abs_f

    # temperature dependent bandgap, high temperature range, neglecting bowing effects
    alpha_x = (5.405 +4.038 * al) * 1e-4                        # [eV/K]
    beta_x = 204. / 370. * (370. +54.*al +22.*al**2)            # [K]
    e_gamma = (1.424 +1.266 * al +0.26 * al**2 +(alpha_x * 298.**2) / (298. +beta_x)) -(alpha_x * temperature**2) / (temperature +beta_x)    # [eV]

    # Urbach tail model
    delta_lambda_urbach = 0.05e-6                               # [m]
    lambda_g = 1.239852066e-6 / e_gamma
    low_bound = lambda_g -delta_lambda_urbach
    high_bound = lambda_g +delta_lambda_urbach


    return n




def e_vegard(al):
    # al content in [1]


    return (1 -al)*1.424 +al*3.02 -al*(1-al)*0.37


def e_varshni(al, e_g, temperature):
    # al content in [1]
    # e_g in [eV]
    # temperature in [K]

    t_c = temperature - 273.15

    alpha = (5.405 +4.038*al)*1e-4                                          # [eV/K]
    beta = 204./370.*(370. +54.*al +22.*al**2)                              # [K]
    e_var = e_g +alpha*298.**2/(298. +beta) -(alpha*t_c**2)/(t_c +beta)     # [eV?]


    return e_var


def johnson_abs_algaas(al, wavelength, temperature):
    # al [1]
    # wavelength [m]
    # temperature [K]


    t_c = temperature -273.15                       # [°C]
    e = 1.239852066e-6 / wavelength                 # [eV?]

    e_g_0 = e_varshni(al, e_vegard(al), 0.)          # [eV]

    theta_nid = 263.
    s_g_nid = 5.98
    s_0_nid = 0.087
    x_nid = 5.2
    a = 8000.

    e_0_t_nid = s_0_nid * KBEV * theta_nid * ((1. +x_nid) / 2. +1. / (np.exp(theta_nid / t_c) -1.))
    e_g_t_nid = e_g_0 -s_g_nid * KBEV * theta_nid * (1. / (np.exp(theta_nid / t_c) -1.))

    alpha_si = a * np.exp((e -e_g_t_nid) / e_0_t_nid)
    k_nid = (wavelength * 1e-6 * 1e-4 / (4 * np.pi)) * alpha_si


    return k_nid, e_g_t_nid


def c_absorbtion(e_g):
    # e_g in [eV]


    return -7.63792e-11 +2.81092e-12 * e_g




def oxidation(super_lattice, eam_mesa=True, vcsel_mesa=False):
    for i in range(len(super_lattice)):
        name = super_lattice.at[i, 'name']

        if eam_mesa :
            if 'eam mesa AlOx 100% Al' in name:
                super_lattice.at[i, 'refractive_index'] = NALOX

        if vcsel_mesa :
            if 'vcsel mesa AlOx 100% Al' in name:
                super_lattice.at[i, 'refractive_index'] = NALOX


    return super_lattice




def confinement_barrier_mean_al_content(super_lattice, i):
    name = super_lattice.at[i +1, 'name']
    al = 0.
    total_thickness = 0.

    while 'barrier' in name:
        al += super_lattice.at[i +1, 'al'] * super_lattice.at[i +1, 'thickness']
        total_thickness += super_lattice.at[i +1, 'thickness']

        i += 1
        name = super_lattice.at[i +1, 'name']

    al = al / total_thickness


    return al
