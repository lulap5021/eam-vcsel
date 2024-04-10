import numpy as np
from scipy.linalg import expm, fractional_matrix_power
from tqdm import tqdm


from model import optic as op
from model import super_lattice_structure as st
from globals import N0, NGAAS




def element_matrix(n, d, wavelength):
    # equation (1.11)
    k0 = 2*np.pi/wavelength
    beta = k0*n*d

    co = np.cos(beta)
    si = np.sin(beta)

    return np.array([[co, 1j*si/n], [1j*si*n, co]])


def reflection(n, d, wavelength):
    # element matrix
    m = np.identity(2)

    for i in range(np.size(n)):
        m = np.matmul(m, element_matrix(n[i], d[i], wavelength))

    # electromagnetic field element amplitude
    [e_m, h_m] = np.matmul(m, np.array([1, NGAAS]))         # eta_s = NGAAS since theta = 0


    # reflection
    return np.abs( ((N0*e_m -h_m) / (N0*e_m +h_m))**2 )


def absorption(n, d, wavelength):
    # element matrix
    m = np.identity(2)

    for i in range(np.size(n)):
        m = np.matmul(m, element_matrix(n[i], d[i], wavelength))

    # electromagnetic field element amplitude
    [e_m, h_m] = np.matmul(m, np.array([1, NGAAS]))         # eta_s = NGAAS since theta = 0


    # absorption
    return np.imag( ((N0*e_m -h_m) / (N0*e_m +h_m))**2 )


def reflectivity(bypass_dbr,
                 electric_field,
                 wavelength):
    r = np.zeros(len(wavelength))

    # wavelength in [m]
    # wavelength must be a numpy array
    for i in tqdm(range(len(wavelength))):
        sl = st.structure_eam_vcsel(bypass_dbr=bypass_dbr)
        sl = op.algaas_super_lattice_refractive_index(sl, electric_field, wavelength[i])

        n = sl['refractive_index'].to_numpy(dtype=np.complex128)
        d = sl['thickness'].to_numpy(dtype=np.complex128)

        r[i] = reflection(n, d, wavelength[i])


    return r




def scattering_matrix_global():
    iden = np.identity(2)
    ze = np.zeros((2, 2))


    return np.array([[ze, iden], [iden, ze]])


def scattering_matrix_layer(vi, vg, xi):
    a = np.identity(2) +np.linalg.inv(vi) @ vg
    b = np.identity(2) -np.linalg.inv(vi) @ vg

    xb = xi @ b
    bab = b @ np.linalg.inv(a) @ b
    xbax = xb @ np.linalg.inv(a) @ xi

    d = a -xbax @ b
    e = xbax @ a - b
    f = xi @ (a - bab)

    s_11_22 = np.linalg.inv(d) @ e
    s_12_21 = np.linalg.inv(d) @ f

    s = np.array([[s_11_22, s_12_21],
                  [s_12_21, s_11_22]])


    return s


def redheffer_star_product(sa, sb):
    ba = sb[0, 0] @ sa[1, 1]
    ab = sa[1, 1] @ sb[0, 0]

    d = sa[0, 1] @ np.linalg.inv(np.identity(2) -ba)
    f = sb[1, 0] @ np.linalg.inv(np.identity(2) -ab)

    s = np.array([[
        sa[0, 0] +d @ sb[0, 0] @ sa[1, 0],
        d @ sb[0, 1]
    ], [
        f @ sa[1, 0],
        sb[1, 1] +f @ sa[1, 1] @ sb[0, 1]
    ]])


    return s


def em_amplitude_scattering_matrix(super_lattice, wavelength, normalize=True):
    # compute the amplitude of the electromagnetic field through the structure
    # using scattering matrix method
    # based on 'TMM using scattering matrices' by EMPossible
    # https://empossible.net/wp-content/uploads/2020/05/Lecture-TMM-Using-Scattering-Matrices.pdf


    # number of layers
    n_layers = super_lattice.shape[0]

    # vacuum wave number
    k_0 = 2 * np.pi / wavelength

    # compute gap medium parameters
    # here since theta = 0, kx = ky = 0
    q_global = np.array([[0., 1.],
                        [-1., 0.]])
    eigen_g = -1j * np.identity(2)
    v_global = eigen_g @ q_global

    # initialize global scattering matrix
    s_global_ini = scattering_matrix_global()

    # empty electromagnetic field
    e = np.zeros(n_layers)

    # empty matrices
    eigen_i = np.zeros((n_layers, 2, 2), dtype=np.complex128)
    v_i = np.zeros((n_layers, 2, 2), dtype=np.complex128)
    s_global = np.zeros((n_layers, 2, 2, 2, 2), dtype=np.complex128)
    x_i = np.zeros((n_layers, 2, 2), dtype=np.complex128)


    # loop forward through layers
    for i in range(n_layers):
        # calculate parameters for layer i
        k_zt_i = super_lattice.at[i, 'refractive_index']
        l = super_lattice.at[i, 'thickness']

        # k_z_i = k_0 * k_zt_i
        eigen_i[i] = 1j * k_zt_i * np.identity(2)

        q_i = np.array([[0.,            k_zt_i**2],
                        [-k_zt_i**2,    0.]])
        v_i[i] = q_i @ np.linalg.inv(eigen_i[i])
        x_i[i] = expm(eigen_i[i] * k_0 * l)

        # calculate scattering matrix for layer i
        s_i = scattering_matrix_layer(v_i[i], v_global, x_i[i])

        # update global scattering matrix
        if i==0 :
            s_global[i] = redheffer_star_product(s_global_ini, s_i)
        else:
            s_global[i] = redheffer_star_product(s_global[i -1], s_i)


    # w for external mode coefficients
    wv_g = np.array([[np.identity(2),  np.identity(2)],
                     [v_global,        -v_global]])
    wv_g = concatenate_2x2x2x2_to_4x4(wv_g)


    # incident and reflection coefficient
    c_inc = np.array([[1], [0]])
    c_ref = s_global[n_layers -1][0][0] @ c_inc


    # loop backward through layers
    for i in range(n_layers):
        # iterator from n to 0
        di = n_layers -i -1

        e_kl = x_i[di]
        e_mkl = fractional_matrix_power(e_kl, -1.)
        el_i = np.array([[e_kl,                np.zeros((2, 2))],
                         [np.zeros((2, 2)),    e_mkl]])

        wv_i = np.array([[np.identity(2),  np.identity(2)],
                         [v_i[di],         -v_i[di]]])


        # external mode coefficients
        c_im = np.linalg.inv(s_global[di -1][0][1]) @ (c_ref -s_global[di -1][0][0] @ c_inc)
        c_ip = s_global[di -1][1][0] @ c_inc +s_global[di -1][1][1] @ c_im
        c_e_i = np.array([c_ip[0],
                          c_ip[1],
                          c_im[0],
                          c_im[1]])


        # internal mode coefficients
        wv_i = concatenate_2x2x2x2_to_4x4(wv_i)
        c_i_i = np.linalg.inv(wv_i) @ wv_g @ c_e_i


        # internal fields
        el_i = concatenate_2x2x2x2_to_4x4(el_i)
        psi = wv_i @ el_i @ c_i_i
        # electromagnetic field amplitude on x axis
        e[di] = np.abs(np.real(psi[0].item()))**2


    # normalize
    if normalize:
        e = e / np.max(e)


    return e




def concatenate_2x2x2x2_to_4x4(arr):
    m1 = np.concatenate([arr[0][0], arr[0][1]], axis=1)
    m2 = np.concatenate([arr[1][0], arr[1][1]], axis=1)

    m = np.concatenate([m1, m2], axis=0)


    return m



