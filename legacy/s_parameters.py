import matplotlib.pyplot as plt
from mpmath import mp
import numpy as np
import os


MPL_SIZE = 24
plt.rc('font', size=MPL_SIZE)
plt.rc('axes', titlesize=MPL_SIZE)
plt.rcParams['font.family'] = 'Calibri'


# solving methods and solvers
def s_params_linear_algebra_method(s_11_open, s_11_short, s_11_load,
                                     gamma_open, gamma_short, gamma_load):
    # Yen-Chung Lin et al.
    # DOI : 10.1109/TMTT.2015.2436400
    # gammas are known terminations
    # rtp is the round-trip path = s_12 * s_21


    # 3x3 matrix from equation (6)
    m_0 = np.array([[1.,    s_11_open * gamma_open,     gamma_open],
                    [1.,    s_11_short * gamma_short,   gamma_short],
                    [1.,    s_11_load * gamma_load,     gamma_load]])

    # calculate the accuracy indicator
    #k = condition_number(m_0)
    #print('Accuracy indicator : k = ' +str(k))

    # measurement matrix from equation (6)
    m_meas = np.array([s_11_open, s_11_short, s_11_load])

    # from equation (4)
    #eigval, eigvec = np.linalg.eig(m_0)
    #a = np.diagflat(eigval) @ m_0
    #b = eigval * m_measure
    #m_unknowns = np.linalg.solve(a, b)
    m_unknowns = np.linalg.solve(m_0, m_meas)


    # s parameter
    s_11 = m_unknowns[0]
    s_22 = m_unknowns[1]
    rtp = m_unknowns[2] + s_11 * s_22


    return s_11, rtp, s_22


def s_params_anharmonic_method(s_11_open, s_11_short, s_11_load,
                                gamma_open, gamma_short, gamma_load):
    # solve using anharmonic ratio (cross-road ratio)
    # based on Invariance of the Cross Ratio Applied to Microwave Network Analysis
    # NBS technical note 623 september 1972
    # R. W. Beatty


    # setting arbitrary arithmetic precision
    mp.dps = 10000


    # casts variables as arbitrary precision arithmetic
    s_11_open = mp.mpc(s_11_open)
    s_11_short = mp.mpc(s_11_short)
    s_11_load = mp.mpc(s_11_load)
    gamma_open = mp.mpc(gamma_open)
    gamma_short = mp.mpc(gamma_short)
    gamma_load = mp.mpc(gamma_load)


    # anharmonic solver
    denom = gamma_open * gamma_short * (s_11_open - s_11_short)
    denom += gamma_short * gamma_load * (s_11_short - s_11_load)
    denom += gamma_load * gamma_open * (s_11_load - s_11_open)

    s_11 = gamma_open * gamma_short * s_11_load * (s_11_open - s_11_short)
    s_11 += s_11_open * gamma_short * gamma_load * (s_11_short - s_11_load)
    s_11 += gamma_open * s_11_short * gamma_load * (s_11_load - s_11_open)
    s_11 /= denom

    s_22 = gamma_open * (s_11_short - s_11_load)
    s_22 += gamma_short * (s_11_load - s_11_open)
    s_22 += gamma_load * (s_11_open - s_11_short)
    s_22 /= denom

    rtp = (s_11_open - s_11_short) * (s_11_short - s_11_load) * (s_11_load - s_11_open)
    rtp *=  (gamma_open - gamma_short) * (gamma_short - gamma_load) * (gamma_load - gamma_open)
    rtp /= denom ** 2


    return s_11, rtp, s_22




# solvers
def cramer_solver(a, b):
    # 0
    ac = a.copy()
    bc = b.copy()

    # 1
    bc[1] -= b[0] * a[1, 0] / a[0, 0]
    ac[1, :] -= a[0, :] * a[1, 0] / a[0,0]
    bc[2] -= b[0] * a[1, 0] / a[0, 0]
    ac[2, :] -= a[0, :] * a[1, 0] / a[0,0]

    # 2
    bc[2] -= bc[1] * ac[2,1] / ac[1, 1]
    ac[2, :] -= ac[1, :] * ac[2, 1] / ac[1, 1]

    # solution
    z = bc[2] / ac[2, 2]
    y = (bc[1] - z * ac[1, 2]) / ac[1, 1]
    x = (bc[0] - y * ac[0, 1] - z * ac[0, 2]) / ac[0, 0]


    return np.array([x, y, z])


def weighted_least_square_solver(a, b):
    # solve using weighted least square method
    # Uncertainty analysis of the weighted least squares VNA calibration
    # 10.1109/ARFTGF.2004.1427566

    # create the weight matrix
    w = np.zeros((3, 3))
    sigmas = np.array([1e10, 1e0, 1e0])
    np.fill_diagonal(w, sigmas)

    # least square method
    x, residuals, rank, s = np.linalg.lstsq(w @ a, w @ b, rcond=None)


    return x




# data import / export functions
def import_ads_data(directory='ads'):
    # reflection coefficients imports
    gamma_open, f = import_s_11('gamma_open_substrate', directory=directory)
    gamma_short, f = import_s_11('gamma_short', directory=directory)
    gamma_load, f = import_s_11('gamma_load_50', directory=directory)

    # dut imports
    s_11_open, f = import_s_11('dut_open_substrate', directory=directory)
    s_11_short, f = import_s_11('dut_short', directory=directory)
    s_11_load, f = import_s_11('dut_load_50', directory=directory)


    return s_11_open, s_11_short, s_11_load, \
           gamma_open, gamma_short, gamma_load, \
           f


def import_s_11(file_name, directory):
    # import from ads data file
    # data_gamma = np.loadtxt(os.getcwd() +'/data/' + directory + '/' + file_name + '.s1p', skiprows=5)
    data_gamma = np.loadtxt(os.getcwd() +'/data/' + directory + '/' + file_name, skiprows=8)

    # short circuit reflection coefficient
    gamma_mag = np.exp(np.log(10) * data_gamma[:, 1] / 20)
    gamma_phase = data_gamma[:, 2]
    gamma = gamma_mag * np.exp(1j * np.deg2rad(gamma_phase))

    # frequency array
    f = data_gamma[:, 0]


    return gamma, f


def import_s_21(file_name, directory):
    # import from ads data file
    # data_gamma = np.loadtxt(os.getcwd() +'/data/' + directory + '/' + file_name + '.s1p', skiprows=5)
    data_gamma = np.loadtxt(os.getcwd() +'/data/' + directory + '/' + file_name, skiprows=8)

    # short circuit reflection coefficient
    gamma_mag = np.exp(np.log(10) * data_gamma[:, 3] / 20)
    gamma_phase = data_gamma[:, 4]
    gamma = gamma_mag * np.exp(1j * np.deg2rad(gamma_phase))

    # frequency array
    f = data_gamma[:, 0]


    return gamma, f


def matrices_setup(n):
    # import data
    s_11_open, s_11_short, s_11_load, \
    gamma_open, gamma_short, gamma_load, \
    f = import_ads_data()


    # 3x3 matrix from equation (6)
    m_0 = np.array([[1.,    s_11_open[n] * gamma_open[n],     gamma_open[n]],
                    [1.,    s_11_short[n] * gamma_short[n],   gamma_short[n]],
                    [1.,    s_11_load[n] * gamma_load[n],     gamma_load[n]]])

    # calculate the accuracy indicator
    k = condition_number(m_0)
    #print('Accuracy indicator : k = ' +str(k))

    # measurement matrix from equation (6)
    m_meas = np.array([s_11_open[n], s_11_short[n], s_11_load[n]])

    # import solution
    data_solution = np.loadtxt(os.getcwd() +'/data/ads/dut.s2p', skiprows=5)
    s_11_solution = data_solution[:, 1] * np.exp(1j * np.deg2rad(data_solution[:, 2]))
    s_21_solution = data_solution[:, 3] * np.exp(1j * np.deg2rad(data_solution[:, 4]))
    s_12_solution = data_solution[:, 5] * np.exp(1j * np.deg2rad(data_solution[:, 6]))
    s_22_solution = data_solution[:, 7] * np.exp(1j * np.deg2rad(data_solution[:, 8]))

    m_sol = np.array([s_11_solution[n], s_22_solution[n], s_21_solution[n] * s_12_solution[n] - s_11_solution[n] * s_22_solution[n]])


    return m_0, m_meas, m_sol, k


def extract_s_from_x(x):
    n = int(len(x) / 3)
    s_11 = np.zeros(n)
    s_22 = np.zeros(n)
    rtp = np.zeros(n)


    for i in range(n):
        s_11[i] = x[3 * i]
        s_22[i] = x[3 * i + 1]
        rtp[i] = x[3 * i + 2]


    return s_11, s_22, rtp


def export_s2p(filename, freq, s_11, s_12, s_21, s_22):
    num_freq_points = len(freq)

    # Write S-parameter data to .s2p file
    with open(filename, 'w') as f:
        f.write('! File created by Python script\n')
        f.write('! Frequency units in Hz\n')
        f.write(f'! Number of ports: 1\n')
        f.write('! S-parameters: S11, S12, S21, S22\n\n')
        f.write('# Hz S DB R 50\n\n')
        f.write('! Frequency magS11 angS11 magS21 angS21 magS12 angS12 magS22 angS22\n')
        for i in range(int(num_freq_points)):
            f.write(f'{freq[i]:.9E} {np.abs(s_11[i]):.5f} {np.angle(s_11[i], deg=True):.5f} {np.abs(s_21[i]):.5f} {np.angle(s_21[i], deg=True):.5f} {np.abs(s_12[i]):.5f} {np.angle(s_12[i], deg=True):.5f} {np.abs(s_22[i]):.5f} {np.angle(s_22[i], deg=True):.5f}\n')




# test functions
def condition_number(m):
    # Matrix analysis and applied linear algebra
    # Author:Carl D. Meyer
    # ISBN:9780898714548, 0898714540
    # P.127

    # This magnification factor k is called a condition number for m. In other
    # words, if k is small relative to 1 (i.e., if m is well conditioned), then a small
    # relative change (or error) in m cannot produce a large relative change (or error)
    # in the inverse, but if k is large (i.e., if m is ill conditioned), then a small rela-
    # tive change (or error) in m can possibly (but not necessarily) result in a large
    # relative change (or error) in the inverse.

    # close to 1 is good
    # bigger is bad

    k = np.linalg.norm(m) * np.linalg.norm(np.linalg.inv(m))


    return k


def test_s_params_analytic_arbitrary_precision():
    # import data
    s_11_open, s_11_short, s_11_load, \
    gamma_open, gamma_short, gamma_load, \
    f = import_ads_data()


    # instantiation by casting of arbitrary arithmetic precision complex numbers
    n = len(s_11_open)
    mp.dps = 10000
    zeroes = [mp.mpc(0, 0) for i in range(n)]
    s_11 = np.array(zeroes)
    rtp = np.array(zeroes)
    s_22 = np.array(zeroes)


    # calculate s parameters
    for i in range(n):
        s_11[i], rtp[i], s_22[i] = s_params_anharmonic_method(s_11_open[i], s_11_short[i], s_11_load[i],
                                                              gamma_open[i], gamma_short[i], gamma_load[i])


    # import solution
    data_solution = np.loadtxt(os.getcwd() +'/data/ads/dut.s2p', skiprows=5)
    s_11_solution = data_solution[:, 1] * np.exp(1j * np.deg2rad(data_solution[:, 2]))
    rtp_solution = data_solution[:, 3] * np.exp(1j * np.deg2rad(data_solution[:, 4]))
    rtp_solution *= data_solution[:, 5] * np.exp(1j * np.deg2rad(data_solution[:, 6]))
    s_22_solution = data_solution[:, 7] * np.exp(1j * np.deg2rad(data_solution[:, 8]))


    # plot
    fig, axs = plt.subplots(3, 3)

    # labels
    axs[0, 0].set_xlabel('Frequency [GHz]')
    axs[0, 0].set_ylabel('Real part')
    axs[1, 0].set_xlabel('Frequency [GHz]')
    axs[1, 0].set_ylabel('Imaginary part')
    axs[2, 0].set_xlabel('Frequency [GHz]')
    axs[2, 0].set_ylabel('log(module error)')

    # S11
    axs[0, 0].plot(f, np.real(s_11_solution), label='S11_solution')
    axs[0, 0].plot(f, np.real(s_11), label='S11')
    axs[0, 0].legend()
    axs[1, 0].plot(f, np.imag(s_11_solution), label='S11_solution')
    axs[1, 0].plot(f, np.imag(s_11), label='S11')
    axs[1, 0].legend()
    axs[2, 0].plot(f, np.log(np.abs((s_11_solution - s_11) / s_11_solution)))
    #axsmith = pp.subplot(1, 1, 1, projection='smith')
    #pp.subplot(1, 1, 1, projection='smith')
    #pp.plot(s_11, label="default", datatype=SmithAxes.Z_PARAMETER)

    # RTP
    axs[0, 1].plot(f, np.real(rtp_solution), label='RTP_solution')
    axs[0, 1].plot(f, np.real(rtp), label='RTP')
    axs[0, 1].legend()
    axs[1, 1].plot(f, np.imag(rtp_solution), label='RTP_solution')
    axs[1, 1].plot(f, np.imag(rtp), label='RTP')
    axs[1, 1].legend()
    axs[2, 1].plot(f, np.log(np.abs((rtp_solution - rtp) / rtp_solution)))

    # S22
    axs[0, 2].plot(f, np.real(s_22_solution), label='S22_solution')
    axs[0, 2].plot(f, np.real(s_22), label='S22')
    axs[0, 2].legend()
    axs[1, 2].plot(f, np.imag(s_22_solution), label='S22_solution')
    axs[1, 2].plot(f, np.imag(s_22), label='S22')
    axs[1, 2].legend()
    axs[2, 2].plot(f, np.log(np.abs((s_22_solution - s_22) / s_22_solution)))


    plt.tight_layout()
    plt.show()


def test_s_params_linear_algebra():
    # import data
    s_11_open, s_11_short, s_11_load, \
    gamma_open, gamma_short, gamma_load, \
    f = import_ads_data()


    # instantiation with numpy
    n = len(s_11_open)
    s_11 = np.zeros(n, dtype=np.complex128)
    rtp = np.zeros(n, dtype=np.complex128)
    s_22 = np.zeros(n, dtype=np.complex128)


    # calculate s parameters
    for i in range(n):
        s_11[i], rtp[i], s_22[i] = s_params_linear_algebra_method(s_11_open[i], s_11_short[i], s_11_load[i],
                                                                  gamma_open[i], gamma_short[i], gamma_load[i])


    # import solution
    data_solution = np.loadtxt(os.getcwd() +'/data/ads/dut.s2p', skiprows=5)
    s_11_solution = data_solution[:, 1] * np.exp(1j * np.deg2rad(data_solution[:, 2]))
    rtp_solution = data_solution[:, 3] * np.exp(1j * np.deg2rad(data_solution[:, 4]))
    rtp_solution *= data_solution[:, 5] * np.exp(1j * np.deg2rad(data_solution[:, 6]))
    s_22_solution = data_solution[:, 7] * np.exp(1j * np.deg2rad(data_solution[:, 8]))


    # plot
    fig, axs = plt.subplots(3, 3)

    # labels
    axs[0, 0].set_xlabel('Frequency [GHz]')
    axs[0, 0].set_ylabel('Real part')
    axs[1, 0].set_xlabel('Frequency [GHz]')
    axs[1, 0].set_ylabel('Imaginary part')
    axs[2, 0].set_xlabel('Frequency [GHz]')
    axs[2, 0].set_ylabel('log(module error)')

    # S11
    axs[0, 0].plot(f, np.real(s_11_solution), label='S11_solution')
    axs[0, 0].plot(f, np.real(s_11), label='S11')
    axs[0, 0].legend()
    axs[1, 0].plot(f, np.imag(s_11_solution), label='S11_solution')
    axs[1, 0].plot(f, np.imag(s_11), label='S11')
    axs[1, 0].legend()
    axs[2, 0].plot(f, np.log(np.abs((s_11_solution - s_11) / s_11_solution)))
    #axsmith = pp.subplot(1, 1, 1, projection='smith')
    #pp.subplot(1, 1, 1, projection='smith')
    #pp.plot(s_11, label="default", datatype=SmithAxes.Z_PARAMETER)

    # RTP
    axs[0, 1].plot(f, np.real(rtp_solution), label='RTP_solution')
    axs[0, 1].plot(f, np.real(rtp), label='RTP')
    axs[0, 1].legend()
    axs[1, 1].plot(f, np.imag(rtp_solution), label='RTP_solution')
    axs[1, 1].plot(f, np.imag(rtp), label='RTP')
    axs[1, 1].legend()
    axs[2, 1].plot(f, np.log(np.abs((rtp_solution - rtp) / rtp_solution)))

    # S22
    axs[0, 2].plot(f, np.real(s_22_solution), label='S22_solution')
    axs[0, 2].plot(f, np.real(s_22), label='S22')
    axs[0, 2].legend()
    axs[1, 2].plot(f, np.imag(s_22_solution), label='S22_solution')
    axs[1, 2].plot(f, np.imag(s_22), label='S22')
    axs[1, 2].legend()
    axs[2, 2].plot(f, np.log(np.abs((s_22_solution - s_22) / s_22_solution)))


    plt.tight_layout()
    plt.show()


def test_deembedding():
    # reflection coefficients imports
    gamma_open, f = import_s_11('gamma_open_substrate', directory='a2c')
    gamma_short, f = import_s_11('gamma_short', directory='a2c')
    gamma_load, f = import_s_11('gamma_load_50', directory='a2c')

    # dut imports
    data_s_11_open = np.loadtxt(os.getcwd() +'/data/a2c/' + 'dut_open_substrate' + '.s1p', skiprows=5)
    data_s_11_short = np.loadtxt(os.getcwd() +'/data/a2c/' + 'dut_short' + '.s1p', skiprows=5)
    data_s_11_load = np.loadtxt(os.getcwd() +'/data/a2c/' + 'dut_load_50' + '.s1p', skiprows=5)

    s_11_open = data_s_11_open[:, 1] + 1j * data_s_11_open[:, 2]
    s_11_short = data_s_11_short[:, 1] + 1j * data_s_11_short[:, 2]
    s_11_load = data_s_11_load[:, 1] + 1j * data_s_11_load[:, 2]

    # instantiation with numpy
    n = len(s_11_open)
    s_11 = np.zeros(n, dtype=np.complex128)
    rtp = np.zeros(n, dtype=np.complex128)
    s_22 = np.zeros(n, dtype=np.complex128)

    # calculate s parameters
    for i in range(n):
        s_11[i], rtp[i], s_22[i] = s_params_linear_algebra_method(s_11_open[i], s_11_short[i], s_11_load[i],
                                                                  gamma_open[i], gamma_short[i], gamma_load[i])


    # plot
    fig, axs = plt.subplots(2, 3)

    # labels
    axs[0, 0].set_xlabel('Frequency [GHz]')
    axs[0, 0].set_ylabel('Magnitude [dB]')
    axs[1, 0].set_xlabel('Frequency [GHz]')
    axs[1, 0].set_ylabel('Phase [degree]')

    # S11
    axs[0, 0].plot(f, 20 * np.log10(np.abs(s_11)), label='S11')
    axs[0, 0].legend()
    axs[1, 0].plot(f, np.angle(s_11, deg=True), label='S11')
    axs[1, 0].legend()

    # RTP
    axs[0, 1].plot(f, 20 * np.log10(np.abs(np.sqrt(rtp))), label='S12')
    axs[0, 1].legend()
    axs[1, 1].plot(f, np.angle(np.sqrt(rtp), deg=True), label='S12')
    axs[1, 1].legend()

    # S22
    axs[0, 2].plot(f, 20 * np.log10(np.abs(s_22)), label='S22')
    axs[0, 2].legend()
    axs[1, 2].plot(f, np.angle(s_22, deg=True), label='S22')
    axs[1, 2].legend()

    plt.tight_layout()
    plt.show()


def check_pnax_deembedding():
    # dut imports
    data_pnax = np.loadtxt(os.getcwd() + '/data/pnax_n5244b/' + 'extractedDUT' + '.s2p', skiprows=3)
    f = data_pnax[:, 0]
    s_11_mag = data_pnax[:, 1]
    s_11_phase = data_pnax[:, 2]
    s_12_mag = data_pnax[:, 5]
    s_12_phase = data_pnax[:, 6]
    s_22_mag = data_pnax[:, 7]
    s_22_phase = data_pnax[:, 8]

    # plot
    fig, axs = plt.subplots(2, 3)

    # labels
    axs[0, 0].set_xlabel('Frequency [GHz]')
    axs[0, 0].set_ylabel('Magnitude [dB]')
    axs[1, 0].set_xlabel('Frequency [GHz]')
    axs[1, 0].set_ylabel('Phase [degree]')

    # S11
    axs[0, 0].plot(f, s_11_mag, label='S11')
    axs[0, 0].legend()
    axs[1, 0].plot(f, s_11_phase, label='S11')
    axs[1, 0].legend()

    # S12
    axs[0, 1].plot(f, s_12_mag, label='S12')
    axs[0, 1].legend()
    axs[1, 1].plot(f, s_12_phase, label='S12')
    axs[1, 1].legend()

    # S22
    axs[0, 2].plot(f, s_22_mag, label='S22')
    axs[0, 2].legend()
    axs[1, 2].plot(f, s_22_phase, label='S22')
    axs[1, 2].legend()

    plt.tight_layout()
    plt.show()


def z_to_gamma(z):
    # of course for 50ohm reference impedance
    z0 = 50
    return (z - z0) / (z + z0)


def data_processing():
    directory = 'ads/ff_z40p_gsg150'

    # dut imports
    s_11_open, f = import_s_11('Cal_DATAs_2PORTS_OSLT_CSR8_OPEN.s2p', directory=directory)
    s_11_short, f = import_s_11('Cal_DATAs_2PORTS_OSLT_CSR8_SHORT.s2p', directory=directory)
    s_11_load, f = import_s_11('Cal_DATAs_2PORTS_OSLT_CSR8_LOAD.s2p', directory=directory)

    # reflection coefficients
    # Formfactor Z Probe
    # Z40-P-GSG-150
    gamma_open = z_to_gamma(0 - 1j / (2 * np.pi * f * 4.8e-15))
    gamma_short = z_to_gamma(0 + 1j * 2 * np.pi * f * 21e-12)
    gamma_load = z_to_gamma(50 + 1j * 2 * np.pi * f * 6e-12)

    # instantiation with numpy
    n = len(s_11_open)
    s_probe_11 = np.zeros(n, dtype=np.complex128)
    s_probe_rtp = np.zeros(n, dtype=np.complex128)
    s_probe_22 = np.zeros(n, dtype=np.complex128)


    # calculate probe s parameters
    for i in range(n):
        s_probe_11[i], s_probe_rtp[i], s_probe_22[i] = s_params_linear_algebra_method(s_11_open[i], s_11_short[i], s_11_load[i],
                                                                              gamma_open[i], gamma_short[i], gamma_load[i])

    # import eam s parameters
    s_11, f = import_s_11('EAM_2668_Cal2PORT_Veam_3V5dc_LaserON.s2p', directory=directory)
    s_21, f = import_s_21('EAM_2668_Cal2PORT_Veam_3V5dc_LaserON.s2p', directory=directory)

    # export probe s-parameters
    s_probe_21 = np.sqrt(np.abs(s_probe_rtp))
    #z_eam = 50. * (1 + s_11) / (1 - s_11)

    h = eam_transfer_function(s_21, s_probe_21, s_probe_22, s_11)
    h_db = 20 * np.log10(np.abs(h))
    s_21_db = 20 * np.log10(np.abs(s_21))

    # plot
    fig, ax = plt.subplots()
    plt.plot(f * 1e-9, h_db, color='dodgerblue', label='Heam')
    plt.plot(f * 1e-9, s_21_db, color='orange', label='S21')
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Gain (dB)')
    ax.legend(loc=2)

    plt.tight_layout()
    plt.show()


def test_s2p():
    directory = 'ads/pn_i40_gsg150'

    # import solution
    data = np.loadtxt(os.getcwd() + '/data/' + directory + '/dut_open_substrate.s2p', skiprows=8)
    f = data[:, 0]
    s_11_open = data[:, 1] * np.exp(1j * np.deg2rad(data[:, 2]))
    data = np.loadtxt(os.getcwd() + '/data/' + directory + '/dut_short.s2p', skiprows=8)
    s_11_short = data[:, 1] * np.exp(1j * np.deg2rad(data[:, 2]))
    data = np.loadtxt(os.getcwd() + '/data/' + directory + '/dut_load_50.s2p', skiprows=8)
    s_11_load = data[:, 1] * np.exp(1j * np.deg2rad(data[:, 2]))

    # reflection coefficients
    # Formfactor Z Probe
    # Z40-P-GSG-150
    gamma_open = z_to_gamma(0 - 1j / (2 * np.pi * f * 4.8e-15))
    gamma_short = z_to_gamma(0 + 1j * 2 * np.pi * f * 21e-12)
    gamma_load = z_to_gamma(50 + 1j * 2 * np.pi * f * 6e-12)

    # instantiation with numpy
    n = len(s_11_open)
    s_11 = np.zeros(n, dtype=np.complex128)
    rtp = np.zeros(n, dtype=np.complex128)
    s_22 = np.zeros(n, dtype=np.complex128)


    # calculate s parameters
    for i in range(n):
        s_11[i], rtp[i], s_22[i] = s_params_linear_algebra_method(s_11_open[i], s_11_short[i], s_11_load[i],
                                                                  gamma_open[i], gamma_short[i], gamma_load[i])

    s_12 = np.sqrt(rtp)
    s_21 = s_12


    # export to s2p
    export_s2p('moulined.s2p', f, s_11, s_12, s_21, s_22)


def test_pointes():
    directory = 'ads/ff_z40p_gsg150'

    # dut imports
    s_11_open, f = import_s_11('Cal_DATAs_2PORTS_OSLT_CSR8_OPEN.s2p', directory=directory)
    s_11_short, f = import_s_11('Cal_DATAs_2PORTS_OSLT_CSR8_SHORT.s2p', directory=directory)
    s_11_load, f = import_s_11('Cal_DATAs_2PORTS_OSLT_CSR8_LOAD.s2p', directory=directory)

    # reflection coefficients
    # Formfactor Z Probe
    # Z40-P-GSG-150
    gamma_open = z_to_gamma(0 - 1j / (2 * np.pi * f * 4.8e-15))
    gamma_short = z_to_gamma(0 + 1j * 2 * np.pi * f * 21e-12)
    gamma_load = z_to_gamma(50 + 1j * 2 * np.pi * f * 6e-12)

    # instantiation with numpy
    n = len(s_11_open)
    s_11 = np.zeros(n, dtype=np.complex128)
    rtp = np.zeros(n, dtype=np.complex128)
    s_22 = np.zeros(n, dtype=np.complex128)


    # calculate s parameters
    for i in range(n):
        s_11[i], rtp[i], s_22[i] = s_params_linear_algebra_method(s_11_open[i], s_11_short[i], s_11_load[i],
                                                                  gamma_open[i], gamma_short[i], gamma_load[i])


    # import solution
    data_solution = np.loadtxt(os.getcwd() +'/data/ads/pn_i40_gsg150/PN I40-A-GSG-150_SN_HT2KF-1_50M-40G-801p.s2p', skiprows=5)
    s_11_solution = data_solution[:, 1] + 1j * data_solution[:, 2]
    rtp_solution = data_solution[:, 3] + 1j * data_solution[:, 4]
    rtp_solution *= (data_solution[:, 5] + 1j * data_solution[:, 6])
    s_22_solution = data_solution[:, 7] + 1j * data_solution[:, 8]


    # plot
    fig, axs = plt.subplots(3, 3)

    # labels
    axs[0, 0].set_xlabel('Frequency [GHz]')
    axs[0, 0].set_ylabel('Real part')
    axs[1, 0].set_xlabel('Frequency [GHz]')
    axs[1, 0].set_ylabel('Imaginary part')
    axs[2, 0].set_xlabel('Frequency [GHz]')
    axs[2, 0].set_ylabel('log(module error)')

    # S11
    axs[0, 0].plot(f, np.real(s_11_solution), label='S11_solution')
    axs[0, 0].plot(f, np.real(s_11), label='S11')
    axs[0, 0].legend()
    axs[1, 0].plot(f, np.imag(s_11_solution), label='S11_solution')
    axs[1, 0].plot(f, np.imag(s_11), label='S11')
    axs[1, 0].legend()
    axs[2, 0].plot(f, np.log(np.abs((s_11_solution - s_11) / s_11_solution)))

    # RTP
    axs[0, 1].plot(f, np.real(rtp_solution), label='RTP_solution')
    axs[0, 1].plot(f, np.real(rtp), label='RTP')
    axs[0, 1].legend()
    axs[1, 1].plot(f, np.imag(rtp_solution), label='RTP_solution')
    axs[1, 1].plot(f, np.imag(rtp), label='RTP')
    axs[1, 1].legend()
    axs[2, 1].plot(f, np.log(np.abs((rtp_solution - rtp) / rtp_solution)))

    # S22
    axs[0, 2].plot(f, np.real(s_22_solution), label='S22_solution')
    axs[0, 2].plot(f, np.real(s_22), label='S22')
    axs[0, 2].legend()
    axs[1, 2].plot(f, np.imag(s_22_solution), label='S22_solution')
    axs[1, 2].plot(f, np.imag(s_22), label='S22')
    axs[1, 2].legend()
    axs[2, 2].plot(f, np.log(np.abs((s_22_solution - s_22) / s_22_solution)))


    plt.tight_layout()
    plt.show()


def eam_transfer_function(s_21, s_probe_21, s_probe_22, gamma_eam):
    #r0 = 50.

    a = s_21 / 2
    #b = 1 + r0 / z_eam
    b = 2 / (1 + gamma_eam)
    c = (1 - s_probe_22 * gamma_eam) / s_probe_21

    return a * b * c
