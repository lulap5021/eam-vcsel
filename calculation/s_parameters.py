import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import os
from scipy.io import savemat
from scipy.signal import savgol_filter




MPL_SIZE = 24
plt.rc('font', size=MPL_SIZE)
plt.rc('axes', titlesize=MPL_SIZE)
plt.rcParams['font.family'] = 'Calibri'




# data import / export functions
def import_cal_set(path):
    data = np.loadtxt(os.getcwd() + path + 'dut_open.s2p', skiprows=8)
    f = data[:, 0]
    #s_11_open = 10 ** (data[:, 1] / 20) * np.exp(1j * np.deg2rad(data[:, 2]))
    s_11_open = db_deg_to_z(data[:, 1], data[:, 2])

    data = np.loadtxt(os.getcwd() + path + 'dut_short.s2p', skiprows=8)
    #s_11_short = 10 ** (data[:, 1] / 20) * np.exp(1j * np.deg2rad(data[:, 2]))
    s_11_short = db_deg_to_z(data[:, 1], data[:, 2])

    data = np.loadtxt(os.getcwd() + path + 'dut_load_50.s2p', skiprows=8)
    #s_11_load = 10 ** (data[:, 1] / 20) * np.exp(1j * np.deg2rad(data[:, 2]))
    s_11_load = db_deg_to_z(data[:, 1], data[:, 2])


    return f, s_11_open, s_11_short, s_11_load

def import_probe_reference(path):
    # import solution
    data = np.loadtxt(os.getcwd() + path, skiprows=5)
    s_11 = data[:, 1] + 1j * data[:, 2]
    rtp = data[:, 3] + 1j * data[:, 4]
    rtp *= (data[:, 5] + 1j * data[:, 6])
    s_22 = data[:, 7] + 1j * data[:, 8]


    return s_11, rtp, s_22

def import_eam_s_parameters(path):
    data = np.loadtxt(os.getcwd() + path, skiprows=8)
    #s_11 = 10 ** (data[:, 1] / 20) * np.exp(1j * np.deg2rad(data[:, 2]))
    #s_21 = 10 ** (data[:, 3] / 20) * np.exp(1j * np.deg2rad(data[:, 4]))
    s_11 = db_deg_to_z(data[:, 1], data[:, 2])
    s_21 = db_deg_to_z(data[:, 3], data[:, 4])
    s_12 = db_deg_to_z(data[:, 5], data[:, 6])
    s_22 = db_deg_to_z(data[:, 7], data[:, 8])

    return s_11, s_12, s_21, s_22

def import_eam_gamma(path):
    data = np.loadtxt(os.getcwd() + path, skiprows=8)
    #gamma = 10 ** (data[:, 1] / 20) * np.exp(1j * np.deg2rad(data[:, 2]))
    gamma = db_deg_to_z(data[:, 1], data[:, 2])

    return gamma




# s-parameters functions
def z_to_gamma(z):
    # of course for 50ohm reference impedance
    z0 = 50
    return (z - z0) / (z + z0)

def gamma_to_z(gamma):
    # of course for 50ohm reference impedance
    z0 = 50
    return z0 * (1 + gamma) / (1 - gamma)

def db_deg_to_z(gain, degree):
    magnitude = 10 ** (gain / 20)
    radian = degree * np.pi / 180

    return magnitude * (np.cos(radian) + 1j * np.sin(radian))

def eam_transfer_function(gamma_eam, s_21_eam, s_probe_21, s_probe_22):
    #r0 = 50.
    #z_eam = gamma_to_z(gamma_eam)

    #a = 0.5 * (1 + r0 / z_eam)
    #b = (1 - s_probe_22 * gamma_eam) / s_probe_21
    c = (1 - s_probe_22 * gamma_eam) / ( (1 + gamma_eam) * s_probe_21)


    #return s_21_eam * a * b
    return s_21_eam * c




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

    # measurement matrix from equation (6)
    m_meas = np.array([s_11_open, s_11_short, s_11_load])

    # from equation (4)
    m_unknowns = np.linalg.solve(m_0, m_meas)

    # s parameter
    s_11 = m_unknowns[0]
    s_22 = m_unknowns[1]
    rtp = m_unknowns[2] + s_11 * s_22

    return s_11, rtp, s_22


def s_to_t(s_11, s_12, s_21, s_22):
    t_11 = 1 / s_21
    t_12 = - s_22 / s_21
    t_21 = s_11 / s_21
    t_22 = - (s_11 * s_22 - s_12 * s_21) / s_21


    return t_11, t_12, t_21, t_22

def t_to_s(t_11, t_12, t_21, t_22):
    s_11 = t_21 / t_11
    s_12 = (t_11 * t_22 - t_12 * t_21) / t_11
    s_21 = 1 / t_11
    s_22 = - t_12 / t_11


    return s_11, s_12, s_21, s_22

def s11_ref_vna_to_gamma_eam(sp_11, sp_12, sp_21, sp_22, s_11, s_12, s_21, s_22):
    # extract the eam reflection coefficient / s_11
    # in the probe reference
    # starting with s_11 in the vna reference
    # and the probe's s parameters
    # sp: probe s parameters
    # s: eam s parameters in the reference of vna
    t_11, t_12, t_21, t_22 = s_to_t(s_11, s_12, s_21, s_22)
    tp_11, tp_12, tp_21, tp_22 = s_to_t(sp_11, sp_12, sp_21, sp_22)
    n = len(t_11)
    se_11 = np.zeros(n)

    # solver
    for i in range(n):
        # t matrices construction
        t_g = np.array([[t_11[i], t_12[i]],
                        [t_21[i], t_22[i]]])
        t_probe = np.array([[tp_11[i], tp_12[i]],
                            [tp_21[i], tp_22[i]]])

        # solving t_g = t_probe @ t_eam
        t_eam = np.linalg.solve(t_g, t_probe)


        se_11[i], se_12, se_21, se_22 = t_to_s(t_eam[0, 0], t_eam[0, 1],
                                               t_eam[1, 0], t_eam[1, 1])


    return se_11

def resize_norm_to_db(vector, start, stop):
    vector = 20 * np.log10(np.abs(vector))
    mean = np.mean(vector[start:stop])
    vector -= mean

    window = 100
    polyorder = 3
    vector_smooth = savgol_filter(vector, window, polyorder)


    return vector, vector_smooth


# probes reflection coefficients
def gamma_probe_z40(f):
    # Formfactor Z Probe
    # Z40-P-GSG-150
    gamma_open = z_to_gamma(0 - 1j / (2 * np.pi * f * 4.8e-15))
    gamma_short = z_to_gamma(0 + 1j * 2 * np.pi * f * 21e-12)
    gamma_load = z_to_gamma(50 + 1j * 2 * np.pi * f * 6e-12)

    return gamma_open, gamma_short, gamma_load

def gamma_probe_i40(f):
    # PN probe
    # PN-I40-A-GSG-150
    gamma_open = z_to_gamma(1e-3 + 1 / (1j * 3.7e-15 * 2 * np.pi * f))
    gamma_short = z_to_gamma(1e-3 + 1j * 8.2e-12 * 2 * np.pi * f)
    gamma_load = z_to_gamma(50 + 1j * 2 * 3.7e-12 * np.pi * f)

    return gamma_open, gamma_short, gamma_load

def gamma_probe_z67(f):
    # Z probe
    # Z67-XVF-GSG150
    gamma_open = z_to_gamma(1j / (2 * np.pi * f * 3.7e-15))
    gamma_short = z_to_gamma(1j * 2 * np.pi * f * 16.9e-12)
    gamma_load = z_to_gamma(50 + 1j * 2 * np.pi * f * 5.6e-12)

    return gamma_open, gamma_short, gamma_load

def s_probe_z67():
    # ff_dir = '\\data\\pnax_n5244b\\ff_z40p_gsg150\\'
    # ff_dir = '\\data\\pnax_n5244b\\pn_i40_gsg150_comparison\\'
    ff_dir = '\\data\\pnax_n5244b\\z67_xvf_gsg150\\'

    # import cal set measurements
    f, s_11_open, s_11_short, s_11_load = import_cal_set(ff_dir)

    # reflection coefficients
    gamma_open, gamma_short, gamma_load = gamma_probe_z67(f)

    # instantiation with numpy
    n = len(s_11_open)
    s_probe_11 = np.zeros(n, dtype=np.complex128)
    rtp_probe = np.zeros(n, dtype=np.complex128)
    s_probe_22 = np.zeros(n, dtype=np.complex128)

    # calculate probe s parameters
    for i in range(n):
        s_probe_11[i], rtp_probe[i], s_probe_22[i] = s_params_linear_algebra_method(s_11_open[i], s_11_short[i],
                                                                                    s_11_load[i],
                                                                                    gamma_open[i], gamma_short[i],
                                                                                    gamma_load[i])
    s_probe_21 = np.sqrt(rtp_probe)
    s_probe_12 = s_probe_21

    return f, s_probe_11, s_probe_12, s_probe_21, s_probe_22


# mains
def test_pointes_i40():
    pn_dir = '/data/pnax_n5244b/pn_i40_gsg150/'


    # import cal set measurements
    f, s_11_open, s_11_short, s_11_load = import_cal_set(pn_dir)


    # reflection coefficients
    # PN Probe
    # PN-I40-A-GSG-150
    gamma_open = z_to_gamma(1e-3 - 1j / (2 * np.pi * f * 3.7e-15))
    gamma_short = z_to_gamma(1e-3 + 1j * 2 * np.pi * f * 8.2e-12)
    gamma_load = z_to_gamma(50 + 1j * 2 * np.pi * f * 3.7e-12)


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
    probe_reference_path = pn_dir + 'PN I40-A-GSG-150_SN_HT2KF-1_50M-40G-801p.s2p'
    s_11_solution, rtp_solution, s_22_solution = import_probe_reference(probe_reference_path)


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


    plt.show()

def h_s(folder, file, gamma_eam, s_probe_21, s_probe_22):
    # import eam s-parameters
    eam_path = folder + file
    s_11_eam, s_12_eam, s_21_eam, s_22_eam = import_eam_s_parameters(eam_path)

    # calculate voltage transfer function
    h_eam = eam_transfer_function(gamma_eam, s_21_eam, s_probe_21, s_probe_22)


    return h_eam, s_21_eam


def eam_processing():
    #ff_dir = '\\data\\pnax_n5244b\\ff_z40p_gsg150\\'
    #ff_dir = '\\data\\pnax_n5244b\\pn_i40_gsg150_comparison\\'
    ff_dir = '\\data\\pnax_n5244b\\z67_xvf_gsg150\\'

    # import cal set measurements
    f, s_11_open, s_11_short, s_11_load = import_cal_set(ff_dir)


    # reflection coefficients
    gamma_open, gamma_short, gamma_load = gamma_probe_z67(f)


    # instantiation with numpy
    n = len(s_11_open)
    s_probe_11 = np.zeros(n, dtype=np.complex128)
    rtp_probe = np.zeros(n, dtype=np.complex128)
    s_probe_22 = np.zeros(n, dtype=np.complex128)


    # calculate probe s parameters
    for i in range(n):
        s_probe_11[i], rtp_probe[i], s_probe_22[i] = s_params_linear_algebra_method(s_11_open[i], s_11_short[i], s_11_load[i],
                                                                  gamma_open[i], gamma_short[i], gamma_load[i])
    s_probe_21 = np.sqrt(rtp_probe)
    #s_probe_12 = s_probe_21


    # import eam s-parameters
    eam_path = ff_dir + 'EAM_2669_Cal2PORT_50GHz_Veam_7Vdc_LaserON.s2p'
    s_11_eam, s_12_eam, s_21_eam, s_22_eam = import_eam_s_parameters(eam_path)


    # calculate voltage transfer function
    gamma_eam = import_eam_gamma(ff_dir + 'EAM_2669_Cal1PORT_50GHz_Veam_5V5dc_LaserOFF_OSL_ISS.s2p')
    h_eam = eam_transfer_function(gamma_eam, s_21_eam, s_probe_21, s_probe_22)


    # resize
    start = 60      # start at 2GHz
    f = f[start:]
    h_eam = h_eam[start:]
    s_21_eam = s_21_eam[start:]


    # normalize
    h_eam /= h_eam[0]
    s_21_eam /= s_21_eam[0]


    # to decibel
    h_eam_db = 20 * np.log10(np.abs(h_eam))
    s_21_db = 20 * np.log10(np.abs(s_21_eam))


    # smooth curve
    window = 100
    polyorder = 3
    h_smooth = savgol_filter(h_eam_db, window, polyorder)
    s_smooth = savgol_filter(s_21_db, window, polyorder)


    # plot
    fig, ax = plt.subplots()
    plt.plot(f * 1e-9, s_21_db, color='lightcoral', label='S21')
    plt.plot(f * 1e-9, s_smooth, color='darkred')
    plt.plot(f * 1e-9, h_eam_db, color='dodgerblue', label='Heam')
    plt.plot(f * 1e-9, h_smooth, color='navy')

    plt.ylim(-15, 3)

    ax.hlines(- 3., f.min() * 1e-9, f.max() * 1e-9, color='black', linestyles='--')
    trans = transforms.blended_transform_factory(
        ax.get_yticklabels()[0].get_transform(), ax.transData)
    ax.text(0, -3., '-3', transform=trans, ha="right", va="center")

    ax.vlines(46.1, -15, - 3., color='black', linestyles='--')
    trans = transforms.blended_transform_factory(
        ax.get_xticklabels()[0].get_transform(), ax.transData)
    ax.text(46, -15 -1, '46', transform=trans, ha="center", va="center")

    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Gain (dB)')
    ax.legend(loc=0)


    plt.tight_layout()
    plt.show()


def eam_multi_processing():
    # import probe s-parameters
    folder = '\\data\\pnax_n5244b\\z67_xvf_gsg150\\'
    f, s_probe_11, s_probe_12, s_probe_21, s_probe_22 = s_probe_z67()

    gamma_eam = import_eam_gamma(folder + 'EAM_2669_Cal1PORT_50GHz_Veam_5V5dc_LaserOFF_OSL_ISS.s2p')

    h_eam_5, s_21_eam_5 = h_s(folder, 'EAM_2669_Cal2PORT_50GHz_Veam_5V5dc_LaserON.s2p', gamma_eam, s_probe_21, s_probe_22)
    h_eam_6, s_21_eam_6 = h_s(folder, 'EAM_2669_Cal2PORT_50GHz_Veam_6Vdc_LaserON.s2p', gamma_eam, s_probe_21, s_probe_22)
    h_eam_7, s_21_eam_7 = h_s(folder, 'EAM_2669_Cal2PORT_50GHz_Veam_7Vdc_LaserON.s2p', gamma_eam, s_probe_21, s_probe_22)
    h_eam_8, s_21_eam_8 = h_s(folder, 'EAM_2669_Cal2PORT_50GHz_Veam_8Vdc_LaserON.s2p', gamma_eam, s_probe_21, s_probe_22)


    # resize, normalize, to dB, and get smooth version
    start = 40          # start at 1GHz
    #start = 60          # start at 2GHz
    stop = 809          # stop at 25GHz
    #stop = 970          # stop at 30GHz
    #f = f[start:]
    h_eam_5, h_eam_5_s = resize_norm_to_db(h_eam_5, start, stop)
    h_eam_6, h_eam_6_s = resize_norm_to_db(h_eam_6, start, stop)
    h_eam_7, h_eam_7_s = resize_norm_to_db(h_eam_7, start, stop)
    h_eam_8, h_eam_8_s = resize_norm_to_db(h_eam_8, start, stop)


    # plot
    fig, ax = plt.subplots()
    #plt.plot(f * 1e-9, h_eam_8, color='plum', label='8V', lw=0.75)
    #plt.plot(f * 1e-9, h_eam_7, color='dodgerblue', label='7V', lw=0.75)
    plt.plot(f * 1e-9, h_eam_6, color='darkseagreen', label='6V', lw=0.75)
    #plt.plot(f * 1e-9, h_eam_5, color='palegoldenrod', label='5V', lw=0.75)


    #plt.plot(f * 1e-9, h_eam_5_s, color='darkgoldenrod')
    plt.plot(f * 1e-9, h_eam_6_s, color='darkgreen')
    #plt.plot(f * 1e-9, h_eam_7_s, color='blue')
    #plt.plot(f * 1e-9, h_eam_8_s, color='darkviolet')

    plt.ylim(-15, 3)

    ax.hlines(- 3., f.min() * 1e-9, f.max() * 1e-9, color='black', linestyles='--')
    trans = transforms.blended_transform_factory(
        ax.get_yticklabels()[0].get_transform(), ax.transData)
    ax.text(0, -3., '-3', transform=trans, ha="right", va="center")


    ax.vlines(47.1, -15, - 3., color='black', linestyles='--')
    trans = transforms.blended_transform_factory(
        ax.get_xticklabels()[0].get_transform(), ax.transData)
    ax.text(47, -15 -1, '47', transform=trans, ha="center", va="center")


    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Gain (dB)')
    ax.legend(loc=0)


    plt.tight_layout()
    plt.show()


    savemat('data/pnax_n5244b/z67_xvf_gsg150/h_eam_V/h_eam_5V.mat', {'h_eam_5V': h_eam_5})
    savemat('data/pnax_n5244b/z67_xvf_gsg150/h_eam_V/h_eam_6V.mat', {'h_eam_6V': h_eam_6})
    savemat('data/pnax_n5244b/z67_xvf_gsg150/h_eam_V/h_eam_7V.mat', {'h_eam_7V': h_eam_7})
    savemat('data/pnax_n5244b/z67_xvf_gsg150/h_eam_V/h_eam_8V.mat', {'h_eam_8V': h_eam_8})
    savemat('data/pnax_n5244b/z67_xvf_gsg150/h_eam_V/frequency.mat', {'frequency': f})

