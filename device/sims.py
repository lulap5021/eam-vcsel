import matplotlib.pyplot as plt
import numpy as np




MPL_SIZE = 24
plt.rc('font', size=MPL_SIZE)
plt.rc('axes', titlesize=MPL_SIZE)
plt.rcParams['font.family'] = 'Calibri'




def sims_result(file_name):
    #file_path = '/home/llaplanche/PycharmProjects/eam-vcsel/data/sims/' + file_name + '.csv'
    file_path = 'C:\\Users\\LCG\\PycharmProjects\\eam-vcsel\\data\\sims\\' + file_name + '.csv'
    data = np.genfromtxt(file_path, skip_header=1, invalid_raise=False)
    x_al_depth = data[:, 3]
    x_al = data[:, 4]

    #file_path = '/home/llaplanche/PycharmProjects/eam-vcsel/data/sims/' + file_name + '_dop.csv'
    file_path = 'C:\\Users\\LCG\\PycharmProjects\\eam-vcsel\\data\\sims\\' + file_name + '_dop.csv'
    data = np.genfromtxt(file_path, skip_header=1, invalid_raise=False)
    si_depth = data[:, 0]
    si_count = data[:, 1]
    c_depth = data[:, 2]
    c_count = data[:, 3]


    # slice arrays
    x_al_depth = x_al_depth[0:find_nearest(x_al_depth, 9.4)]
    si_depth = si_depth[0:find_nearest(si_depth, 9.4)]
    c_depth = c_depth[0:find_nearest(c_depth, 9.4)]

    x_al = x_al[0:len(x_al_depth)]
    si_count = si_count[0:len(si_depth)]
    c_count = c_count[0:len(c_depth)]


    # flip arrays
    x_al_depth = -np.flipud(x_al_depth - x_al_depth[-1])
    si_depth = -np.flipud(si_depth - si_depth[-1])
    c_depth = -np.flipud(c_depth - c_depth[-1])

    x_al = np.flipud(x_al)
    si_count = np.flipud(si_count)
    c_count = np.flipud(c_count)


    # plot
    # Create figure
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot()
    ax1.set_yscale("log", nonpositive='clip')
    ax1.set_ylim([2e14, 2e18])
    ax1.set_xlim([0, 9.4])


    ax1.set_ylabel('Doping (at/cm3)')
    ax1.set_xlabel('Depth (um)')
    ax1.plot(si_depth, si_count, color='seagreen', label='Si')
    ax1.plot(c_depth, c_count, color='royalblue', label='C')
    ax1.legend(loc='lower left')
    # Add a horizontal grid
    ax1.grid(axis='y', which='both')
    plt.tight_layout()

    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot()
    ax2.set_ylim([0, 100])
    ax2.set_xlim([0, 9.4])

    ax2.set_xlabel('Depth (um)')
    ax2.set_ylabel('Al (%)')

    ax2.plot(x_al_depth, x_al * 100, color='slateblue')
    # Add a horizontal grid
    ax2.grid(axis='y')

    plt.tight_layout()
    plt.show()




def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()


    return idx


