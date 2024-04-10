from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as mplt
import numpy as np




MPL_SIZE = 18
mplt.rc('font', size=MPL_SIZE)
mplt.rc('axes', titlesize=MPL_SIZE)
mplt.rcParams['font.family'] = 'Calibri'




def in_situ_reflectivity(sample_name):
    # sample_name is supposed to be in format like 'A1397'
    time_refl = np.genfromtxt('data/epitaxy/' + sample_name + '/time_refl.csv') / 3600
    wavelength = np.genfromtxt('data/epitaxy/' + sample_name + '/wavelength.csv')
    reflectivity = np.genfromtxt('data/epitaxy/' + sample_name + '/reflectivity.csv',
                              delimiter=';', invalid_raise=False)
    reflectivity = reflectivity.T
    data_curv = np.genfromtxt('data/epitaxy/' + sample_name + '/curvature.csv', delimiter=';')
    time_curv = data_curv[:, 0] / 3600
    curvature = data_curv[:, 1]


    # find index at 700nm
    idx_min = np.absolute(wavelength - 700).argmin()
    # find index at time ...s
    idx_time_stop = np.absolute(time_refl - 50000 / 3600).argmin()
    idx_start_curv = np.absolute(time_curv - 1.7).argmin()
    # reduce vectors range
    wavelength = wavelength[idx_min:]
    time_refl = time_refl[:idx_time_stop]
    reflectivity = reflectivity[idx_min:, :idx_time_stop]
    time_curv = time_curv[idx_start_curv:]
    curvature = curvature[idx_start_curv:]


    # correct time vector
    time_refl -= time_refl[0]
    time_curv -= time_curv[0]
    idx_time_stop = np.absolute(time_curv - time_refl.max()).argmin()
    time_curv = time_curv[:idx_time_stop]
    curvature = curvature[:idx_time_stop]
    # normalize reflectivity
    reflectivity /= reflectivity.max()


    # plot
    # setup
    fig = mplt.figure(constrained_layout=True)
    gs = GridSpec(3, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1:, :])


    # plot
    ax1.plot(time_curv, curvature, linewidth=0.8)
    ax1.set_xlim(time_curv[0], time_curv[-1])
    ax1.set_ylabel('Curvature (km$^{-1}$)')
    mesh = ax2.pcolormesh(time_refl, wavelength, reflectivity, cmap='PuBu_r')
    ax2.set_xlabel('Growth time (h)')
    ax2.set_ylabel('Wavelength (nm)')
    cbar = mplt.colorbar(mesh)
    cbar.set_label('Relative Reflectivity')


    # add annotation
    at = AnchoredText(
        'a)', prop=dict(size=18), frameon=False, loc='upper left')
    ax1.add_artist(at)
    at = AnchoredText(
        'b)', prop=dict(size=18), frameon=False, loc='upper left')
    ax2.add_artist(at)


    mplt.show()
    mplt.savefig('curv-refl.png', bbox_inches='tight')


def check_curvature():
    data_curv = np.genfromtxt('data/epitaxy/A1397/curvature.csv', delimiter=';')

    time_curv = data_curv[:, 0] / 3600
    curvature = data_curv[:, 1]

    idx_start_curv = np.absolute(time_curv - 2).argmin()
    idx_stop_curv = np.absolute(time_curv - 13.5).argmin()

    time_curv = time_curv[idx_start_curv:idx_stop_curv]
    curvature = curvature[idx_start_curv:idx_stop_curv]

    # correct time vector
    time_curv -= time_curv[0]

    # plot
    fig, ax = mplt.subplots()
    ax.plot(time_curv, curvature)
    mplt.show()


def bowing(sample_name):
    # import
    file_path = '/home/llaplanche/PycharmProjects/eam-vcsel/data/curvature/' + sample_name + '.csv'
    data = np.genfromtxt(file_path, skip_header=1)
    x = data[:, 0]
    y = data[:, 1]
    z = data[0:, 2:]


    # plot
    # setup
    fig, ax = mplt.subplots()


    # plot
    mesh = ax.pcolormesh(x, y, z, cmap='coolwarm')
    ax.set_xlabel('(cm)')
    ax.set_ylabel('(cm)')
    cbar = mplt.colorbar(mesh)
    cbar.set_label('Bowing')


    mplt.show()
    mplt.savefig('bow.png', bbox_inches='tight')



def capital(base, rate, years):
    total = 0
    rate = 1 + rate / 100

    for n in range(int(years)):
        total += base * rate ** n

    return total