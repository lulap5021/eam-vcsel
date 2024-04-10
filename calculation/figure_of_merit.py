import numpy as np
import scipy.ndimage
from scipy import interpolate
from tqdm import tqdm


import matplotlib.pyplot as plt
from model import sqw_lengyel_absorption as lengy


MPL_SIZE = 16
plt.rc('font', size=MPL_SIZE)
plt.rc('axes', titlesize=MPL_SIZE)




def figure_of_merit(wavelength=850e-9,
                    e_field_on=60.,
                    e_field_off=80.):
    # 'ON' state : '1' logic state -> eam-vcsel emits light -> eam doesn't absorb -> low electric field (no QCSE)
    # 'OFF' state : '0' logic state -> eam-vcsel emits less light -> eam absorb -> high electric field (QCSE)
    # this implies the following :
    # e_field_off > e_field_on
    # alpha_off > alpha_on
    # wavelength in [m]
    # e_field in [kV/cm]


    # send exceptions instead of warnings in order for the try/except statements to work
    np.seterr(all='raise')


    # unit conversion
    # [kV/cm] -> [V/m]
    e_field_on = e_field_on * 1e5
    e_field_off = e_field_off * 1e5


    # resolution
    n_pts = 1000
    # amount of aluminium in confinement barriers
    al_x = np.linspace(0.1, 0.4, num=n_pts)
    # quantum well width
    qw_width = np.linspace(7e-9, 9e-9, num=n_pts)
    # factor of merit m
    m = np.zeros([n_pts, n_pts])


    # Lengyel model is unstable, these are default values in case of NaN returned
    temp_val_alp_on = 1e-10
    temp_val_alp_off = 1e-10


    for i in tqdm(range(n_pts)):
        for j in range(n_pts):
            # quantum well absorption
            try:
                alpha_on = lengy.gaas_sqw_absorption_at_wavelength(al_x[j], qw_width[i], e_field_on, wavelength)
                alpha_off = lengy.gaas_sqw_absorption_at_wavelength(al_x[j], qw_width[i], e_field_off, wavelength)
            except:
                # keep last values if Lengyel model failed
                alpha_on = temp_val_alp_on
                alpha_off = temp_val_alp_off


            # update temporary last values calculated
            temp_val_alp_on = alpha_on
            temp_val_alp_off = alpha_off


            # unit conversion
            # [cm-1] -> [m-1]
            alpha_on = alpha_on * 1e2
            alpha_off = alpha_off * 1e2


            # factor of merit m
            delta_alpha_sq = ( alpha_off -alpha_on ) ** 2
            delta_e_sq = (e_field_off -e_field_on) ** 2
            m[j, i] = delta_alpha_sq / (alpha_on * delta_e_sq)


    # save data
    np.savez_compressed('data/fom/figure_of_merit_' + str(n_pts) + '_pts', al_x=al_x, qw_width=qw_width, m=m)


    # plot
    fig, ax = plt.subplots()

    # labels
    ax.set_xlabel('Quantum well width [nm]')
    ax.set_ylabel('Aluminium in barrier [%]')

    img = ax.contourf(qw_width * 1e9, al_x * 1e2, m, levels=100, cmap='magma')
    fig.colorbar(img, ax=ax, label='Factor of merit')

    plt.tight_layout()
    plt.show()


def plot_fom():
    # resolution
    n_pts = 1000
    # amount of aluminium in confinement barriers
    #al_x = np.linspace(0.1, 0.4, num=n_pts)
    # quantum well width
    #qw_width = np.linspace(7e-9, 10e-9, num=n_pts)


    # factor of merit m
    # load data
    data = np.load('data/fom/figure_of_merit_' + str(n_pts) + '_pts.npz')
    al_x = data['al_x']
    qw_width = data['qw_width']
    m = data['m']


    # remove erratic data from limits of lengyel model
    for i in range(n_pts):
        for j in range(n_pts):
            if al_x[j] * 1e2 < (-10 * qw_width[i] * 1e9 +115) / 3:
                m[j][i] = 0


    # plot
    fig, ax = plt.subplots()

    # labels
    ax.set_xlabel('Quantum well width (nm)')
    ax.set_ylabel('Aluminium in barrier (%)')

    m = scipy.ndimage.gaussian_filter(m, sigma=5)
    img = ax.contourf(qw_width * 1e9, al_x * 1e2, m * 1e7, levels=100, cmap='RdYlBu_r')
    fig.colorbar(img, ax=ax, label='Factor of merit $(10^{-7})$', format='%1.1f')

    plt.tight_layout()
    plt.show()


def plot_3d_fom():
    # resolution
    n_pts = 1000
    # amount of aluminium in confinement barriers
    #al_x = np.linspace(0.1, 0.4, num=n_pts)
    # quantum well width
    #qw_width = np.linspace(7, 10, num=n_pts)


    # factor of merit m
    # load data
    data = np.load('data/fom/figure_of_merit_' + str(n_pts) + '_pts.npz')
    al_x = data['al_x']
    qw_width = data['qw_width'] * 1e9
    m = data['m']

    m = scipy.ndimage.gaussian_filter(m, sigma=5)

    # interpolation
    x2 = np.linspace(al_x.min(), al_x.max(), 1000)
    y2 = np.linspace(qw_width.min(), qw_width.max(), 1000)
    f = interpolate.interp2d(al_x, qw_width, m, kind='cubic')

    xx2, yy2 = np.meshgrid(x2, y2)
    zz2 = f(x2, y2)


    fig = plt.figure('3D surface')
    ax = fig.add_subplot(111, projection='3d')
    #al_x, qw_width = np.meshgrid(al_x, qw_width)
    surf = ax.plot_surface(xx2, yy2, zz2 * 1e7, cmap='RdYlBu_r')


    # Add a color bar which maps values to colors.
    fig.colorbar(surf , shrink=0.5, aspect=8, label='Factor of merit $(10^{-7})$', format='%1.1f')

    # labels
    ax.set_ylabel('Quantum well width (nm)')
    ax.set_xlabel('Aluminium in barrier (%)')


    plt.show()
