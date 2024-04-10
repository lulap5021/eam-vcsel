import matplotlib.pyplot as plt
import numpy as np


MPL_SIZE = 20
plt.rc('font', size=MPL_SIZE)
plt.rc('axes', titlesize=MPL_SIZE)




def generate_550C_polynomial_regression():
    # this function generate a polynomial function that fit the x y response
    # to use it store the coefficients in m, store them
    # and call it using :
    # zz = polyval2d(xx, yy, m)


    # import data
    data = np.load('../data/550C_AlGaAs_refractive_indices/indices_arrays.npz')

    al = data['al_array']
    wavelength = data['wavelength_array']
    n = data['n_array']
    k = data['k_array']


    # reshape the arrays
    al, wavelength, n = reshape_xyz_matrices_into_vectors(al, wavelength, n)
    k = np.reshape(k, k.shape[0] * k.shape[1])


    # Fit a 3rd order, 2d polynomial
    n = polyfit2d(al, wavelength, n)
    k = polyfit2d(al, wavelength, k)


    # save the coeffs
    np.savez('data/550C_AlGaAs_refractive_indices/model.npz', n=n, k=k, allow_pickle=True)




def polyfit2d(x, y, z, deg=np.array([4, 4])):
    # create the vandermonde matrix which the least square will be applied on
    vander = np.polynomial.polynomial.polyvander2d(x, y, deg)

    # least square regression
    # c contains the coefficient
    # must be used by calling the function polyval
    c = np.linalg.lstsq(vander, z)


    return c




def reshape_xyz_matrices_into_vectors(x, y, z):
    # x, y, z must be numpy arrays
    # x, y must be vectors of lenght n and m
    # z must be a nxm matrix


    # get the length of x and y
    x_length = x.shape[0]
    y_length = y.shape[0]
    total_length = x_length * y_length

    # reshape z matrix into a vector
    z = np.reshape(z, total_length)


    # reshape x as a vector [x1 x2 x3 ... xn x1 x2 x3 ... xn]
    x_new = np.zeros([total_length])
    for i in range(y_length):
        for j in range(x_length):
            x_new[i * x_length +j] = x[j]

    x = x_new


    # reshape y as a vector [y1 y1 y1 ... y2 y2 y2 ... yn yn yn]
    y_new = np.zeros([total_length])
    for i in range(y_length):
        for j in range(x_length):
            y_new[i * x_length + j] = y[i]

    y = y_new


    return x, y, z




def empty_plot():
    x = np.linspace(0, 14, num=16)
    y = np.linspace(700, 1000, num=16)
    # plot
    fig, ax = plt.subplots()
    ax.set_xlabel('Time of epitaxy [hours]')
    ax.set_ylabel('Wavelength [nm]')
    ax.plot(x, y, color='tab:blue')

    plt.tight_layout()
    plt.show()