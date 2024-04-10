import numpy as np
import matplotlib.pyplot as plt




MPL_SIZE = 18
plt.rc('font', size=MPL_SIZE)
plt.rc('axes', titlesize=MPL_SIZE)




def import_vertex_70_data_raw(file, document_folder=True, linux=False):
    if document_folder:
        if linux:
            file = '/home/llaplanche/Documents/mesures/ftir/' +file
        else:
            file = 'C:\\Users\\LCG\\PycharmProjects\\eam-vcsel\\data\\ftir\\' +file

    wavelength = np.loadtxt(file, skiprows=2, max_rows=1, dtype=str, comments=None)
    wavelength = clean_wavelength_data(wavelength)
    reflectivity = np.loadtxt(file, skiprows=3, usecols=0)

    # flip the array because for some reason opus goes decrescendo
    wavelength = np.flip(wavelength)
    reflectivity = np.flip(reflectivity)


    return wavelength, reflectivity


def import_vertex_70_data_dpt(file, document_folder=True, linux=False, normalize=True):
    if document_folder:
        if linux:
            file = '/home/llaplanche/Documents/mesures/ftir/' +file
        else:
            file = 'C:\\Users\\LCG\\PycharmProjects\\eam-vcsel\\data\\ftir\\' +file

    wavelength = np.loadtxt(file, delimiter='\t', usecols=0)
    wavelength = wavelength * 1e-9
    reflectivity = np.loadtxt(file, delimiter='\t', usecols=1)

    # flip the array because for some reason opus goes decrescendo
    wavelength = np.flip(wavelength)
    reflectivity = np.flip(reflectivity)

    # normalization
    if normalize:
        reflectivity = reflectivity / np.max(reflectivity) * 0.99


    return wavelength, reflectivity


def plot_ftir(file, title):
    w, r = import_vertex_70_data_dpt(file)

    # plot
    fig, ax = plt.subplots()

    # labels
    ax.set_xlabel('Wavelength [nm]')
    ax.set_ylabel('Reflection [1]')
    ax.set_title(title)

    # find index at 800 and 900 nm
    idx_800 = np.absolute(w - 800e-9).argmin()
    idx_900 = np.absolute(w - 900e-9).argmin()



    ax.plot(w[idx_800:idx_900] * 1e9, r[idx_800:idx_900], linewidth=2.0)

    plt.tight_layout()
    plt.show()




def clean_wavelength_data(array):
    # remove the #c comment
    array = np.delete(array, 0)

    # convert as float
    array = array.astype(float)
    array = array * 1e-9                        # [nm] -> [m]


    return array




def import_ftir_mbe_maison(file, document_folder=True, linux=True):
    if document_folder:
        if linux:
            file = '/home/llaplanche/Documents/mesures/ftir_salle_blanche/' +file
        else:
            file = 'Z:\\Documents\\mesures\\ftir_salle_blanche\\' +file

    wavelength = np.array(np.loadtxt(file, usecols=0)) * 1e-9                   # [nm] -> [m]
    reflectivity = np.array(np.loadtxt(file, usecols=1))


    return wavelength, reflectivity



def capital(money, interest, years):
    capital = 0
    interest = 1 + interest / 100

    for n in range(int(years)):
        capital += money * interest ** n

    return capital