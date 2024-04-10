import numpy as np
import scipy.signal as ss


from tool import plot as plt




def import_horiba_ev20_data(file, document_folder=True):
    if document_folder:
        file = 'Z:\\Documents\\' +file +'.exp'

    time = np.loadtxt(file, delimiter='\t', skiprows=2, max_rows=1, dtype=str)
    time = clean_time_data(time)
    wavelength = np.loadtxt(file, delimiter='\t', skiprows=3, usecols=0)
    # select every columns except the first and last one that has a ghost character
    cols = [i +1 for i in range(time.shape[0] -2)]
    data = np.loadtxt(file, delimiter='\t', skiprows=3, usecols=cols)


    return time, wavelength, data




def convert_horiba_ev20_data_to_csv(file, document_folder=True):
    time, wavelength, data = import_horiba_ev20_data(file, document_folder=document_folder)

    data = np.vstack((time, data))
    wavelength = np.append(np.nan, wavelength)
    data = np.column_stack((wavelength, data))

    if document_folder:
        data.tofile('Z:\\Documents\\' +file +'.csv', sep = ',')
    else:
        data.tofile(file, sep = ',')




def clean_time_data(array):
    # remove the lambdas comment
    array = np.delete(array, 0)
    # remove the 'ghost' character
    array = np.delete(array, array.shape[0] -1)

    # remove the strings
    array = np.char.strip(array, chars='Spec ')
    array = np.char.strip(array, chars=' ms')

    # convert as float
    array = array.astype(float)
    array = array/1000.                                 # [ms] -> [s]


    return array




def import_data_march_2021():
    time, wavelength, data_1 = import_horiba_ev20_data('preclean_1')
    time, wavelength, data_2 = import_horiba_ev20_data('preclean_2')
    time, wavelength, data_3 = import_horiba_ev20_data('preclean_3')
    time, wavelength, data_4 = import_horiba_ev20_data('preclean_4')
    time, wavelength, data_empty = import_horiba_ev20_data('preclean_vide')

    data_set = [data_1, data_2, data_3, data_4, data_empty]


    # slice array to be of the same size
    data_set = slice_data_arrays(data_set)

    n = data_set[0].shape[0]
    time = time[:n]


    # mean data with a sample
    data_mean_sample = (data_set[0] +data_set[1] +data_set[2] +data_set[3]) / 5.

    data_diff = data_mean_sample -2.45*data_set[4] +1.


    # get a list of the peaks
    peaks = peak_list(data_mean_sample)


    return time, wavelength, peaks, data_mean_sample, data_set[4], data_diff




def normalize(data):
    return data/100.                            # [%] -> [1]


def peak_list(data):
    data = normalize(data)                      # [%] -> [1]

    # temporal mean of every line (wavelength)
    mean = np.mean(data, axis=1)

    # find the peaks
    peaks = ss.find_peaks(mean, height=0.01, distance=4, width=5)


    return peaks


def slice_data_arrays(data_set):
    # slice data sets in same size arrays
    # get the size of the smallest array
    mini = data_set[0].shape[1]

    for i in range(len(data_set)):
        if data_set[i].shape[1]<mini:
            mini = data_set[i].shape[1]

    # slice the arrays
    for i in range(len(data_set)):
        data_set[i] = data_set[i][:, :mini]


    return data_set


def temporal_mean(data):
    # temporal mean of every line (wavelength)
    mean = np.mean(data, axis=1)

    return mean




def oes_test():
    time, wavelength, peaks, data_mean_sample, data_empty, data_diff = import_data_march_2021()
    data_mean_sample = temporal_mean(data_mean_sample)
    data_empty = temporal_mean(data_empty)
    data_diff = temporal_mean(data_diff)
    plt.plot_eos_spectrum_gold_plate(wavelength, data_mean_sample, data_empty, data_diff)