import numpy as np
import os




def resistance_contour(foldername):
    # os.listdir returns an array of all the file names in the folder
    file_list = os.listdir('data/mpi/' + foldername)

    # remove files that are not in csv file format from the list
    file_list = remove_non_csv(file_list)

    # return the resistance through linear regression at each location
    position, r = extract_resistance(file_list, foldername)

    # reorder the array in order
    r = reorder_array(position, r)

    # print bloublou
    print('bloublou')




def remove_non_csv(file_list):
    n = len(file_list)
    i = 0


    while i < n:
        # remove the files that are not csv from the list
        if not 'csv' in file_list[i]:
            file_list.remove(file_list[i])

            i -= 1
            n -= 1
        else:
            i += 1


    return file_list


def extract_resistance(file_list, foldername):
    n = len(file_list)
    r = np.zeros(n, dtype=float)
    position = np.zeros(n, dtype=float)


    for j in range(n):
        file_name = file_list[j]
        # import colons v1 and i1
        data = np.genfromtxt('data/mpi/' + foldername + '/' + file_name, delimiter=',', skip_header=261, usecols=(1, 3))
        v = data[:, 0]
        i = data[:, 1]

        # get the coeff of the linear regression : resistance
        m = np.polyfit(v, i, 1)
        r[j] = 1 / m[0]

        # get the number of reference of the measurement
        data = np.genfromtxt('data/mpi/' + foldername + '/' + file_name, delimiter=',', skip_header=120, usecols=2, max_rows=2)
        position[j] = data[0]


    return position, r


def reorder_array(pos, val):
    n = len(pos)

    new_val = np.zeros(int(np.nanmax(pos)))

    for i in range(n):
        if not np.isnan(pos[i]):
            new_val[int(pos[i]) - 1] = val[i]


    return new_val