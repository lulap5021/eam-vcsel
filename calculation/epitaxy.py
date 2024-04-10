import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
from tqdm import tqdm

from device import ftir as ft
from tool import pandas_tools as pdt, structure_macro as stm
from model import super_lattice_structure as sls, optic as op, transfer_matrix_method as tmm


def reflectivity_from_growth_speed_2x2(delta_speed=0.05,
                                       start_wavelength=700e-9,
                                       stop_wavelength=1000e-9,
                                       electric_field=0.,
                                       n_points=300):
    # reflectivity computation parameters
    wavelength = np.linspace(start_wavelength, stop_wavelength, num=n_points)
    r = np.zeros(len(wavelength))

    # epitaxy structure parameters
    speed_array = np.array([1. -delta_speed, 1. +delta_speed])
    v_ga6 = 100 * speed_array
    v_ga11 = 850 * speed_array
    v_al5 = 900 * speed_array
    v_al12 = 150 * speed_array

    cell_list = [v_ga6, v_ga11, v_al5, v_al12]
    cell_names = ['v_ga6', 'v_ga11', 'v_al5', 'v_al12']

    # figure parameters
    # create a figure
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=(cell_names[0], cell_names[1], cell_names[2], cell_names[3]))

    for i in tqdm(range(len(cell_list))):
        for j in range(len(cell_list[i])):
            argument = stm.eam_vcsel_classic_arguments() | {cell_names[i]: cell_list[i][j]}

            sl = sls.structure_eam_vcsel(**argument)

            for k in range(len(wavelength)):
                sl = op.algaas_super_lattice_refractive_index(sl, electric_field, wavelength[k])

                n = sl['refractive_index'].to_numpy(dtype=np.complex128)
                d = sl['thickness'].to_numpy(dtype=np.complex128)

                r[k] = tmm.reflection(n, d, wavelength[k])


            # add the traces
            fig.add_trace(
                go.Scatter(
                    x=wavelength,
                    y=r,
                    name=cell_names[i] +' = ' +str(int(cell_list[i][j])) +' um/h'),
                row=row_num(i),
                col=col_num(i),
            )

            # create axis objects
            fig.update_layout(
                xaxis=dict(
                    title='wavelength [m]'
                ),
                yaxis=dict(
                    title='reflectivity [1]',
                    titlefont=dict(
                        color='#1f77b4'
                    ),
                    tickfont=dict(
                        color='#1f77b4'
                    )
                )
            )


    # update layout properties
    fig.update_layout(
        title_text='reflectivity as a function of wavelength',
        font_size=20,
        width=1600,
    )


    # show the figure
    fig.show()


def reflectivity_from_growth_speed_1x2(start_wavelength=700e-9,
                                       stop_wavelength=1000e-9,
                                       electric_field=0.,
                                       n_points=200):
    # reflectivity computation parameters
    wavelength = np.linspace(start_wavelength, stop_wavelength, num=n_points)
    r = np.zeros(len(wavelength))

    # epitaxy structure parameters
    speed_array = np.array([1.02, 1.05])
    v_ga11 = 850 * speed_array
    v_al5 = 900 * speed_array

    cell_list = [v_ga11, v_al5]
    cell_names = ['v_ga11', 'v_al5']

    # figure parameters
    # create a figure
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=(cell_names[0], cell_names[1]))

    for i in tqdm(range(len(cell_list))):
        for j in range(len(cell_list[i])):
            argument = stm.eam_vcsel_classic_arguments() | {cell_names[i]: cell_list[i][j]}

            sl = sls.structure_eam_vcsel(**argument)


            for k in range(len(wavelength)):
                sl = op.algaas_super_lattice_refractive_index(sl, electric_field, wavelength[k])

                n = sl['refractive_index'].to_numpy(dtype=np.complex128)
                d = sl['thickness'].to_numpy(dtype=np.complex128)

                r[k] = tmm.reflection(n, d, wavelength[k])

            # add the traces
            fig.add_trace(
                go.Scatter(
                    x=wavelength,
                    y=r,
                    name=cell_names[i] +' = ' +str(cell_list[i][j]) +' um/h'),
                row=row_num(i),
                col=col_num(i),
            )

            # create axis objects
            fig.update_layout(
                xaxis=dict(
                    title='wavelength [m]'
                ),
                yaxis=dict(
                    title='reflectivity [1]',
                    titlefont=dict(
                        color='#1f77b4'
                    ),
                    tickfont=dict(
                        color='#1f77b4'
                    )
                )
            )


    # update layout properties
    fig.update_layout(
        title_text='reflectivity as a function of wavelength',
        width=1600,
    )


    # show the figure
    fig.show()


def reflectivity_from_growth_speed(v_ga6, v_ga11, v_al5, v_al12,
                                   start_wavelength=700e-9,
                                   stop_wavelength=1000e-9,
                                   electric_field=0.,
                                   n_points=200,
                                   figure=True):
    # reflectivity computation parameters
    wavelength = np.linspace(start_wavelength, stop_wavelength, num=n_points)
    r = np.zeros(len(wavelength))


    cells_arg = {'v_ga6' : v_ga6,
                 'v_ga11' : v_ga11,
                 'v_al5' : v_al5,
                 'v_al12' : v_al12,}
    argument = stm.eam_vcsel_classic_arguments() | cells_arg

    sl = sls.structure_eam_vcsel(**argument)


    for k in tqdm(range(len(wavelength))):
        sl = op.algaas_super_lattice_refractive_index(sl, electric_field, wavelength[k])

        n = sl['refractive_index'].to_numpy(dtype=np.complex128)
        d = sl['thickness'].to_numpy(dtype=np.complex128)

        r[k] = tmm.reflection(n, d, wavelength[k])


    # experimental data
    wavelength_exp, r_exp = ft.import_ftir_mbe_maison('A1397_centre.dat')
    # slice and shrink the arrays
    wavelength_exp, r_exp = resize_w_r_length(wavelength_exp, r_exp, start_wavelength, stop_wavelength, n_points)
    # normalize
    r_exp = r_exp / np.max(r_exp)


    if figure:
        # figure parameters
        # create a figure
        fig = go.Figure()

        # add the traces
        fig.add_trace(
            go.Scatter(
                x=wavelength,
                y=r,
                name='v_ga6 = ' +str(v_ga6) +' , '
                +'v_ga11 = ' +str(v_ga11) +' , '
                +'v_al5 = ' +str(v_al5) +' , '
                +'v_al12 = ' +str(v_al12),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=wavelength_exp,
                y=r_exp,
                name='ftir measurement'
            )
        )

        # create axis objects
        fig.update_layout(
            xaxis=dict(
                title='wavelength [m]'
            ),
            yaxis=dict(
                title='reflectivity [1]',
                titlefont=dict(
                    color='#1f77b4'
                ),
                tickfont=dict(
                    color='#1f77b4'
                )
            )
        )


        # update layout properties
        fig.update_layout(
            title_text='reflectivity as a function of wavelength',
            width=1600,
        )

        # show the figure
        fig.show()


def reflectivity_from_growth_speed_slider(delta_speed=0.2,
                                          n_points_slider=40,
                                          start_wavelength=700e-9,
                                          stop_wavelength=1000e-9,
                                          tmm_resolution=300):
    # reflectivity computation parameters
    wavelength = np.linspace(start_wavelength, stop_wavelength, num=tmm_resolution)
    r = np.zeros(len(wavelength))


    # create the vector of different speeds [1]
    speed_vector = np.linspace(1. -delta_speed, 1. +delta_speed, num=n_points_slider)
    v_al12_arr = 150.*speed_vector


    # create figure
    fig = go.Figure()


    # add traces, one for each slider step
    for v_al12 in tqdm(v_al12_arr):
        # arguments to create the corresponding structure
        cells_arg = {'v_al12': v_al12,}
        argument = stm.eam_vcsel_classic_arguments() | cells_arg

        # create the structure
        sl = sls.structure_eam_vcsel(**argument)

        # calculate the reflectivity
        for k in range(len(wavelength)):
            sl = op.algaas_super_lattice_refractive_index(sl, 0., wavelength[k])

            n = sl['refractive_index'].to_numpy(dtype=np.complex128)
            d = sl['thickness'].to_numpy(dtype=np.complex128)

            r[k] = tmm.reflection(n, d, wavelength[k])

        # add the trace with the current growth speed configuration
        fig.add_trace(
            go.Scatter(
                visible=False,
                x=wavelength,
                y=r,
                name='v_al12 = ' +str(v_al12),
            )
        )


    # make first trace visible
    fig.data[1].update(visible=True)


    fig.update_layout(
        sliders=slider_generator('Al12', fig.data, n_points_slider)
    )


    fig.show()


def reflectivity_from_growth_speed_slider_fixed_fp(fp_resonnance_wavelength,
                                                   delta_speed=0.2,
                                                   n_points_slider=40,
                                                   start_wavelength=700e-9,
                                                   stop_wavelength=1000e-9,
                                                   tmm_resolution=300):
    # reflectivity computation parameters
    wavelength = np.linspace(start_wavelength, stop_wavelength, num=tmm_resolution)
    r = np.zeros(len(wavelength))


    # create the vector of different speeds [1]
    speed_vector = np.linspace(1. -delta_speed, 1. +delta_speed, num=n_points_slider)
    v_ga11_arr = 850. * speed_vector * 1.05
    v_ga11_fp = 850.
    v_al5_arr = 900. * speed_vector


    # create figure
    fig = go.Figure()


    # add traces, one for each slider step
    for v_al5 in tqdm(v_al5_arr):
        # determine the pair of growth speed that gives the right fp resonnance
        for v_ga11 in np.flip(v_ga11_arr, 0):
            cells_arg = {'v_al5': v_al5,
                         'v_ga11': v_ga11,}
            argument = stm.eam_vcsel_classic_arguments() | cells_arg

            sl = sls.structure_eam_vcsel(**argument)

            delta_w = 6e-9
            wavelength_valley_set = np.linspace(fp_resonnance_wavelength -delta_w, fp_resonnance_wavelength +delta_w, num=7)

            for k in range(len(wavelength_valley_set)):
                sl = op.algaas_super_lattice_refractive_index(sl, 0., wavelength_valley_set[k])

                n = sl['refractive_index'].to_numpy(dtype=np.complex128)
                d = sl['thickness'].to_numpy(dtype=np.complex128)

                r[k] = tmm.reflection(n, d, wavelength_valley_set[k])

            if is_valley(r):
                v_ga11_fp = v_ga11
                print('v_ga11 = ', v_ga11_fp)
                break


        # arguments to create the corresponding structure
        cells_arg = {'v_al5': v_al5,
                     'v_ga11': v_ga11_fp,}
        argument = stm.eam_vcsel_classic_arguments() | cells_arg

        # create the structure
        sl = sls.structure_eam_vcsel(**argument)

        # calculate the reflectivity
        for k in range(len(wavelength)):
            sl = op.algaas_super_lattice_refractive_index(sl, 0., wavelength[k])

            n = sl['refractive_index'].to_numpy(dtype=np.complex128)
            d = sl['thickness'].to_numpy(dtype=np.complex128)

            r[k] = tmm.reflection(n, d, wavelength[k])

        # add the trace with the current growth speed configuration
        fig.add_trace(
            go.Scatter(
                visible=False,
                x=wavelength,
                y=r,
                name='v_al5 = ' +str(np.floor(v_al5)) +' v_ga11 = ' +str(np.floor(v_ga11_fp)),
            )
        )


    # make first trace visible
    fig.data[1].update(visible=True)


    fig.update_layout(
        sliders=slider_generator('Al5', fig.data, n_points_slider)
    )


    fig.show()




def growth_speed_fit(delta_speed=0.2,
                     start_wavelength=780e-9,
                     stop_wavelength=1000e-9,
                     tmm_resolution=200):
    # initial guess
    v_ga6 = 100.
    #v_ga6 = 90.
    v_ga11 = 850.
    #v_ga11 = 916.
    v_al5 = 900.
    #v_al5 = 790.
    v_al12 = 150.
    #v_al12 = 165.

    # initial guess object
    v0 = v_ga6, v_ga11, v_al5, v_al12

    # lower boundaries
    v_ga6_min = (1 -delta_speed) * v_ga6
    v_ga11_min = (1 -delta_speed) * v_ga11
    v_al5_min = (1 -delta_speed) * v_al5
    v_al12_min = (1 -delta_speed) * v_al12

    # higher boundaries
    v_ga6_max = (1 +delta_speed) * v_ga6
    v_ga11_max = (1 +delta_speed) * v_ga11
    v_al5_max = (1 +delta_speed) * v_al5
    v_al12_max = (1 +delta_speed) * v_al12

    # boundaries object
    bounds = ([v_ga6_min, v_ga11_min, v_al5_min, v_al12_min], [v_ga6_max, v_ga11_max, v_al5_max, v_al12_max])


    # curve fit
    # don't forget to choose in reflectivity_alias(...) the correct sl structure
    # according to the type of data you import from the ftir (it should be the same structure of course !)
    wavelength, r_to_fit = ft.import_ftir_mbe_maison('A1397_centre.dat')
    # slice and shrink the arrays
    wavelength, r_to_fit = resize_w_r_length(wavelength, r_to_fit, start_wavelength, stop_wavelength, tmm_resolution)
    # normalize
    r_to_fit = r_to_fit / np.max(r_to_fit)
    values = curve_fit(reflectivity_alias, wavelength, r_to_fit, v0, bounds=bounds)


    # print computed values
    print(values)


def reflectivity_alias(wavelength, v_ga6, v_ga11, v_al5, v_al12):
    # reflectivity array
    r = np.zeros(len(wavelength))


    # super lattice structure
    # arguments
    cells_arg = {'v_ga6' : v_ga6,
                 'v_ga11' : v_ga11,
                 'v_al5' : v_al5,
                 'v_al12' : v_al12,}
    argument = stm.eam_vcsel_classic_arguments() | cells_arg

    # create the structure
    sl = sls.structure_eam_vcsel(**argument)


    # calculate the reflectivity
    for k in range(len(wavelength)):
        sl = op.algaas_super_lattice_refractive_index(sl, 0., wavelength[k])

        n = sl['refractive_index'].to_numpy(dtype=np.complex128)
        d = sl['thickness'].to_numpy(dtype=np.complex128)

        r[k] = tmm.reflection(n, d, wavelength[k])


    # return the reflectivity
    return r




def reflectivity_heatmap(bypass_dbr=True,
                         eam_only=False,
                         start_wavelength=700e-9,
                         stop_wavelength=1000e-9,
                         electric_field=0.,
                         n_wavelength=9,
                         n_time=16,
                         v_ga6=100,
                         v_ga11=850,
                         v_al5=900,
                         v_al12=150):
    # create the wavelength array
    wavelength = np.linspace(start_wavelength, stop_wavelength, num=n_wavelength)

    # create the super lattice
    sl = sls.structure_eam_vcsel(bypass_dbr=bypass_dbr,
                                 eam_only=eam_only,
                                 v_ga6=v_ga6,
                                 v_ga11=v_ga11,
                                 v_al5=v_al5,
                                 v_al12=v_al12)

    # add the depth column
    sl = pdt.add_depth_column(sl)

    # add time column
    sl = pdt.add_epitaxial_time_columns(sl,
                                        v_ga6=v_ga6,
                                        v_ga11=v_ga11,
                                        v_al5=v_al5,
                                        v_al12=v_al12)

    # get the total time of epitaxy
    total_epitaxy_time = sl.at[0, 'stop_time']

    # create the time array
    time = np.linspace(1.0, total_epitaxy_time, num=n_time)

    # create the reflectivity array
    r = np.zeros([len(wavelength), len(time)])

    # calculate the reflectivity
    for i in tqdm(range(n_time)):
        sl_i = pdt.cut_sl_at_time(sl, time[i])

        # wavelength in [m]
        # wavelength must be a numpy array
        for j in range(len(wavelength)):
            sl_j = op.algaas_super_lattice_refractive_index(sl_i, electric_field, wavelength[j], temperature=550+273.15, lengyel=False, only_real=True)

            n = sl_j['refractive_index'].to_numpy(dtype=np.complex128)
            d = sl_j['thickness'].to_numpy(dtype=np.complex128)


            r[j, i] = tmm.reflection(n, d, wavelength[j])





    return time, wavelength, r




def row_num(x):
    if x>=2:
        return 2
    else:
        return 1


def col_num(x):
    # even
    if x%2==0:
        return 1
    # odd
    else:
        return 2


def slider_generator(slider_name, slider_array, n_points_slider):
    steps = []
    for i in range(len(slider_array)):
        step = dict(
            method='update',
            args=[{'visible': [False] * len(slider_array)},
                  {'title': slider_name + str(slider_array[i])}],  # layout attribute
        )
        step['args'][0]['visible'][i] = True  # toggle i'th trace to 'visible'
        steps.append(step)

    slider = [dict(
        active=10,
        currentvalue={'prefix': 'Growth speed: '},
        pad={'t': n_points_slider},
        steps=steps
    )]

    return slider


def resize_w_r_length(w, r, start_wavelength, stop_wavelength, tmm_resolution):
    # get the indices of the boundaries
    # calculate the difference array
    difference_array = np.absolute(w - start_wavelength)
    # find the index of minimum element from the array
    idx_start = difference_array.argmin()

    difference_array = np.absolute(w - stop_wavelength)
    idx_stop = difference_array.argmin()


    # slice the arrays
    w = w[idx_start:idx_stop]
    r = r[idx_start:idx_stop]


    # get the length of the array
    l = w.shape[0]

    # create an evenly spaced array of indices of length tmm_resolution
    if tmm_resolution<l:
        indices = np.floor(np.linspace(0, l -1, num = tmm_resolution)).astype(int)

        # shrink the arrays
        w = w[indices]
        r = r[indices]


    return w, r


def is_valley(r):
    l = r.shape[0]
    mid = int((l -1) / 2)

    for i in range(1, mid +1):
        if r[i] > r[i -1] :
            return False

    for i in range(mid, l):
        if r[i] < r[i -1] :
            return False


    return True
