# -*- coding: utf-8 -*-


import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


import pandas_tools as pt




pio.renderers.default = 'browser'




def plot_xy(x, y):
    # create a figure
    fig = go.Figure()


    # add the traces
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        name='curve'
    ))


    # create axis objects
    fig.update_layout(
        xaxis=dict(
            title='x'
        ),
        yaxis=dict(
            title='y',
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
        title_text='y as a function of x',
        width=1600,
    )


    # show the figure
    fig.show()




def plot_refra_doping(sl):
    # remove the substrate layer
    sl = sl.drop(sl.shape[0] -1, axis=0)


    # check wether the depth column exist
    if not 'depth' in sl:
        sl.insert(sl.shape[1], 'depth', value=np.nan)


    # duplicate each layer with a 0 thickness layer
    # in order to have two points for each layer
    # at the start and the end of each layer
    for i in range(sl.shape[0]):
        j = 2*i

        row_to_insert = sl.loc[j]
        row_to_insert['thickness'] = 0.

        sl = pt.insert_row(j, sl, row_to_insert)


    # calculate the depth of every layer
    for i in range(sl.shape[0]):
        sl.at[i, 'depth'] = sl.loc[0:i, 'thickness'].sum()


    # create a figure
    fig = go.Figure()


    # add the traces
    fig.add_trace(go.Scatter(
        x=sl['depth'],
        y=sl['refractive_index'].apply(np.real),
        name='refractive index'
    ))

    fig.add_trace(go.Scatter(
        x=sl['depth'],
        y=sl['na'],
        name='P doping',
        yaxis='y2'
    ))

    fig.add_trace(go.Scatter(
        x=sl['depth'],
        y=sl['nd'],
        name='N doping',
        yaxis='y2'
    ))


    # create axis objects
    fig.update_layout(
        xaxis=dict(
            title='depth [m]'
        ),
        yaxis=dict(
            title='refractive index',
            titlefont=dict(
                color='#1f77b4'
            ),
            tickfont=dict(
                color='#1f77b4'
            )
        ),
        yaxis2=dict(
            title='doping [at/cm³]',
            titlefont=dict(
                color='#ff7f0e'
            ),
            tickfont=dict(
                color='#ff7f0e'
            ),
            anchor='x',
            overlaying='y',
            side='right',
        )
    )


    # update layout properties
    fig.update_layout(
        title_text='refractive index and doping along z axis',
        width=1600,
    )


    # show the figure
    fig.show()


def plot_refra(sl):
    # remove the substrate layer
    sl = sl.drop(sl.shape[0] -1, axis=0)


    # check wether the depth column exist
    if not 'depth' in sl:
        sl.insert(sl.shape[1], 'depth', value=np.nan)


    # duplicate each layer with a 0 thickness layer
    # in order to have two points for each layer
    # at the start and the end of each layer
    for i in range(sl.shape[0]):
        j = 2*i

        row_to_insert = sl.loc[j]
        row_to_insert['thickness'] = 0.

        sl = pt.insert_row(j, sl, row_to_insert)


    # calculate the depth of every layer
    for i in range(sl.shape[0]):
        sl.at[i, 'depth'] = sl.loc[0:i, 'thickness'].sum()


    # create a figure
    fig = go.Figure()


    # add the traces
    fig.add_trace(go.Scatter(
        x=sl['depth'],
        y=sl['refractive_index'].apply(np.real),
        name='refractive index'
    ))


    # create axis objects
    fig.update_layout(
        xaxis=dict(
            title='depth [m]'
        ),
        yaxis=dict(
            title='refractive index',
            titlefont=dict(
                color='#1f77b4'
            ),
            tickfont=dict(
                color='#1f77b4'
            )
        ),
    )


    # update layout properties
    fig.update_layout(
        title_text='refractive index along z axis',
        width=1600,
    )


    # show the figure
    fig.show()


def plot_al_doping(sl):
    # remove the substrate layer
    sl = sl.drop(sl.shape[0] -1, axis=0)


    # check wether the depth column exist
    if not 'depth' in sl:
        sl.insert(sl.shape[1], 'depth', value=np.nan)


    # duplicate each layer with a 0 thickness layer
    # in order to have two points for each layer
    # at the start and the end of each layer
    for i in range(sl.shape[0]):
        j = 2*i
        row_to_insert = sl.loc[j]
        row_to_insert['thickness'] = 0.
        sl = pt.insert_row(j, sl, row_to_insert)


    # calculate the depth of every layer
    for i in range(sl.shape[0]):
        sl.at[i, 'depth'] = sl.loc[0:i, 'thickness'].sum()


    # create a figure
    fig = go.Figure()


    # add the traces
    fig.add_trace(go.Scatter(
        x=sl['depth'],
        y=sl['al'].apply(np.real),
        name='aluminium'
    ))

    fig.add_trace(go.Scatter(
        x=sl['depth'],
        y=sl['na'],
        name='P doping',
        yaxis='y2'
    ))

    fig.add_trace(go.Scatter(
        x=sl['depth'],
        y=sl['nd'],
        name='N doping',
        yaxis='y2'
    ))


    # create axis objects
    fig.update_layout(
        xaxis=dict(
            title='depth [m]'
        ),
        yaxis=dict(
            title='aluminium content [1]',
            titlefont=dict(
                color='#1f77b4'
            ),
            tickfont=dict(
                color='#1f77b4'
            )
        ),
        yaxis2=dict(
            title='doping [at/cm³]',
            titlefont=dict(
                color='#ff7f0e'
            ),
            tickfont=dict(
                color='#ff7f0e'
            ),
            anchor='x',
            overlaying='y',
            side='right',
        )
    )


    # update layout properties
    fig.update_layout(
        title_text='aluminium content and doping along z axis',
        width=1600,
    )


    # show the figure
    fig.show()


def plot_refra_em(sl, text_to_display=''):
    # remove the substrate layer
    sl = sl.drop(sl.shape[0] -1, axis=0)


    # check wether the depth column exist
    if not 'depth' in sl:
        sl.insert(sl.shape[1], 'depth', value=np.nan)


    # calculate the depth of every layer
    for i in range(sl.shape[0]):
        sl.at[i, 'depth'] = sl.loc[0:i, 'thickness'].sum()


    # create a figure
    fig = go.Figure()


    # add the traces
    fig.add_trace(go.Scatter(
        x=sl['depth'],
        y=sl['refractive_index'].apply(np.real),
        name='refractive index'
    ))

    fig.add_trace(go.Scatter(
        x=sl['depth'],
        y=sl['electromagnetic_amplitude'],
        name='em amplitude',
        yaxis='y2'
    ))


    # create axis objects
    fig.update_layout(
        xaxis=dict(
            title='depth [m]'
        ),
        yaxis=dict(
            title='refractive index',
            titlefont=dict(
                color='#1f77b4'
            ),
            tickfont=dict(
                color='#1f77b4'
            )
        ),
        yaxis2=dict(
            title='electromagnetic amplitude [1]',
            titlefont=dict(
                color='#ff7f0e'
            ),
            tickfont=dict(
                color='#ff7f0e'
            ),
            anchor='x',
            overlaying='y',
            side='right',
        )
    )


    # update layout properties
    fig.update_layout(
        title_text='refractive index and electromagnetic amplitude along z axis ' +text_to_display,
        width=1600,
    )


    # show the figure
    fig.show()


def plot_refra_clad_coupling_em(sl1, sl2, sl3):
    # sl[sl['name'].str.contains('cladding')].reset_index(drop=True).iloc[0]['thickness']
    # remove the substrate layer
    sl1 = sl1.drop(sl1.shape[0] -1, axis=0)
    sl2 = sl2.drop(sl2.shape[0] - 1, axis=0)
    sl3 = sl3.drop(sl3.shape[0] - 1, axis=0)


    # check wether the depth column exist
    if not 'depth' in sl1:
        sl1.insert(sl1.shape[1], 'depth', value=np.nan)
        sl2.insert(sl2.shape[1], 'depth', value=np.nan)
        sl3.insert(sl3.shape[1], 'depth', value=np.nan)


    # calculate the depth of every layer
    for i in range(sl1.shape[0]):
        sl1.at[i, 'depth'] = sl1.loc[0:i, 'thickness'].sum()
        sl2.at[i, 'depth'] = sl2.loc[0:i, 'thickness'].sum()
        sl3.at[i, 'depth'] = sl3.loc[0:i, 'thickness'].sum()


    # define xaxes, yaxes
    fig = make_subplots(
        rows=3,
        cols=1,
        specs=[[{'secondary_y': True}],
               [{'secondary_y': False}],
               [{'secondary_y': False}]]
    )


    # add traces
    fig.add_scatter(
        x=sl1['depth'],
        y=sl1['electromagnetic_amplitude'],
        name='em amplitude1',
        row=1,
        col=1
    )

    fig.add_scatter(
        x=sl1['depth'],
        y=sl1['refractive_index'].apply(np.real),
        name='refractive index',
        row=1,
        col=1,
        secondary_y=True
    )

    fig.add_scatter(
        x=sl2['depth'],
        y=sl2['electromagnetic_amplitude'],
        name='em amplitude2',
        row=2,
        col=1
    )

    fig.add_scatter(
        x=sl3['depth'],
        y=sl3['electromagnetic_amplitude'],
        name='em amplitude3',
        row=3,
        col=1
    )


    # update layout properties
    fig.update_layout(
        width=1600,
        height=900,
    )


    # show the figure
    fig.show()



def plot_reflectivity_depth(al, r, depth):
    # create a figure
    fig = go.Figure()


    # create the subplots
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=('reflectivity', 'al content [1]'))


    # add first trace
    fig.add_trace(go.Scatter(
        x=depth,
        y=np.real(r),
        name='reflectivity'),
    row=1,
    col=1
    )
    fig.update_layout(
        xaxis=dict(
            title='depth [m]'
        ),
        yaxis=dict(
            title='reflectivity',
            titlefont=dict(
                color='#1f77b4'
            ),
            tickfont=dict(
                color='#1f77b4'
            )
        )
    )


    # add second trace
    fig.add_trace(go.Scatter(
        x=depth,
        y=al,
        name='al content',
        yaxis='y2'),
    row=2,
    col=1
    )
    fig.update_layout(
        xaxis=dict(
            title='depth [m]'
        ),
        yaxis=dict(
            title='al content [1]',
            titlefont=dict(
                color='#ff7f0e'
            ),
            tickfont=dict(
                color='#ff7f0e'
            ),
            anchor='x',
            overlaying='y',
            side='right',
        )
    )


    # update layout properties
    fig.update_layout(
        title_text='reflectivity and al content as a function of depth',
        width=1600,
    )


    # show the figure
    fig.show()



def plot_reflectivity(wavelength, r):
    # create a figure
    fig = go.Figure()


    # add the traces
    fig.add_trace(go.Scatter(
        x=wavelength,
        y=r,
        name='reflectance'
    ))


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


def plot_mult_reflectivity(wavelength, r1, wavelength2, r2):
    # create a figure
    fig = go.Figure()


    # add the traces
    fig.add_trace(go.Scatter(
        x=wavelength,
        y=r1,
        name='measurement'
    ))
    fig.add_trace(go.Scatter(
        x=wavelength2,
        y=r2,
        name='theory'
    ))


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


def plot_std_heatmap(x, y, r):
    fig = go.Figure(data=
                    go.Heatmap(x=x, y=y, z=r, colorscale='jet'))


    # show the figure
    fig.show()


def plot_2_std_heatmaps(x1, y1, r1, x2, y2, r2):
    # define xaxes, yaxes
    fig = go.Figure(data=
                    go.Heatmap(x=x1, y=y1, z=r1, colorscale='jet'))

    # show the figure
    fig.show()


    fig = go.Figure(data=
                    go.Heatmap(x=x2, y=y2, z=r2, colorscale='jet'))


    # show the figure
    fig.show()


def plot_reflectivity_heatmap(time, wavelength, r):
    fig = go.Figure(data=
                    go.Heatmap(x=time, y=wavelength, z=r, colorscale='jet'))


    # show the figure
    fig.show()




def plot_psie(lz=0.01e-9):
    # import arrays
    v_e = np.load('v_e.npz')['arr_0']
    psi_e = np.load('psi_e.npz')['arr_0']


    # create a figure
    fig = go.Figure()


    # create depth array
    depth = np.arange(0., lz*v_e.shape[0], lz)


    # add the traces
    fig.add_trace(go.Scatter(
        x=depth,
        y=v_e,
        name='potential_barrier'
    ))

    fig.add_trace(go.Scatter(
        x=depth,
        #y=psi_e[0, :],
        y=psi_e,
        name='wavefunction',
        yaxis = 'y2'
    ))


    # create axis objects
    fig.update_layout(
        xaxis=dict(
            title='depth [m]'
        ),
        yaxis=dict(
            title='conduction band [eV]',
            titlefont=dict(
                color='#1f77b4'
            ),
            tickfont=dict(
                color='#1f77b4'
            )
        ),
        yaxis2 = dict(
            title='electron wavefunction [1]',
            titlefont=dict(
                color='#ff7f0e'
            ),
            tickfont=dict(
                color='#ff7f0e'
            ),
            anchor='x',
            overlaying='y',
            side='right',
        )
    )


    # update layout properties
    fig.update_layout(
        title_text='electron wavefunction vs depth',
        width=1600,
    )


    # show the figure
    fig.show()




def plot_refra_bench(al, wavelength, indices_array):
    # create a figure
    fig = go.Figure()


    # add the traces
    for x in range(al.size):
        fig.add_trace(go.Scatter(
            x=wavelength,
            y=indices_array[x],
            name='refracitve_index_al=' +str(al[x])
        ))


    # create axis objects
    fig.update_layout(
        xaxis=dict(
            title='wavelength [m]'
        ),
        yaxis=dict(
            title='refractive index [1]',
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
        title_text='refractive indices as a function of wavelength',
        width=1600,
    )


    # show the figure
    fig.show()




def plot_eos_spectrum_gold_plate(wavelength, data_mean_sample, data_empty, data_diff):
    fig = make_subplots(rows=1, cols=2)

    fig.add_trace(
        go.Scatter(x=wavelength,
                   y=data_mean_sample),
                   name = 'With sample',
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=wavelength,
                   y=data_empty),
                   name='Without sample (scaled)',
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=wavelength,
                   y=data_diff),
                   name='Delta',
        row=1, col=2
    )

    fig.update_layout(width=1600, title_text="Optical emission spectroscopy")
    fig.show()




# function that returns default yaxis domain for each subplot and the additional yaxes positions
def xyaxes_dom_yaxes_pos(gap=0.1, rows=2):
    if rows < 2:
        raise ValueError('This function works for subplots with  rows>2 and cols=1')
    h_window = (1 - gap) / rows  # window height
    d = 3 / 10 / 2
    # xaxis{k} has the domain [w[2],w[-3]] k=1,...rows
    # w[1], w[-2] give the left, resp right yaxes position associated to the default yaxis of the plot window
    yd = []
    for k in range(rows):
        start = k * (h_window + gap)
        end = start + h_window
        yd.append([start, end])
    w = [0, d, 2 * d, 1 - 2 * d, 1 - d, 1]

    return w, yd[::-1]  # yd[::-1] contains the domains of the default yaxes