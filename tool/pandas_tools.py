from decimal import Decimal
import numpy as np
import pandas as pd
from tqdm import tqdm

from model import super_lattice_structure as slt


def insert_row(row_number, df, row_value):
    # Function to insert row in the dataframe
    # Starting value of upper half
    start_upper = 0

    # End value of upper half
    end_upper = row_number

    # Start value of lower half
    start_lower = row_number

    # End value of lower half
    end_lower = df.shape[0]

    # Create a list of upper_half index
    upper_half = [*range(start_upper, end_upper, 1)]

    # Create a list of lower_half index
    lower_half = [*range(start_lower, end_lower, 1)]

    # Increment the value of lower half by 1
    lower_half = [x.__add__(1) for x in lower_half]

    # Combine the two lists
    index_ = upper_half + lower_half

    # Update the index of the dataframe
    df.index = index_

    # Insert a row at the end
    df.loc[row_number] = row_value

    # Sort the index labels
    df = df.sort_index()

    # return the dataframe
    return df



def etch_super_lattice_from_top(sl, etch_depth):
    # etch the super_lattice of etch_depth [m]
    # removes layers and/or adjust top layer thickness


    # removes the air layer
    if 'air' in sl.iloc[0]['name']:
        sl = sl.drop(index=0)
        sl = sl.reset_index(drop=True)


    # check wether the depth column exist
    if not 'depth' in sl:
        sl.insert(sl.shape[1], 'depth', value=np.nan)
    # calculate the depth of every layer
    for i in range(sl.shape[0]):
        sl.at[i, 'depth'] = sl.loc[0:i, 'thickness'].sum()


    # get the position of the layer
    # get index of corresponding layer
    idx = find_nearest_index(sl['depth'].to_numpy(), etch_depth)
    if sl.loc[idx, 'depth'] < etch_depth and idx < sl.shape[0] :
        idx += 1

    corrected_thickness = sl.loc[0:idx, 'thickness'].sum() -etch_depth

    # copy and remove the unneeded
    sl = sl.loc[idx:sl.shape[0]]

    # reset the indices so the following lines can work
    sl = sl.reset_index(drop=True)

    # modify first layer thickness
    if corrected_thickness > 0. :
        sl.at[0, 'thickness'] = corrected_thickness
    else :
        sl.at[0, 'thickness'] = 0.

    # re-add the air layer
    air_layer = pd.DataFrame([['air', 1000e-9, 0., 0., 0., 1., False, False, False, False]],
                                columns=['name', 'thickness', 'al', 'na', 'nd', 'refractive_index', 'ga6', 'ga11', 'al5', 'al12'])
    sl = pd.concat([air_layer, sl]).reset_index(drop=True)


    return sl


def cut_in_equal_layers_thickness(super_lattice, step, ignore_air=True, progress_bar=False):
    if ignore_air:
        i = 0
    else:
        i = 0

    # get a list of thicknesses
    total = super_lattice['thickness'].copy()
    # drop the substrate layer
    total = total[:-1]
    # compute the total lenght to cut
    total = total.sum() / step

    # set the progress bar total
    if progress_bar:
        pbar = tqdm(total=total)

    while i < (super_lattice.shape[0] - 1) :
        ini = i

        # get the thickness of current layer
        l = super_lattice.at[ini, 'thickness']

        if l <= step :
            # update iterator
            i += 1
        else:
            # create the layer that will be inserted
            # copy() avoids a warning
            row_to_insert = super_lattice.copy().loc[ini]
            row_to_insert['thickness'] = step

            # insert the layers
            for j in range(int(np.floor(l/step))):
                i += 1
                super_lattice = insert_row(i, super_lattice, row_to_insert)

            # remainder of the euclidean division
            l_remain = l % step

            # insert last layer
            row_to_insert['thickness'] = l_remain
            i += 1
            super_lattice = insert_row(i, super_lattice, row_to_insert)

            # delete the current cut layer
            super_lattice = super_lattice.drop(super_lattice.index[[ini]])
            super_lattice = super_lattice.reset_index(drop=True)

        if progress_bar:
            pbar.update(i -ini)
    if progress_bar:
        pbar.close()


    return super_lattice


def cut_sl_at_time(sl, time):
    # get index of corresponding layer
    idx = find_nearest_index(sl['stop_time'].to_numpy(), time)
    if time > sl.at[idx, 'stop_time']:
        idx -= 1

    # calculate its modified thickness
    if idx > 0:
        ratio = (time - sl.at[idx, 'start_time']) / (sl.at[idx, 'stop_time'] - sl.at[idx, 'start_time'])
    # air case
    else:
        ratio = 1.
    corrected_thickness = ratio * sl.at[idx, 'thickness']

    # copy and remove the unneeded
    sl_cut = sl.loc[idx:sl.shape[0]]

    # reset the indices so the following lines can work
    sl_cut = sl_cut.reset_index(drop=True)

    # modify first layer thickness
    sl_cut.at[0, 'thickness'] = corrected_thickness

    # modify first layer stop time
    sl_cut.at[0, 'stop_time'] = time


    return sl_cut


def cut_sl_at_first_layer_named(sl, name):
    # get index of corresponding layer
    idx = sl.loc[sl['name'].str.contains(name)].head().index.values[0]

    # copy and remove the unneeded
    sl_cut = sl.loc[0:idx]

    # reset the indices so the following lines can work
    sl_cut = sl_cut.reset_index(drop=True)


    return sl_cut


def remove_slice_exclusive(sl, beginning_name, end_name, grading_type, period):
    idx_1 = sl.index[sl.name == beginning_name][0]
    idx_2 = sl.index[sl.name == end_name][0]

    sl_cut = sl.truncate(after=idx_1)
    if 'linear' in grading_type:
        sl_cut = sl_cut.append(sl.truncate(before=idx_2 -int(period/2)),  ignore_index=True)
    else:
        sl_cut = sl_cut.append(sl.truncate(before=idx_2), ignore_index=True)


    return sl_cut




def add_depth_column(super_lattice):
    # check wether the depth column exist
    if not 'depth' in super_lattice:
        super_lattice.insert(super_lattice.shape[1], 'depth', value=np.nan)


    # calculate the depth of every layer
    for i in range(super_lattice.shape[0]):
        super_lattice.at[i, 'depth'] = super_lattice.loc[0:i, 'thickness'].sum()


    return super_lattice


def add_epitaxial_time_columns(super_lattice,
                               v_ga6 = 100,
                               v_ga11 = 850,
                               v_al5 = 900,
                               v_al12 = 150):
    # check wether the start_time column exist
    if not 'start_time' in super_lattice:
        super_lattice.insert(super_lattice.shape[1], 'start_time', value=np.nan)

    # check wether the stop_time column exist
    if not 'stop_time' in super_lattice:
        super_lattice.insert(super_lattice.shape[1], 'stop_time', value=np.nan)


    # reverse order of rows
    super_lattice = super_lattice[::-1]
    super_lattice = super_lattice.reset_index(drop=True)


    # calculate epitaxy time
    # layer 0
    lz = super_lattice.loc[0, 'thickness']

    v = 0.
    if super_lattice.at[0, 'ga6']:
        v += v_ga6
    if super_lattice.at[0, 'ga11']:
        v += v_ga11
    if super_lattice.at[0, 'al5']:
        v += v_al5
    if super_lattice.at[0, 'al12']:
        v += v_al12
    # [nm/h] -> [m/s]
    v = slt.nmph_to_mps(v)

    if v==0.:
        t = 0.
    else:
        t = lz / v

    super_lattice.at[0, 'start_time'] = 0.
    super_lattice.at[0, 'stop_time'] = t

    # run through the layers and compute the epitaxial time
    for i in range(1, super_lattice.shape[0]):
        lz = super_lattice.loc[i, 'thickness']

        v = 0.
        if super_lattice.at[i, 'ga6']:
            v += v_ga6
        if super_lattice.at[i, 'ga11']:
            v += v_ga11
        if super_lattice.at[i, 'al5']:
            v += v_al5
        if super_lattice.at[i, 'al12']:
            v += v_al12
        # [nm/h] -> [m/s]
        v = slt.nmph_to_mps(v)

        if v == 0.:
            t = 0.
        else:
            t = lz / v

        super_lattice.at[i, 'start_time'] = super_lattice.loc[i -1, 'stop_time']
        super_lattice.at[i, 'stop_time'] = super_lattice.loc[i, 'start_time'] + t


    # reverse order of rows
    super_lattice = super_lattice[::-1]
    super_lattice = super_lattice.reset_index(drop=True)


    return super_lattice




def extract_arrays_from_super_lattice(super_lattice):
    name_array = super_lattice['name'].to_numpy()
    al_array = super_lattice['al'].to_numpy(dtype=float)
    thickness_array = super_lattice['thickness'].to_numpy(dtype=float)
    r_array = super_lattice['refractive_index'].to_numpy(dtype=np.complex128)

    return [name_array, al_array, thickness_array, r_array]



def fexp(number):
    (sign, digits, exponent) = Decimal(number).as_tuple()
    return len(digits) + exponent - 1


def fman(number):
    return Decimal(number).scaleb(-fexp(number)).normalize()


def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()


    return idx