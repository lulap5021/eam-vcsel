import numpy as np
from tqdm import tqdm

from model import super_lattice_structure as sls


def command_mesh_points(thickness, n_points):
    return 'grid length=' +str(thickness) +' points=' +str(n_points) +'\n'


def command_mesh_coarse(thickness):
    step = thickness/10.

    command = 'grid length=' +str(step) +' points=' +str(20) +'\n'
    command += 'grid length=' +str(thickness -2*step) +' points=' +str(20) +'\n'
    command += 'grid length=' +str(step) +' points=' +str(20) +'\n'


    return command


def command_layer_structure(thickness, al):
    if al==0.:
        return 'structure material=gaas length=' +str(thickness) +'\n'
    else:
        return 'structure material=gaas alloy=al length=' +str(thickness) +' conc=' +str(al) +'\n'


def command_doping(thickness, na, nd):
    if na==nd:
        return 'doping length=' +str(thickness) +'\n'
    elif na>nd:
        return 'doping length=' +str(thickness) +' Na=' +str(na) +'\n'
    else:
        return 'doping length=' +str(thickness) +' Nd=' +str(nd) +'\n'




def convert_dataframe_to_dev(df, name='super_lattice'):
    # convert dataframe to numpy float32
    df['thickness'].astype(np.float32)
    df['al'].astype(np.float32)
    df['na'].astype(np.float32)
    df['nd'].astype(np.float32)

    # create a new file
    f = open(name +'.dev', 'w')

    # clean the file
    # absolute file positioning
    f.seek(0)
    # to erase all data
    f.truncate()

    # mesh
    for i in tqdm(range(df.shape[0])):
        # get the properties of the current layer
        l = df.at[i, 'thickness']

        # write the mesh line
        f.write(command_mesh_coarse(l))

    f.write('\n')

    # structure
    for i in tqdm(range(df.shape[0])):
        # get the properties of the current layer
        l = df.at[i, 'thickness']
        al = df.at[i, 'al']

        # write the mesh line
        f.write(command_layer_structure(l, al))

    f.write('\n')

    # doping
    for i in tqdm(range(df.shape[0])):
        # get the properties of the current layer
        l = df.at[i, 'thickness']
        na = df.at[i, 'na']
        nd = df.at[i, 'nd']

        # write the mesh line
        f.write(command_doping(l, na, nd))


    # close the file
    f.close()




def generate_super_lattice_file(name='super_lattice',
                                air = False,
                                mqw = True,
                                top_dbr = True,
                                bot_dbr = True,
                                bypass_dbr = True,
                                shared_dbr = False,
                                contact = True,
                                substrate = False,
                                amount_qw = 25,
                                l_qw = 8.27e-9,
                                l_cb_low_al = 2.2e-9,
                                l_cb_high_al = 0.3e-9,
                                l_clad_low_al = 2.2e-9,
                                l_clad_high_al = 0.3e-9,
                                top_dbr_period = 6,
                                bot_dbr_period = 8,
                                shared_dbr_period = 12,
                                grading_type = 'linear',
                                grading_width = 20e-9,
                                grading_period = 10,
                                v_ga6 = 100,
                                v_ga11 = 850,
                                v_al5 = 900,
                                v_al12 = 150):
    df = sls.structure_eam(air = air,
                           mqw = mqw,
                           top_dbr = top_dbr,
                           bot_dbr = bot_dbr,
                           bypass_dbr = bypass_dbr,
                           shared_dbr = shared_dbr,
                           contact = contact,
                           substrate = substrate,
                           amount_qw = amount_qw,
                           l_qw = l_qw,
                           l_cb_low_al = l_cb_low_al,
                           l_cb_high_al = l_cb_high_al,
                           l_clad_low_al = l_clad_low_al,
                           l_clad_high_al = l_clad_high_al,
                           top_dbr_period = top_dbr_period,
                           bot_dbr_period = bot_dbr_period,
                           shared_dbr_period = shared_dbr_period,
                           grading_type = grading_type,
                           grading_width = grading_width,
                           grading_period = grading_period,
                           v_ga6 = v_ga6,
                           v_ga11 = v_ga11,
                           v_al5 = v_al5,
                           v_al12 = v_al12)


    convert_dataframe_to_dev(df, name=name)
