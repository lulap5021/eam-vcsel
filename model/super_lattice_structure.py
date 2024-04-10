import numpy as np
import pandas as pd


from globals import L_15_AL_DBR, L_90_AL_DBR
from model import optic as op




def structure_eam_vcsel(vcsel_only = False,
                        eam_only = False,

                        air = True,
                        top_contact = True,
                        top_eam_dbr = True,
                        eam_alox = False,
                        eam_mqw = True,
                        bypass_dbr = False,
                        bot_eam_dbr = True,
                        middle_contact = True,
                        shared_dbr = True,
                        vcsel_alox = True,
                        vcsel_mqw = True,
                        bot_vcsel_dbr = True,
                        substrate = True,

                        amount_eam_qw = 25,
                        eam_mean_al = 0.22,
                        l_eam_qw = 8.27e-9,
                        l_eam_cb = 10.3e-9,
                        l_eam_clad = 6e-9,

                        amount_vcsel_qw = 3,
                        vcsel_mean_al = 0.22,
                        l_vcsel_qw = 8.27e-9,
                        l_vcsel_cb = 10.3e-9,
                        l_vcsel_clad = 15e-9,

                        top_eam_dbr_period = 6,
                        bot_eam_dbr_period = 8,
                        shared_dbr_period = 12,
                        shared_dbr_period_bypass = 20,
                        bot_vcsel_period = 35,

                        mqw_alloy_type = 'digital',
                        grading_type = 'linear digital',
                        grading_width = 20e-9,
                        grading_period = 10,

                        v_ga6 = 100,
                        v_ga11 = 850,
                        v_al5 = 900,
                        v_al12 = 150):
    # times are in [s]
    # speeds are in [nm/h]


    if eam_only:
        vcsel_alox = False
        vcsel_mqw = False
        bot_vcsel_dbr = False
        shared_dbr = False


    # super lattice
    sl = pd.DataFrame(columns = ['name', 'thickness', 'al', 'na', 'nd', 'refractive_index', 'ga6', 'ga11', 'al5', 'al12'])


    # air
    if air:
        sl = sl.append(pd.DataFrame([['air', 1000e-9, 0., 0., 0., False, False, False, False]],
                                    columns=['name', 'thickness', 'al', 'na', 'nd', 'ga6', 'ga11', 'al5', 'al12']),
                                    ignore_index=True)


    # top contact
    if top_contact:
        sl = sl.append(pd.DataFrame([['top contact', 50e-9, 0., 0., 5e18, False, True, False, False]],
                                    columns=['name', 'thickness', 'al', 'na', 'nd', 'ga6', 'ga11', 'al5', 'al12']),
                                    ignore_index=True)


    # top distributed Bragg reflector
    if top_eam_dbr:
        dbr = structure_dbr(period = top_eam_dbr_period,
                            nd = 2e18,
                            grading_type = grading_type,
                            grading_width = grading_width,
                            grading_period = grading_period,
                            next_structure_is_eam_cladding = True,
                            v_ga6 = v_ga6,
                            v_ga11 = v_ga11,
                            v_al5 = v_al5,
                            v_al12 = v_al12)
        sl = sl.append(dbr, ignore_index=True)


    # aluminium oxide aperture
    if eam_alox:
        sl.drop(sl.tail(1).index, inplace=True)
        alox_layers = structure_alox(name='eam mesa')
        sl = sl.append(alox_layers, ignore_index=True)


    # eam mqw
    if eam_mqw:
        mqw = structure_mqw(name = 'eam',
                            mqw_alloy_type = mqw_alloy_type,
                            amount_qw = amount_eam_qw,
                            mean_al = eam_mean_al,
                            l_qw = l_eam_qw,
                            l_cb = l_eam_cb,
                            l_clad = l_eam_clad,
                            v_ga6 = v_ga6,
                            v_ga11 = v_ga11,
                            v_al12 = v_al12)
        sl = sl.append(mqw, ignore_index = True)


    # middle contact
    if middle_contact and bypass_dbr:
        mid_contact, previous_layer_is_dbr = structure_middle_contact(bypass_dbr = bypass_dbr,
                                                                      grading_type = grading_type,
                                                                      grading_width = grading_width,
                                                                      grading_period = grading_period)
        sl = sl.append(mid_contact, ignore_index=True)


    # bottom distributed Bragg reflector
    if bot_eam_dbr and (not bypass_dbr):
        dbr = structure_dbr(period = bot_eam_dbr_period,
                            na = 2e18,
                            grading_type = grading_type,
                            grading_width = grading_width,
                            grading_period = grading_period,
                            previous_structure_is_eam_cladding = True,
                            next_structure_is_mid_contact = True,
                            v_ga6 = v_ga6,
                            v_ga11 = v_ga11,
                            v_al5 = v_al5,
                            v_al12 = v_al12)
        sl = sl.append(dbr, ignore_index=True)
    elif bot_eam_dbr and bypass_dbr:
        dbr = structure_dbr(period = bot_eam_dbr_period,
                            na = 2e18,
                            grading_type = grading_type,
                            grading_width = grading_width,
                            grading_period = grading_period,
                            next_structure_is_dbr = True,
                            v_ga6 = v_ga6,
                            v_ga11 = v_ga11,
                            v_al5 = v_al5,
                            v_al12 = v_al12)
        sl = sl.append(dbr, ignore_index=True)


    # middle contact
    if middle_contact and (not bypass_dbr):
        mid_contact, previous_layer_is_dbr = structure_middle_contact(bypass_dbr = bypass_dbr,
                                                                      vcsel_only = vcsel_only,
                                                                      grading_type = grading_type,
                                                                      grading_width = grading_width,
                                                                      grading_period = grading_period)
        sl = sl.append(mid_contact, ignore_index=True)


    # shared distributed Bragg reflector
    if shared_dbr:
        if bypass_dbr:
            dbr = structure_dbr(period = shared_dbr_period_bypass,
                                na = 2e18,
                                grading_type = grading_type,
                                grading_width = grading_width,
                                grading_period = grading_period,
                                eam_only = eam_only,
                                next_structure_is_alox = True,
                                v_ga6 = v_ga6,
                                v_ga11 = v_ga11,
                                v_al5 = v_al5,
                                v_al12 = v_al12)
            sl = sl.append(dbr, ignore_index=True)
        else:
            if shared_dbr:
                dbr = structure_dbr(period=shared_dbr_period,
                                    na=2e18,
                                    grading_type=grading_type,
                                    grading_width=grading_width,
                                    grading_period=grading_period,
                                    eam_only=eam_only,
                                    next_structure_is_alox=True,
                                    v_ga6=v_ga6,
                                    v_ga11=v_ga11,
                                    v_al5=v_al5,
                                    v_al12=v_al12)
                sl = sl.append(dbr, ignore_index=True)


    # aluminium oxide aperture
    if vcsel_alox:
        alox_layers = structure_alox(name='vcsel mesa')
        sl = sl.append(alox_layers, ignore_index=True)


    # vcsel mqw
    if vcsel_mqw:
        linear_grading_low_to_high_al = structure_linear_grading(grading_type=grading_type,
                                                                 grading_width=90e-9,
                                                                 period=40,
                                                                 low_al = 0.3,
                                                                 high_al = 0.6,
                                                                 na=0.,
                                                                 nd=0.,
                                                                 v_ga11=v_ga11,
                                                                 v_al5=v_al5)
        linear_grading_high_to_low_al = linear_grading_low_to_high_al[::-1].copy()
        mqw = structure_mqw(name = 'vcsel',
                            mqw_alloy_type = mqw_alloy_type,
                            amount_qw = amount_vcsel_qw,
                            mean_al = vcsel_mean_al,
                            l_qw = l_vcsel_qw,
                            l_cb = l_vcsel_cb,
                            l_clad = l_vcsel_clad,
                            v_ga6 = v_ga6,
                            v_ga11 = v_ga11,
                            v_al12 = v_al12)
        sl = sl.append(linear_grading_high_to_low_al, ignore_index=True)
        sl = sl.append(mqw, ignore_index=True)
        sl = sl.append(linear_grading_low_to_high_al, ignore_index=True)


    # bottom distributed Bragg reflector
    if bot_vcsel_dbr:
        dbr = structure_dbr(period=bot_vcsel_period,
                            na=0.,
                            nd=2e18,
                            grading_type=grading_type,
                            grading_width=grading_width,
                            grading_period=grading_period,
                            previous_structure_is_vcsel_cavity=True,
                            v_ga6=v_ga6,
                            v_ga11=v_ga11,
                            v_al5=v_al5,
                            v_al12=v_al12)
        sl = sl.append(dbr, ignore_index=True)


    # substrate
    if substrate:
        sl = sl.append(pd.DataFrame([['substrate', 600e-6, 0., 0., 0., False, False, False, False]],
                                    columns=['name', 'thickness', 'al', 'na', 'nd', 'ga6', 'ga11', 'al5', 'al12']),
                                    ignore_index=True)


    return sl








def structure_mqw(name = 'vcsel',
                  mqw_alloy_type = 'digital',
                  amount_qw = 3,
                  mean_al = 0.3,
                  l_clad = 15e-9,
                  l_cb = 10e-9,
                  l_qw = 8.5e-9,
                  v_ga6 = 100,
                  v_ga11 = 850,
                  v_al12 = 150):
    # times are in [s]
    # speeds are in [nm/h]


    # super lattice
    sl = pd.DataFrame(columns = ['name', 'thickness', 'al', 'na', 'nd', 'refractive_index', 'ga6', 'ga11', 'al5', 'al12'])

    # layers definitions
    if 'digital' in mqw_alloy_type:
        # cladding
        clad = structure_standard_digital_alloy_15_60_hlh(l_clad,
                                                          mean_al=mean_al,
                                                          layer_name=name +' cladding',
                                                          v_ga6=v_ga6,
                                                          v_ga11=v_ga11,
                                                          v_al12=v_al12)
        # confinement barrier
        cb = structure_standard_digital_alloy_15_60_hlh(l_cb,
                                                        mean_al=mean_al,
                                                        layer_name=name +' confinement barrier',
                                                        v_ga6=v_ga6,
                                                        v_ga11=v_ga11,
                                                        v_al12=v_al12)
    else:
        # cladding
        clad = pd.DataFrame([[name +' cladding', l_clad, mean_al, 0., 0., False, False, False, False]],
                                columns=['name', 'thickness', 'al', 'na', 'nd', 'ga6', 'ga11', 'al5', 'al12'])
        # confinement barrier
        cb = pd.DataFrame([[name +' confinement barrier', l_cb, mean_al, 0., 0., False, False, False, False]],
                                columns=['name', 'thickness', 'al', 'na', 'nd', 'ga6', 'ga11', 'al5', 'al12'])


    # quantum well
    l_qw = true_epitaxy_thickness(l_qw, v_ga6, ga6=True)
    quantum_well = pd.DataFrame([[name +' quantum well', l_qw, 0., 0., 0., True, False, False, False]],
                                columns=['name', 'thickness', 'al', 'na', 'nd', 'ga6', 'ga11', 'al5', 'al12'])


    # super lattice
    # cladding
    sl = sl.append(clad, ignore_index = True)

    # mqw
    for i in range(amount_qw -1):
        # quantum well
        sl = sl.append(quantum_well, ignore_index = True)

        # confinement barrier
        sl = sl.append(cb, ignore_index = True)

    # quantum well
    sl = sl.append(quantum_well, ignore_index = True)

    #cladding
    sl = sl.append(clad, ignore_index = True)


    return sl




def structure_dbr(period = 6,
                  na = 0.,
                  nd = 0.,
                  grading_type= 'none',
                  grading_width = 20e-9,
                  grading_period = 10,
                  previous_structure_is_eam_cladding = False,
                  next_structure_is_eam_cladding = False,
                  previous_structure_is_top_contact = False,
                  previous_structure_is_vcsel_cavity = False,
                  next_structure_is_mid_contact = False,
                  next_structure_is_dbr = False,
                  next_structure_is_alox = False,
                  eam_only = False,
                  l_dbr_low_al = L_15_AL_DBR,
                  l_dbr_high_al = L_90_AL_DBR,
                  v_ga6 = 100,
                  v_ga11 = 850,
                  v_al5 = 900,
                  v_al12 = 150):
    # times are in [s]
    # speeds are in [nm/h]


    # super lattice
    sl = pd.DataFrame(columns = ['name', 'thickness', 'al', 'na', 'nd', 'refractive_index', 'ga6', 'ga11', 'al5', 'al12'])


    # layers definition
    if na > nd:
        doping_type = 'P'
    elif nd > na:
        doping_type = 'N'
    else:
        doping_type = '?'

    # al content
    low_al = v_al12/(v_al12 +v_ga11)                        # [1] 15% Al si tout va bien
    high_al = v_al5/(v_al5 +v_ga6)                          # [1] 90% Al si tout va bien

    # dbr layers thickness
    l_dbr_low_al = true_epitaxy_thickness(l_dbr_low_al, v_al12 +v_ga11, ga11=True, al12=True)
    l_dbr_high_al = true_epitaxy_thickness(l_dbr_high_al, v_al5 +v_ga6, ga6=True, al5=True)

    # grading definition
    linear_grading_low_to_high_al = structure_linear_grading(grading_type = grading_type,
                                                             grading_width = grading_width,
                                                             period = grading_period,
                                                             low_al = low_al,
                                                             high_al = high_al,
                                                             na = na,
                                                             nd = nd,
                                                             v_ga11 = v_ga11,
                                                             v_al5 = v_al5)
    linear_grading_high_to_low_al = linear_grading_low_to_high_al[::-1].copy()
    # not sure if needed
    grading_width = linear_grading_low_to_high_al['thickness'].sum()

    # dbr layers
    if 'linear' or 'mean' in grading_type:
        low_al_layer = pd.DataFrame([['dbr ' + doping_type + ' low Al ~ 15%', l_dbr_low_al -grading_width, low_al, na, nd, False, True, False, True]],
                              columns=['name', 'thickness', 'al', 'na', 'nd', 'ga6', 'ga11', 'al5', 'al12'])
        high_al_layer = pd.DataFrame([['dbr ' + doping_type + ' high Al ~ 90%', l_dbr_high_al -grading_width, high_al, na, nd, True, False, True, False]],
                               columns=['name', 'thickness', 'al', 'na', 'nd', 'ga6', 'ga11', 'al5', 'al12'])

    else:
        low_al_layer = pd.DataFrame([['dbr ' + doping_type + ' low Al ~ 15%', l_dbr_low_al, low_al, na, nd, False, True, False, True]],
                              columns=['name', 'thickness', 'al', 'na', 'nd', 'ga6', 'ga11', 'al5', 'al12'])
        high_al_layer = pd.DataFrame([['dbr ' + doping_type + ' high Al ~ 90%', l_dbr_high_al, high_al, na, nd, True, False, True, False]],
                               columns=['name', 'thickness', 'al', 'na', 'nd', 'ga6', 'ga11', 'al5', 'al12'])

    # special layers
    high_al_three_half_grading = pd.DataFrame([['dbr ' + doping_type + ' high Al ~ 90%', l_dbr_high_al -3*grading_width/2, high_al, na, nd, True, False, True, False]],
                                    columns=['name', 'thickness', 'al', 'na', 'nd', 'ga6', 'ga11', 'al5', 'al12'])
    high_al_one_side_grading = pd.DataFrame([['dbr ' + doping_type + ' high Al ~ 90%', l_dbr_high_al -grading_width/2, high_al, na, nd, True, False, True, False]],
                                            columns=['name', 'thickness', 'al', 'na', 'nd', 'ga6', 'ga11', 'al5', 'al12'])


    # dbr
    if 'linear' or 'mean' in grading_type:
        # beginning
        if previous_structure_is_top_contact:
            sl = sl.append(linear_grading_low_to_high_al, ignore_index=True)
            sl = sl.append(high_al_three_half_grading, ignore_index=True)
        elif previous_structure_is_eam_cladding:
            sl = sl.append(high_al_one_side_grading, ignore_index=True)
        elif previous_structure_is_vcsel_cavity:
            sl = sl.append(high_al_layer, ignore_index=True)
        else:
            sl = sl.append(linear_grading_low_to_high_al, ignore_index=True)
            sl = sl.append(high_al_layer, ignore_index=True)

        sl = sl.append(linear_grading_high_to_low_al, ignore_index=True)

        # loop
        for i in range(period -1):
            sl = sl.append(low_al_layer, ignore_index=True)
            sl = sl.append(linear_grading_low_to_high_al, ignore_index=True)
            sl = sl.append(high_al_layer, ignore_index=True)
            sl = sl.append(linear_grading_high_to_low_al, ignore_index=True)

        sl = sl.append(low_al_layer, ignore_index=True)
        sl = sl.append(linear_grading_low_to_high_al, ignore_index=True)

        # ending
        if next_structure_is_mid_contact:
            sl = sl.append(high_al_layer, ignore_index=True)
            sl = sl.append(linear_grading_high_to_low_al, ignore_index=True)
        elif next_structure_is_eam_cladding:
            sl = sl.append(high_al_one_side_grading, ignore_index=True)
        elif next_structure_is_dbr:
            sl = sl.append(high_al_layer, ignore_index=True)
            sl = sl.append(linear_grading_high_to_low_al, ignore_index=True)
            sl = sl.append(low_al_layer, ignore_index=True)
        elif next_structure_is_alox:
            if eam_only:
                if 'digital' in grading_type:
                    sl.drop(sl.tail(grading_period).index, inplace=True)
                else:
                    sl.drop(sl.tail(int(grading_period/2.)).index, inplace=True)

        else:
            sl = sl.append(high_al_layer, ignore_index=True)
            sl = sl.append(linear_grading_high_to_low_al, ignore_index=True)

    elif next_structure_is_alox:
        sl = sl.append(high_al_layer, ignore_index=True)

        for i in range(period -1):
            sl = sl.append(low_al_layer, ignore_index=True)
            sl = sl.append(high_al_layer, ignore_index=True)

        sl = sl.append(low_al_layer, ignore_index=True)

    else:
        sl = sl.append(high_al_layer, ignore_index=True)

        for i in range(period):
            sl = sl.append(low_al_layer, ignore_index=True)
            sl = sl.append(high_al_layer, ignore_index=True)


    return sl




def structure_linear_grading(grading_type = 'linear digital',
                             grading_width = 20e-9,
                             period = 10,
                             low_al = 0.,
                             high_al = 1.,
                             na = 2e18,
                             nd = 0.,
                             v_ga11 = 850,
                             v_al5 = 900):
    # width is in [m]
    # al contents are unitary [1]
    # dopings are in [at/cm3]
    # speeds are in [nm/h]


    # super lattice
    sl = pd.DataFrame(columns = ['name', 'thickness', 'al', 'na', 'nd', 'refractive_index', 'ga6', 'ga11', 'al5', 'al12'])

    if na > nd:
        doping_type = 'P'
    elif nd > na:
        doping_type = 'N'
    else:
        doping_type = '?'


    # grading
    if 'linear digital' in grading_type:
        for i in range(period):
            layer_al_content = low_al +((i +1) * (high_al -low_al) / (period +1))                   # [1]
            bi_layer_thickness = grading_width / period                                             # [m]

            al_thickness = layer_al_content * bi_layer_thickness                                    # [m]
            ga_thickness = (1 -layer_al_content) * bi_layer_thickness                               # [m]

            sl = sl.append(pd.DataFrame([['dbr ' +doping_type +' linear digital grading', true_epitaxy_thickness(al_thickness, v_al5, al5=True), 1., na, nd, False, False, True, False]],
                                        columns=['name', 'thickness', 'al', 'na', 'nd', 'ga6', 'ga11', 'al5', 'al12']),
                                        ignore_index=True)
            sl = sl.append(pd.DataFrame([['dbr ' +doping_type +' linear digital grading', true_epitaxy_thickness(ga_thickness, v_ga11, ga11=True), 0., na, nd, False, True, False, False]],
                                        columns=['name', 'thickness', 'al', 'na', 'nd', 'ga6', 'ga11', 'al5', 'al12']),
                                        ignore_index=True)
    elif 'linear slope' in grading_type:
        # case this is a dbr grading
        # al content are modified to be a mean
        if low_al == 0. and high_al == 1.:
            low_al = 0.15                                                                           # [1]
            high_al = 0.9                                                                           # [1]

        layer_thickness = grading_width / period                                                    # [m]

        for i in range(period):
            al_content = low_al +((i +1) * (high_al -low_al) / (period +1))                         # [1]

            sl = sl.append(pd.DataFrame([['dbr ' +doping_type +' linear slope grading', layer_thickness, al_content, na, nd, False, False, False, False]],
                                        columns=['name', 'thickness', 'al', 'na', 'nd', 'ga6', 'ga11', 'al5', 'al12']),
                                        ignore_index=True)
    elif 'mean' in grading_type:
        al_content = (low_al + high_al) / 2.  # [1]
        grading_width = grading_width / period  # [m]

        for i in range(period):
            sl = sl.append(pd.DataFrame([['dbr ' + doping_type + ' linear slope grading', grading_width, al_content, na, nd, False, False, False, False]],
                                        columns=['name', 'thickness', 'al', 'na', 'nd', 'ga6', 'ga11', 'al5', 'al12']),
                                        ignore_index=True)
    else:
        pass


    return sl


def structure_middle_contact(bypass_dbr = False,
                             vcsel_only = False,
                             grading_type = 'linear digital',
                             grading_width = 20e-9,
                             grading_period = 10,
                             l_low_al = 3*L_15_AL_DBR,
                             l_high_al = L_90_AL_DBR,
                             v_ga6 = 100,
                             v_ga11 = 850,
                             v_al5 = 900,
                             v_al12 = 150):
    # times are in [s]
    # speeds are in [nm/h]


    # super lattice
    sl = pd.DataFrame(columns = ['name', 'thickness', 'al', 'na', 'nd', 'refractive_index', 'ga6', 'ga11', 'al5', 'al12'])


    # layers definition
    l_low_al = true_epitaxy_thickness(l_low_al, v_al12 +v_ga11, al12=True, ga11=True)
    l_high_al = true_epitaxy_thickness(l_high_al, v_al5 +v_ga6, al5=True, ga6=True)

    # al content
    low_al = v_al12/(v_al12 +v_ga11)                    # [1] 15% Al si tout va bien
    high_al = v_al5/(v_al5 +v_ga6)                      # [1] 90% Al si tout va bien

    # grading definition
    #linear_grading_low_to_high_al = structure_linear_grading(grading_type=grading_type,
    #                                                         grading_width=grading_width,
    #                                                         period=grading_period,
    #                                                         v_ga11=v_ga11,
    #                                                         v_al5=v_al5)
    linear_grading_high_to_low_al = structure_linear_grading(grading_type = grading_type,
                                                             grading_width = grading_width,
                                                             period = grading_period,
                                                             v_ga11 = v_ga11,
                                                             v_al5 = v_al5)[::-1]

    if 'linear' in grading_type:
        grading_width = linear_grading_high_to_low_al['thickness'].sum()
    elif 'mean' in grading_type:
        pass
    else:
        grading_width = 0.


    # bypass contact
    if bypass_dbr:
        sl = sl.append(pd.DataFrame([['low index layer for middle contact', l_high_al -grading_width/2, high_al, 2e18, 0., True, False, True, False]],
                                    columns=['name', 'thickness', 'al', 'na', 'nd', 'ga6', 'ga11', 'al5', 'al12']),
                                    ignore_index=True)

        if 'linear' or 'mean' in grading_type:
            sl = sl.append(linear_grading_high_to_low_al, ignore_index=True)

            sl = sl.append(pd.DataFrame([['middle contact', l_low_al -grading_width, low_al, 5e18, 0., False, True, False, True]],
                                        columns=['name', 'thickness', 'al', 'na', 'nd', 'ga6', 'ga11', 'al5', 'al12']),
                                        ignore_index=True)

        else:
            sl = sl.append(pd.DataFrame([['middle contact', l_low_al, low_al, 5e18, 0., False, True, False, True]],
                                        columns=['name', 'thickness', 'al', 'na', 'nd', 'ga6', 'ga11', 'al5', 'al12']),
                                        ignore_index=True)

    # normal contact
    else:
        if 'linear' or 'mean' in grading_type:
            if vcsel_only:
                if 'digital' in grading_type:
                    sl = sl.append(linear_grading_high_to_low_al[grading_period -1 : 2*grading_period -1], ignore_index=True)
                else:
                    sl = sl.append(linear_grading_high_to_low_al[int(grading_period/2) - 1 : grading_period - 1], ignore_index=True)
                sl = sl.append(pd.DataFrame([['middle contact', l_low_al -grading_width, low_al, 5e18, 0., False, True, False, True]],
                                        columns=['name', 'thickness', 'al', 'na', 'nd', 'ga6', 'ga11', 'al5', 'al12']),
                                        ignore_index=True)
            else:
                sl = sl.append(pd.DataFrame([['middle contact', l_low_al -grading_width, low_al, 5e18, 0., False, True, False, True]],
                                        columns=['name', 'thickness', 'al', 'na', 'nd', 'ga6', 'ga11', 'al5', 'al12']),
                                        ignore_index=True)
        else:
            sl = sl.append(pd.DataFrame([['middle contact', l_low_al, low_al, 5e18, 0., False, True, False, True]],
                                        columns=['name', 'thickness', 'al', 'na', 'nd', 'ga6', 'ga11', 'al5', 'al12']),
                                        ignore_index=True)


    previous_layer_is_dbr = False

    return sl, previous_layer_is_dbr


def structure_alox(name = '??? mesa',
                   period = 10,
                   alox_thickness = 30e-9,
                   end_thickness = 28e-9,
                   alox_mean_al = 0.98,
                   low_al = 0.9,
                   high_al = 1.,
                   v_ga6 = 100,
                   v_al5 = 900):
    # width is in [m]
    # al contents are unitary [1]
    # dopings are in [at/cm3]
    # speeds are in [nm/h]


    # super lattice
    sl = pd.DataFrame(columns = ['name', 'thickness', 'al', 'na', 'nd', 'refractive_index', 'ga6', 'ga11', 'al5', 'al12'])


    # parameters
    al_90 = v_al5 / (v_al5 + v_ga6)                                                     # [1] 90% Al si tout va bien

    bi_layer_thickness = alox_thickness / period                                        # [m]

    al_thickness = bi_layer_thickness * (alox_mean_al - low_al) / (high_al - low_al)    # [m]
    al_90_thickness = bi_layer_thickness - al_thickness                                 # [m]


    # digital alloy
    for i in range(period):
        sl = sl.append(pd.DataFrame([[name +' AlOx 90% Al', true_epitaxy_thickness(al_90_thickness, v_al5 + v_ga6, al5=True, ga6=True), al_90, 2e18, 0., True, False, True, False]],
                                    columns=['name', 'thickness', 'al', 'na', 'nd', 'ga6', 'ga11', 'al5', 'al12']),
                                    ignore_index=True)
        sl = sl.append(pd.DataFrame([[name +' AlOx 100% Al', true_epitaxy_thickness(al_thickness, v_al5, al5=True), 1., 2e18, 0., False, False, True, False]],
                                    columns=['name', 'thickness', 'al', 'na', 'nd', 'ga6', 'ga11', 'al5', 'al12']),
                                    ignore_index=True)

    sl = sl.append(pd.DataFrame([[name +' AlOx 90% Al',true_epitaxy_thickness(end_thickness, v_al5 + v_ga6, al5=True, ga6=True), al_90, 2e18, 0., True, False, True, False]],
                                  columns=['name', 'thickness', 'al', 'na', 'nd', 'ga6', 'ga11', 'al5', 'al12']),
                                  ignore_index=True)


    return sl


def structure_standard_digital_alloy_15_60_hlh(total_thickness,
                                               mean_al = 0.5,
                                               na = 0.,
                                               nd = 0.,
                                               layer_name = 'digital alloy',
                                               v_ga6=100,
                                               v_ga11=850,
                                               v_al12=150):
    # for digital alloy where mean al content is within the range of 15% to 60%
    # start and end with high al content layers
    # width is in [m]
    # al contents are unitary [1]
    # dopings are in [at/cm3]
    # speeds are in [nm/h]


    # super lattice
    sl = pd.DataFrame(columns = ['name', 'thickness', 'al', 'na', 'nd', 'refractive_index', 'ga6', 'ga11', 'al5', 'al12'])


    # name parameters
    if na > nd:
        doping_type = 'P'
    elif nd > na:
        doping_type = 'N'
    else:
        doping_type = '?'

    layer_name = layer_name +' ' +doping_type +' '


    # physical parameters
    low_al = v_al12/(v_al12 +v_ga11)                                                            # [1] 15% Al si tout va bien
    high_al = v_al12/(v_al12 +v_ga6)                                                            # [1] 60% Al si tout va bien

    low_al_thickness_coeff = (high_al - mean_al) / (high_al - low_al)                           # [1]
    high_al_thickness_coeff = 1. - low_al_thickness_coeff                                       # [1]

    period = optimized_period_amount(total_thickness, low_al_thickness_coeff, high_al_thickness_coeff)

    low_al_thickness = total_thickness * low_al_thickness_coeff / period                        # [m]
    high_al_thickness = total_thickness * high_al_thickness_coeff / (period +1)                 # [m]

    l_low_al = true_epitaxy_thickness(low_al_thickness, v_ga11 +v_al12, ga11=True, al12=True)
    l_high_al = true_epitaxy_thickness(high_al_thickness, v_ga6 +v_al12, ga6=True, al12=True)


    # digital alloy
    for i in range(period):
        sl = sl.append(pd.DataFrame([[layer_name +' ~60% Al', l_high_al, high_al, na, nd, True, False, False, True]],
                                    columns=['name', 'thickness', 'al', 'na', 'nd', 'ga6', 'ga11', 'al5', 'al12']),
                                    ignore_index=True)
        sl = sl.append(pd.DataFrame([[layer_name +' ~15% Al', l_low_al, low_al, na, nd, False, True, False, True]],
                                    columns=['name', 'thickness', 'al', 'na', 'nd', 'ga6', 'ga11', 'al5', 'al12']),
                                    ignore_index=True)

    sl = sl.append(pd.DataFrame([[layer_name +' ~60% Al',l_high_al, high_al, na, nd, True, False, False, True]],
                                  columns=['name', 'thickness', 'al', 'na', 'nd', 'ga6', 'ga11', 'al5', 'al12']),
                                  ignore_index=True)


    return sl




def structure_zandbergen():
    # super lattice
    sl = pd.DataFrame(columns = ['name', 'thickness', 'al', 'na', 'nd', 'refractive_index', 'ga6', 'ga11', 'al5', 'al12'])


    # refractive indices
    n_gaas = op.afromovitz_real_algaas_refractive_index(0., 1e-6)
    n_algaas = op.afromovitz_real_algaas_refractive_index(1., 1e-6)

    # layer thicknesses
    l_gaas = 1e-6 / (4 * n_gaas)
    l_algaas = 1e-6 / (4 * n_algaas)
    l_cavity = 6 * l_gaas


    # air
    sl = sl.append(pd.DataFrame([['air', 500e-9, 0., 0., 0.]],
                                columns=['name', 'thickness', 'al', 'na', 'nd', 'ga6', 'ga11', 'al5', 'al12']),
                                ignore_index=True)


    # distributed Bragg reflector
    for i in range(14):
        sl = sl.append(pd.DataFrame([['gaas', l_gaas, 0., 0., 0.]],
                                    columns=['name', 'thickness', 'al', 'na', 'nd', 'ga6', 'ga11', 'al5', 'al12']),
                                    ignore_index=True)
        sl = sl.append(pd.DataFrame([['algaas', l_algaas, 1., 0., 0.]],
                                    columns=['name', 'thickness', 'al', 'na', 'nd', 'ga6', 'ga11', 'al5', 'al12']),
                                    ignore_index=True)

    # micro-cavity
    sl = sl.append(pd.DataFrame([['gaas', l_cavity, 0., 0., 0.]],
                                columns=['name', 'thickness', 'al', 'na', 'nd', 'ga6', 'ga11', 'al5', 'al12']),
                                ignore_index=True)

    # distributed Bragg reflector
    sl = sl.append(pd.DataFrame([['algaas', l_algaas, 1., 0., 0.]],
                                columns=['name', 'thickness', 'al', 'na', 'nd', 'ga6', 'ga11', 'al5', 'al12']),
                                ignore_index=True)
    for i in range(15):
        sl = sl.append(pd.DataFrame([['gaas', l_gaas, 0., 0., 0.]],
                                    columns=['name', 'thickness', 'al', 'na', 'nd', 'ga6', 'ga11', 'al5', 'al12']),
                                    ignore_index=True)
        sl = sl.append(pd.DataFrame([['algaas', l_algaas, 1., 0., 0.]],
                                    columns=['name', 'thickness', 'al', 'na', 'nd', 'ga6', 'ga11', 'al5', 'al12']),
                                    ignore_index=True)


    # substrate
    sl = sl.append(pd.DataFrame([['substrate', 250e-9, 1., 0., 0.]],
                                columns=['name', 'thickness', 'al', 'na', 'nd', 'ga6', 'ga11', 'al5', 'al12']),
                                ignore_index=True)


    return sl




def thick(growth_speed, growth_time):
    # time is in [s]
    # speed is in [nm/h]
    # return layer thickness in [m]

    return growth_speed * (growth_time / 3600) * 1e-9


def time(growth_speed, thickness):
    # return epitaxial time in [s]
    # speed is in [nm/h]
    # thickness is in [m]

    return thickness * 3600 / (growth_speed * 1e-9)


def true_epitaxy_thickness(expected_thickness,
                           corrected_growth_speed,
                           ga6=False,
                           ga11=False,
                           al5=False,
                           al12=False):
    # the epitaxial growth rate is not the same as the commanded growth rate
    # this function convert the command expected thickness
    # to the true epitaxial growth rate
    # using the corrected growth speed


    # expected growth speed that will be considered by the epitaxy software
    expected_growth_speed = 0.

    # speeds are in [nm/h]
    # those are the default growth rates for each cell
    if ga6:
        expected_growth_speed += 100.
    if ga11:
        expected_growth_speed += 850.
    if al5:
        expected_growth_speed += 900.
    if al12:
        expected_growth_speed += 150.


    # actual corrected growth speed
    epitaxial_time = time(expected_growth_speed, expected_thickness)

    thickness = thick(corrected_growth_speed, epitaxial_time)


    return thickness


def nmph_to_mps(v):
    # nm/h to m/s
    v = v*1e-9                                          # [nm/h] -> [m/h]
    v = v/3600.                                         # [m/h] -> [m/s]


    return v


def nmph_to_nmps(v):
    # nm/h to m/s
    v = v/3600.                                         # [nm/h] -> [nm/s]


    return v


def optimized_period_amount(total_thickness, low_al_thickness_coeff, high_al_thickness_coeff):
    min_coeff = np.min([low_al_thickness_coeff, high_al_thickness_coeff])
    period = total_thickness * min_coeff / 1e-9


    return np.ceil(period).astype(int)



