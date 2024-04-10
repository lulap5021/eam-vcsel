import pandas as pd

from model import super_lattice_structure as st, optic as op


def eam_structures_description(document_folder=True):
    # create the structures
    eam = st.structure_eam_vcsel()
    eam_dbr_bypass = st.structure_eam_vcsel(bypass_dbr=True)

    # structures list
    eams = [eam, eam_dbr_bypass]

    # calculate the refraction index
    for i in range(len(eams)):
        eams[i] = op.algaas_super_lattice_refractive_index(eams[i], 0., 850e-9)

    # create the xlsx document
    if document_folder:
        writer = pd.ExcelWriter('Z:\\Documents\\' +'eam_structures_layers.xlsx', engine='xlsxwriter')
    else:
        writer = pd.ExcelWriter('data/xlsx/eam_structures_layers.xlsx', engine='xlsxwriter')


    # sheets list
    sheets = ['eam 1', 'eam 2 bypass dbr']


    for i in range(len(eams)):
        # write the sheets
        eams[i].to_excel(writer, sheet_name=sheets[i])


    # output the file
    writer.save()


def vcsel_eam_structures_description(document_folder=False):
    # create the structures
    vcsel_eam = st.structure_eam_vcsel(bypass_dbr=False, grading_type='linear digital', mqw_alloy_type='digital')
    vcsel_eam_dbr_bypass = st.structure_eam_vcsel(bypass_dbr=True, grading_type='linear digital', mqw_alloy_type='digital')
    vcsel_eam_double_alox = st.structure_eam_vcsel(bypass_dbr=False, eam_alox=True, grading_type='linear digital', mqw_alloy_type='digital')

    # structures list
    vcsel_eams = [vcsel_eam, vcsel_eam_dbr_bypass, vcsel_eam_double_alox]

    # calculate the refraction index
    for i in range(len(vcsel_eams)):
        vcsel_eams[i] = op.algaas_super_lattice_refractive_index(vcsel_eams[i], 0., 850e-9)


    # create the xlsx document
    if document_folder:
        writer = pd.ExcelWriter('Z:\\Documents\\' +'vcsel_eam_structures_layers.xlsx', engine='xlsxwriter')
    else:
        writer = pd.ExcelWriter('data/xlsx/vcsel_eam_structures_layers.xlsx', engine='xlsxwriter')


    # sheets list
    sheets = ['vcsel-eam 1', 'vcsel-eam 2 bypass dbr', 'vcsel-eam 3 double alox']


    for i in range(len(vcsel_eams)):
        # write the sheets
        vcsel_eams[i].to_excel(writer, sheet_name=sheets[i])


    # output the file
    writer.save()


def structure_to_xlsx(structure, file_name):
    # calculate the refraction index
    structure = op.algaas_super_lattice_refractive_index(structure, 0., 850e-9)

    writer = pd.ExcelWriter('data/xlsx/data/xlsx/' + file_name +'.xlsx', engine='xlsxwriter')

    structure.to_excel(writer)

    writer.save()