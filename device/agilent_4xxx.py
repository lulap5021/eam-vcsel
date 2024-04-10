import pandas as pd




def import_agilent_4294a_data(file, document_folder=True):
    if document_folder:
        file = 'Z:\\Documents\\' +file +'.txt'

    df = pd.read_table(file, header=30, skip_blank_lines=False)


    return df




def convert_agilent_4294a_data_to_csv(file, document_folder=True):
    df = import_agilent_4294a_data(file, document_folder=document_folder)

    if document_folder:
        df.to_csv('Z:\\Documents\\' +file +'.csv')
    else:
        df.to_csv(file)




def convert_agilent_data_to_impedance_csv(file, document_folder=True):
    # for use with the impedance.py library
    df = import_agilent_4294a_data(file, document_folder=document_folder)

    if document_folder:
        df.to_csv('Z:\\Documents\\' +file +'.csv', header=False, index=False, columns=['f (Hz)', 'Re (ohm)', 'Im (ohm)'])
    else:
        df.to_csv(file, header=False, index=False, columns=['f (Hz)', 'Re (ohm)', 'Im (ohm)'])