from model import super_lattice_structure as sls


# general dictionnary
def structure_boolean_arg_dict(config):
    # boolean setup
    # overwrite default config
    if 'vcsel_only' in config :
        arg_dict = {
            'vcsel_only': True,
            'eam_only': False,
            'air' : True,
            'top_contact' : False,
            'top_eam_dbr' : False,
            'eam_alox' : False,
            'eam_mqw' : False,
            'bypass_dbr' : False,
            'bot_eam_dbr' : False,
            'middle_contact' : True,
            'shared_dbr' : True,
            'vcsel_alox' : True,
            'vcsel_mqw' : True,
            'bot_vcsel_dbr' : True,
            'substrate' : True
        }
    elif 'eam_only' in config :
        arg_dict = {
            'vcsel_only': False,
            'eam_only': True,
            'air' : True,
            'top_contact' : True,
            'top_eam_dbr' : True,
            'eam_alox': False,
            'eam_mqw' : True,
            'bypass_dbr': False,
            'bot_eam_dbr' : True,
            'middle_contact' : True,
            'shared_dbr' : True,
            'vcsel_alox' : False,
            'vcsel_mqw' : False,
            'bot_vcsel_dbr' : False,
            'substrate' : True
        }
    elif 'eam_mqw_only' in config :
        arg_dict = {
            'vcsel_only' : False,
            'eam_only' : True,
            'air' : False,
            'top_contact' : False,
            'top_eam_dbr' : False,
            'eam_alox' : False,
            'eam_mqw' : True,
            'bypass_dbr': False,
            'bot_eam_dbr' : False,
            'middle_contact' : False,
            'shared_dbr' : False,
            'vcsel_alox' : False,
            'vcsel_mqw' : False,
            'bot_vcsel_dbr' : False,
            'substrate' : False
        }
    else:
        arg_dict = {
            'vcsel_only': False,
            'eam_only': False,
            'air' : True,
            'top_contact' : True,
            'top_eam_dbr' : True,
            'eam_alox' : False,
            'eam_mqw' : True,
            'bypass_dbr'  : False,
            'bot_eam_dbr' : True,
            'middle_contact' : True,
            'shared_dbr' : True,
            'vcsel_alox' : True,
            'vcsel_mqw' : True,
            'bot_vcsel_dbr' : True,
            'substrate' : True
        }


    return arg_dict



# structures
def eam_classic_structure():
    arg_dict = structure_boolean_arg_dict('eam_only')


    return sls.structure_eam_vcsel(**arg_dict)


def eam_bypass_structure():
    arg_dict = structure_boolean_arg_dict('eam_only')
    arg_dict['bypass_dbr'] = True

    return sls.structure_eam_vcsel(**arg_dict)


def eam_alox_structure():
    arg_dict = structure_boolean_arg_dict('eam_only')
    arg_dict['eam_alox'] = True
    arg_dict['bypass_dbr'] = True


    return sls.structure_eam_vcsel(**arg_dict)




def eam_vcsel_classic_structure():
    arg_dict = structure_boolean_arg_dict('eam_vcsel')


    return sls.structure_eam_vcsel(**arg_dict)


def eam_vcsel_bypass_structure():
    arg_dict = structure_boolean_arg_dict('eam_vcsel')
    arg_dict['bypass_dbr'] = True


    return sls.structure_eam_vcsel(**arg_dict)


def eam_vcsel_alox_structure():
    arg_dict = structure_boolean_arg_dict('eam_vcsel')
    arg_dict['eam_alox'] = True
    arg_dict['bypass_dbr'] = True


    return sls.structure_eam_vcsel(**arg_dict)




def vcsel_structure():
    arg_dict = structure_boolean_arg_dict('vcsel_only')


    return sls.structure_eam_vcsel(**arg_dict)




# arguments
def eam_classic_arguments():
    arg_dict = structure_boolean_arg_dict('eam_only')


    return arg_dict


def eam_bypass_arguments():
    arg_dict = structure_boolean_arg_dict('eam_only')
    arg_dict['bypass_dbr'] = True

    return arg_dict


def eam_alox_arguments():
    arg_dict = structure_boolean_arg_dict('eam_only')
    arg_dict['eam_alox'] = True
    arg_dict['bypass_dbr'] = True


    return arg_dict




def eam_vcsel_classic_arguments():
    arg_dict = structure_boolean_arg_dict('eam_vcsel')


    return arg_dict


def eam_vcsel_bypass_arguments():
    arg_dict = structure_boolean_arg_dict('eam_vcsel')
    arg_dict['bypass_dbr'] = True


    return arg_dict


def eam_vcsel_alox_arguments():
    arg_dict = structure_boolean_arg_dict('eam_vcsel')
    arg_dict['eam_alox'] = True
    arg_dict['bypass_dbr'] = True


    return arg_dict




def vcsel_arguments():
    arg_dict = structure_boolean_arg_dict('vcsel_only')


    return arg_dict