import pyvisa




def check_ressources():
    rm = pyvisa.ResourceManager()
    ressources = rm.list_resources()
    print(ressources)

    for r in ressources:
        my_instrument = rm.open_resource(r)
        print(my_instrument.query('*IDN?'))
