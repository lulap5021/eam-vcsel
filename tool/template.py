from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as mplt
import numpy as np
import scipy.optimize
from tqdm import tqdm


from device import ftir
from model import optic as op
from model import transfer_matrix_method as tmm
from tool import pandas_tools as pt
from model import super_lattice_structure as sls




MPL_SIZE = 24
mplt.rc('font', size=MPL_SIZE)
mplt.rc('axes', titlesize=MPL_SIZE)
mplt.rcParams['font.family'] = 'Calibri'



def graph_template_1():
    # plot
    # define subplots
    fig, ax = mplt.subplots()


    # add x-axis and y-axis label
    ax.set_xlabel('Lateral (um)')
    ax.set_ylabel('Height (um)')
    ax.set_ylim([0, 11*8])
    ax.set_ylim([-0.29, 2.54])


    mplt.tight_layout()
    mplt.show()

