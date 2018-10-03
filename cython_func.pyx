import sys

import numpy as np
cimport numpy as np
cimport cython
import config

cdef int M = config.M
cdef float delta_M = config.delta_M
cdef float mass_H2O = config.mass_H2O
cdef float mass_NH3 = config.mass_NH3
cdef float mass_H = config.mass_H
cdef float mass_CO = config.mass_CO

def get_ions_mz_index(float peptide_mass, float prefix_mass):
    """

    :param peptide_mass: the total mass of the peptide
    :param prefix_mass: the accumulated mass of AAs from the left hand side
    :return: the possible 18 combination of ions locations. (a,b,y) x (charge 1, charge 2) x (0, -H2O, -NH3)
    """
    b_ion_mass = prefix_mass + mass_H
    a_ion_mass = b_ion_mass - mass_CO
    y_ion_mass = peptide_mass - prefix_mass + mass_H

    # b-ions
    b_H2O = b_ion_mass - mass_H2O
    b_NH3 = b_ion_mass - mass_NH3
    #b_plus2_charge1 = (b_ion_mass + mass_H) / 2

    # a-ions
    a_H2O = a_ion_mass - mass_H2O
    a_NH3 = a_ion_mass - mass_NH3
    #a_plus2_charge1 = (a_ion_mass + mass_H) / 2

    # y-ions
    y_H2O = y_ion_mass - mass_H2O
    y_NH3 = y_ion_mass - mass_NH3
    #y_plus2_charge1 = (y_ion_mass + mass_H) / 2

    # charge state 1 9-ions:
    b_ions = [b_ion_mass,
              b_H2O,
              b_NH3]
    a_ions = [a_ion_mass,
              a_H2O,
              a_NH3]
    y_ions = [y_ion_mass,
              y_H2O,
              y_NH3]
    charge1_ions = b_ions + a_ions + y_ions
    charge1_ions_mz = np.array(charge1_ions)  # length 9 1-d array
    charge2_ions_mz = (charge1_ions_mz + mass_H) / 2
    ions_mz = np.concatenate((charge1_ions_mz, charge2_ions_mz))  # length 18 1-d array that stores the mz value

    ions_index = np.floor(ions_mz / delta_M).astype(np.int32)  # the output can be negative or greater than M, masking in tensorflow graph

    return ions_index
