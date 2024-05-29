r"""
Definition
----------

This is the reparameterisation of the core shell bicelle in terms
such that it may be applied to the modeling of a phospholipid
nanodisc.

The reparameterisation is in terms of the SLD of the face and rim regions.
These are considered as the lipid heads and polymer or protein belt respectively
and therefore it is necessary to allow the SLD of them to vary with the
fractional solvation.


References
----------

.. [  # ] Idini, Ilaria. Polymer Stabilised Phospholipid Nanodiscs. Thesis
    (Ph.D.) - University of Bath, 2014., 2014. `Available from Ethos
    < https://ethos.bl.uk/OrderDetails.do?did = 1 & uin = uk.bl.ethos.629677 >`_
.. [  # ] Tognoloni, Cecilia. Formation of polymer-lipid nanodiscs for membrane
    protein studies Thesis(Ph.D.) - University of Bath, 2017., 2017.


Authorship and Verification
----------------------------

* **Author:** Andrew R. McCluskey **Date:** November 29, 2018
"""

import numpy as np
from numpy import inf, sin, cos, pi

name = "nanodisc"
title = "Reparameterisation of the core shell bicelle"
description = """
    A reparameterisation of the core shell bicelle in terms of a phophoslipid
    nanodisc.

    lipid_radius: Radius of core
    tails_thick: Thickness of lipid tail bilayer
    belt_thick: Thickness of polymer rim
    heads_thick: Thickness of lipid heads
    tails_sld: Scattering length density of lipid tails
    belt_sld: Scattering length density of polymer belt
    belt_solv: Fractional solvation of polymer belt
    heads_sld: Scattering length density of lipid heads
    heads_solv: Fractional solvation of lipid heads
    solvent_sld: Scattering length density of solvent
    """
category = "shape:cylinder"
opencl = True

# pylint: disable=bad-whitespace, line-too-long
#             ["name", "units", default, [lower, upper], "type", "description"],
parameters = [
    ["lipid_radius",    "Ang",          80, [0, inf],       "volume",       "Radius of core"],
    ["tails_thick",     "Ang",          50, [0, inf],       "volume",       "Thickness of lipid tail bilayer"],
    ["belt_thick",      "Ang",          10, [0, inf],       "volume",       "Thickness of polymer rim"],
    ["heads_thick",     "Ang",          10, [0, inf],       "volume",       "Thickness of lipid heads"],
    ["tails_sld",       "1e-6/Ang^2",   1,  [-inf, inf],    "sld",          "Scattering length density of lipid tails"],
    ["belt_sld",        "1e-6/Ang^2",   4,  [-inf, inf],    "sld",          "Scattering length density of polymer belt"],
    ["belt_solv",       "None",         0,  [0, 1],         "sld",          "Fractional solvation of polymer belt"],
    ["heads_sld",       "1e-6/Ang^2",   4,  [-inf, inf],    "sld",          "Scattering length density of lipid heads"],
    ["heads_solv",      "None",         0,  [0, 1],         "sld",          "Fractional solvation of lipid heads"],
    ["solvent_sld",     "1e-6/Ang^2",   1,  [-inf, inf],    "sld",          "Scattering length density of solvent"],
    ["theta",           "degrees",      90, [-360, 360],    "orientation",  "cylinder axis to beam angle"],
    ["phi",             "degrees",      0,  [-360, 360],    "orientation",  "rotation about beam"]
]

# pylint: enable=bad-whitespace, line-too-long

source = ["lib/sas_Si.c", "lib/polevl.c", "lib/sas_J1.c", "lib/gauss76.c",
          "nanodisc.c"]


def random():
    pars = dict(
        lipid_radius=10**np.random.uniform(1.3, 3),
        tails_thick=10**np.random.uniform(1.3, 4),
        belt_thick=10**np.random.uniform(0, 1.7),
        heads_thick=10**np.random.uniform(0, 1.7),
    )
    return pars


demo = dict(scale=1, background=0,
            lipid_radius=20.0,
            tails_thick=400.0,
            belt_thick=10.0,
            heads_thick=10.0,
            tails_sld=1.0,
            belt_sld=4.0,
            belt_solv=0,
            heads_sld=4.0,
            heads_solv=0,
            solvent_sld=1.0,
            theta=90,
            phi=0)
q = 0.1
# april 6 2017, rkh add unit tests, NOT compared with any other calc method, assume correct!
qx = q*cos(pi/6.0)
qy = q*sin(pi/6.0)
tests = [
    [{}, 0.05, 7.4883545957],
    [{'theta': 80., 'phi': 10.}, (qx, qy), 2.81048892474]
]
del qx, qy  # not necessary to delete, but cleaner
