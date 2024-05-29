r"""
Definition
----------

This is the Pringle-Schmidt equation for fitting the helical form factor of an infinitely long helix formed from two helical tapes wrapped around each other at an angle $\epsilon$. The two helices are assumed to have the same width and thickness. Please see Figure 1
in Reference [1]. Note that this figure uses $\phi$ in place of the $\epsilon$ used here (because $\phi$ has another meaning in SasView.

**This model can also be used to model a single helical tape**. To do this, set $\epsilon$ = 0.

.. math::

    I(q) = \frac{\pi}{q L} \sum^{\inf}_{n = 0} \epsilon_{n} \cos^2 \left( \frac{n \epsilon}{2} \right) \frac{\sin^{2} \left( n \omega / 2 \right)}{\left( n \omega / 2 \right)^2} \left[ g_{n} \left( q, R, a \right) \right]^2

where

.. math::

    g_{n} \left( q, R, a \right) = 2 R^{-2} \left(1 - a^{2} \right) \times \int^{R}_{aR} r dr J_{n} \left[ q r \left( 1 - q^{2}_{n}) \right)^{1/2} \right]

.. math::

    q_{n} = \frac{2 \pi n}{P q} .

and $L$ is the total length of the tape, $\epsilon$ is the angle of separation between the helices, $\omega$ is the angle of the helical
cross section occupied by a tape, $n$ is the order of the layer line, $R$ is the outer radius of the tape, $aR$ is the inner radius
of the tape, and $P$ is the helical pitch.

References
----------

1) O. A. Pringle and P. W. Schmidt, Journal of Applied Crystallography, 1971, 4, 290-293, DOI: 10.1107/S002188987100699X
2) C. V. Teixeira, H. Amenitsch, T. Fukushima et al., Journal of Applied Crystallography, 2010, 43, 850-857, DOI: 10.1107/S0021889810015736

The fitting equation can be found in Reference [2] as Equations 15 & 16.


Authorship and Verification
----------------------------

* **Author:** Tim Snow **Date:** November 25, 2016
* **Last Modified by:** Tim Snow **Date:** January 23, 2016
* **Last Reviewed by:** Steve King **Date:** November 18, 2022
"""

import numpy
from numpy import inf
from scipy.integrate import quad
from scipy.special import jve

name = "pringle_schmidt_helices"
title = "Pringle-Schmidt helical form factor"
description = """\
      I(q) = (pi/qL) * sum[0 -> inf] e_n * cos^2(n epsilon^2/ 2) * (sin^2 (n * omega / 2) / (n * omega / 2) * (2 R^-2 (1 - a^2) * int[R -> aR] r dr J_n(1 - (2 * pi * n / P * q)^2 )^1/2 )^2

      L = Total length of the tape
      epsilon = Angle of separation between the helices
      omega = Angle of the cross section occupied by a tape
      n = Order of the layer line
      R = radius = Outer radius of the tape
      aR = radius_core = Inner radius of the tape
      P = Helical pitch

      """
category = "shape:cylinder"

# pylint: disable=bad-whitespace, line-too-long
#             ["name", "units", default, [lower, upper], "type", "description"],
parameters = [
        ["length", "um", 5.0, [0, inf], "", "Total length of the tape"],
        ["epsilon", "degrees", 100, [-360, 360], "", "Angle of separation between the helices"],
        ["omega", "degrees", 250.0, [-360, 360], "", "Angle of the helical cross section occupied by a tape"],
        ["n", "", 4.0, [0, inf], "", "Order of the layer line"],
        ["radius", "nm", 6.0, [0, inf], "", "Outer radius of the tape"],
        ["radius_core", "nm", 2.5, [0, inf], "", "Inner radius of the tape"],
        ["pitch", "nm", 1000.0, [0, inf], "", "Helical pitch"],
      ]
# pylint: enable=bad-whitespace, line-too-long


def ps_bessel_integral(x, q, n, pitch):
    # Work out q_n
    q_n = (2 * numpy.pi * n) / (pitch * q)
    # Do the bessel
    bessel_q = q * x * numpy.power((1 - numpy.power(q_n, 2)), 0.5)
    # Return it
    return jve(n, bessel_q)

def ps_bessel_function(radius_core, radius, q_loops, num_loops_array, pitch):
    # To facilitate vectorization
    return quad(ps_bessel_integral, radius_core, radius, args = (q_loops, num_loops_array, pitch))[0]


def Iq(q,
       length = 5.0,
       epsilon = 100.0,
       omega = 250.0,
       n = 4.0,
       radius = 6.0,
       radius_core = 2.5,
       pitch = 1000.0):
    """
    :param q:              Input q-value
    :param length:         Total length of the tape
    :param epsilon:        Angle of separation between the helices
    :param omega:          Angle of the helical cross section occupied by a tape
    :param n:              Order of the layer line
    :param radius:         Outer radius of the tape
    :param radius_core:    Inner radius of the tape
    :param pitch:          Helical pitch
    :return:               Calculated intensity
    """

    # Correct 1 from A to nm
    q = q * 10

    # First prepare some arrays for later broadcasting
    num_loops = n
    num_loops_array = numpy.arange(1, num_loops)
    num_loops_array = num_loops_array[:, None]

    q_loops = numpy.zeros((int(num_loops) - 1, len(q)))
    q_loops[:] = q

    # Vectorize the bessel function
    vec_bessel = numpy.vectorize(ps_bessel_function)

    # Now declare all the 'fixed' prefactors
    epsilon_n = 2
    epsilon_zero = 1
    n_epsilon = (num_loops_array * epsilon) / 2
    n_omega = (num_loops_array * omega) / 2
    radius_core_factor = radius_core / radius

    summation_prefactor = numpy.pi / (q_loops * (length * 1e3))
    integral_prefactor = 2 * numpy.power(radius, -2) * numpy.power((1 - numpy.power(radius_core_factor, 2)), -1)

    # Perform the initial summation
    summation = epsilon_n * numpy.power(numpy.cos(n_epsilon), 2) * (numpy.power(numpy.cos(n_omega), 2) / numpy.power(n_omega, 2)) # * the bessel bit
    
    # Perform the integral function
    integral = vec_bessel(radius_core, radius, q_loops, num_loops_array, pitch)

    # Multiply these together
    summation = numpy.sum((summation * integral_prefactor * integral), axis = 0)

    # Prepare to return the product
    returnValue = summation_prefactor[0] * summation

    # Return it!
    return returnValue

# For SASmodles/SASview
Iq.vectorized = True  # Iq accepts an array of q values

# A demo, to preview the bessel/keissig like behaviour, without going to extremes
demo = dict(scale = 1, background  = 0, length = 1.0, epsilon = 26.0, n = 6.0, omega = 328.0, radius = 17.8, radius_core = 0.5, pitch = 22.0)

# A simple unit test
tests = [
    [{'length' : 1.0,
       'epsilon' : 26.0,
       'n' : 6.0,
       'omega' : 328.0,
       'radius' : 17.8,
       'radius_core' : 0.5,
       'pitch' : 22.0}, 
       1.12, 3.673839e-06]
     ]