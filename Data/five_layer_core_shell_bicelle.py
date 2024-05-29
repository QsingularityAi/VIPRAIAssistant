#Last update: January 24, 2021

r"""
Definition
----------

This model calculates the form factor for a core-shell circular cylinder. The core includes 
three layers, two methylene and one methyl, which creates a five-layer model when combined 
with the two headgroup layers. The shell on the walls and ends of the model can be of 
different thicknesses and scattering length densities. Normalization of the form factor 
is done using the particle volume. Cylindrical symmetry is assumed for this model.

Given the scattering length densities (sld) $ho_{mlene}$, the methylene sld, $ho_{myl}$,
the methyl sld, $ho_f$, the face sld, $ho_r$, the rim sld and $ho_s$ the solvent sld, 
the scattering length density variation along the cylinder axis is:

    egin{align} 
    ho(r) = 
      egin{cases} 
      &ho_{mlene} 	ext{ for } 0 lt r lt R; -L_c/2 lt zlt L_c/2 \[1.5ex]
      &ho_{myl} 	ext{ for } 0 lt r lt R; -L_{myl}/2 lt zlt L_{myl}/2 \[1.5ex]
      &ho_f 	ext{ for } 0 lt r lt R; -(L_c+2t_f)/2 lt zlt -L; L lt zlt (L_c+2t_f)/2 \[1.5ex]
      &ho_r	ext{ for } R lt r lt R+t_r; -(L_c+2t_f)/2 lt zlt -L; L lt zlt (L_c+2t_f)/2
      end{cases}
    end{align}

Cylindrical coordinates are used for this model, where $alpha$ is the angle between the 
$Q$ vector and the cylinder axis, to give:

    egin{align} 
    I(Q,alpha) = frac{	ext{scale}}{V_t} cdot
        F(Q,alpha)^2.sin(alpha) + 	ext{background}
    end{align}

where

.. math::

        egin{align}    
    F(Q,alpha) = &igg[ 
    (ho_{myl} - ho_{mlene}) V_{myl} frac{2J_1(QRsinalpha)}{QRsinalpha}frac{sin(Q(L_{myl})cosalpha/2)}{Q(L_{myl}/2)cosalpha} \
    &+(ho_{mlene} - ho_f) V_c frac{2J_1(QRsinalpha)}{QRsinalpha}frac{sin(QL_ccosalpha/2)}{Q(L_c/2)cosalpha} \
    &+(ho_f - ho_r) V_{c+f} frac{2J_1(QRsinalpha)}{QRsinalpha}frac{sin(Q(L_c/2+t_f)cosalpha)}{Q(L_c/2+t_f)cosalpha} \
    &+(ho_r - ho_s) V_t frac{2J_1(Q(R+t_r)sinalpha)}{Q(R+t_r)sinalpha}frac{sin(Q(L_c/2+t_f)cosalpha)}{Q(L_c/2+t_f)cosalpha}
    igg]
    end{align}
        egin{align}
    L_c = &igg[
        2L_{mlene}+L_{myl}
    igg]
    end{align}

where $V_t$ is the total volume of the bicelle, $V_c$ the volume of the core,
$V_{c+f}$ the volume of the core plus the volume of the faces, $R$ is the radius
of the core, $L_c$ the length of the core, $L_{mlene} is the length of the methylene
layer, $L_{myl}$ the length of the methyl layer, $t_f$ the thickness of the face, 
$t_r$ the thickness of the rim and $J_1$ the usual first order bessel function.

The 1D scattering intensity for randomly oriented cylinders is calculated by 
integrating over all possible $	heta$ and $phi$.

The 1D output does not use the *theta* and *phi* parameters. The scattering kernel 
and the 1D scattering intensity use the c-library from NIST.

References
----------

.. [#] D Singh (2009). *Small angle scattering studies of self assembly in
   lipid mixtures*, John's Hopkins University Thesis (2009) 223-225. `Available
   from Proquest <http://search.proquest.com/docview/304915826?accountid
   =26379>`_
   [#] C Cheu, L Yang, M P Nieh (2020). *Refining internal bilayer structure of bicelles resolved 
   by extended-q small angle X-ray scattering*, Chemistry and Physics of Lipids, 231
   (2020) 104945.

Authorship and Verification
----------------------------

* **Author:** Catherine Cheu **Date:** 2019
"""

from numpy import inf, sin, cos

name = "five_layer_core_shell_bicelle"
title = "Circular cylinder with a three-layer core-shell scattering length density profiles.."
description = """
    P(q,alpha)= (scale/Vs)*f(q)^(2) + bkg,  where: 
    f(q)= Vt(sld_rim - sld_solvent)* sin[qLt.cos(alpha)/2]
    /[qLt.cos(alpha)/2]*J1(qRout.sin(alpha))
    /[qRout.sin(alpha)]+
    2*(sld_methylene-sld_face)*Vl1*sin[qLccos(alpha)/2][[qLc
    *cos(alpha)/2]*J1(qRc.sin(alpha))
    /qRc.sin(alpha)]+
    (sld_methyl-sld_methylene)*Vl2*sin[qlength2cos(alpha)/2][[qlength2
    *cos(alpha)/2]*J1(qRc.sin(alpha))
    /qRc.sin(alpha)]
    (sld_face-sld_rim)*(Vl1+Vl2+Vf)*sin[q(Lc+2.thick_face).
    cos(alpha)/2][[q(Lc+2.thick_face)*cos(alpha)/2]*
    J1(qRc.sin(alpha))/qRc.sin(alpha)]

    alpha:is the angle between the axis of
    the cylinder and the q-vector
    Vt = pi.(Rc + thick_rim)^2.Lt : total volume
    Vl1 = pi.Rc^2.(2.length1 + length2) :the volume of the core
    Vl2 = pi.Rc^2.length2 :the volume of the methyl group
    Vf = 2.pi.Rc^2.thick_face : the volume of the face
    Rc = radius: the core radius
    Lc = 2.length1 + length2: the length of the core
    Lt = Lc + 2.thick_face: total length
    length1: thickness of methylene group
    length2: thickness of methyl group
    thick_face: thickness of face
    thick_rim: thickness of rim
    Rout = radius + thick_rim
    sld_methylene, sld_methyl, sld_rim, sld_face:scattering length
    densities within the particle
    sld_solvent: the scattering length density
    of the solvent
    bkg: the background
    J1: the first order Bessel function
    theta: axis_theta of the cylinder
    phi: the axis_phi of the cylinder...
        """
category = "shape:cylinder"

# pylint: disable=bad-whitespace, line-too-long
#             ["name", "units", default, [lower, upper], "type", "description"],
parameters = [
    ["radius",         "Ang",       80, [0, inf],    "volume",      "Cylinder core radius"],
    ["thick_rim",  "Ang",       10, [0, inf],    "volume",      "Rim shell thickness"],
    ["thick_face", "Ang",       10, [0, inf],    "volume",      "Cylinder face thickness"],
    ["methylene_length",         "Ang",      25, [0, inf],    "volume",      "Methylene length of one side"],
    ["methyl_length",         "Ang",      0, [0, inf],    "volume",      "Methyl core length"],
    ["sld_methylene",       "1e-6/Ang^2", 1, [-inf, inf], "sld",         "Methylene scattering length density"],
    ["sld_methyl",       "1e-6/Ang^2", 1, [-inf, inf], "sld",         "Methyl core scattering length density"],
    ["sld_face",       "1e-6/Ang^2", 4, [-inf, inf], "sld",         "Cylinder face scattering length density"],
    ["sld_rim",        "1e-6/Ang^2", 4, [-inf, inf], "sld",         "Cylinder rim scattering length density"],
    ["sld_solvent",    "1e-6/Ang^2", 1, [-inf, inf], "sld",         "Solvent scattering length density"],
    ["theta",          "degrees",   90, [-inf, inf], "orientation", "In plane angle"],
    ["phi",            "degrees",    0, [-inf, inf], "orientation", "Out of plane angle"],
    ]

# pylint: enable=bad-whitespace, line-too-long

source = ["lib/sas_Si.c", "lib/polevl.c", "lib/sas_J1.c", "lib/gauss76.c",
          "five_layer_core_shell_bicelle.c"]

demo = dict(scale=1, background=0,
            radius=20.0,
            thick_rim=10.0,
            thick_face=10.0,
            methylene_length=100.0,
            methyl_length = 200.0,
            sld_methylene=1.0,
            sld_methyl=1.0,
            sld_face=4.0,
            sld_rim=4.0,
            sld_solvent=1.0,
            theta=90,
            phi=0)

#qx, qy = 0.4 * cos(pi/2.0), 0.5 * sin(0)
