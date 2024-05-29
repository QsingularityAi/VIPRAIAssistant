r"""

Definition
------------

This model is tailored for fitting the equatorial intensity profile from wood samples (Penttila et al., 2019). The model consists of three independent contributions:
1) Scattering in the plane perpendicular to the long axis of infinite cylinders packed in a hexagonal lattice with paracrystalline distortion (based on Hashimoto et al., 1994)
2) Gaussian function centered at $q = 0$
3) Power law scattering

The fitted function is
[ I(q) = A I_{cyl}(q,ar{R},Delta R/ar{R},a,Delta a /a) + B exp{-q^2/(2sigma^2)} + C q^{-alpha} + background ,]
where the cylinder radius $R$ has a Gaussian distribution with mean $ar{R}$ and standard deviation $Delta R$, and the paracrystalline distortion of the distance $a$ between the cylinders' center points is characterized by $Delta a$. 

The cylinder contribution is
[ I_{cyl}(q) = frac{1}{2pi} int_{0}^{2pi} I_{perp}(q,psi) dpsi , ]
where $psi$ is the rotational angle around the cylinder axis and 
[ I_{perp}(q,psi) = leftlangle left| f^2 ight| ightangle - left| leftlangle f ightangle ight| ^2 + left| leftlangle f ightangle ight| ^2 Z_1 Z_2 .]

The form factor of an infinitely long cylinder is
[ f(q, R) = A_{cyl} frac{J_1(qR)}{qR} = pi R frac{J_1(qR)}{q} ,]
where $J_1$ is the Bessel function of the first kind and $A_{cyl}$ the cross-sectional area of the cylinder. 

The terms with averaging are 
[  leftlangle left| f^2(q) ight| ightangle = frac{int_{0}^{infty} P(R) f^2(q,R) dR }{int_{0}^{infty} P(R) dR} ]
and
[ left| leftlangle f(q) ightangle ight| ^2 = left( frac{int_{0}^{infty} P(R) f(q,R) dR }{int_{0}^{infty} P(R) dR} ight)^2 ,]
where the Gaussian distribution of the radius is
[ P(R) propto exp left[ -frac{(R-ar{R})^2}{2(Delta R)^2} ight].]

The paracrystalline lattice factors $Z_1$ and $Z_2$ for a hexagonal lattice with lattice vectors a$_1$ and a$_2$ are
[ Z_k(q) = frac{1- left| F_k ight|^2}{1 - 2left| F_k ight| cos(mathbf{q cdot a_k}) + left| F_k ight|^2} ,]
where 
[ left| F_k ight| = exp left{ -frac{1}{2} left( Delta a/a ight)^2 left[ left( mathbf{q cdot a_1 } ight)^2 + left( mathbf{q cdot a_2 } ight)^2 ight] ight} ,]
[ mathbf{q cdot a_1} = -a q cos{(psi-frac{pi}{6})} ,]
[ mathbf{q cdot a_2 } = a q sin{psi} .]

The lattice factor $Z_k(q)$ has been modified according to Penttila et al, 2019:
[ Z_k =egin{cases}
Z_k(q_0), & 	ext{if $q leq  7.061 	imes 10^{-5} a^2 - 0.007413a + 0.2465$} \
Z_k(q) 	ext{ as in Hashimoto et al., 1994}, & 	ext{if $q> 7.061 	imes 10^{-5} a^2 - 0.007413a + 0.2465$}
end{cases}]

A detailed description of the model is given in reference Penttila et al., 2019.

For the model to work properly, the scaling parameter of SasView should be fixed to 1.0 and $da/a$ should be larger than 0. The output intensity is given in arbitrary units (not in cm$^{-1}$!).


References
---------------

Hashimoto, T., Kawamura, T., Harada, M., & Tanaka, H. (1994). Macromolecules, 27, 3063-3072. DOI: 10.1021/ma00089a025
Penttila, P. A., Rautkari, R., Osterberg, M., & Schweins, R. (2019). Journal of Applied Crystallography, 52. DOI: 10.1107/S1600576719002012


Authorship and Verification
---------------------------

* **Author:** Paavo Penttila **Date:** March 15, 2019

"""
from numpy import cos, sin, exp, square, divide, multiply, linspace, power, pi, zeros, inf
from scipy.special import j1
from scipy.integrate import trapz
from scipy.stats import norm

name = "woodsas"
title = "Model tailored for wood samples, based on hexagonally packed cylinders"
description = """
    I(q) = A_scale*I_cyl(q,radius,dR_ratio,a,da_ratio) + 
           B_scale*exp[-q^2/(2*sigma^2)] + 
           C_scale*q^(-alpha) + background

    I_cyl(q): Scattering in the plane perpendicular to the long axis of infinite 
              cylinders packed in a hexagonal lattice with paracrystalline distortion

    List of parameters:
    scale = 1.0 (do not change)
    background = Constant background
    A_scale = Scaling factor of cylinders
    radius = Mean cylinder radius
    dR_ratio = Polydispersity of cylinder radius
    a = Distance between cylinder axes in hexagonal lattice
    da_ratio = Relative paracrystallline lattice distortion
    B_scale = Scaling factor of Gaussian function
    sigma = Width parameter of Gaussian function
    C_scale = Scaling factor of power law
    alpha = Power law exponent

    Output in arbitrary units
"""

category = "shape:cylinder"

single = True

openCL = False

structure_factor = False

# pylint: disable=bad-whitespace, line-too-long
#   ["name",        "units", default, [min, max], "type",    "description"],
parameters = [
    ["A_scale", "arb. u.", 1, [0, inf], "", "Scaling constant for cylinders"],
    ["radius", "Ang", 13, [5, 15], "", "Mean cylinder radius"],
    ["dR_ratio", "None", 0.2, [0.1, 0.3], "", "dR/R"],
    ["a", "Ang", 40, [15, 50], "", "Distance a"],
    ["da_ratio", "None", 0.35, [0.1, 0.5], "", "da/a"],
    ["B_scale", "arb. u.", 0, [0, inf], "", "Gaussian scaling constant"],
    ["sigma", "1/Ang", 0.01, [0.01, 0.1], "", "Gaussian width"],
    ["C_scale", "arb. u.", 1e-8, [0, inf], "", "Power law scaling constant"],
    ["alpha", "None", 4, [3, 5], "", "Power law exponent"]
]
# pylint: enable=bad-whitespace, line-too-long

def calc_Zq( q, a, da_ratio, phi):
    # Eqs. 11, 12 in Penttila et al. (2019):
    qa1 = -a*q*cos(phi-pi/6.)
    qa2 = a*q*sin( phi )
    # Eq. 10 in Penttila et al. (2019):
    F_abs = exp( -0.5 * (da_ratio)**2 * ( square( qa1 ) + square( qa2 ) ))
    # Eq. 9 in Penttila et al. (2019):
    Z1 = divide( 1 - square(F_abs) ,  1 - 2*F_abs*cos(qa1) + square(F_abs) )
    Z2 = divide( 1 - square(F_abs) ,  1 - 2*F_abs*cos(qa2) + square(F_abs) )    
    return multiply(Z1,Z2)

def calc_f( q, radius ):
    qR = q*radius
    return pi*square(radius) * divide( j1(qR), qR )

def Iq(q, A_scale, radius, dR_ratio, a, da_ratio, B_scale, sigma, C_scale, alpha):
# q is a single value, dR_ratio=dR/R, da_ratio=da/a
# Based on Hashimoto et al. (1994) and Penttila et al. (2019)

    dR = dR_ratio*radius
    phi = linspace(0,2*pi,1001)
    # Eq. 13 in Penttila et al. (2019):
    if q > 7.061e-05*a**2 - 0.007413*a + 0.2465:# force ZZ to a finite value at low q
        ZZ = calc_Zq( q, a, da_ratio, phi)
    else:
        ZZ = calc_Zq( 7.061e-05*a**2 - 0.007413*a + 0.2465, a, da_ratio, phi)
    if dR == 0: # special case for monodisperse cylinder radius
        f_abs_sq_av = square( calc_f( q, radius ) )
        f_av_abs_sq = f_abs_sq_av
    else:
        # Gaussian distribution with 11 points between mean+-3*sigma, cut away negative values
        R_all = linspace(radius-3*dR, radius+3*dR, num = 11)
        R_all = R_all[R_all > 0]
        # Eq. 8 in Penttila et al. (2019):
        PR = norm.pdf(R_all, loc=radius, scale=dR)
        fqR_all = calc_f(q, R_all)        
        # Eqs. 6, 7 in Penttila et al. (2019)
        trapzPR = trapz( PR, R_all )
        f_abs_sq_av = trapz( PR * square( fqR_all ), R_all ) / trapzPR
        f_av_abs_sq  = square( trapz( PR * fqR_all, R_all ) / trapzPR )
    
    # Eq. 4 in Penttila et al. (2019):
    Iperp_q_phi =  f_abs_sq_av - f_av_abs_sq + f_av_abs_sq * ZZ

    # Intensity perpendicular to the cylinder axis (Eq. 3 in Penttila et al. (2019)):
    Iperp = A_scale/(2*pi)*trapz( Iperp_q_phi, phi )

    Iperp /= (pi*power(radius,4))# to scale approximately I(0)=1 (arbitrary units)
    Iperp += B_scale*norm.pdf(q, loc=0, scale=sigma)/norm.pdf(0, loc=0, scale=sigma)
    Iperp += C_scale * power(q, -alpha)
    return Iperp

Iq.vectorized = False

demo = dict(scale=1., background=0.05, A_scale=1., radius=13, 
      dR_ratio=0.2, a=40, a_ratio=0.35, B_scale=0.1, sigma=0.04,
      C_scale=1e-8, alpha=4.)

tests = [
    [{"scale": 1., "background": 0.05, "A_scale": 1., "radius": 13, 
      "dR_ratio": 0.2, "a": 40, "a_ratio": 0.35,
      "B_scale": 0.1, "sigma": 0.04,
      "C_scale": 1e-8, "alpha": 4.},0.08,0.341487],
]