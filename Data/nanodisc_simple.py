r"""

Definition
----------
This model provides the form factor for a circular cylinder with a
core-shell scattering length density profile. Thus this is a variation
of a core-shell cylinder or disc where the shell on the walls and ends
may be of different thicknesses and scattering length densities. The form
factor is normalized by the particle volume.

.. _core-shell-bicelle-geometry:

.. figure:: img/core_shell_bicelle_geometry.png

    (Graphic from DOI: 10.1039/C0NP00002G, note however that the model here
    calculates for rectangular, not curved, rims.)

The output of the 1D scattering intensity function for randomly oriented
cylinders is then given by the equation above.

The *theta* and *phi* parameters are not used for the 1D output.
Our implementation of the scattering kernel and the 1D scattering intensity
use the c-library from NIST.

.. figure:: img/cylinder_angle_definition.jpg

    Definition of the angles for the oriented core shell bicelle tmodel.

.. figure:: img/cylinder_angle_projection.jpg

    Examples of the angles for oriented pp against the detector plane.

References
----------

L A Feigin and D I Svergun,
*Structure Analysis by Small-Angle X-Ray and Neutron Scattering,*
Plenum Press, New York, (1987)

"""

import os
import sys
import copy
import collections

from ctypes import c_int, WINFUNCTYPE, windll
from ctypes.wintypes import HWND, LPCSTR, UINT
prototype = WINFUNCTYPE(c_int, HWND, LPCSTR, LPCSTR, UINT)
paramflags = (1, "hwnd", 0), (1, "text", "Hi"), (1, "caption", None), (1, "flags", 0)
MessageBox = prototype(("MessageBoxA", windll.user32), paramflags)

import numpy

from sas.sascalc.fit.pluginmodel import Model1DPlugin
from sasmodels.sasview_model import find_model

class Model(Model1DPlugin):
    name = os.path.splitext(os.path.basename(__file__))[0]
    is_multiplicity_model = False
    def __init__(self, multiplicity=1):
        Model1DPlugin.__init__(self, name='')
        
        # Load the core_shell_bicelle model that is included with SasView
        P1 = find_model('core_shell_bicelle')
        p_model1 = P1()

        # List of parameters that are derived from the fit parameters (i.e. they
        # are not direct fit parameters, but are a function of fit parameters).
        # These are taken as input to the built-in core_shell_bicelle model.
        self.derived_params = ['p1_sld_core', 'p1_sld_face', 'p1_sld_rim', 'p1_sld_solvent']

        ## Setting  model name model description
        self.description = 'The core_shell_bicelle model with SLDs derived from molecular parameters'
        self.name = 'nanodisc_simple'
        
        ## Define parameters
        self.params = collections.OrderedDict()

        ## Parameter details [units, min, max]
        self.details = {}

        ## Magnetic Panrameters
        self.magnetic_params = []

        # non-fittable parameters
        self.non_fittable = p_model1.non_fittable

        ##models
        self.p_model1=p_model1

        ## Polydispersity
        self._set_dispersion()

        ## Define parameters
        self._set_params()

        # These are initialized to zero but true initial values will be calculated later
        # based on molecular parameters, e.g. number of lipids of a given scattering
        # length in a given layer volume defines the core SLD, etc.
        self.sld_core = 0
        self.sld_face = 0
        self.sld_rim = 0
        self.sld_solvent = 0
        self.b_water = 0

        # These are the sum of scattering lenghts of the specified molecule. Scattering
        # lengths are for 6 Angstrom neutrons and should be modified for x-ray, etc.
        # The numbers below are for DMPC lipids and MSP1D1-like belt proteins.
        self.params['b_tail'] = -283.36
        self.params['b_head'] = 602.13
        self.params['b_msp_exch_H'] = 57795.66
        self.params['b_msp_exch_D'] = 102145.04
        self.params['b_H2O'] = -16.47
        self.params['b_D2O'] = 190.77
        
        # These are ballpark initial guesses.
        self.params['n_wat_per_hg'] = 1.0
        self.params['n_lipids'] = 150
        self.params['n_wat_per_MSP'] = 200
        self.params['n_MSPs'] = 2.0
        
        # Change according to the experiment. Will be combined with b_H2O and b_D2O to
        # calculate the SLD of the solvent. For x-ray, b_H2O and b_D2O are almost the
        # same so this won't matter.
        self.params['frac_D2O'] = 1.0

        ## Set approximate geometric parameters for nanodiscs
        self.setParam('p1_radius', 30)
        self.setParam('p1_thick_rim', 10)
        self.setParam('p1_thick_face', 7)
        self.setParam('p1_length', 26)

        ## Parameter details [units, min, max]
        self._set_details()

        self.details['b_tail'] = ['1e-6 Ang', -numpy.inf, numpy.inf]
        self.details['b_head'] = ['1e-6 Ang', -numpy.inf, numpy.inf]
        self.details['b_msp_exch_H'] = ['1e-6 Ang', -numpy.inf, numpy.inf]
        self.details['b_msp_exch_D'] = ['1e-6 Ang', -numpy.inf, numpy.inf]
        self.details['b_H2O'] = ['1e-6 Ang', -numpy.inf, numpy.inf]
        self.details['b_D2O'] = ['1e-6 Ang', -numpy.inf, numpy.inf]
        self.details['n_wat_per_hg'] = ['', 0, numpy.inf]
        self.details['n_lipids'] = ['', 0, numpy.inf]
        self.details['n_wat_per_MSP'] = ['', 0, numpy.inf]
        self.details['n_MSPs'] = ['', 0, numpy.inf]
        self.details['frac_D2O'] = ['', 0, 1]

        # This calculates all the SLDs given the specified molecular quantities (e.g.
        # number of DMPC molecules) and geometry of the disc. Basically, it just does:
        # SLD = (sum of scattering lengths of all atoms in the layer) / (layer volume)
        self._setDerivedParams()

        #list of parameter that can be fitted
        self._set_fixed_params()

        ## parameters with orientation
        self.orientation_params = []
        for item in self.p_model1.orientation_params:
            new_item = "p1_" + item
            if not new_item in self.orientation_params:
                self.orientation_params.append(new_item)

        ## magnetic params
        self.magnetic_params = []
        for item in self.p_model1.magnetic_params:
            new_item = "p1_" + item
            if not new_item in self.magnetic_params:
                self.magnetic_params.append(new_item)

        # get multiplicity if model provide it, else 1.
        try:
            multiplicity1 = p_model1.multiplicity
        except:
            multiplicity1 = 1

        ## functional multiplicity of the model
        self.multiplicity1 = multiplicity1
        self.multiplicity_info = []

    def _clone(self, obj):
        obj.params     = copy.deepcopy(self.params)
        obj.description     = copy.deepcopy(self.description)
        obj.details    = copy.deepcopy(self.details)
        obj.dispersion = copy.deepcopy(self.dispersion)
        obj.p_model1  = self.p_model1.clone()
        obj.sld_core = copy.deepcopy(self.sld_core)
        obj.sld_face = copy.deepcopy(self.sld_face)
        obj.sld_rim = copy.deepcopy(self.sld_rim)
        obj.sld_solvent = copy.deepcopy(self.sld_solvent)
        obj.b_water = copy.deepcopy(self.b_water)
        obj.derived_params = copy.deepcopy(self.derived_params)

        #obj = copy.deepcopy(self)
        return obj

    def _get_name(self, name1, name2):
        p1_name = self._get_upper_name(name1)
        if not p1_name:
            p1_name = name1
        name = p1_name
        name += "_and_"
        p2_name = self._get_upper_name(name2)
        if not p2_name:
            p2_name = name2
        name += p2_name
        return name

    def _get_upper_name(self, name=None):
        if name == None:
            return ""
        upper_name = ""
        str_name = str(name)
        for index in range(len(str_name)):
            if str_name[index].isupper():
                upper_name += str_name[index]
        return upper_name

    def _set_dispersion(self):
        self.dispersion = collections.OrderedDict()
        ##set dispersion only from p_model
        for name , value in self.p_model1.dispersion.iteritems():
            #if name.lower() not in self.p_model1.orientation_params:
            new_name = "p1_" + name
            self.dispersion[new_name]= value

    def function(self, x=0.0):
        return 0

    def getProfile(self):
        try:
            x,y = self.p_model1.getProfile()
        except:
            x = None
            y = None

        return x, y

    def _set_params(self):
        # params_text = ''
        for name , value in self.p_model1.params.iteritems():
            new_name = "p1_" + name
            # params_text += new_name + ': ' + str(value) + '
'
            if new_name not in self.derived_params:
                self.params[new_name]= value
        # MessageBox(text=params_text)

    def _set_details(self):
        for name ,detail in self.p_model1.details.iteritems():
            new_name = "p1_" + name
            if new_name not in self.derived_params:
                self.details[new_name]= detail

    def _set_BackGround(self):
        pass

    # This function will be called to update the SLDs any time any
    # of the other parameters are changed.
    def _setDerivedParams(self):
        
        # Retrieve all the molecular/geometric parameters
        frac_D2O = self.params['frac_D2O']
        b_D2O = self.params['b_D2O']
        b_H2O = self.params['b_H2O']
        n_MSPs = self.params['n_MSPs']
        b_msp_exch_D = self.params['b_msp_exch_D']
        b_msp_exch_H = self.params['b_msp_exch_H']
        n_wat_per_MSP = self.params['n_wat_per_MSP']
        p1_length = self.params['p1_length']
        p1_thick_face = self.params['p1_thick_face']
        p1_radius = self.params['p1_radius']
        p1_thick_rim = self.params['p1_thick_rim']
        n_lipids = self.params['n_lipids']
        b_tail = self.params['b_tail']
        b_head = self.params['b_head']
        n_wat_per_hg = self.params['n_wat_per_hg']

        # Do the SLD calculations
        b_water = frac_D2O * b_D2O + (1.0 - frac_D2O) * b_H2O
        sld_solvent = b_water / 30.0
        sld_core = n_lipids * b_tail / (numpy.pi * (p1_radius ** 2) * p1_length)
        sld_face = n_lipids * (b_head + n_wat_per_hg * b_water) / (2.0 * numpy.pi * (p1_radius ** 2) * p1_thick_face)
        sld_rim = n_MSPs 
                     * (frac_D2O * b_msp_exch_D + (1.0 - frac_D2O) * b_msp_exch_H + n_wat_per_MSP * b_water) 
                     / (numpy.pi * (p1_length + 2.0 * p1_thick_face) * ((p1_radius + p1_thick_rim) ** 2 - p1_radius ** 2))


        # Update the object's SLD parameters with the calculated values
        self.b_water = b_water
        self.sld_solvent = sld_solvent
        self.sld_core = sld_core
        self.sld_face = sld_face
        self.sld_rim = sld_rim

        # Update the core_shell_bicelle model object with the new SLD parameters
        self.p_model1.setParam('sld_solvent', sld_solvent)
        self.p_model1.setParam('sld_core', sld_core)
        self.p_model1.setParam('sld_face', sld_face)
        self.p_model1.setParam('sld_rim', sld_rim)

    def setParam(self, name, value):
        # MessageBox(text=name + ' ' + str(value))
        # set param to this (p1, p2) model

        self._setParamHelper(name, value)
        self._setDerivedParams()

        model_pre = ''
        new_name = ''
        name_split = name.split('_', 1)
        if len(name_split) == 2:
            model_pre = name.split('_', 1)[0]
            new_name = name.split('_', 1)[1]
        if model_pre == "p1":
            if new_name in self.p_model1.getParamList():
                self.p_model1.setParam(new_name, value)

    def getParam(self, name):
        # Look for dispersion parameters
        toks = name.split('.')
        if len(toks)==2:
            for item in self.dispersion.keys():
                # 2D not supported
                if item.lower()==toks[0].lower():
                    for par in self.dispersion[item]:
                        if par.lower() == toks[1].lower():
                            return self.dispersion[item][par]
        else:
            # Look for standard parameter
            for item in self.params.keys():
                if item.lower()==name.lower():
                    return self.params[item]
        return
        #raise ValueError, "Model does not contain parameter %s" % name

    def _setParamHelper(self, name, value):
        # Look for dispersion parameters
        toks = name.split('.')
        if len(toks)== 2:
            for item in self.dispersion.keys():
                if item.lower()== toks[0].lower():
                    for par in self.dispersion[item]:
                        if par.lower() == toks[1].lower():
                            self.dispersion[item][par] = value
                            return
        else:
            # Look for standard parameter
            for item in self.params.keys():
                if item.lower()== name.lower():
                    self.params[item] = value
                    # MessageBox(text=name + ' ' + str(value))
                    return

        raise ValueError, "Model does not contain parameter %s" % name

    def _set_fixed_params(self):
        self.fixed = []
        for item in self.p_model1.fixed:
            new_item = "p1" + item
            self.fixed.append(new_item)
        self.fixed.sort()

    def run(self, x = 0.0):
        self._set_BackGround()
        return (self.p_model1.run(x))

    def runXY(self, x = 0.0):
        self._set_BackGround()
        return (self.p_model1.runXY(x))
    ## Now (May27,10) directly uses the model eval function
    ## instead of the for-loop in Base Component.
    def evalDistribution(self, x = []):
        self._set_BackGround()
        return (self.p_model1.evalDistribution(x))
    def set_dispersion(self, parameter, dispersion):
        value= None
        new_pre = parameter.split("_", 1)[0]
        new_parameter = parameter.split("_", 1)[1]
        try:
            if new_pre == 'p1' and new_parameter in self.p_model1.dispersion.keys():
                value= self.p_model1.set_dispersion(new_parameter, dispersion)
            self._set_dispersion()
            return value
        except:
            raise
    def fill_description(self, p_model1, p_model2):
        description = ""
        description += "This model gives the summation or multiplication of"
        description += "%s and %s. "% ( p_model1.name, p_model2.name )
        self.description += description

if __name__ == "__main__":
    m1= Model()
    out1 = m1.runXY(0.01)

    m2= Model()
    out2 = m2.p_model1.runXY(0.01)

    print "My name is %s."% m1.name
    print out1, " = ", out2
    if out1 == out2:
        print "===> Simple Test: Passed!"
    else:
        print "===> Simple Test: Failed!"

