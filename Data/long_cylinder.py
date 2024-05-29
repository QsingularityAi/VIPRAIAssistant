r"""

Cylinder model for long cylinders.

Background
----------
The default numerical integration scheme in SasView leads to numerical instabilities
in the calculation of the cylinder form factor when the length of the cylinders
becomes too long. For practical purposes, the limit of applicability of the normal
cylinder model is when $length$ > 1000 Ang (which is not so long).

This Model
----------
This model computes the **same** cylinder form factor as the normal cylinder model,
but does so with a 501 point Gaussian integration scheme in place of the normal
76 point integration scheme. This leads to a vastly improved calculation which has
been tested for $length$ = 400000 Ang.

This model *also* requires the file gauss501.c 

Limitations
-----------
The cylinder form factor is not appropriate for so-called 'infinitely thin'
cylinders or rods. These would be better dealt with by models using appropriate
limiting forms of the normal form factor. Some discussion of this can be found in
our GitHub issues, for example, https://github.com/SasView/sasmodels/issues/109

Authorship and Verification
---------------------------

* **Author:** Paul Kienzle **Date:** 23/06/2020
* **Last Modified by:** Steve King **Date:** 24/06/2020 (added Description)
* **Last Reviewed by:** Wojciech Potrzebowski **Date:** 23/06/2020

"""
from pathlib import Path
from copy import copy
from sasmodels.core import load_model_info
from sasmodels.sasview_model import make_model_from_info
from sasmodels.gengauss import gengauss

ngauss = 501
gausspath = Path(__file__).parent / ("gauss%d.c" % ngauss)
if not gausspath.exists():
    gengauss(ngauss, gausspath)

model_info = copy(load_model_info('cylinder'))
model_info.source = [str(gausspath) if lib.startswith('lib/gauss') else lib
                     for lib in model_info.source]
model_info.id = model_info.name = 'long_cylinder'
model_info.filename = __file__
Model = make_model_from_info(model_info)
