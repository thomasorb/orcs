.. _examples:

Examples
========

First basic examples
--------------------

These examples show the basic usage of ORCS

.. toctree::
   :maxdepth: 1

   examples/fit_a_single_spectrum.ipynb
   examples/velocity_parameter_precision.ipynb
   examples/deep_wcs.ipynb
   examples/fit_region.ipynb
   examples/sincgauss_vs_2_sinc.ipynb

[test](examples/fit_a_single_spectrum.ipynb)


Bayesian fitting vs. classical fitting
--------------------------------------

Here are more advanced examples that show step-by-step the differences
between a classical fit and a Bayesian fit on a model spectrum. You
will thus first learn how to model a spectrum and then fit a model
spectrum with one line, two resolved lines and two unresolved lines:
this is when the bayesian fitting algorithm becomes intersting ;)

.. toctree::
   :maxdepth: 1

   examples/model+fit_1_line.ipynb
   examples/model+fit_2_lines.ipynb
   examples/model+fit_2_lines_bayes.ipynb

Calibrating your data
---------------------

A data cube can be recalibrated using mode specific algorithm which
depends on the type of data you have observed. You might want to give
a try to these examples to see if you can get a better calibration for
your data.

.. toctree::
   :maxdepth: 1

   examples/wavelength_calibration.ipynb
   examples/image_registration.ipynb
   examples/hst_flux_calibration.ipynb
   examples/use_calibration_outputs.ipynb

Advanced fitting
----------------

These examples show advanced fitting procedures for:
* constraining line ratios (e.g. [NII]6548,6584 or [OIII]5007,4959)
* fitting regions with mapped input parameters (e.g. velocity/broadening maps as input)

.. toctree::
   :maxdepth: 1

   examples/constaining_line_ratios.ipynb
   examples/fit_region_w_mapped_params.ipynb

Other Tools
-----------

.. toctree::
   :maxdepth: 1

   examples/automatic_source_detection.ipynb
   examples/heliocentric_velocity.ipynb
