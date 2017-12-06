.. _examples:

Examples
========

First basic examples
--------------------

These examples show the basic usage of ORCS

.. toctree::
   :maxdepth: 1
	      
   script_example_fit_a_single_spectrum.ipynb
   script_example_velocity_parameter_precision.ipynb
   script_example_deep_wcs.ipynb
   script_example_fit_region.ipynb
   

Bayesian fitting vs. classical fitting
--------------------------------------

Here are more advanced examples that show step-by-step the differences
between a classical fit and a Bayesian fit on a model spectrum. You
will thus first learn how to model a spectrum and then fit a model
spectrum with one line, two resolved lines and two unresolved lines:
this is when the bayesian fitting algorithm becomes intersting ;)

.. toctree::
   :maxdepth: 1
	      
   script_example_model+fit_1_line.ipynb
   script_example_model+fit_2_lines.ipynb
   script_example_model+fit_2_lines_bayes.ipynb

Calibrating your data
---------------------

A data cube can be recalibrated using mode specific algorithm which
depends on the type of data you have observed. You might want to give
a try to these examples to see if you can get a better calibration for
your data.

.. toctree::
   :maxdepth: 1

   script_example_wavelength_calibration.ipynb
   script_example_image_registration.ipynb
   script_example_hst_flux_calibration.ipynb

Advanced fitting
----------------

These examples show advanced fitting procedures.

.. toctree::
   :maxdepth: 1

   script_example_constaining_line_ratios.ipynb

Other Tools
-----------

.. toctree::
   :maxdepth: 1
	      
   script_example_automatic_source_detection.ipynb
   script_example_heliocentric_velocity.ipynb
