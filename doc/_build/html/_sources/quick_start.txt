Quick start Guide
#################

.. contents::

Step 1: Create your option file
-------------------------------

Like ORBS, ORCS uses option files to load all the parameters
needed. Here's en example.

.. literalinclude:: option_file.orc
   :language: python

Basic parameters
~~~~~~~~~~~~~~~~

:option:`CUBE_PATH` Path to the spectral cube

:option:`LINES` Emission Lines names, enter an integer or a float to
use a non_recorded emission line. See :py:class:`~core.ORCSTools` to
get a list of the available lines.

:option:`SHIFT` Spectral shift of the lines [in nm]

:option:`FWHM` Rough FWHM of the lines [in nm]

:option:`FILTER_EDGES` Edges of the filter filter_min,filter_max [in nm]

:option:`POLY_ORDER` Order of the polynomial used to fit continuum

:option:`ROI` Region of interest : Xmin,Xmax,Ymin,Ymax. Must be integers.

Heliocentric velocity correction parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:option:`OBS_LAT` Observatory latitude

:option:`OBS_LONG` Observatory longitude

:option:`OBS_ALT` Observatory altitude

:option:`DATE` Observation date

:option:`HOUR_UT` Observation hour

Step 2: Run it
--------------

.. warning:: Don't forget to put the correct path to ORBS in orcs/data.py

To run the option you must create a python script like this one:

.. code-block:: python
  :linenos:

  # orcs_script.py

  import sys, os
  sys.path.append(os.path.expanduser("/path/to/Orcs")) # set Orcs Path

  from orcs.orcs import * # import orcs module

  sc = SpectralCube("option_file.orc") # Init extracting class
  sc.extract_lines_maps() # start map extraction

You can then run this script with the command:

.. code-block:: console

  python orcs_script.py
