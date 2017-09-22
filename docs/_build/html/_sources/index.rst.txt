.. Orbs documentation master file, created by
   sphinx-quickstart on Sat May 26 01:02:35 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


ORCS Documentation
##################


.. image:: logo.png
   :width: 35%
   :align: center

.. topic:: Welcome to ORCS documentation !

.. contents::


Features
--------

	   
**ORCS** (*Outils de Réduction de Cubes Spectraux*) is an analysis engine for SITELLE_ spectral cubes. With ORCS_ you can:

* extract integrated spectra
  
* fit the sinc emission lines
  
* recalibrate your data in wavelength, astrometry and flux
  
* choose between a Bayesian or a classical fitting algorithm

* and much more ...

**Examples** in Jupyter format can be found in :ref:`examples`.
    
   
.. warning:: ORCS is always in fast development. You might see bugs or
 strange behaviours ! You might also want desperatly to have new
 features. In all cases please send an email:
 thomas.martin.1@ulaval.ca



Quick Start
-----------

.. toctree::
   :maxdepth: 1

   installation
   introduction
   examples

Documentation
-------------

.. toctree::
   :maxdepth: 1

   core_module
	   
   process_module

   rvcorrect_module

   utils_module



Known issues
------------

.. toctree::
   :maxdepth: 1

   known_issues

   
Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _CFHT: http://www.cfht.hawaii.edu/
.. _Python: http://www.python.org/
.. _ORB: https://github.com/thomasorb/orb
.. _ORCS: https://github.com/thomasorb/orcs
.. _SITELLE: http://www.cfht.hawaii.edu/Instruments/Sitelle
