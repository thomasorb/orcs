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


Publications
------------

* Martin, Drissen, Melchior (2017). A SITELLE view of M31's central region - I: Calibrations and radial velocity catalogue of nearly 800 emission-line point-like sources. MNRAS (accepted)
  http://adsabs.harvard.edu/abs/2017arXiv170701366M


* Martin, Drissen (2017). SITELLE's level-1 calibration. MNRAS (submitted)
  http://adsabs.harvard.edu/abs/2017arXiv170603230M

* Martin, Prunet, Drissen (2016). Optimal fitting of Gaussian-apodized or under-resolved emission lines in Fourier transform spectra providing new insights on the velocity structure of NGC 6720. MNRAS
  http://adsabs.harvard.edu/abs/2016MNRAS.463.4223M


* Martin, Drissen, Joncas (2015). ORBS, ORCS, OACS, a Software Suite for Data Reduction and Analysis of the Hyperspectral Imagers SITELLE and SpIOMM. PASP
  http://adsabs.harvard.edu/abs/2015ASPC..495..327M
 

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
