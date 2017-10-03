ORCS
####

**ORCS** (*Outils de Réduction de Cubes Spectraux*) is an analysis engine for SITELLE_ spectral cubes. With ORCS_ you can:

* extract integrated spectra
  
* fit the sinc emission lines
  
* recalibrate your data in wavelength, astrometry and flux
  
* choose between a Bayesian or a classical fitting algorithm

* and much more ...

**Examples** in Jupyter format can be found in the documentation:

http://celeste.phy.ulaval.ca/orcs-doc/examples.html
    
   
.. warning:: ORCS is always in fast development. You might see bugs or
 strange behaviours ! You might also want desperatly to have new
 features. In all cases please send an email:
 thomas.martin.1@ulaval.ca



Documentation
=============

You can find the up-to-date documentation here:

http://celeste.phy.ulaval.ca/orcs-doc/index.html




Publications
============

* Martin, Drissen, Melchior (2017). A SITELLE view of M31's central region - I: Calibrations and radial velocity catalogue of nearly 800 emission-line point-like sources. MNRAS (accepted)
  http://adsabs.harvard.edu/abs/2017arXiv170701366M


* Martin, Drissen (2017). SITELLE's level-1 calibration. MNRAS (submitted)
  http://adsabs.harvard.edu/abs/2017arXiv170603230M

* Martin, Prunet, Drissen (2016). Optimal fitting of Gaussian-apodized or under-resolved emission lines in Fourier transform spectra providing new insights on the velocity structure of NGC 6720. MNRAS
  http://adsabs.harvard.edu/abs/2016MNRAS.463.4223M


* Martin, Drissen, Joncas (2015). ORBS, ORCS, OACS, a Software Suite for Data Reduction and Analysis of the Hyperspectral Imagers SITELLE and SpIOMM. PASP
  http://adsabs.harvard.edu/abs/2015ASPC..495..327M
 

Installation
============

.. contents::

Install ORB
-----------
   
ORCS_ depends on ORB_ which must be installed first.

The archive and the installation instructions for ORB_ can be found on github::
  
  https://github.com/thomasorb/orb


Install / update ORCS
---------------------

With pip
~~~~~~~~

you just have to do (but don't forget to update ORB_ version as well !)::

  pip install orcs


Without pip
~~~~~~~~~~~

Then ORCS_ can be downloaded and installed from github also::
  
  https://github.com/thomasorb/orcs

Once the archive has been downloaded (from github just click on the
green button `clone or download` and click on `Download ZIP`) you may
extract it in a temporary folder. Then cd into the extracted folder
and type::

  python setup.py install



.. _ORB: https://github.com/thomasorb/orb
.. _ORCS: https://github.com/thomasorb/orcs
.. _SITELLE: http://www.cfht.hawaii.edu/Instruments/Sitelle
.. _CFHT: http://www.cfht.hawaii.edu/
.. _Python: http://www.python.org/
