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
