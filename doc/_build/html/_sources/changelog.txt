Changelog
#########

	
v0.9 ORCS integration to ORB suite
**********************************

Major architecture changes to respect ORBS concept.

* creation of :py:class:`orcs.orcs.Orcs` as a user-interface

* :py:class:`orcs.orcs.SpectralCube` becomes a processing class based on
  :py:class:`orb.core.Cube`.

* wavenumber / wavelength and calibrated / uncalibrated
  extraction. The best extraction is thus based on a wavenumber
  uncalibrated spectrum cube which spectra have never been
  interpolated so their spectral information has not been
  corrupted. The 3 other configurations are also possible.

* Modification of the option file keywords

* Creation of the executable script **orcs** which run the extraction
  process directly

.. note:: Up to the v1.0 some secondaries functionalities have been
   deactivated for they have to be rewritten because of the
   architecture change . We will add them on the next release.
