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

* Creation of the executable script **orcs** which runs the extraction
  process directly

.. note:: Up to v1.0 some secondaries functionalities have been
   deactivated because they have to be rewritten since the
   architecture change. They will be added on future releases.

v0.9.1
------

Recoded methods:
~~~~~~~~~~~~~~~~

* py:meth:`orcs.orcs.SpectralCube.extract_raw_lines_maps` and its
  wrapper py:meth:`orcs.orcs.Orcs.extract_raw_lines_maps`.


bug fix:
~~~~~~~~

* Conversion of the fwhm from cm-1/nm to pixels before fit

v0.9.2
------

bug fix: Treat cubes without astrometrical calibration


v0.9.3
------

**scripts/orcs**: Command line call changed in order to switch from
the deprecated getopt module to the argparse module and make it
similar to the **orbs** script of ORBS.

bug fix: WCS headers of the original cube is transfered to the output
maps.
