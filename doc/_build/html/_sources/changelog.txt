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

v0.9.4
------

* ORCS now handles HDF5 cubes only (the last ORBS reduction step must
  thus be redone)

* keyword DIRSPEC has been changed to CUBEPATH 

* bug fix: a lorentzian function was added to a sinc function since
  v0.9.2 resulting in poorer fit and a bad SNR (roughly divided by
  2). This is no more the case.


v0.10 New fit class
*******************

A new fit module has been designed in ORB (orb/fit.py). ORCS has been
changed to use this new module.

Most of the changes are related to this major upgrade.

v0.10.0
-------

New operations
~~~~~~~~~~~~~~

* new operation in orcs command: **'check'**, to fit the integrated
  spectrum of one region. The idea is to select a small region with
  high SNR to test if the fitting procedure works as expected before
  launching a long fit process on a larger region.

* new operation in orcs command: **'skymap'** : fit the sky velocity
  on a high number of points and returns an interpolated map of the
  velocity zero point.


Binning
~~~~~~~

Binning before a fit is now possible. The cube does not have to be
binned, ORCS does it on the fly. Created binned maps are used by ORCS
to fit a cube with a smaller binning. For example it can be more
robust to start the first fit with a high binning value (e.g. 10x10)
and lower the binning down to 1x1 (e.g. 10x10 then 6x6 then 3x3 then
1x1). This way the last computed parameters (especially the velocity)
will be used in the next fit: 10x10 parameters will be used by the 6x6
fit, then 6x6 parameters will be used by the 3x3 fit etc.


Velocity range
~~~~~~~~~~~~~~

A brute force procedure can be run on each spectrum to guess the real
velocity in a certain range around the object mean velocity. Works
very well on spectrum with a good SNR (> 10). But may be wrong on low
SNR spectra. This function is best used with binned fit.

Note that if the velocity has already been computed with a higher
binning, this value will be used as an initial guess and the velocity
range won't be taken into account. This way, using the velocity range
with a 10x10 binning is the best way to recover the general velocity
profile of a galaxy and use it as an initial guess fot a smaller
binning fit.

The guess of a higher binning fit wil be used only if its uncertainty
is small enough (around 1/5 of the resolution)


Multiple fit
~~~~~~~~~~~~

Multiple fits can be done. The newly fitted pixels will replace the
previous ones and the other pixels will be kept. This way multiple
regions with different velocities can be fitted one after the other.

Miscellaneous
~~~~~~~~~~~~~

* ORCS uses the internal calibration map of the HDF5 cube by default.
  If it exists CALIB_MAP keyword is not used, so in general this
  keyword can be removed if the cube has been exported recently.

* SNR and CHI2 maps have been removed

* FLUX maps are computed and written in output
