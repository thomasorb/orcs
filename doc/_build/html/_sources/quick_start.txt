Quick Start Guide
#################

.. contents::


A few words on ORCS
-------------------

ORCS is a fitting engine for SpIOMM and SITELLE's data. It is designed
to fit each spectrum with a model in a parallel. The core fitting
procedure relies on a classical least-square Levenberg-Marquardt
algorithm. Which means that the statistical paradigm of ORCS is
frequentist.

ORCS has been built to accept any number of models (even grid
models). Up to now only three models are available (and used by
default):

- Emission lines (sinc convoluted with a Gaussian giving a certain
  broadening to the sinc)

.. image:: images/emission-line-params.png
   :width: 70%
   :align: center
  
- Background (treated as a polynomial)
    
- Filter

Emission lines and background model parameters are defined in an
'option file' which is described below. Emission line model can be
constrained in velocity and broadening: multiple groups of lines can
share the same velocity and/or the same broadening.

.. image:: images/sky-spectrum.png
   :width: 100%
   :align: center


The uncertainties on the returned parameters are based on the
assumption that noise distribution is Gaussian and that there are not
correlated. I have checked those assumptions by analyzing the
distribution of the posterior probability on each parameter with a
Monte-Carlo-Markov-Chain algorithm and found that they are very
reasonable. The uncertainties returned by the MCMC algorithm are also
very close to the one returned by our algorithm (less than a few
percents).


   
Step 1: Create your option file
-------------------------------

Like ORBS, ORCS uses option files to load all the parameters. Here's
en example.

.. literalinclude:: option_file.orc
   :language: python

**REQUIRED** parameters
~~~~~~~~~~~~~~~~~~~~~~~

- ``CUBEPATH``: Path to the hdf5 cube spectral cube.

- ``LINES``: Emission Lines names. Enter an integer or a float to use
  a non_recorded emission line (line wavelength must be given in
  nm). See :ref:`list-lines` to get a list of the available lines. Sky
  lines can be added with SKY (be careful with fitting skylines, it's
  incredibly longer and generally not useful as they can be simply
  removed). The same line can appear multiple time if ha a different
  velocity (but it must be assigned a different velocity group with
  the keyword ``COV_SIGMA``)

- ``OBJECT_VELOCITY``: Mean object velocity in km.s-1. This parameter
  is very important and must be known precisely. If the object have a
  large velocity range, use velocity_range option.

- ``OBJ_REG``: Path to a ds9 region path defining the regions of the
  cube which have to be fitted (region parameters must be recorded in
  pixel coordinates, region shape can be a circle, a box or a
  polygon). Multiple regions can be defined at the same time. To
  remove a particular region its line can be commented (with a '#') in
  the region file.

- ``POLY_ORDER``: Order of the polynomial used to fit continuum (avoid
  high order polynomials : 0, 1 or 2 are generally enough, to remove
  the sky it is better to use the keyword ``SKY_REG``)


Optional parameters
~~~~~~~~~~~~~~~~~~~

- ``COV_LINES``: Lines velocity groups. Give the same number to the
  lines with the same expected velocity (e.g. same atom). **If nothing
  is given all the lines are considered to have the same velocity**.

- ``COV_SIGMA``: Lines broadening groups. Give the same number to the
  lines with the same expected broadening (e.g. same atom). **If nothing
  is given all the lines are considered to have the same broadening**.

- ``VELOCITY_RANGE``: A brute force procedure can be run on each
  spectrum to guess the real velocity in a certain range around the
  object mean velocity. Works very well on spectrum with a good SNR (>
  10). But may be wrong on low SNR spectra. This function is best used
  with binned fit (see ``BINNING`` option below).

  Note that if the velocity has already been computed with a higher
  binning, this value will be used as an initial guess and the
  velocity range won't be taken into account. This way, using the
  velocity range with a 10x10 binning is the best way to recover the
  general velocity profile of a galaxy and use it as an initial guess
  fot a smaller binning fit.
    
- ``SKY_REG``: Path to a ds9 region path defining the region of the
  cube which is integrated to create the sky spectrum (regions
  parameters must be recorded in pixel coordinates, region shape can
  be a circle, a box or a polygon). The sky spectrum will be removed
  from the fitted spectra.

- ``BINNING``: Data can be binned 'on the fly' before being
  fitted. This option can be used to enhance the SNR of the
  spectra. Created binned maps are used by ORCS to fit a cube with a
  smaller binning. For example it can be more robust to start the
  first fit with a high binning value (e.g. 10x10) and lower the
  binning down to 1x1 (e.g. 10x10 then 6x6 then 3x3 then 1x1). This
  way the last computed parameters (especially the velocity) will be
  used in the next fit: 10x10 parameters will be used by the 6x6 fit,
  then 6x6 parameters will be used by the 3x3 fit etc. *This fitting
  procedure is especially useful for objects with a large velocity
  range (e.g. nearly edge-on galaxies)*.

.. _list-lines:
	
List of available lines
~~~~~~~~~~~~~~~~~~~~~~~

============ ================
    NAME       Air Wavelength
============ ================
[OII]3726    372.603
[OII]3729    372.882
[NeIII]3869  386.875
Hepsilon     397.007
Hdelta       410.176
Hgamma       434.047
[OIII]4363   436.321
Hbeta        486.133
[OIII]4959   495.891
[OIII]5007   500.684
HeI5876      587.567
[OI]6300     630.030
[SIII]6312   631.21
[NII]6548    654.803
Halpha       656.280
[NII]6583    658.341
HeI6678      667.815
[SII]6716    671.647
[SII]6731    673.085
HeI7065      706.528
[ArIII]7136  713.578
[OII]7120    731.965
[OII]7130    733.016
[ArIII]7751  775.112
============ ================

Step 2: Sanity Checks
---------------------

Before running a 12 hours fitting procedure it can be safe to check if
the fitting parameters are correct. You can try a fit on a spectrum
integrated over a small circular aperture with the option: **check**.

.. code-block:: console

  orcs sitelle option_file.orc check X Y R

X, Y being respectively the center of the region (in pixels) along X
and Y and R being the radius of the region (in pixels).

This command will output a plot of the spectrum and its fit. Use it
first over a high SNR region. If the fit does not work maybe the
velocity is not well defined. Try the option ``VELOCITY_RANGE`` to
give a larger fit range and look for the parameter ``v`` in the terminal output.

Another sanity check can be done by defining a small region in the
region file (OBJ_REG) and testing the fitting procedure (see step 3).

Finally, the ``BINNING`` option can be used to try a fit over a binned
cube (e.g. 3x3) which reduces a lot the fitting time (approximately by
the square of the binning).

Step 3: Run it
--------------

.. code-block:: console

  orcs sitelle option_file.orc start


Step 4: Check the results
-------------------------

Maps of the parameters of the fit can be found in the directory
created by ORCS: ``OBJECT_NAME_FILTER.ORCS/MAPS/``.

Each line has 5 parameters (which gives 5 maps): height, amplitude,
velocity, fwhm, sigma. Height and amplitude are given in
ergs/cm^2/s/A. Velocity and broadening are given in km/s. FWHM is
given in cm^-1.

The flux map is also computed (from fwhm, amplitude and sigma
parameters) and given in ergs/cm^2/s.

Each fitted parameter is associated an uncertainty (*_err maps) given
in the same unit.
