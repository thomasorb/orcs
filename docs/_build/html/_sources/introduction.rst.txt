Introduction
============

.. contents::
   
A few words on ORCS
-------------------

ORCS is a Python module which gives you the keys to the analysis of
your SITELLE data cube.

It is composed of calibration methods, extracting tools and a powerful
fitting engine specifically designed for interferometric data. When
fitting spectra, both a classical (frequentist) and a Bayesian
paradigm are available.


The fitting engine
~~~~~~~~~~~~~~~~~~

ORCS ftting engine has been built to accept any number of models (even grid
models). Up to now only three models are available (and used by
default):

- Emission lines (sinc convoluted with a Gaussian giving a certain
  broadening to the sinc)

.. image:: images/emission-line-params.png
   :width: 70%
   :align: center
  
- Background (treated as a polynomial)
    
- Filter

Emission lines and background model parameters are always defined via
a list of keywords which must be passed to the fitting functions (see
`Jupyter Examples`_). Emission line model can be constrained in
velocity and broadening: multiple groups of lines can share the same
velocity and/or the same broadening. They can also be constrained in
amplitude ratio.

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


Models parameters
~~~~~~~~~~~~~~~~~

Each time you want to use a fitting method you must pass a list of
fitting parameters as keywords. The exact list of fitting parameters
depends on the models you are going to use. You only have to enter a
keyword if you want to change its default behaviour.

In general 3 models are used which are all implemented an documented
in the orb.fit module:

- py:class:`orb.fit.Cm1LinesModel`
  
- py:class:`orb.fit.ContinuumModel`

- py:class:`orb.fit.FilterModel`

Here's the list of the available keywords and their use.

  
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
