# Introduction


**ORCS** (*Outils de Réduction de Cubes Spectraux*) is an analysis engine for [SITELLE](http://www.cfht.hawaii.edu/Instruments/Sitelle) spectral cubes. With [ORCS](https://github.com/thomasorb/orcs) you can:

* extract integrated spectra
  
* fit the sinc emission lines
  
* recalibrate your data in wavelength, astrometry and flux
  
* choose between a Bayesian or a classical fitting algorithm

* and much more ...

    
   
**warning** ORCS is always in fast development. You might see bugs or
 strange behaviours ! You might also want desperatly to have new
 features. In all cases please send an email:
 thomas.martin.1@ulaval.ca



## Documentation

You can find the up-to-date documentation here: https://orcs.readthedocs.io/


## Examples
**Examples** in Jupyter notebook format can be found in the documentation:

https://github.com/thomasorb/orcs/tree/master/docs/examples


### First basic examples

These examples show the basic usage of ORCS

* [Example of a single spectrum fit](docs/examples/fit_a_single_spectrum.ipynb)
* [How precise must be the input velocity parameter ?](docs/examples/velocity_parameter_precision.ipynb)
* [Extract the deep frame and use the WCS](docs/examples/deep_wcs.ipynb)
* [Make a fit over an entire region of the field](docs/examples/fit_region.ipynb)
* [Differences between fitting a sincgauss model and two sinc lines](docs/examples/sincgauss_vs_2_sinc.ipynb)
     
### Bayesian fitting vs. classical fitting

Here are more advanced examples that show step-by-step the differences between a classical fit and a Bayesian fit on a model spectrum. You will thus first learn how to model a spectrum and then fit a model spectrum with one line, two resolved lines and two unresolved lines: this is when the bayesian fitting algorithm becomes intersting ;)

* [Modelling and fitting a single line spectrum](docs/examples/model+fit_1_line.ipynb)
* [Modelling and fitting a spectrum with two resolved lines](docs/examples/model+fit_2_lines.ipynb)
* [Modelling and fitting two unresolved emission lines with a bayesian approach](docs/examples/model+fit_2_lines_bayes.ipynb)

### Calibrating your data

A data cube can be recalibrated using mode specific algorithm which depends on the type of data you have observed. You might want to give a try to these examples to see if you can get a better calibration for your data.

 
* [Wavelength recalibration with the sky lines (Mendel OH bands)](docs/examples/wavelength_calibration.ipynb)
* [Image registration](docs/examples/image_registration.ipynb)
* [Flux Calibration Example Using HST image](docs/examples/hst_flux_calibration.ipynb)

### Advanced fitting

These examples show advanced fitting procedures.

* [Constraining line ratios](docs/examples/constaining_line_ratios.ipynb)

### Other Tools
* [Automatic source detection](docs/examples/automatic_source_detection.ipynb)
* [Radial Velocity Correction](docs/examples/heliocentric_velocity.ipynb)


## Related Publications

* Martin, Drissen, Melchior (2017). A SITELLE view of M31's central region - I: Calibrations and radial velocity catalogue of nearly 800 emission-line point-like sources. MNRAS (accepted)
  http://adsabs.harvard.edu/abs/2017arXiv170701366M

* Martin, Drissen (2017). SITELLE's level-1 calibration. MNRAS (submitted)
  http://adsabs.harvard.edu/abs/2017arXiv170603230M

* Martin, Prunet, Drissen (2016). Optimal fitting of Gaussian-apodized or under-resolved emission lines in Fourier transform spectra providing new insights on the velocity structure of NGC 6720. MNRAS
  http://adsabs.harvard.edu/abs/2016MNRAS.463.4223M

* Martin, Drissen, Joncas (2015). ORBS, ORCS, OACS, a Software Suite for Data Reduction and Analysis of the Hyperspectral Imagers SITELLE and SpIOMM. PASP
  http://adsabs.harvard.edu/abs/2015ASPC..495..327M
 

## Installation


### 1. install [ORB](https://github.com/thomasorb/orb)

Follow the installation instructions [here](https://github.com/thomasorb/orb)

### 2. add orcs module

clone [ORCS](https://github.com/thomasorb/orcs)
```bash
mkdir orcs-stable # do it where you want to put orcs files
cd orcs-stable
git clone https://github.com/thomasorb/orcs.git
```

in the downloaded folder
```bash
conda activate orb # you don't need to do it if you are already in the orb environment
cd path/to/orcs-stable/orcs
python setup.py install
```

Test it:
```bash
conda activate orb # you don't need to do it if you are already in the orb environment
python -c 'import orcs.process'
```


