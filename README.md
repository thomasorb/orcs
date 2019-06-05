# Introduction


**ORCS** (*Outils de Réduction de Cubes Spectraux*) is an analysis engine for [SITELLE](http://www.cfht.hawaii.edu/Instruments/Sitelle) spectral cubes. With [ORCS](https://github.com/thomasorb/orcs) you can:

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



## Documentation

You can find the up-to-date documentation here:

http://celeste.phy.ulaval.ca/orcs-doc/index.html



## Publications

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


