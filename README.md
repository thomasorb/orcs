# ORCS


**ORCS** (*Outils de Réduction de Cubes Spectraux*) is an analysis engine for [SITELLE](http://www.cfht.hawaii.edu/Instruments/Sitelle) spectral cubes. With [ORCS](https://github.com/thomasorb/orcs) you can:

* extract integrated spectra
  
* fit the sinc emission lines
  
* recalibrate your data in wavelength, astrometry and flux
  
* choose between a Bayesian or a classical fitting algorithm

* and much more ...

**Examples** in Jupyter format can be found in the documentation:

https://orcs.readthedocs.io/en/latest/examples.html
    
   
**Warning:** ORCS is always in fast development. You might see bugs or
 strange behaviours ! You might also want desperatly to have new
 features. In all cases please send an email:
 thomas.martin.1@ulaval.ca



## Documentation


You can find the up-to-date documentation here:

https://orcs.readthedocs.io/en/latest/index.html


## Installation


### Install ORB

   
ORCS depends on [ORB](https://github.com/thomasorb/orb) which must be installed first.

The archive and the installation instructions for ORB can be found on github::
  
  https://github.com/thomasorb/orb


### Install / update ORCS


[ORCS](https://github.com/thomasorb/orcs) can be downloaded and installed from github also::
  
  https://github.com/thomasorb/orcs
  
Following the instructions given to install ORB you can do the following to install ORCS.

clone [ORCS](https://github.com/thomasorb/orcs)
```bash
mkdir orcs-stable # do it where you want to put orb files
cd orcs-stable
git clone https://github.com/thomasorb/orcs.git
```

in the downloaded folder
```bash
conda activate orb3 # you don't need to do it if you are already in the orb3 environment
cd path/to/orcs-stable
python setup.py install
```

Test it:
```bash
conda activate orb3 # you don't need to do it if you are already in the orb3 environment
python -c 'import orcs.core'
```

## Troubleshooting

### Check the version of the packages

- [Here](docs/orb3-ver.txt) is a list of all the versions of the packages on a working installation ([docs/orbs3/ver.txt](docs/orb3-ver.txt)). Higher versions generally work but not in some cases.

- [Here](docs/orb3-env.txt) is an environment file that can be used directly with conda to install the correct versions of the packages ([docs/orbs3/env.txt](docs/orb3-env.txt)). 

## Publications


* Martin, Drissen, Melchior (2017). A SITELLE view of M31's central region - I: Calibrations and radial velocity catalogue of nearly 800 emission-line point-like sources. MNRAS (accepted)
  http://adsabs.harvard.edu/abs/2017arXiv170701366M


* Martin, Drissen (2017). SITELLE's level-1 calibration. MNRAS (submitted)
  http://adsabs.harvard.edu/abs/2017arXiv170603230M

* Martin, Prunet, Drissen (2016). Optimal fitting of Gaussian-apodized or under-resolved emission lines in Fourier transform spectra providing new insights on the velocity structure of NGC 6720. MNRAS
  http://adsabs.harvard.edu/abs/2016MNRAS.463.4223M


* Martin, Drissen, Joncas (2015). ORBS, ORCS, OACS, a Software Suite for Data Reduction and Analysis of the Hyperspectral Imagers SITELLE and SpIOMM. PASP
  http://adsabs.harvard.edu/abs/2015ASPC..495..327M

