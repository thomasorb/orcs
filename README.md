# ORCS


**ORCS** (*Outils de Réduction de Cubes Spectraux*) is an analysis engine for [SITELLE](http://www.cfht.hawaii.edu/Instruments/Sitelle) spectral cubes. With [ORCS](https://github.com/thomasorb/orcs) you can:

* extract integrated spectra
  
* fit the sinc emission lines

* get a fast and robust estimation fo the velocity and fluxes of all the emission lines in you cube (even for multiple components, see [our paper](https://doi.org/10.1093/mnras/staa4046)).
  
* recalibrate your data in wavelength, astrometry and flux

* and much more ...

**Examples** in Jupyter format can be found in the documentation:

https://orcs.readthedocs.io/en/latest/examples.html
    
   
**Warning:** ORCS is always in fast development. You might see bugs or
 strange behaviours ! You might also want desperatly to have new
 features. In all cases please send an email: thomas dot martin dot 1 at ulaval dot ca

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

- [Here](docs/orb3-ver.txt) is a list of all the versions of the packages on a working installation ([docs/orbs3-ver.txt](docs/orb3-ver.txt)). Higher versions generally work but not in some cases.

- [Here](docs/orb3-env.txt) is an environment file that can be used directly with conda to install the correct versions of the packages ([docs/orbs3-env.txt](docs/orb3-env.txt)). 

## Publications

* Martin, Milisavljevic, Drissen (2021), 3D mapping of the Crab Nebula with SITELLE – I. Deconvolution and kinematic reconstruction, MNRAS, Volume 502, Issue 2, Pages 1864–1881, https://doi.org/10.1093/mnras/staa4046

* Martin, Drissen, Prunet (2021), **Data reduction and calibration accuracy** of the imaging Fourier transform spectrometer SITELLE, MNRAS, Volume 505, Issue 4, Pages 5514–5529, https://doi.org/10.1093/mnras/stab1656

* Martin, Drissen, Melchior (2017). A SITELLE view of M31's central region - I: Calibrations and radial velocity catalogue of nearly 800 emission-line point-like sources. MNRAS
  http://adsabs.harvard.edu/abs/2017arXiv170701366M

* Martin, Prunet, Drissen (2016). Optimal fitting of Gaussian-apodized or under-resolved emission lines in Fourier transform spectra providing new insights on the velocity structure of NGC 6720. MNRAS
  http://adsabs.harvard.edu/abs/2016MNRAS.463.4223M

* Martin, Drissen, Joncas (2015). ORBS, ORCS, OACS, a Software Suite for Data Reduction and Analysis of the Hyperspectral Imagers SITELLE and SpIOMM. PASP
  http://adsabs.harvard.edu/abs/2015ASPC..495..327M

