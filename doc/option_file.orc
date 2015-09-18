# option_file.orc

## ORCS option file

INCLUDE /Path/to/ORBS_option_file.opt
CUBEPATH /Path/to/spectrum/cube.hdf5
WAVENUMBER 1
WAVE_CALIB 0
APOD 2.0
LINES [NII]6548,Halpha,[NII]6583,[SII]6716,[SII]6731
COV_LINES 1,2,1,1,1
OBJECT_VELOCITY 180
POLY_ORDER 0
ROI 165,260,155,255 #Xmin,Xmax,Ymin,Ymax
SKY_REG sky.reg 
CALIBMAP /Path/to/calibration_laser_map.fits
