# option_file.orc

## ORCS option file

# Cube path
CUBEPATH /cube/path/NGC000_SN3.merged.cm1.1.0.hdf5

# Emission-lines parameters
LINES [NII]6548,Halpha,[NII]6583,HeI6678,[SII]6716,[SII]6731 # 1

# (Optional) Velocity groups (same symbol for same group)
COV_LINES 1,2,1,2,1,1

# (Optional) Broadening groups (same symbol for same group)
COV_SIGMA 1,2,1,2,1,1

# Object mean velocity in km.s-1
OBJECT_VELOCITY 500

# (Optional) Velocity range to check for the initial guess
VELOCITY_RANGE 1000

# Path to a ds9 region file (with pixel coordinates) defining the
# fitting region
OBJ_REG ngc000.reg

# Order of the polynomial used to fit continuum
POLY_ORDER 0

# (Optional) Sky regions ds9 file path
SKY_REG ngc000-sky.reg

# (Optional) 'on the fly' binning of the data
BINNING 10