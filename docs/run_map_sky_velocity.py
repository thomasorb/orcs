#from orcs.process import SpectralCube
from orb.cube import SpectralCube
cube = SpectralCube('/home/thomas/M31_SN3.merged.cm1.1.0.hdf5')
#cube.map_sky_velocity(80, div_nb=10, exclude_reg_file_path='m31_exclude.reg', no_fit=False) # the mean velocity bias is set around 80 km/s
