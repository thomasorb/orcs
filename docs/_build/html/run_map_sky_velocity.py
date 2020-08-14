from orcs.process import SpectralCube
if __name__ == "__main__": # this is very important when running parallel processes
    cube = SpectralCube('/home/thomas/M31_SN3.merged.cm1.1.0.hdf5')
    cube.map_sky_velocity(80, div_nb=10, exclude_reg_file_path='m31_exclude.reg', no_fit=False, plot=False) # the mean velocity bias is set around 80 km/s
