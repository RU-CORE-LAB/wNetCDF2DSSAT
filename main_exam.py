from wNetCDF2DSSAT.convert import WTH
ncObj = WTH()
ncObj.debug_level = "0"
ncObj.define_input_path("ESGF_Downloads/MPI-M-MPI-ESM-MR/historical/")
ncObj.define_description(rcm_ver = "RegCM4.7", model="MPI-ESM-MR", scenario="historical")
ncObj.define_coordinates("lat","lon","time")
ncObj.define_climate_variable("tas","tasmax","tasmin","hurs","pr","sfcWind","rsds")
ncObj.define_domain(16,14,102,105)
ncObj.define_reanalysis_path("OBS/ERA5/")
ncObj.define_reanalysis_coordinates("latitude","longitude")
ncObj.define_reanalysis_variable("t2m","mx2t","mn2t","tp")
ncObj.bias_correction(bias_method = "quantile_mapping", n_quantiles=250, group="", kind="+")
ncObj.convert2dssat(output_path="DSSAT_output/",period="historical",
                    start_year=1970,end_year=2005,single_year_file = False)
