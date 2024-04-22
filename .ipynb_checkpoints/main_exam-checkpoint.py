from wNetCDF2DSSAT.convert import WTH
ncObj = WTH()
ncObj.debug_level = "1"
#MPI-M-MPI-ESM-MR ICHEC-EC-EARTH  MOHC-HadGEM2-ES Ensemble
ncObj.define_input_path("../input/ICHEC-EC-EARTH/rcp85/")
ncObj.define_description(rcm_ver = "RegCM4.7", model="EC-EARTH", scenario="historical")
ncObj.define_coordinates_name("lat","lon","time")
ncObj.define_climate_variable("pr","tas","tasmax","tasmin","rsds","hurs",wind="sfcWind")
#ncObj.define_domain(15.2,18.8,98.1,104.8)
ncObj.define_domain(5.2,12.7,97,103)

#ncObj.define_domain(18,18.8,98.1,99) # area1
#ncObj.define_domain(15.2,15.6,104,104.8)  # area2
ncObj.convert2dssat(output_path = "../output/South_Thailand/", period="projection",start_year=2006, end_year=2030, single_year_file = False)

exit()


# 1. include wNetCDF2DSSAT 
from wNetCDF2DSSAT.convert import WTH
# 2. Created a class wNetCDF2DSSAT Object, assign to ncOBJ or another variable name.
ncObj = WTH()
#ncObj.topo()
ncObj.debug_level = "1"
#ncObj.ncdump("_init_data_testing/ESGF/MPI-ESM-MR/pr/pr_SEA-22_MPI-M-MPI-ESM-MR_historical_r1i1p1_ICTP-RegCM4-3_v4_day_1970010112-1970013112.nc")
# 3. Define the folder path of data source by the define_input_path methods.
ncObj.define_input_path("F:/_GIT/DATA/ESGF_downloads/")

# 4. Define the moodel description for detail in header of the WTH file by the define_description methods.
ncObj.define_description("RegCM4", "MPI-ESM-MR", "Historical")

# 5. Define tpythohe coordinates name netCDF by the define_coordinates_name methods.
#
ncObj.define_coordinates_name("xlat","xlon","time")
# if unknown can use this command call ncdump methods. 
# >> ncOBj.ncdump()  

# 6. Define climate variables name by the define_climate_variable methods
# # (rain,tmean,tmax,tmin,rad)
#ncObj.define_climate_variable("pr","tas","tasmax","tasmin","rsns")
ncObj.define_climate_variable("pr","tas","tasmax","tasmin","rsns","hurs","ps","sfcWind")
# 7. if you want to selection a sub domian, define by define_domain methods pi
#ncObj.define_domain(14,16,99,100.5) # (LAT1, LAT2, LON1, LON2)
ncObj.define_domain(15.2,19,99.,100.5)
# 8. convert netcdf data to dssat format by convert2dssat methods. 
# !! Note that the convert2dssat methods must go through all previous steps.
# ncObj.convert2dssat(output_path = "./Folder_output/", start_year=yyyy, end_year=yyyy)
ncObj.convert2dssat(output_path = "F:/_GIT/DSSAT_output/MPI/hist/", start_year=1970, end_year=2005, single_year_file = False)




