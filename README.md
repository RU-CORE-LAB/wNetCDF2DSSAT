# wDSSAT2NetCDF
>>> from wNetCDF2DSSAT.convert import WTH
>>> ncObj = WTH()
>>> ncObj.define_input_path("ESGF_downloads/MPI-ESM-MR/")
>>> ncObj.define_description("RegCM4.7", "MPI-ESM-MR", "Historical")
>>> ncObj.define_coordinates_name("xlat","xlon","time")
>>> ncObj.define_climate_variable("pr","tas","tasmax","tasmin","rsns") 
>>> ncObj.define_domain(11,12,100,100.5)
>>> ncObj.convert2dssat(output_path = "./DSSAT_output/", start_year=1970, end_year=1971)

Step >> 
1. include wNetCDF2DSSAT library
2. Created a class wNetCDF2DSSAT Object, assign to ncOBJ or another variable name.
3. Define the folder path of the data source by the define_input_path() function.
4. Define the model description for detail in the header of the (.wth) file by the define_description() function.
5. Define the coordinates name of netCDF by the define_coordinates_name() function.
   !! If you don't know the name, you can call ncdump() function.
>>> ncObj.ncdump("ESGF_downloads/ESGF/MPI-ESM-MR/pr/pr_SEA-22_MPI-M-MPI-ESM-MR_historical_r1i1p1_ICTP-RegCM4-3_v4_day_1970010112-1970013112.nc")
Output[1] : Coordinates:
  * time     (time) datetime64[ns] 1970-01-01T12:00:00 ... 1970-01-31T12:00:00
    lon      (y, x) float64 ...
    lat      (y, x) float64 ...
  * x        (x) float64 -2.3e+06 -2.275e+06 -2.25e+06 ... -1.375e+06 -1.35e+06
  * y        (y) float64 -1.5e+05 -1.25e+05 -1e+05 ... 1.625e+06 1.65e+06
     Data variables:
    time_bnds  (time, bnds) datetime64[ns] ...
    crs        int32 ...
    pr         (time, y, x) float32 ...
6. Define climate variables name by the define_climate_variable() function.
7. if you want to select a subdomain, define it by define_domain() function.
8. convert NetCDF data to DSSAT format by convert2dssat() function. 
   !! Note that the convert2dssat() function must go through all previous steps.

define_input_path("a")
a = input folder path
define_description("a","b","c")
a = Model simulation name
b = Model name
c = Experiment or Scenario
define_coordinates_name("a","b","c")
a = Variable name of latitude
b = Variable name of Longitude
c = Variable name of time
define_climate_variable("a","b","c","d","e") 

arg	DSSAT Var_name	Climate Var_name	Climate_standard_name
a	Rain	pr	Precipitaion
b	Tmean	tas	air_temperature
c	Tmax	tasmax	Maximum Near-Surface Air Temperature
d	Tmin	tasmin	Minimum Near-Surface Air Temperature
e	rad	rsns	net_downward_shortwave_flux

define_domain("a","b","c","d")
a = Minimum latitude value
b = Maximum latitude value
c = Minimum longitude value
d = Maximum longitude value
convert2dssat(output_path = "a", start_year=b, end_year=c)
a = Output folder path
b = start year convert
c = end year convert

 
Elevation : 
-	0.0625 x 0.0625 degree
-	Grids size 5760 x 2880
