
# wDSSAT2NetCDF
The wNetCDF2DSSAT is a Python library for converting Networked Common Data Form (NetCDF) to DSSAT input data format.

## Overview



## Installation
```
pip install git+https://github.com/RU-CORE-LAB/wNetCDF2DSSAT.git
```

## User Functions.
(i) The “define_input_path("a")” function is used to let the library recognize the folders that store the climate variable data in NetCDF format.
(ii) The “define_description(rcm_ver = "b", model = "c", scenario = "d")” function is used to provide the data information to appear on the header of the DSSAT input file.
(iii) The “define_coordinates_name("lat","long","time")” function is used to let the library recognize the variable names in NetCDF meta-data for latitude, longitude and time. The users can use the ncdump() function to view the data information in the NetCDF file, for example, the data information of “pr_SEA-22_MPI-M-MPI-ESM-MR_historical_r1i1p1_ICTP-RegCM4-3_v4_day_1970010112-1970013112.nc” as shown in Figure 3. In this meta-data file, the latitude, longitude and date&time are named as “lat”, “long” and “time” respectively. 
(iv) The “define_climate_variable("T2M","TMAX","TMIN","RAIN","RH2M","WIND", "SRAD")” function is used to let the library recognize the variable names in NetCDF meta-data that match with the DSSAT variable names as given in Table 1. The order of arguments of this function must be “T2M”, “TMAX”, “TMIN”, “RAIN”, “RH2M”, “WIND” and “SRAD”. This is one important step as the wNetCDF2DSSAT will get the climate variables from the sub-folders with the same variable names given in the arguments of this function stored inside the folder assigned by the define_input_path("a") function as Figure 2. The users must name the subfolders used to store the variables exactly the same as specified in the function arguments. In addition, the users are recommended to use the ncdump() function to view the climate variable names in the NetCDF file, as in the example in Figure 3, to assure that all variables are matched. In this example, the DSSAT variable “RAIN” matches with “pr” of NetCDF. 
(v) The “define_domain(TLat,BLat,LLong,RLong)” function is used to set the coordinates of the subdomain where TLat is the top latitude, BLat is the bottom latitude, LLong is the left longitude and RLong is the right longitude of the subdomain. Since the climate data stored in NetCDF from ESGF nodes covers the whole study domain of the data providers, this function is used to extract data for the specific subdomain of interested. 
(vi) The “convert2dssat(output_path=“f”, period=“g”, start_year=h, end_year=i, single_year_file=j)” function is used to convert climate variables from NetCDF to DSSAT ASCII format and store them in the output folder “f”. The second argument, or period=“g”, will let the library recognize the period of the climate data set to be converted. “g” of this argument can only be “historical” or “projection”. As the header of the DSSAT input file requires data on i) long term average near-surface air temperature (TAV) and ii) long term average of difference between monthly near-surface maximum and minimum temperature (AMP) over the historical or base line period, the library will calculate both TAV and AMP from the data set in the “historical” folder assigned by define_input_path("a") function and make the information appear in the header of input files for both historical and projection periods. Since the climate data from the ESGF node stored in NetCDF may cover a wider time period than the one of interest, the third and fourth argument, (start_year=h and end_year=i), are used to select the specific time period from year h to i. The last argument of this function is the “single_year_file=j”. The users can select the DSSAT input as single file by assigning TRUE to this argument and vice versa. 

## Basic Usage

> **Step1.** Import library.
```
>>> from wNetCDF2DSSAT.convert impot WTH
```
> **Step2.** create a class of wNetCDF2DSSAT objects and assigned it to a variable named “ncObj”. The users can also assign the object to a different variable name
```
>>> ncObj = WTH()
``` 
> **Step3.** Define the folder path of the data source.
```
>>> ncObj.define_input_path("ESGF_downloads/MPI-M-MPI-ESM-MR/historical/")
```
> **Step4.** Provide the information in the header of the DSSAT ASCII input file.
```
>>> ncObj.define_description(rcm_ver=“RegCM4.7”, model=“MPI-ESM-MR”, scenario=“historical”)
```
> **Step5.** Define the variable names for latitude, longitude and time in the NetCDF file.
```
>>> ncObj.define_coordinates_name(“lat”,“lon”,“time”)
```
> **Step6.** Define the variable names for near-surface air temperature, minimum near-surface air temperature, maximum near-surface air temperature, precipitation, near-surface relative humidity, near-surface wind speed and surface downwelling shortwave flux in the NetCDF files. 
```
>>> ncObj.define_climate_variable("tas","tasmax","tasmin","hurs","pr","sfcWind","rsds")
```
> **Step7.** Define the coordinates of the subdomain
```
>>> ncObj.define_domain(14,16,99,100.5)
```

> **Step8.** Convert the data is used to transform the climate variables with a specific time period from the NetCDF to DSSAT ASCII format and stored them in the specified folder.
```
>>> ncObj.convert2dssat(output_path="DSSAT_output/", period= "historical", start_year=1970, end_year=2005,single_year_file = False
```

