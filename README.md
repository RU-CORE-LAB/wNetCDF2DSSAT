
# wDSSAT2NetCDF
The wNetCDF2DSSAT is a Python library for converting Networked Common Data Form (NetCDF) to DSSAT input data format.

## Overview



## Installation
```
pip install git+https://github.com/RU-CORE-LAB/wNetCDF2DSSAT.git
```

## Metthod Uage

## Basic Usage

> **Step1.** Import library.
```
from wNetCDF2DSSAT.convert impot WTH
```
> **Step2.** create a class of wNetCDF2DSSAT objects and assigned it to a variable named “ncObj”. The users can also assign the object to a different variable name
```
ncObj = WTH()
``` 
> **Step3.** Define the folder path of the data source.
```
ncObj.define_input_path("ESGF_downloads/MPI-M-MPI-ESM-MR/historical/")
```
> **Step4.** Provide the information in the header of the DSSAT ASCII input file.
```
ncObj.define_description(rcm_ver=“RegCM4.7”, model=“MPI-ESM-MR”, scenario=“historical”)
```
> **Step5.** Define the variable names for latitude, longitude and time in the NetCDF file.
```
ncObj.define_coordinates_name(“lat”,“lon”,“time”)
```
> **Step6.** Define the variable names for near-surface air temperature, minimum near-surface air temperature, maximum near-surface air temperature, precipitation, near-surface relative humidity, near-surface wind speed and surface downwelling shortwave flux in the NetCDF files. 
```
ncObj.define_climate_variable("tas","tasmax","tasmin","hurs","pr","sfcWind","rsds")
```
> **Step7.** Define the coordinates of the subdomain
```
ncObj.define_domain(14,16,99,100.5)
```

> **Step8.** Convert the data is used to transform the climate variables with a specific time period from the NetCDF to DSSAT ASCII format and stored them in the specified folder.
```
ncObj.convert2dssat(output_path="DSSAT_output/", period= "historical", start_year=1970, end_year=2005,single_year_file = False
```

