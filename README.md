
# wDSSAT2NetCDF
The wNetCDF2DSSAT is a Python library for converting Networked Common Data Form (NetCDF) to DSSAT input data format.

## Overview



## Installation
```
pip install git+https://github.com/RU-CORE-LAB/wNetCDF2DSSAT.git
```


## Basic Usage

> **Step1.** Import library.
```
>>> from wNetCDF2DSSAT.convert impot WTH
```
This command line is used to include wNetCDF2DSSAT library to the system. 

> **Step2.** create a class of wNetCDF2DSSAT objects and assigned it to a variable named “ncObj”. The users can also assign the object to a different variable name.
```
>>> ncObj = WTH()
```
This command line is used to create a class of wNetCDF2DSSAT objects and assigned it to a variable named “ncObj”. The users can also assign the object to a different variable name. 

> **Step3.** Define the folder path of the data source.
```
>>> ncObj.define_input_path("ESGF_downloads/MPI-M-MPI-ESM-MR/historical/")
```
This command line is used to define the folder path of the data source. In this example, the climate data is stored in the folder named “ESGF_downloads/MPI-M-MPI-ESM-MR/historical/”. The MPI-ESM-MR is the GCMs applied as the initial condition and boundary conditions for the dynamical downscaling processes. 

> **Step4.** Provide the information in the header of the DSSAT ASCII input file.
```
>>> ncObj.define_description(rcm_ver=“RegCM4.7”, model=“MPI-ESM-MR”, scenario=“historical”)
```
This command line is used to provide the information in the header of the DSSAT ASCII input file. In this example, “RegCM4.7” is the Regional Climate Model version 4.7 used to downscale the GCMs (https://www.ictp.it/research/esp/models/regcm4.aspx). MPI-ESM-MR, stands for Max Planck Institute Earth System Model, medium resolution (https://mpimet.mpg.de/en/science/models/mpi-esm) and “historical” refers to the historical or baseline period of the model.

> **Step5.** Define the Dimession names for latitude, longitude and time in the NetCDF file.
```
>>> ncObj.define_coordinates_name(“lat”,“lon”,“time”)
```
This command line is used to let the library recognize the variable names for latitude, longitude and time in the NetCDF file.

> **Step6.** Define the variable names.
```
>>> ncObj.define_climate_variable("tas","tasmax","tasmin","hurs","pr","sfcWind","rsds")
```
This command line is used to let the library recognize the variable names for near-surface air temperature, maximum near-surface air temperature, minimum near-surface air temperature, near-surface relative humidity, precipitation, near-surface wind speed and surface downwelling shortwave flux in the NetCDF files. 

> **Step7.** Define the coordinates of the subdomain
```
>>> ncObj.define_domain(14,16,99,100.5)
```
This command line is used to set the coordinates of the subdomain. In this example, the top latitude is 15.2 degree north, the bottom latitude is 15.6 degree north, the left and right longitude of the subdomain are 104 degree east and 104.8 degree east respectively. 

> **Step8.** Define the folder path of the ReAnalysis data
```
>>> ncObj.define_reanalysis_path("OBS/ERA5")
```
This command line is used to define the folder path of the ReAnalysis data source. In this example, the ReAnalysis data is stored in the folder named “OBS/ERA5/”. 

> **Step9.** Define the Dimession names for latitude, longitude of the ReAnalysis data
```
>>> ncObj.define_reanalysis_coordinates("latitude","longitude")
```
This command line is used to let the library recognize the variable names for latitude and longitude of ReAnalysis data files in the NetCDF format.

> **Step10.** Define the variable names of the ReAnalysis data
```
>>> ncObj.define_reanalysis_variable("t2m","mx2t","mn2t","tp")
```
This command line is used to let the library recognize the variable names for near-surface air temperature, maximum near-surface air temperature, minimum near-surface air temperature and precipitation of ReAnalysis data files in the NetCDF format. 

> **Step11.** Define the variable names of the ReAnalysis data
```
>>> nObj.bias_correction(bias_method = "quantile_mapping", n_quantiles = 250, group = "", kind = "+") 
```
This command line is used to bias correct the climate data by using ReAnalysis data as reference. Step 8 – Step 11 need to be carried out before proceed with Step 12 in case that the model simulation data exhibits substantial systematic biases and bias correction cannot be disregarded. However, these steps can be omitted if the model simulation data have high performance to reproduce the historical climate. In this example, the quantile mapping technique is applied to bias adjust the climate data using ReAnalysis-ERA5 as reference. 

> **Step12.** Covert the data.
```
>>> ncObj.convert2dssat(output_path="DSSAT_output/", period= “historical”, start_year=1970, end_year=2005,single_year_file = False)
```
This command line is used to transform the climate variables with a specific time period from the NetCDF to DSSAT ASCII format and stored them in the specified folder. In this example, the files are stored in the folder “/DSSAT_output/”). The data set is the historical period from 1970 – 2005 and stored in individual files. 

## Data Source (inp files)
> Regional climate model Data.
https://esg-dn1.nsc.liu.se/search/cordex/
Search Constraints:   SEA-22 | esg-dn1.ru.ac.th | historical,rcp45,rcp85 | hus,pr,rsds,sfcWind,tas,tasmax,tasmin | RU-CORE | MPI-M-MPI-ESM-MR | r1i1p1 | day | RegCM4-7

> ReAnalysis Data (EAR5)
https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels

>> variables : t2m, mn2t, mn2t, tp
