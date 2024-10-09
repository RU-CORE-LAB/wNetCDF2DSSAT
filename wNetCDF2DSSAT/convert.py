# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Start up @ Aug 2022
# _wNetCDF2DSSAT.py Version 1.4
# Written by Nick Ratchanan (RU-CORE).
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::

from datetime import datetime, date ,timedelta
import cftime
from operator import index
import xarray as xr
import numpy  as np
import pandas as pd
import os, shutil
from os import path
import glob
from tqdm import tqdm
from time import process_time
import inspect
import math
from matplotlib import pyplot as plt
import shapefile as shp
import geopandas as gpd
from shapely.geometry import Polygon
import warnings
from cmethods import adjust
warnings.simplefilter('ignore')

### CLASS :: Initial Global variables and Method
class mainDefind:
    # initial variabled
    def __init__(self):
        self.GB_input_path                = ""
        
        self.GB_output_path               = ""
        self.GB_climate_variable_name     = {}
        '''
        self.GB_climate_variable_name     = {"rain" : "",
                                            "tmean" : "", 
                                            "tmax":"",
                                            "tmin":"",
                                            "rad" : "",
                                            "wind" : "",
                                            "pa" : "",
                                            "rh" : ""}
        '''
        self.GB_climate_coordinates_name  = {"lat":"lat", "lon":"lon", "time":"time"}
        self.GB_dropvars                  = {"time_bnds","crs","height","bnds"}
        self.debug_level                  = "0"
        self.GB_list_file_process         = pd.DataFrame()
        self.GB_dataset_1y                = pd.DataFrame()
        self.resolution_raw_lat           = 0
        self.resolution_raw_lon           = 0
        self.GB_lonlatbox                 = {"lat_min":None, "lat_max":None, "lon_min":None, "lon_max":None}
        self.GB_unique_lat                = []
        self.GB_unique_lon                = []
        #self.GB_data_latlon               = {"dlat":[],"dlon":[]}
        self.GB_model_detail              = {"model-simulation":"TEST","model-name":"TEST","scenario":"RF_TEST"}
        self.blank                        = " " # " " : is space.
        self.file_wth_name                = ""  # wth file name.
        self.file_wth_proc                = ""  # wth location + filename.
        self.GB_OBS_bias_status           = False
        self.GB_OBS_path                  = ""
        self.GB_OBS_coordinates_name      = {}
        self.GB_OBS_variables_name        = {}
        self.GB_OBS_bias_option           = {}
        self.GB_experiment                = ""
        self.GB_startYear                 = ""
        self.GB_endYear                   = ""
        self.GB_data_topo                 = pd.DataFrame()
        self.GB_lat_topo                  = []
        self.GB_lon_topo                  = []
        self.load_topo                    = False
        self.GB_grids_no                  = []
        self.latlon_convert               = pd.DataFrame()
        self.time_in_file                 = False
        self.startYear                    = 0
        self.endYear                      = 0 
        self.written_header               = False
        self.check_variable_convert       = { "WIND" : False, "AVP" : False, "DEWP" : False}
        self.check_new_process            = False
        self.experiment_process           = ""
        self.tb_historical_global_T       = pd.DataFrame()
        self.TAV                          = {}
        self.TAMP                         = {}
        

    # Display for Debug        
    def display(self,obj , mode="0"):
        if  (self.debug_level == "1"):
            print(obj)
    
    def check_file_exist(self, f):
        self.display("File exists: " + f + ": " + str(path.isfile(f)))
        return path.isfile(f)
    
    def xnc_dataset(self,file,drop_var=""):
        # lib : h5netcdf ; pip install h5netcdf
        # lib : h5py     ; pip install h5py
        # lib : hdf5     ; conda install -c anaconda hdf5
        try:
            return xr.open_dataset(file,drop_variables=drop_var)
        except:
            raise ValueError("Incomplete file: %s"%file)
            
        
    def xnc_to_dataframe(self,nco):
        return nco.to_dataframe()               # xarray open_dataset to dataframe

    def convert_unit(self,v,data):
        MJ       = 0.0864           # 3hr : 3*60*60/10^6 for radiation ::: 0.0864
        K        = 273.15           # Kelvin (1 celsius)
        Pr_daily = 60*60*24         # daily precipitation (kg m-2 s-1 to mm)
        km_day   = 86.4
        Kilo    = 1000
        percent  = 100
        
        v = list(self.GB_climate_variable_name.values()).index(v)
        v = list(self.GB_climate_variable_name.keys())[v]
        unit ={
                'rain'  : data * Pr_daily,      
                'tmean' : data - K,     
                'tmax'  : data - K,       
                'tmin'  : data - K,
                'rad'   : data * MJ,
                'wind'  : data,     # * km_day,
                'pa'    : data / Kilo,
                'rh'    : data / percent
            }
        return unit.get(v, data)
    
    def _mkdir(self,dir):
        try: 
            os.makedirs(dir, exist_ok=True) 
        except OSError as error: 
            pass 

    def print_attrs(title, attrs):
        print(f"\n{title}:")
        for key, value in attrs.items():
            print(f"  {key}: {value}")
    
    def ncdump(self,f):
        nco = self.xnc_dataset(f)
        var_name = nco.data_vars
        dimen = nco.coords
        #xr.set_options(display_width=1000, display_expand_data=True)
        # แสดงข้อมูลทั้งหมดของ Dataset
        #print(nco)
        print (dimen)
        print (var_name)
        
        #print(nco.attrs)
        
        # แสดง attributes ของตัวแปรแต่ละตัว
        for var_name in nco.data_vars:
            print(f"\nAttributes of variable '{var_name}':")
            print(nco[var_name].attrs)
        #print (dimen)
        #print (var_name)
       
        print ("")
        
    def map_plot(self,lon,lat,var_name,data):
        data.plot.scatter(x=lon,y=lat,c=var_name,cmap='YlOrBr') #colormap='viridis'
        #data.plot(kind = 'scatter', x = 'xlon', y = 'xlat')
        plt.show()
        
    def polygons_from_custom_xy_string(df_column):
    
        def chunks(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i:i + n]
        
        def xy_list_from_string(s):
            # 'x y x y ...' -> [[x, y], [x, y], ...]
            return list(chunks([float(i) for i in s.split()], 2))
        
        def poly(s):
            """ returns shapely polygon from point list"""
            ps = xy_list_from_string(s)
            return Polygon([[p[0], p[1]] for p in ps])

        polygons = [poly(r) for r in df_column]

        return polygons
    
### CLASS :: Check data source       
class getFileProcess(mainDefind):
    def __init__(self):
        mainDefind.__init__(self)
            
    def convert_to_dt(x):
        return datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S')   
        
    def get_file_list(self,sYR,eYR):
        #self.GB_input_path
        self.GB_list_file_process = pd.DataFrame()
        for var_name in self.GB_climate_variable_name.values():
            print(f"get file {var_name}")
            glob_arg        = self.GB_input_path + "/" + var_name + "/"
            lfile = getFileProcess.get_files_sorted_by_date_range(self,glob_arg, sYR, eYR, var_name,False)
                     
            df = pd.DataFrame(lfile, columns=[var_name, 'year', 'month'])
            df = df[['year', 'month',var_name]]
            
            if(len(self.GB_list_file_process) == 0 ):
                self.GB_list_file_process = df
            else:
                self.GB_list_file_process = pd.merge(self.GB_list_file_process, df, on=['year', 'month'], how='outer')
        #print(self.GB_list_file_process)
            if ((self.GB_OBS_bias_status == True) and (self.GB_experiment == "historical")):
                obs_var = ""
                if( var_name == self.GB_climate_variable_name['rain']):
                    obs_var = self.GB_OBS_variables_name['pr']
                elif( var_name == self.GB_climate_variable_name['tmean']):
                    obs_var = self.GB_OBS_variables_name['tas']
                elif( var_name == self.GB_climate_variable_name['tmax']):
                    obs_var = self.GB_OBS_variables_name['tmax']
                elif( var_name == self.GB_climate_variable_name['tmin']):
                    obs_var = self.GB_OBS_variables_name['tmin']
                else:
                    obs_var = ""
    
                if (obs_var !=""):
                    for in_filename, in_year, in_month in lfile:
                        rcm_ = xr.open_dataset(f"input_preprocess/{var_name}/{in_filename}")
                        
                        formatted_month = f"{in_month:02d}"
                        
                        obs_path = self.GB_OBS_path + obs_var
                        obs_file = os.path.join(obs_path, f"{obs_var}_{in_year}_{formatted_month}.nc")
                        obs_lat = self.GB_OBS_coordinates_name['lat']
                        obs_lon = self.GB_OBS_coordinates_name['lon']
                        rcm_lat_name = self.GB_climate_coordinates_name["lat"]
                        rcm_lon_name = self.GB_climate_coordinates_name["lon"]
                        #print(obs_file)
                        try:
                            obs_data = xr.open_dataset(obs_file)
                            
                            obs_data = obs_data.rename({obs_lat: rcm_lat_name, obs_lon: rcm_lon_name, obs_var: obs_var})
                            data_interpolated = obs_data[obs_var].interp(lat=rcm_[rcm_lat_name], lon=rcm_[rcm_lon_name])
    
                            if(var_name == "pr"):
                                data_obs = ((data_interpolated*1000)/86400).resample(time='D').sum()
                            else:
                                data_obs = ((data_interpolated)).resample(time='D').mean()
                            
                            new_obs_file = xr.Dataset({
                                rcm_lat_name: rcm_[rcm_lat_name],
                                rcm_lon_name: rcm_[rcm_lon_name],
                                #'time' : rcm_[self.GB_climate_coordinates_name["time"]],
                                var_name: data_obs
                            })
                            
                            file_to = f"input_preprocess/obs/{var_name}"
                            self._mkdir(file_to)
                            new_obs_file.to_netcdf(f"{file_to}/OBS_{var_name}_{in_year}_{formatted_month}.nc")
        
                        except Exception as e:
                            print(f"Error processing {obs_file}: {e}")
                            
                        rcm_.close()
                        obs_data.close()
        self.display(self.GB_list_file_process)
        
    def crop_dromain_to_ncfile(self,rcm,filename,var_name):
        lat1, lat2, lon1, lon2 = self.GB_lonlatbox.values()
        rcm_lat_name = self.GB_climate_coordinates_name["lat"]
        rcm_lon_name = self.GB_climate_coordinates_name["lon"]
        lat_bnds = [lat1, lat2]
        lon_bnds = [lon1, lon2]
        lat_mask = (rcm[rcm_lat_name] >= lat_bnds[0]) & (rcm[rcm_lat_name] <= lat_bnds[1])
        lon_mask = (rcm[rcm_lon_name] >= lon_bnds[0]) & (rcm[rcm_lon_name] <= lon_bnds[1])
        mask = lat_mask & lon_mask
        rcm_cropped = rcm.where(mask, drop=True)
        
        encoding = {
            'time': {
                'units': 'seconds since 1949-12-01T00:00:00+00:00',
                'dtype': 'float64'  # dtype is float64
            }
        }
        #self.GB_input_path = "input_preprocess"
        tmp_path = f"input_preprocess/{var_name}"
        
        self._mkdir(tmp_path)
        rcm_cropped.to_netcdf(f"{tmp_path}/{filename}", encoding=encoding)

        
    def get_files_sorted_by_date_range(self, folder_path, start_year, end_year, keyword, re_get_file):
      pattern = os.path.join(folder_path, '*.nc')
      all_files = glob.glob(pattern)
      
      files_with_dates = []
      
      for f in all_files:
          if keyword in os.path.basename(f):
              try:
                  ds = xr.open_dataset(f)  # open nc file to check the period.
                  time = ds.time
                  if len(time) > 0:
                      # Handling different time formats
                      first_time_value = time.values[0]
                      
                      # Check if the time value is a cftime object
                      if isinstance(first_time_value, cftime.datetime):
                          date = pd.to_datetime(first_time_value.strftime('%Y-%m-%d'))
                      else:
                          date = pd.to_datetime(first_time_value)
                      
                      year = date.year
                      month = date.month
                      
                      if start_year <= year <= end_year:
                          if re_get_file:
                              file_name = f"{os.path.basename(f)}"
                              files_with_dates.append((file_name, year, month))
                          else:
                              file_name = f"inp_{os.path.basename(f)}"
                              getFileProcess.crop_dromain_to_ncfile(self, ds, file_name, keyword)
                              files_with_dates.append((file_name, year, month))
                              
              except Exception as e:
                  print(f"Error reading {f}: {e}")
      
      files_with_dates.sort(key=lambda x: (x[1], x[2]))
      sorted_files = [(f[0], f[1], f[2]) for f in files_with_dates]
      
      # Generate the full date range for the specified years
      full_date_range = pd.date_range(start=f'{start_year}-01', end=f'{end_year}-12', freq='MS')
      
      # Check for missing dates
      actual_dates = set(pd.to_datetime([f'{year}-{month:02d}' for _, year, month in files_with_dates]))
      missing_dates = set(full_date_range) - actual_dates
      
      if missing_dates:
          raise FileNotFoundError(f"File not found. {folder_path} : {missing_dates}")
      
      return sorted_files
    
    # Create table path file
    def get_file_list2(self,sYR,eYR):
        head_name = ["year","month"] + list(self.GB_climate_variable_name.values())
        self.GB_list_file_process = pd.DataFrame(columns=head_name, index=range(0))
        
        for loop_year in range(sYR,eYR):
            
            for var_name in self.GB_climate_variable_name.values():
                glob_arg        = self.GB_input_path + "/" + var_name + "/" + var_name + "*" + str(loop_year) + "*.nc"
                ls              = sorted(glob.glob(glob_arg))
                
                for strName in ls :
                    list_file   = strName.replace("\\", "/")
                    temp_open   = self.xnc_dataset(list_file,self.GB_dropvars).to_dataframe()
                    
                    # date_index  = temp_open.index.get_level_values(self.GB_climate_coordinates_name["time"])[0]
                    date_index  = temp_open.index.get_level_values(self.GB_climate_coordinates_name["time"])[0]
                    yy=0;mm=0
                    if(isinstance(date_index, cftime._cftime.Datetime360Day)) :
                        yy          = date_index.year
                        mm          = date_index.month
                    else :
                        yy          = date_index.date().year
                        mm          = date_index.date().month

                    if (yy == loop_year) :
                        d = pd.DataFrame(data={'year': [yy], 'month': [mm] , var_name : [list_file]})
                    
                        sel = self.GB_list_file_process.loc[(self.GB_list_file_process["year"] == yy) & (self.GB_list_file_process["month"] == mm)]
                        if (len(sel) == 0):
                            self.GB_list_file_process = pd.concat([self.GB_list_file_process,d], ignore_index=True)
                        else:
                            self.GB_list_file_process.loc[(self.GB_list_file_process["year"] == yy) & (self.GB_list_file_process["month"] == mm),[var_name]] = list_file  
            
                self.display("Dataset Year %s :%s/%s"%(var_name,np.amax(self.GB_list_file_process[['year']].to_numpy()), self.endYear))
            
            self.display(self.GB_list_file_process)
            
            getFileProcess.check_list_file_process(self,sYR,eYR)
    
    def check_list_file_process(self,sYR,eYR):
        getFileProcess.select_list_file_process_year(self,sYR,eYR)
        for var_name in self.GB_climate_variable_name.values():
            get_nan_index = self.GB_list_file_process[self.GB_list_file_process[var_name].isnull()].index.values
            if (len(get_nan_index > 0)):
                for inx in get_nan_index :
                    print ("file not found (%s): Year: %s - Month: %s "%(var_name, self.GB_list_file_process["year"][inx], self.GB_list_file_process["month"][inx]))

    def select_list_file_process_year(self,sYR,eYR):
        if(sYR == ""):
            self.GB_list_file_process = self.GB_list_file_process
        elif(eYR < sYR): 
            print("Your input range year incorrect!!")
        else:
            if(eYR == ""):
                eYR = sYR
            YEAR = [*range(sYR,eYR+1)]
            self.GB_list_file_process = self.GB_list_file_process[self.GB_list_file_process['year'].isin(YEAR)]

class netcdf2dataframe(mainDefind):
    def __init__(self):
        mainDefind.__init__(self)

    def bias_correction_run(self,fobs,fhist,fproj,clivar,adjust_method,N,kind):
        obsh = xr.open_dataset(fobs)
        simh = xr.open_dataset(fhist)
        if(fhist == fproj):
            simp = simh
        else:
            simp = xr.open_dataset(fproj)
        
        simh = simh.assign_coords(time=pd.DatetimeIndex(simh.time).normalize())
        simp = simp.assign_coords(time=pd.DatetimeIndex(simp.time).normalize())
    
        if((adjust_method == "quantile_mapping") or (adjust_method == "quantile_delta_mapping")):
            result = adjust(
                method=adjust_method,
                obs=obsh[clivar],
                simh=simh[clivar],
                simp=simp[clivar],
                n_quantiles=N,
                kind=kind
            )
        else:
            N = f"time.{N}"
            print(N)
            result = adjust(
                method=adjust_method,
                obs=obsh[clivar],
                simh=simh[clivar],
                simp=simp[clivar],
                group=f"{N}",
                kind=kind
            )
        obsh.close()
        simh.close()
        simp.close()
        return result
        
    
    def bias_correction_prepare(self):
        self._mkdir("input_preprocess/file_merge/")
        #filtered_columns = [col for col in self.GB_list_file_process.columns if 'year' not in col and 'month' not in col]
        #filtered_columns = ["pr","tasmax","tasmin"]

        filtered_columns = [self.GB_climate_variable_name["rain"],
                            self.GB_climate_variable_name["tmean"],  
                            self.GB_climate_variable_name["tmax"],
                            self.GB_climate_variable_name["tmin"] ]
        for clivar in filtered_columns:
            filenames =  f"input_preprocess/{clivar}/" + self.GB_list_file_process[clivar]
            
            ds = xr.open_mfdataset(filenames, combine='by_coords')
            sc = self.GB_model_detail["scenario"]
            output_path = f'input_preprocess/file_merge/rcm_{clivar}_{sc}.nc'
            ds.to_netcdf(output_path)
            ds.close()

            if(self.GB_experiment == "historical"):
                obs_var = clivar
                # OBS_{obs_var}_{in_year}_{formatted_month}.nc
                obs_path = f"input_preprocess/obs/{obs_var}/"
                pattern = os.path.join(obs_path, '*.nc')
                all_files = glob.glob(pattern)
    
                ds = xr.open_mfdataset(all_files, combine='by_coords')
                output_path = f'input_preprocess/file_merge/obs_{obs_var}.nc'
                ds.to_netcdf(output_path)
                ds.close()
            
            sc = self.GB_model_detail["scenario"]

            
            folder_path = f"input_preprocess/{clivar}/"

            try:
                # Remove the folder and all its contents
                shutil.rmtree(folder_path)
                #print(f"Folder '{folder_path}' and all its contents have been removed.")
            except Exception as e:
                print(f"Error: {e}")

    def split_netcdf_by_year_month(input_file, output_folder,clivar):
        # เปิดไฟล์ NetCDF หลัก
        ds = xr.open_dataset(input_file)
        
        # ตรวจสอบว่ามีตัวแปรเวลา
        if 'time' not in ds:
            raise ValueError("The dataset does not contain a 'time' coordinate.")
        
        # แปลงพิกัดเวลาเป็น datetime
        ds['time'] = pd.to_datetime(ds['time'].values)
        
        # แปลงพิกัดเวลาเป็น Series ของ pandas
        time_series = pd.Series(ds['time'].values)
        
        # หาค่าที่ไม่ซ้ำกันของปีและเดือน
        years = time_series.dt.year.unique()
        months = time_series.dt.month.unique()
        
        # สร้างโฟลเดอร์สำหรับการบันทึกไฟล์ใหม่
        if not os.path.exists(f"{output_folder}/{clivar}/"):
            os.makedirs(f"{output_folder}/{clivar}/")
        
        # แยกข้อมูลตามปีและเดือน
        for year in years:
            for month in range(1, 13):  # เดือน 1 ถึง 12
                # สร้าง mask สำหรับปีและเดือนที่ต้องการ
                time_mask = (ds['time'].dt.year == year) & (ds['time'].dt.month == month)
                
                if time_mask.any():  # ตรวจสอบว่ามีข้อมูลหรือไม่
                    # เลือกข้อมูลตาม mask
                    ds_subset = ds.sel(time=time_mask)
                    
                    # สร้างชื่อไฟล์สำหรับปีและเดือน
                    output_file = os.path.join(f"{output_folder}/{clivar}/", f"{clivar}_{year}_{month:02d}.nc")
                    
                    # บันทึกข้อมูลที่เลือกลงในไฟล์ NetCDF ใหม่
                    ds_subset.to_netcdf(output_file)
                    #print(f"Saved {output_file}")       
            
            #f_obs  = f_obs
    def dataframe_format(self):
        
        if self.GB_OBS_bias_status == True:
            print("Bias Correction !!")
            netcdf2dataframe.bias_correction_prepare(self)
            filtered_columns = [self.GB_climate_variable_name["rain"],
                            self.GB_climate_variable_name["tmean"],  
                            self.GB_climate_variable_name["tmax"],
                            self.GB_climate_variable_name["tmin"]]

            
            for clivar in filtered_columns:
                #clivar = "pr"
                fobs  = f'input_preprocess/file_merge/obs_{clivar}.nc'
                fhist = f'input_preprocess/file_merge/rcm_{clivar}_historical.nc'

                
                if(self.GB_experiment == "historical"):
                    fproj = f'input_preprocess/file_merge/rcm_{clivar}_historical.nc'
                else:
                    sc = self.GB_model_detail["scenario"]
                    fproj = f'input_preprocess/file_merge/rcm_{clivar}_{sc}.nc'

                
                adjust_method = self.GB_OBS_bias_option['bias_method']
                N = self.GB_OBS_bias_option['n_quantiles']
                kind = self.GB_OBS_bias_option['kind']
                
                if((adjust_method == "quantile_mapping") or (adjust_method == "quantile_delta_mapping")):
                    N = self.GB_OBS_bias_option['n_quantiles']
                else:
                    N = self.GB_OBS_bias_option['group']
                    
                data_bias = netcdf2dataframe.bias_correction_run(self,fobs,fhist,fproj,clivar,adjust_method,N,kind)
                
                output_path = f'input_preprocess/bias_{clivar}_historical.nc'
                data_bias.to_netcdf(output_path)
                data_bias.close()
                input_file = output_path
                output_folder = f'input_preprocess/'
                netcdf2dataframe.split_netcdf_by_year_month(input_file, output_folder,clivar)
            self.GB_list_file_process = pd.DataFrame()
            
            for var_name in self.GB_climate_variable_name.values():
                print(f"get file {var_name}")
                glob_arg        = "input_preprocess/" + var_name + "/"
                sYR = self.startYear
                eYR = self.endYear
                lfile = getFileProcess.get_files_sorted_by_date_range(self,glob_arg, sYR, eYR, var_name,True)
                
                df = pd.DataFrame(lfile, columns=[var_name, 'year', 'month'])
                df = df[['year', 'month',var_name]]
                if(len(self.GB_list_file_process) == 0 ):
                    self.GB_list_file_process = df
                else:
                    self.GB_list_file_process = pd.merge(self.GB_list_file_process, df, on=['year', 'month'], how='outer')
            self.display(self.GB_list_file_process)
            netcdf2dataframe.query_data2dataframe(self)
        else:
            netcdf2dataframe.query_data2dataframe(self)
        
        # Insert TAV and TAMP
        if (self.experiment_process == "historical" and self.time_in_file == False) :
            Pout = self.GB_model_detail['scenario'] +"_"+str(self.startYear)+"-"+str(self.endYear)+"/"
            folder = self.GB_output_path + Pout
            nYear       = self.endYear - self.startYear + 1
            nYear_str   = "%02d" %(nYear)
            start_year  = str(self.startYear)[2:4]
            self.display(" -- Insert TAV and TAMP")
            
            table_t = { "INSI" : self.TAV.keys(), "TAV" : self.TAV.values(), "TAMP":self.TAMP.values()}
            df =  pd.DataFrame(table_t)
            df["TAV"] = [ float(x)/nYear  for x in df["TAV"] ]
            df["TAMP"] = [ float(x)/nYear  for x in df["TAMP"] ]
            #print(df)

            df.to_csv(self.GB_output_path + self.experiment_process+"_TAV_table.csv")

            for k in self.TAV.keys():
                filename = folder + k + start_year + nYear_str+".WTH"
                tav = str(round(float(self.TAV[k]) / nYear,1))
                amp = str(round(float(self.TAMP[k]) / nYear,1))
                netcdf2dataframe.find_and_replace_T(self,filename,tav,amp)

        self.display("Job done.")

    def find_and_replace_T(self,f,tav,amp):
        with open(f,'r') as file:
            filedata = file.read()

        filedata = filedata.replace('tav?',tav)
        filedata = filedata.replace('amp?',amp)

        with open(f,'w') as file:
            file.write(filedata)
    #def query_data2dataframe2(self):
   #     pass:
        
    def query_data2dataframe(self):
        self.GB_input_path = "input_preprocess"
        self.local_data_set = list()                                 # Clean temp data_set
        control_false   = [(False)]        # 1 Jan xxxx (Start file process) (T|F) follow position $(var_name)
        control_true    = [(True)]
        for i in range (1,len(self.GB_climate_variable_name)):
            control_false.append(False)
            control_true.append(True)

        control_true        = tuple(control_true)
        control_false       = tuple(control_false)
        control_first_day   = control_false
        control_next_month  = control_first_day        # for checking next month
        control_end_year    = control_first_day        # for end of year

        last_month  = ""
        for index, row in (self.GB_list_file_process.iterrows()):
            for idx,var_name in enumerate(self.GB_climate_variable_name.values()):
                var_path        = var_name
                df_DATASET = self.xnc_dataset(f"{self.GB_input_path}/{var_name}/{row[var_path]}",self.GB_dropvars)             # xr.open_dataset(row[var_path],drop_variables=self.dropvars)
                #print(df_DATASET)
                temp_df = self.xnc_to_dataframe(df_DATASET)                               # self.GB_DATASET.to_dataframe()

                if (control_first_day[idx] == False) :
                    chk_pass                    = netcdf2dataframe.check_1_jan(self,df_DATASET)
                    list_control_first_day      = list(control_first_day)
                    list_control_first_day[idx] = chk_pass
                    control_first_day           = tuple(list_control_first_day)
                    if (chk_pass):
                        self.local_data_set.append(temp_df)
                        if(isinstance(temp_df.index.get_level_values(self.GB_climate_coordinates_name["time"])[0], cftime._cftime.Datetime360Day)) :
                            last_month = temp_df.index.get_level_values(self.GB_climate_coordinates_name["time"])[0].month
                            self.display("load dataset [%s]: %s/%s"%(var_name,temp_df.index.get_level_values(self.GB_climate_coordinates_name["time"])[0].year,self.endYear))
                        else:
                            last_month = temp_df.index.get_level_values(self.GB_climate_coordinates_name["time"])[0].date().month
                            self.display("load dataset [%s]: %s/%s"%(var_name,temp_df.index.get_level_values(self.GB_climate_coordinates_name["time"])[0].date().year,self.endYear))
                    else:
                        raise FileNotFoundError("Please Files in directory : %s" %(row[var_path]))
                        
                    if(control_first_day == control_true):
                        self.GB_dataset_1y  = netcdf2dataframe.merge_df(self, self.local_data_set)

                        self.local_data_set = list()
                else:
                    if(isinstance(temp_df.index.get_level_values(self.GB_climate_coordinates_name["time"])[0], cftime._cftime.Datetime360Day)) :
                        temp_month  = temp_df.index.get_level_values(self.GB_climate_coordinates_name["time"])[0].month
                    else:
                        temp_month  = temp_df.index.get_level_values(self.GB_climate_coordinates_name["time"])[0].date().month
                    # Check next month
                    if last_month == temp_month - 1:
                        #print(temp_df)
                        self.local_data_set.append(temp_df)
                        #self.display("load dataset [%s]: %s/%s"%(var_name,temp_df.index.get_level_values(self.GB_climate_coordinates_name["time"])[0].date().year,temp_df.index.get_level_values(self.GB_climate_coordinates_name["time"])[0].date().month))
                        list_control_next_month         = list(control_next_month)
                        list_control_next_month[idx]    = True
                        control_next_month              = tuple(list_control_next_month)
                        # Check next month
                        if(control_next_month == control_true):
                            last_month          = temp_month
                            control_next_month  = control_false #(False,False,False,False,False,False,False,False)
                            #self.display("load dataset [%s]: %s/%s"%(var_name,temp_df.index.get_level_values(self.GB_climate_coordinates_name["time"])[0].date().year,temp_df.index.get_level_values(self.GB_climate_coordinates_name["time"])[0].date().month))
                            #self.dataset = self.dataset.append(self.merge_df(),sort=False)
                            self.GB_dataset_1y  = pd.concat([self.GB_dataset_1y,netcdf2dataframe.merge_df(self, self.local_data_set)],sort=False)
                            self.local_data_set = list()

                    # Check End of Year :> goto dataframe_format(self.merge_df())
                    if (control_end_year[idx] == False and temp_month == 12):
                        list_control_end_year       = list(control_end_year)
                        list_control_end_year[idx]  = True
                        control_end_year            = tuple(list_control_end_year)
                    #print(control_end_year)

                    if (control_end_year == control_true):
                        ##print("done.")
                        self.display("Converting ... ")
                        if(isinstance(temp_df.index.get_level_values(self.GB_climate_coordinates_name["time"])[0], cftime._cftime.Datetime360Day)) :
                            this_year = temp_df.index.get_level_values(self.GB_climate_coordinates_name["time"])[0].year
                        else:
                            this_year = temp_df.index.get_level_values(self.GB_climate_coordinates_name["time"])[0].date().year

                        if(self.load_topo == False):
                            # if  self.load_topo    = True  :When Created Shapefile
                            self.display("-- TOPO --")
                            topology_control.load_topology(self,self.GB_dataset_1y)


                        #self.GB_dataset_1y = netcdfCropArea.cropDomain(self, data=self.GB_dataset_1y)
                        sort_lon = self.GB_climate_coordinates_name["lon"]
                        sort_lat = self.GB_climate_coordinates_name["lat"]
                        sort_tim = self.GB_climate_coordinates_name["time"]
                        self.GB_dataset_1y.sort_values(by=[sort_lat,sort_lon,sort_tim], inplace=True, ascending = (False, True, True))

                        self.GB_unique_lat = list(np.unique(self.GB_dataset_1y[self.GB_climate_coordinates_name["lat"]]))
                        self.GB_unique_lon = list(np.unique(self.GB_dataset_1y[self.GB_climate_coordinates_name["lon"]]))
                        self.resolution_raw_lat = abs(self.GB_unique_lat[0] - self.GB_unique_lat[1])
                        self.resolution_raw_lon = abs(self.GB_unique_lon[0] - self.GB_unique_lon[1])
                        lat_data = sorted(self.GB_unique_lat , reverse = True)
                        lon_data = sorted(self.GB_unique_lon , reverse = False)
                        self.GB_dataset_1y["topo"] = np.nan
                        latlon_tmp_convert = list()

                        grid_count = 0
                        for lat_data_1 in lat_data:
                            for lon_data_1 in lon_data:
                                    temp_topo = topology_control.get_topo(self,lat_data_1,lon_data_1)
                                    if(math.isnan(temp_topo) == False):
                                        grids_no = nc2wth.num_of_grid(self,grid_count)
                                        grid_count += 1
                                        latlon_tmp_convert.append([grids_no,lat_data_1,lon_data_1,grids_no,grids_no])


                                    self.GB_dataset_1y.loc[((self.GB_dataset_1y[self.GB_climate_coordinates_name["lon"]] ==  lon_data_1)    \
                                                &(self.GB_dataset_1y[self.GB_climate_coordinates_name["lat"]] ==  lat_data_1)), 'topo'] = temp_topo

                        self.latlon_convert = pd.DataFrame(latlon_tmp_convert, columns=['Grid_NO','Latitude','Longitude','Input_FDD','WSTA'])

                        self.GB_dataset_1y = self.GB_dataset_1y.dropna(subset = ["topo"])
                        
                        if(self.load_topo == False):
                            self.load_topo    = True

                            self._mkdir(self.GB_output_path+"/shp/")

                            # ---------------------------------

                            #Extent
                            #99.2190170288085938,14.0640048980712891 : 100.3504028320312500,15.8129625320434570
                            #lon_L = 99.2190170288086
                            #lon_R = 100.3504028320312
                            #lat_T = 15.812962532043457
                            #lat_B = 14.064004898071289
                            res_lat = (self.resolution_raw_lat/2)
                            res_lon = (self.resolution_raw_lon/2)
                            list_geometry = list()
                            for inx, row in self.latlon_convert.iterrows():
                                #print(row["Longitude"])
                                lon_L = row["Longitude"] - res_lon
                                lon_R = row["Longitude"] + res_lon
                                lat_T = row["Latitude"] + res_lat
                                lat_B = row["Latitude"] - res_lat

                                #print("%s, %s, %s, %s"%(lon_L, lon_R,lat_T, lat_B))
                                grid_polygon = [[lon_L, lat_T], [lon_R, lat_T], [lon_R, lat_B], [lon_L,lat_B], [lon_L, lat_T]]
                                list_geometry.append(Polygon(grid_polygon))

                            self.latlon_convert['geometry'] = list_geometry

                            crs = 'epsg:4326'
                            #gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x=df.lon, y=df.lat), crs = 'EPSG:4326')
                            polygon = gpd.GeoDataFrame(self.latlon_convert, crs=crs, geometry=self.latlon_convert.geometry)

                            polygon.to_file(filename=self.GB_output_path + "/shp" + '/referance_polygon.shp', driver="ESRI Shapefile")

                            #p = self.GB_dataset_1y.loc['1972-01-01']
                            #self.map_plot('xlon','xlat','topo',p)
                            #plt.show()

                        nc2wth.convert2wth(self,this_year)
                        #print(self.tb_historical_global_T)
                        #self.to_dssat()
                        control_end_year    = control_false #(False,False,False,False,False)
                        control_first_day   = control_false #(False,False,False,False,False)
                        self.GB_dataset_1y       = list()
                        self.file_wth_proc.close()
                        self.display(" done.")


    def merge_df(self,local_data_set):
        local_dataset = pd.DataFrame(self.local_data_set[2]) 
        for k, v in enumerate(self.GB_climate_variable_name.values()):
            data_b_convert = self.convert_unit(v,self.local_data_set[k][[v]].to_numpy()) 
            local_dataset[v] = data_b_convert     
        return local_dataset
        
    def check_1_jan(self,df_DATASET):
        temp_df = df_DATASET.to_dataframe()

        # date_index  = temp_df.index.get_level_values(self.GB_climate_coordinates_name["time"])[0]
        date_index = temp_df.index.get_level_values(self.GB_climate_coordinates_name["time"])[0]
        dd=0
        mm=0
        if(isinstance(date_index, cftime._cftime.Datetime360Day)) :
            dd          = date_index.day
            mm          = date_index.month
        else :
            dd          = date_index.date().day
            mm          = date_index.date().month
        if(dd == 1 and mm == 1):        
            return True
        else:
            return False
    
    def check_year_exist(self,year):
        df_year = self.file_process.loc[self.file_process["year"] == year]
        if (len(df_year) == 0):
            print("Year : %s not exist dataset !!"%(year))
            return False
        else:
            print("Year : %s "%(year))
            return True
        
class   netcdfCropArea(mainDefind):
    def __init__(self):
        mainDefind.__init__(self)  
                
    def mk_int(self,s):
        if isinstance(s, str):
            s = s.strip()        
            return int(s) if s else 0
        else:
            return s
    
    def cropDomain(self, data,status="original",topo="") :
        la1, la2, lo1, lo2 = self.GB_lonlatbox.values()
        self.display(" -- crop domain")
        
        col_lat_name = self.GB_climate_coordinates_name["lat"]
        col_lon_name = self.GB_climate_coordinates_name["lon"]
        arr_LAT      = data[col_lat_name].to_numpy()
        arr_LON      = data[col_lon_name].to_numpy()   
        
        if(status == "original"):  
            col_lat_name = self.GB_climate_coordinates_name["lat"]
            col_lon_name = self.GB_climate_coordinates_name["lon"]
            
        elif(status == "topo"): # global coordinate
            col_lat_name = "lat"
            col_lon_name = "lon"
            data = topo
        arr_LAT_max  = np.amax(arr_LAT)
        arr_LAT_min  = np.amin(arr_LAT)
        arr_LON_max  = np.amax(arr_LON)
        arr_LON_min  = np.amin(arr_LON)   
        
        if(la1 == None or la2 == None or lo1 == None or lo2 == None):
            la1         = float(arr_LAT_min)
            la2         = float(arr_LAT_max)
            lo1         = float(arr_LON_min)
            lo2         = float(arr_LON_max)
        else:
            la1         = float(self.GB_lonlatbox["lat_min"])
            la2         = float(self.GB_lonlatbox["lat_max"])
            lo1         = float(self.GB_lonlatbox["lon_min"])
            lo2         = float(self.GB_lonlatbox["lon_max"])
        
        if(la1 >= -90 and la1 <= 90 and la2 >= -90 and la2 <= 90 and la2 >= la1):
            if(lo1 >= -180 and lo1 <= 180 and lo2 >= -180 and lo2 <= 180 and lo2 >= lo1 ):
                if(la1 < arr_LAT_min or la1 > arr_LAT_max): 
                    raise ValueError("!!! LAT1 out of area : %s Latitude(min, max):(%s, %s)" %(la1,arr_LAT_min,arr_LAT_max))
                    
                elif(la2 < arr_LAT_min or la2 > arr_LAT_max): 
                    raise ValueError("!!! LAT2 out of area : %s Latitude(min, max):(%s, %s)" %(la2,arr_LAT_min,arr_LAT_max))
                   
                elif(lo1 < arr_LON_min or lo1 > arr_LON_max): 
                    raise ValueError("!!! LON1 out of area : %s Longitude(min, max):(%s, %s)" %(lo1,arr_LON_min,arr_LON_max))
                    
                elif(lo2 < arr_LON_min or lo2 > arr_LON_max):
                    raise ValueError("!!! LON2 out of area : %s Longitude(min, max):(%s, %s)" %(lo2,arr_LON_min,arr_LON_max))
                  
                else:
                    self.display("-- select Area")
                    self.display("la1=%s la2=%s, lo1=%s, lo2=%s" %(la1,la2,lo1,lo2))
                    
                    data = data.loc[(data[col_lat_name] >= la1) \
                        &(data[col_lat_name] <= la2) \
                            &(data[col_lon_name] >= lo1) \
                                & (data[col_lon_name] <= lo2)]
                    
                    if (len(data) == 0):
                        print("Please extend cooradinate becuse selected grid = 0")
                    else:
                        return data
            else:
                print("!!!")
                print("LON1 : %s" %(lo1))
                print("LON2 : %s" %(lo2))
                print("* Please check range of longitude in degrees is -180 and 180 and must be LOT1 <= LON2")
                print("** Longitude(min,max) = (%s,%s)"%(arr_LON_min,arr_LON_max))
                print("!!!")
                raise ValueError()
        else:
                print("!!!")
                print("LAT1 : %s" %(la1))
                print("LAT2 : %s" %(la2))
                print("* Please check range of latitude in degrees is -90 and 90 and must be LAT1 <= LAT2")
                print("** Latitude(min,max) = (%s,%s)"%(arr_LAT_min,arr_LAT_max))
                print("!!!")
                raise ValueError()

class nc2wth(mainDefind):
    def __init__(self):
        mainDefind.__init__(self)
        
        
    def decToHex(self,decimal):
        conversion_table = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
                        5: '5', 6: '6', 7: '7',
                        8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C',
                        13: 'D', 14: 'E', 15: 'F'}
        hexadecimal = ''
        while(decimal > 0):
            remainder   = decimal % 16
            hexadecimal = conversion_table[remainder] + hexadecimal
            decimal     = decimal // 16
        format_hex = hexadecimal
        for p in range(0,4 - len(hexadecimal)):         # 4 positions
            format_hex = "0" + format_hex
        return format_hex
    
    def num_of_grid(self, inx):
        # 2D 
        #num_of_grid(self, lat_data, lon_data, lat_data_1, lon_data_1)
        #nlon        = len(lon_data)
        #clat        = lat_data.index(lat_data_1)
        #clon        = lon_data.index(lon_data_1)
        #limit_row   = (nlon * clat)   

        return nc2wth.decToHex(self,inx + 1)          # GET gridnumber Ex. 010F | 0001 
    
    def wth_header(self,ngrid):
        fl = "$WEATHER DATA: "+ngrid+", Grid "+ngrid+" DATA, ("+self.GB_model_detail["model-simulation"]+":"+self.GB_model_detail["model-name"]+","+self.GB_model_detail["scenario"]+")"
        
        fl += "\n"
        fl += "\n"
        fl += "! T2M  : Temperature at 2 Meters (C)  \n"
        fl += "! TMIN : Temperature at 2 Meters Minimum (C) \n"
        fl += "! TMAX : Temperature at 2 Meters Maximum (C) \n"
        fl += "! TDEW : Dew/Frost Point at 2 Meters (C) \n"
        fl += "! RAIN : Precipitation Corrected (mm/day) \n" 
        fl += "! RH2M : Relative Humidity at 2 Meters (%) \n"
        fl += "! WIND : Wind Speed at 2 Meters (m/s) \n"
        fl += "! SRAD : All Sky Surface Shortwave Downward Irradiance (MJ/m^2/day)\n"
        
        self.display(fl)
        self.file_wth_proc.write(fl+"\n")
        
    def secondLine(self,IN, SI, lat, lon, elev, tav, amp, tmht, wndht) :
        header   = nc2wth.splitType(self,"@ INSI", 5)
        header  += nc2wth.splitType(self,self.blank, 1) + nc2wth.splitType(self,"LAT", 8)
        header  += nc2wth.splitType(self,self.blank, 1) + nc2wth.splitType(self,"LONG", 8)
        header  += nc2wth.splitType(self,self.blank, 1) + nc2wth.splitType(self,"ELEV", 5)
        header  += nc2wth.splitType(self,self.blank, 1) + nc2wth.splitType(self,"TAV", 5)
        header  += nc2wth.splitType(self,self.blank, 1) + nc2wth.splitType(self,"AMP", 5)
        header  += nc2wth.splitType(self,self.blank, 1) + nc2wth.splitType(self,"REFHT", 5)
        header  += nc2wth.splitType(self,self.blank, 1) + nc2wth.splitType(self,"WNDHT", 5)
        header  += "\n"
        strLine  = header
        strLine += nc2wth.splitType(self,IN, "IN")
        strLine += nc2wth.splitType(self,SI, "SI")
        strLine += nc2wth.splitType(self,lat, "LAT")
        strLine += nc2wth.splitType(self,lon, "LONG")
        strLine += nc2wth.splitType(self,elev, "ELEV")
        strLine += nc2wth.splitType(self,tav, "TAV")
        strLine += nc2wth.splitType(self,amp, "AMP")
        strLine += nc2wth.splitType(self,tmht, "TMHT")
        strLine += nc2wth.splitType(self,wndht, "WMHT")
        strLine += "\n"
        self.display(strLine)
        self.file_wth_proc.write(strLine+"\n")  
        
    def splitType(self,value, types) :
        t = types
        if (t == "IN") :
            return nc2wth.formatSet(self,2,'C',2,0,value)
        elif (t == "SI") :
            return nc2wth.formatSet(self,0,'C',2,0,value)
        elif (t == "LAT") or (t == "LONG") :
            return nc2wth.formatSet(self,1,'R',8,3,value)
        elif (t == "ELEV") :
            return nc2wth.formatSet(self,1,'R',5,0,value) 
        elif (t == "TAV") or (t == "AMP") or (t == "TMHT") or \
            (t == "WMHT") or (t == "SRAD") or (t == "TMAX") or \
            (t == "TMIN") or (t == "RAIN") or (t == "TDEW") or \
            (t == "PAR") or (t == "WIND") :
            if isinstance(t, str) :
                return nc2wth.formatSet(self,1,'C',5,1,value)
            else:
                return nc2wth.formatSet(self,1,'R',5,1,value)
        elif (t == "DATE") :
            return nc2wth.formatSet(self,0,'I',5,0,value)
        elif (t >= 0) and (t <= 9) :
            t = t - len(value) + 1
            for i in range(1, t) :
                value = self.blank + value
            return value
        else :
            return self.blank
        
    def formatSet(self, space, types, quantity, decimal, value) :
        f       = ""
        decimal = '.'+str(decimal)+'f'
        #print str(v) + " :: " + str(type(v)) 
        if (types == 'R') and (value != -99 ) :
            f = f + str(format(value, decimal))
        else :
            f = value
        l = len(str(f))
        space += int(quantity) - l + 1 #other space 
        #print value
        for i in range(1, space) :
            f = self.blank + str(f)            
        return f
    
    def num_of_row_df1grid(self,inx):
        #dd=0;mm=0;yy=0;YRD="";f_yy="";d_of_y=""

        if(isinstance(inx, cftime._cftime.Datetime360Day)) :
            dd              = inx.day 
            mm              = inx.month
            yy              = inx.year
            f_yy            = str(yy)[0:4]
            d_of_Y          = "%03d" %(inx.dayofyr)
            YRD             = str(f_yy) + str(d_of_Y)
            return YRD
        else:
            dd              = inx.date().day
            mm              = inx.date().month
            yy              = inx.date().year
            f_yy            = str(yy)[0:4]#f_yy            = str(yy)[2:4]
            d_of_Y          = "%03d" %(date(yy,mm,dd).timetuple().tm_yday)      # select date of year
            YRD             = str(f_yy) + str(d_of_Y)  
        
            return YRD
    
    
    def format_str_value(self,vv):
        l = [ ("%05s"%(a)) for a in list((np.around(np.array(list(pd.to_numeric(vv))),1)))]
        return l
    
    def clean_data_in_new_folder(self, folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e)) 

    def avg_Tmaxmin(self,Tmax,Tmin) :
        
        #print(Tmax.index.get_level_values(self.GB_climate_coordinates_name["time"])[0].month)
        Tmax_chk=0;Tmin_chk=0
        #print(Tmax)
        if(isinstance(Tmax.index.get_level_values(self.GB_climate_coordinates_name["time"])[0], cftime._cftime.Datetime360Day)) : 
            Tmax_chk = [i.month for i in  Tmax.index.get_level_values(self.GB_climate_coordinates_name["time"])]
            Tmin_chk = [i.month for i in  Tmin.index.get_level_values(self.GB_climate_coordinates_name["time"])]
        else:
            Tmax_chk = [i.date().month for i in  Tmax.index.get_level_values(self.GB_climate_coordinates_name["time"])]
            Tmin_chk = [i.date().month for i in  Tmin.index.get_level_values(self.GB_climate_coordinates_name["time"])]
        
        localTmax = Tmax.to_frame().reset_index(drop=True) 
        localTmin = Tmin.to_frame().reset_index(drop=True)
        
        localTmax["chk"] = Tmax_chk
        localTmin["chk"] = Tmin_chk
    
        #monthly_mx = list()
        #monthly_mn = list()
        monthly_mm = list()
        #print(np.max((localTmax[localTmax["chk"] == 2][self.GB_climate_variable_name["tmax"]]).to_numpy()))

        for mm in range (1,13):
            #print(Tmax[Tmax["chk"] == mm][self.GB_climate_variable_name["tmax"]])
            #monthly_mx.append(np.max((localTmax[localTmax["chk"] == mm][self.GB_climate_variable_name["tmax"]]).to_numpy()))
            #monthly_mn.append(np.min((localTmin[localTmin["chk"] == mm][self.GB_climate_variable_name["tmin"]]).to_numpy()))
            
            monthly_mm.append( np.max((localTmax[localTmax["chk"] == mm][self.GB_climate_variable_name["tmax"]])) - np.min((localTmin[localTmin["chk"] == mm][self.GB_climate_variable_name["tmin"]])))
        #Tmax = np.mean(monthly_mx)
        #Tmin = np.mean(monthly_mn)
        Tmaxmin = np.mean(monthly_mm)
        return Tmaxmin
        #return Tmax - Tmin

    def convert2wth(self,this_year):
        # --- setting directory and file output ----
        Pout = self.GB_model_detail['scenario'] +"_"+str(self.startYear)+"-"+str(self.endYear)+"/"
        folder = self.GB_output_path + Pout
        self._mkdir(self.GB_output_path + Pout)
        
        if self.check_new_process == False :
            self.check_new_process = True
            nc2wth.clean_data_in_new_folder(self,folder)
       
        file_wth_name=""
        # --- END setting
        self.display(self.GB_dataset_1y) 
        #self.GB_dataset_1y.to_csv("tmp.csv")       
        
        for inx, latlon_convert in tqdm(self.latlon_convert.iterrows(),total=(len(self.latlon_convert))):
            lat_data_1 = latlon_convert["Latitude"]
            lon_data_1 = latlon_convert["Longitude"]
            
            df1grid = self.GB_dataset_1y[(self.GB_dataset_1y[self.GB_climate_coordinates_name["lat"]] == lat_data_1) \
            & (self.GB_dataset_1y[self.GB_climate_coordinates_name["lon"]] == lon_data_1)]
            
            ngrid           = latlon_convert["Grid_NO"]
            IN              = ngrid[0:2]
            SI              = ngrid[2:4]
            elev            = float(np.unique(df1grid["topo"]))            # Elevation
            #print("lat = %s, lon = %s, elev = %s" %(lat,lon,elev))
            # TAMP 
            # T_max_year            = Monthly maximum average : [ max(Jan), max(Feb), ... , max(Dec) ] / 12
            # T_min_year            = Monthly minimum average : [ min(Jan), min(Feb), ... , min(Dec) ] / 12
            # average_max_min_year  = (T_max_year + T_min_year) / 2 
            # TAMP                  = sum(average_max_min_year) / historical long-term
            mock_tav = "tav?"
            mock_amp = "amp?"
            
            if (self.experiment_process == "historical"):
                T_mean_avg_year = np.mean(df1grid[self.GB_climate_variable_name["tmean"]].to_numpy())        # annual average of T_mean daily
                T_max           = df1grid[self.GB_climate_variable_name["tmax"]]
                T_min           = df1grid[self.GB_climate_variable_name["tmin"]]
                amp = nc2wth.avg_Tmaxmin(self,T_max,T_min)

                grid_year = ngrid # + str(self.startYear)[2:4]
            
                if grid_year in self.TAV: 
                    self.TAV[grid_year] += T_mean_avg_year
                    self.TAMP[grid_year] += amp
                else:
                    self.TAV[grid_year] = T_mean_avg_year
                    self.TAMP[grid_year] = amp
            elif (self.experiment_process == "projection"):
                print( self.tb_historical_global_T)
                print(ngrid)
                INSI_historical = self.tb_historical_global_T[self.tb_historical_global_T['INSI'] == ngrid]
                mock_tav = str(round(float(INSI_historical["TAV"]),1))
                mock_amp = str(round(float(INSI_historical["TAMP"]),1))
            
             # amp             = T_max_avg_year - T_min_avg_year     

            tmht            = 2.0                                # m
            wndht           = 2.0                                # m
            
            inx_df1grid     = [nc2wth.num_of_row_df1grid(self,i) for i in  df1grid.index.get_level_values(self.GB_climate_coordinates_name["time"])] 
            
            # Solve : Blank space first
            # df_test_test = pd.DataFrame(inx_df1grid)
            # string_repr = df_test_test.to_string(index=False,line_width=1000000)
            # string_repr = '\n'.join([line.lstrip() for line in string_repr.split('\n')])
            # print(string_repr)
            # with open("test.WTH", 'w') as f:
            #    f.write(string_repr)
            
            subfolder = self.GB_output_path + Pout
            
            post_name = "01"
            if self.time_in_file == False:
                post_name = "%02d" %(self.endYear - self.startYear + 1)
                self.file_wth_name = ngrid + str(self.startYear)[2:4] + post_name + ".WTH"
            
            else:
                subfolder = subfolder + str(this_year) +"/"
                self._mkdir(subfolder)
                self.file_wth_name = ngrid + str(this_year)[2:4] + post_name + ".WTH"
                
                T_mean_avg_year = np.mean(df1grid[self.GB_climate_variable_name["tmean"]].to_numpy())        # annual average of T_mean daily
                T_max_avg_year  = np.mean(df1grid[self.GB_climate_variable_name["tmax"]].to_numpy())         # annual average of T_max daily
                T_min_avg_year  = np.mean(df1grid[self.GB_climate_variable_name["tmin"]].to_numpy())  
                
                mock_tav = str(round(T_mean_avg_year,1))
                mock_amp = str(round((T_max_avg_year - T_min_avg_year),1))

            
            location_wth_file = subfolder + "/" +self.file_wth_name 
            isExist = os.path.exists(location_wth_file)
            
            file_action = "w"
            
            if(isExist == True):
                file_action = "a"
                self.written_header = False
            else:
                file_action = "w"
                self.written_header = True
            
            self.file_wth_proc = open(location_wth_file, file_action) 
            
            self.display(self.file_wth_name)
            
            local_tmean     = nc2wth.format_str_value(self,df1grid[self.GB_climate_variable_name["tmean"]])
            local_rain      = nc2wth.format_str_value(self,df1grid[self.GB_climate_variable_name["rain"]])
            local_tmax      = nc2wth.format_str_value(self,df1grid[self.GB_climate_variable_name["tmax"]])
            local_tmin      = nc2wth.format_str_value(self,df1grid[self.GB_climate_variable_name["tmin"]])
            local_rad       = nc2wth.format_str_value(self,df1grid[self.GB_climate_variable_name["rad"]])
            
            # **** local_pa        = df1grid[self.GB_climate_variable_name["pa"]]
            local_rh        = df1grid[self.GB_climate_variable_name["rh"]]
            local_wind      = nc2wth.format_str_value(self,df1grid[self.GB_climate_variable_name["wind"]])
        
            
            #AVP = [(6.11 * pow(10,(7.5*float(T))/(237.3+float(T))))*0.1 for T in local_tmean] * local_rh
            # **** AVP = nc2wth.format_str_value(self,local_pa)            
            RH2M = nc2wth.format_str_value(self,local_rh*100)  
            # if <= 0 ; math.exp(46.494 - ( 6545.8/(float(T)+278)))/pow(float(T)+868,2)/1000
            
            ps = [(math.exp(34.494 - ( 4924.99/(float(T)+237.1)))/pow(float(T)+105,1.57)/1000) for T in local_tmean] * local_rh
            
            T_dew = nc2wth.format_str_value(self,[(116.91+237.3*math.log(float(T)))/(16.78-math.log(float(T))) for T in ps]) 
            
            #=EXP(34.494 - ( 4924.99/(T+237.1)))/(T+105)^1.57
            
            #[ print(x) for x in inx_df1grid]
            
            df_wth_data = pd.DataFrame(
                {"@  DATE":inx_df1grid,
                "T2M" :local_tmean,  
                "TMAX":local_tmax,
                "TMIN":local_tmin,
                "TDEW":T_dew,
                "RH2M":RH2M,
                "RAIN":local_rain,
                "WIND":local_wind,
                "SRAD":local_rad
                #"AVP" :AVP                
                })
            self.display(df_wth_data)
             
            if(self.written_header == True):          
                nc2wth.wth_header(self,ngrid)
                nc2wth.secondLine(self,IN, SI, lat_data_1, lon_data_1, elev, mock_tav, mock_amp, tmht, wndht)  # (v?) = TAV, (p?) = TAMP
                
                
            self.file_wth_proc.close()


            with open(location_wth_file, 'a') as f:

                if(self.written_header ==  True) :
                    string_repr = df_wth_data.to_string(index=False,header=True)
                else:
                    f.write("\n")
                    string_repr = df_wth_data.to_string(index=False,header=False)
                    if(self.time_in_file == True):
                        self.written_header = True
                    else:
                        self.written_header = False
                
                string_repr = '\n'.join([line.lstrip() for line in string_repr.split('\n')])
                f.write(string_repr)
                
                    
            self.file_wth_proc.close()
        
class topology_control(mainDefind):
    def __init__(self):
        mainDefind.__init__(self)
        
    def load_topology(self,climate_data):
        lib_path             = inspect.getfile(WTH)
        topo_file            = os.path.dirname(os.path.abspath(lib_path)) + "/topology/topo_land.nc"
        #self.GB_data_topo    = self.xnc_dataset(topo_file,self.GB_dropvars).to_dataframe()       
        #self.GB_data_topo.reset_index(inplace=True)
        
        rcm = xr.open_dataset(topo_file) 
        print(rcm)
        lat1, lat2, lon1, lon2 = self.GB_lonlatbox.values()
        rcm_lat_name = "lat"
        rcm_lon_name = "lon"
        lat_bnds = [lat1, lat2]
        lon_bnds = [lon1, lon2]
        lat_mask = (rcm[rcm_lat_name] >= lat_bnds[0]) & (rcm[rcm_lat_name] <= lat_bnds[1])
        lon_mask = (rcm[rcm_lon_name] >= lon_bnds[0]) & (rcm[rcm_lon_name] <= lon_bnds[1])
        mask = lat_mask & lon_mask
        
        rcm_cropped = rcm.where(mask, drop=True)
        rcm_cropped.to_netcdf(f"topo.nc")
        
        self.GB_data_topo    = self.xnc_dataset("topo.nc",self.GB_dropvars).to_dataframe()       
        self.GB_data_topo.reset_index(inplace=True)
        
        self.display(self.GB_data_topo)
        #self.GB_data_topo    = netcdfCropArea.cropDomain(self,status="topo", topo=self.GB_data_topo,data=climate_data).dropna(subset = ["topo"])    
        #self.map_plot('lon','lat','topo',self.GB_data_topo)
        
    def get_topo(self,lat,lon):
        
        idx_lat = topology_control.nearestIDX(self,self.GB_data_topo["lat"], lat)
        idx_lon = topology_control.nearestIDX(self,self.GB_data_topo["lon"], lon)
        return_topo = self.GB_data_topo.loc[\
                    (self.GB_data_topo["lat"] == idx_lat) \
                    &(self.GB_data_topo["lon"] == idx_lon)]
        
        if(len(return_topo["topo"]) == 0):
            return(np.NaN)
        else:
            return(float(return_topo["topo"]))
    
    def nearestIDX(self,array, value):
        array   = np.asarray(array)
        idx     = (np.abs(array - value)).argmin()  
        #self.display(array + " " +array.shape + " " + type(array)+ " " + value)  
        return array[idx]
        
    
class WTH(mainDefind):
    def __init__(self):
        mainDefind.__init__(self)
    # efine_climate_variable("tas","tasmax","tasmin","hurs","pr","sfcWind","rsds")
    def define_climate_variable(self, 
                                tas="",
                                tasmax="",
                                tasmin="",
                                hurs="",
                                pr="",
                                wind="",
                                rsns="",
                                ps=""):
        # 
        if pr       != "" : self.GB_climate_variable_name["rain"]   = pr
        if tas      != "" : self.GB_climate_variable_name["tmean"]  = tas    
        if tasmax   != "" : self.GB_climate_variable_name["tmax"]   = tasmax 
        if tasmin   != "" : self.GB_climate_variable_name["tmin"]   = tasmin 
        if rsns     != "" : self.GB_climate_variable_name["rad"]    = rsns   
        if hurs     != "" : self.GB_climate_variable_name["rh"]     = hurs   
        if ps       != "" : self.GB_climate_variable_name["pa"]     = ps     
        if wind     != "" : self.GB_climate_variable_name["wind"]   = wind
        
    
    def define_coordinates(self, x="lat",y="lon",t='time'):
        self.GB_climate_coordinates_name = {"lat":x, "lon":y, "time":t}

    def define_input_path(self, input_path=""):
        self.GB_input_path = input_path
    
    def define_domain(self, la1=0, la2=0, lo1=0, lo2=0):
        self.GB_lonlatbox = {"lat_min":la2, "lat_max":la1, "lon_min":lo1, "lon_max":lo2}

    def define_description(self, rcm_ver="", model="", scenario="" ):
        self.GB_model_detail = {"model-simulation":rcm_ver,"model-name":model,"scenario":scenario}
        
    def define_reanalysis_path(self, obs_path=""):
        self.GB_OBS_path = obs_path

    def define_reanalysis_coordinates(self, lat = "latitude", lon = "longitude"):
        self.GB_OBS_coordinates_name = {"lat":lat, "lon":lon, "time":"time"}

    def define_reanalysis_variable(self, tas="t2m", tmax="tmax", tmin="tmin", pr="tp"):
        self.GB_OBS_variables_name = {"tas":tas, "tmax":tmax, "tmin":tmin,"pr":pr}

    def bias_correction(self, bias_method = "quantile_mapping", n_quantiles=250, group="", kind="+"):
        
        self.GB_OBS_bias_status = True
        self.GB_OBS_bias_option =  {"bias_method":bias_method, 
                                    "n_quantiles":n_quantiles, 
                                    "group":group,
                                    "kind":kind} 
    def topo(self):
        topology_control.load_topology(self)  

    #ncObj.convert2dssat(output_path="DSSAT_output/",period="projection",start_year=2030,end_year=2060,single_year_file = False)
    def convert2dssat(self,
                      output_path="",
                      period="", 
                      start_year="",
                      end_year="",
                      single_year_file = False,
                      ):
        
        self.GB_output_path = output_path+"/"+self.GB_model_detail['model-name']+"/"
        
        experiment_check=["historical","projection","force"]
        experiment = period
        self.GB_experiment = experiment
        if experiment not in experiment_check :
            raise ValueError("Experiment input Error!!")
            
        else:
            self.experiment_process = experiment

        if experiment == "projection" :
            self.tb_historical_global_T = pd.read_csv(self.GB_output_path + "historical_TAV_table.csv",dtype={'INSI' : str})
            print("check file table Tav (histotical first)")
        
        t1_start = process_time() 
        self._mkdir(self.GB_output_path)
        self.time_in_file   = single_year_file    
        self.startYear      = start_year
        self.endYear        = end_year
        self.GB_startYear      = start_year
        self.GB_endYear        = end_year

        self.display(print("Start"))
        main_folder_path = "input_preprocess"
        excluded_folders = ["file_merge"]
        
        try:
            # Iterate through the contents of the main folder
            for item in os.listdir(main_folder_path):
                item_path = os.path.join(main_folder_path, item)
                if item not in excluded_folders:
                    if os.path.isdir(item_path):
                        # Remove the folder and its contents
                        shutil.rmtree(item_path)
                    else:
                        # Remove files directly
                        os.remove(item_path)
            #print(f"Selected contents of '{main_folder_path}' have been removed, excluding {excluded_folders}.")
        except Exception as e:
            print(f"Error: {e}")
            
        
            
        getFileProcess.get_file_list(self,start_year,end_year) 
        #print(self.GB_list_file_process)
        
        
        netcdf2dataframe.dataframe_format(self)
        
        t1_stop = process_time()
        #print("\n\nElapsed time:", t1_stop, t1_start) 
        print("\nProcessed in seconds:",t1_stop-t1_start) 
        
    
        
