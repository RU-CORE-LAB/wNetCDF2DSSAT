# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Start up @ Aug 2022
# _wNetCDF2DSSAT.py Version 1.3
# Written by Nick Ratchanan (RU-CORE).
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::

from datetime import datetime, date ,timedelta
from operator import index
import xarray as xr
import numpy  as np
import pandas as pd
import os
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
import fiona

### CLASS :: Initial Global variables and Method
class mainDefind:
    # initial variabled
    def __init__(self):
        self.GB_input_path                = ""
        self.GB_output_path               = ""
        self.GB_climate_variable_name     = {"rain" : "pr","tmean" : "tas", "tmax":"tasmax","tmin":"tasmin","rad" : "rsns"}
        self.GB_climate_coordinates_name    = {"lat":"lat", "lon":"lon", "time":"time"}
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
        
        self.GB_data_topo                 = pd.DataFrame()
        self.GB_lat_topo                  = []
        self.GB_lon_topo                  = []
        self.load_topo                    = False
        self.GB_grids_no                  = []
        self.latlon_convert               = pd.DataFrame()
                        
    # Display for Debug        
    def display(self,obj , mode="0"):
        if  (self.debug_level == "1"):
            print(obj)
    
    def check_file_exist(self, f):
        self.display("File exists: " + f + ": " + str(path.isfile(f)))
        return path.isfile(f)
    
    def xnc_dataset(self,file,drop_var=""):
        try:
            return xr.open_dataset(file,drop_variables=drop_var)
        except:
            print("Incomplete file: %s"%file)
            exit()
        
    def xnc_to_dataframe(self,nco):
        return nco.to_dataframe()               # xarray open_dataset to dataframe

    def convert_unit(self,v,data):
        MJ       = 0.0108           # 3hr : 3*60*60/10^6 for radiation
        K        = 273.15           # Kelvin (1 celsius)
        Pr_daily = 60*60*24         # daily precipitation (kg m-2 s-1 to mm)
        
        v = list(self.GB_climate_variable_name.values()).index(v)
        v = list(self.GB_climate_variable_name.keys())[v]
        unit ={
                'rain'  : data * Pr_daily,      
                'tmean' : data - K,     
                'tmax'  : data - K,       
                'tmin'  : data - K,
                'rad'   : data * MJ   
            }
        return unit.get(v, data)
    
    def _mkdir(self,dir):
        try: 
            os.makedirs(dir, exist_ok=True) 
        except OSError as error: 
            pass 
    
    def ncdump(self,f):
        nco = self.xnc_dataset(f)
        var_name = nco.data_vars
        dimen = nco.coords
        
        print (dimen)
        print (var_name)
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
        
    # Create table path file
    def get_file_list(self,sYR,eYR):
        head_name = ["year","month"] + list(self.GB_climate_variable_name.values())
        self.GB_list_file_process = pd.DataFrame(columns=head_name, index=range(0))

        for loop_year in range(sYR,eYR+1):
            for var_name in self.GB_climate_variable_name.values():
                glob_arg        = self.GB_input_path + "/" + var_name + "/" + var_name + "*" + str(loop_year) + "*.nc"
                ls              = sorted(glob.glob(glob_arg))
                #list_files      = {}
                for strName in ls :
                    list_file   = strName.replace("\\", "/")
                    temp_open   = self.xnc_dataset(list_file,self.GB_dropvars).to_dataframe()
                    yy          = temp_open.index[0][0].date().year
                    mm          = temp_open.index[0][0].date().month
                    d = pd.DataFrame(data={'year': [yy], 'month': [mm] , var_name : [list_file]})
                    sel = self.GB_list_file_process.loc[(self.GB_list_file_process["year"] == yy) & (self.GB_list_file_process["month"] == mm)]
                    if (len(sel) == 0):
                        self.GB_list_file_process = pd.concat([self.GB_list_file_process,d], ignore_index=True)
                    else:
                        self.GB_list_file_process.loc[(self.GB_list_file_process["year"] == yy) & (self.GB_list_file_process["month"] == mm),[var_name]] = list_file  
                self.display("Dataset Year %s :%s - %s"%(var_name,np.amin(self.GB_list_file_process[['year']].to_numpy()),np.amax(self.GB_list_file_process[['year']].to_numpy())))
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
        
    def dataframe_format(self):
        netcdf2dataframe.query_data2dataframe(self)
        
    def query_data2dataframe(self):
        self.local_data_set = list()                                 # Clean temp data_set
        control_first_day   = (False,False,False,False,False)        # 1 Jan xxxx (Start file process) (T|F) follow position $(var_name)
        control_next_month  = (False,False,False,False,False)        # for checking next month
        control_end_year    = (False,False,False,False,False)        # for end of year  
        last_month  = ""
        for index, row in (self.GB_list_file_process.iterrows()):
            for idx,var_name in enumerate(self.GB_climate_variable_name.values()):
                var_path        = var_name                 
                df_DATASET = self.xnc_dataset(row[var_path],self.GB_dropvars)             # xr.open_dataset(row[var_path],drop_variables=self.dropvars)
                temp_df = self.xnc_to_dataframe(df_DATASET)                               # self.GB_DATASET.to_dataframe()
                if (control_first_day[idx] == False) :
                    chk_pass                    = netcdf2dataframe.check_1_jan(self,df_DATASET)
                    list_control_first_day      = list(control_first_day)
                    list_control_first_day[idx] = chk_pass
                    control_first_day           = tuple(list_control_first_day)
                    if (chk_pass):
                        self.local_data_set.append(temp_df) 
                        last_month = temp_df.index[0][0].date().month
                        self.display("load dataset [%s]: %s/%s"%(var_name,temp_df.index[0][0].date().year,temp_df.index[0][0].date().month))
                    else:
                        print("Please Files in directory : %s" %(row[var_path]))
                        exit()
                    if(control_first_day == (True,True,True,True,True)):
                        self.GB_dataset_1y  = netcdf2dataframe.merge_df(self, self.local_data_set)
                        
                        self.local_data_set = list()
                else:
                    temp_month  = temp_df.index[0][0].date().month
                    # Check next month
                    if last_month == temp_month - 1:
                        #print(temp_df)
                        self.local_data_set.append(temp_df)
                        self.display("load dataset [%s]: %s/%s"%(var_name,temp_df.index[0][0].date().year,temp_df.index[0][0].date().month))
                        list_control_next_month         = list(control_next_month)
                        list_control_next_month[idx]    = True
                        control_next_month              = tuple(list_control_next_month)
                        # Check next month 
                        if(control_next_month == (True,True,True,True,True)):
                            last_month          = temp_month 
                            control_next_month  = (False,False,False,False,False)  
                            self.display("load dataset [%s]: %s/%s"%(var_name,temp_df.index[0][0].date().year,temp_df.index[0][0].date().month))                          
                            #self.dataset = self.dataset.append(self.merge_df(),sort=False)  
                            self.GB_dataset_1y  = pd.concat([self.GB_dataset_1y,netcdf2dataframe.merge_df(self, self.local_data_set)],sort=False)                         
                            self.local_data_set = list()
                            
                    # Check End of Year :> goto dataframe_format(self.merge_df())
                    if (control_end_year[idx] == False and temp_month == 12):
                        list_control_end_year       = list(control_end_year)
                        list_control_end_year[idx]  = True
                        control_end_year            = tuple(list_control_end_year)
                    #print(control_end_year)
                    
                    if (control_end_year == (True,True,True,True,True)):
                        ##print("done.")
                        self.display("Converting ... ")
                        this_year = temp_df.index[0][0].date().year
                        
                        if(self.load_topo == False):
                            # if  self.load_topo    = True  :When Created Shapefile                      
                            topology_control.load_topology(self,self.GB_dataset_1y) 
                            
                    
                        self.GB_dataset_1y = netcdfCropArea.cropDomain(self, data=self.GB_dataset_1y) 
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
                                                        
                            '''
                            w = shp.Writer(self.GB_output_path + "/shp" + '/referance_points')
                            w.autoBalance = 1
                            for fieldname in self.latlon_convert.columns:
                                w.field(fieldname, "C")
                                
                            for k, row in self.latlon_convert.iterrows():
                                w.point(row["Longitude"],row["Latitude"])
                                w.record(*row.tolist())
                            w.close()
                            
                            schema = {
                                    'geometry':'Point',
                                    'properties':[('Name','str')]
                                }
                            pointShp = fiona.open(self.GB_output_path + "/shp" + '/referance_points2.shp', mode='w',\
                                driver='ESRI Shapefile', schema = schema, crs = "EPSG:4326")
                            #get list of points
                            xyList = []
                            for index, row in self.latlon_convert.iterrows():
                                rowDict = {
                                    'geometry' : {'type':'Point',
                                                'coordinates': (row.Longitude,row.Latitude)},
                                    'properties': {'Name' : row.Grid_NO},
                                }
                                pointShp.write(rowDict)
                            #close fiona object
                            pointShp.close()
                            '''
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
                        #self.to_dssat()
                        control_end_year    = (False,False,False,False,False)
                        control_first_day   = (False,False,False,False,False)
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
        dd      = temp_df.index[0][0].date().day
        mm      = temp_df.index[0][0].date().month
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
                    print("!!! LAT1 out of area : %s Latitude(min, max):(%s, %s)" %(la1,arr_LAT_min,arr_LAT_max))
                    exit()
                elif(la2 < arr_LAT_min or la2 > arr_LAT_max): 
                    print("!!! LAT2 out of area : %s Latitude(min, max):(%s, %s)" %(la2,arr_LAT_min,arr_LAT_max))
                    exit()
                elif(lo1 < arr_LON_min or lo1 > arr_LON_max): 
                    print("!!! LON1 out of area : %s Longitude(min, max):(%s, %s)" %(lo1,arr_LON_min,arr_LON_max))
                    exit()
                elif(lo2 < arr_LON_min or lo2 > arr_LON_max):
                    print("!!! LON2 out of area : %s Longitude(min, max):(%s, %s)" %(lo2,arr_LON_min,arr_LON_max))
                    exit()
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
                exit()
        else:
                print("!!!")
                print("LAT1 : %s" %(la1))
                print("LAT2 : %s" %(la2))
                print("* Please check range of latitude in degrees is -90 and 90 and must be LAT1 <= LAT2")
                print("** Latitude(min,max) = (%s,%s)"%(arr_LAT_min,arr_LAT_max))
                print("!!!")
                exit()

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
        fl = "*WEATHER : "+ngrid+", Grid "+ngrid+" ("+self.GB_model_detail["model-simulation"]+":"+self.GB_model_detail["model-name"]+","+self.GB_model_detail["scenario"]+")"
        fl += "\n"
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
            (t == "TMIN") or (t == "RAIN") or (t == "DEWP") or \
            (t == "PAR") or (t == "WIND") :
            if isinstance(t, str) :
                return nc2wth.formatSet(self,1,'R',5,1,value)
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
    
    def H_OrherLine(self):
        header  = nc2wth.splitType(self,"@DATE", 5)# splitType("@YRDAY", 5)
        header += nc2wth.splitType(self,self.blank, 1) + nc2wth.splitType(self,"SRAD", 5)
        header += nc2wth.splitType(self,self.blank, 1) + nc2wth.splitType(self,"TMAX", 5)
        header += nc2wth.splitType(self,self.blank, 1) + nc2wth.splitType(self,"TMIN", 5)
        header += nc2wth.splitType(self,self.blank, 1) + nc2wth.splitType(self,"RAIN", 5)
        #header += splitType(dot, 1) + splitType("DEWP", 5)
        #header += splitType(dot, 1) + splitType("WIND", 5)
        #header += splitType(dot, 1) + splitType("PAR", 5)
        self.display(header)
        self.file_wth_proc.write(header + "\n")  
        
    def num_of_row_df1grid(self,inx):
        
        dd              = inx.date().day 
        mm              = inx.date().month
        yy              = inx.date().year
        f_yy            = str(yy)[2:4]
        d_of_Y          = "%03d" %(date(yy,mm,dd).timetuple().tm_yday)      # select date of year
        YRD             = str(f_yy) + str(d_of_Y)  
        return YRD
    
    def convert2wth(self,this_year):
        # --- setting directory and file output ----
        self._mkdir(self.GB_output_path)
        
        file_wth_name=""
        # --- END setting
        self.display(self.GB_dataset_1y)        
        
        for inx, latlon_convert in tqdm(self.latlon_convert.iterrows()):
            lat_data_1 = latlon_convert["Latitude"]
            lon_data_1 = latlon_convert["Longitude"]
            df1grid = self.GB_dataset_1y[(self.GB_dataset_1y[self.GB_climate_coordinates_name["lat"]] == lat_data_1) \
            & (self.GB_dataset_1y[self.GB_climate_coordinates_name["lon"]] == lon_data_1)]
            
            ngrid           = latlon_convert["Grid_NO"]
            IN              = ngrid[0:2]
            SI              = ngrid[2:4]
            T_mean_avg_year = np.mean(df1grid[self.GB_climate_variable_name["tmean"]].to_numpy())        # annual average of T_mean daily
            T_max_avg_year  = np.mean(df1grid[self.GB_climate_variable_name["tmax"]].to_numpy())         # annual average of T_max daily
            T_min_avg_year  = np.mean(df1grid[self.GB_climate_variable_name["tmin"]].to_numpy())         # annual average of T_min daily
            elev            = float(np.unique(df1grid["topo"]))            # Elevation
            #print("lat = %s, lon = %s, elev = %s" %(lat,lon,elev))
            tav             = T_mean_avg_year                    # 
            amp             = T_max_avg_year - T_min_avg_year    # 
            tmht            = 2.0                                # m
            wndht           = 2.0                                # m
            
            inx_df1grid     = [nc2wth.num_of_row_df1grid(self,i[0]) for i in  df1grid.index] 
            
            subfolder = self.GB_output_path + "/data/" + str(this_year) +"/"
        
            self._mkdir(subfolder)
            self.file_wth_name = ngrid + str(this_year)[2:4] + "01" + ".WTH"
            
            location_wth_file = subfolder + "/" +self.file_wth_name 
            self.file_wth_proc = open(location_wth_file, "w") 
            
            self.display(self.file_wth_name)
            
            local_rain      = list(np.around(np.array(list(pd.to_numeric(df1grid[self.GB_climate_variable_name["rain"]]))),1))
            local_tmax      = list(np.around(np.array(list(pd.to_numeric(df1grid[self.GB_climate_variable_name["tmax"]]))),1))
            local_tmin      = list(np.around(np.array(list(pd.to_numeric(df1grid[self.GB_climate_variable_name["tmin"]]))),1))
            local_rad       = list(np.around(np.array(list(pd.to_numeric(df1grid[self.GB_climate_variable_name["rad"]]))),1))
            df_wth_data = pd.DataFrame(
                {"@DATE":inx_df1grid,
                "SRAD":local_rad,
                "TMAX":local_tmax,
                "TMIN":local_tmin,
                "RAIN":local_rain
                })
            self.display(df_wth_data)
            
            
            nc2wth.wth_header(self,ngrid)
            nc2wth.secondLine(self,IN, SI, lat_data_1, lon_data_1, elev, tav, amp, tmht, wndht)
            self.file_wth_proc.close()
            with open(location_wth_file, 'a') as f:
                f.write(df_wth_data.to_string(index=False))
            self.file_wth_proc.close()
                
class topology_control(mainDefind):
    def __init__(self):
        mainDefind.__init__(self)
        
    def load_topology(self,climate_data):
        lib_path             = inspect.getfile(WTH)
        topo_file            = os.path.dirname(os.path.abspath(lib_path)) + "/topology/topo_land.nc"
        self.GB_data_topo    = self.xnc_dataset(topo_file,self.GB_dropvars).to_dataframe()       
        self.GB_data_topo.reset_index(inplace=True)
        self.GB_data_topo    = netcdfCropArea.cropDomain(self,status="topo", topo=self.GB_data_topo,data=climate_data).dropna(subset = ["topo"])    
        self.map_plot('lon','lat','topo',self.GB_data_topo)
        
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
    
    def define_climate_variable(self, pr="pr",tas="tas",tasmax="tasmax",tasmin="tasmin",rsns="rsns"):
        self.GB_climate_variable_name = {"rain" : pr, "tmean" : tas, "tmax":tasmax, "tmin":tasmin, "rad" : rsns}
        
    def define_coordinates_name(self, x="lat",y="lon",t='time'):
        self.GB_climate_coordinates_name = {"lat":x, "lon":y, "time":t}

    def define_input_path(self, input_path=""):
        self.GB_input_path = input_path
    
    def define_domain(self, la1=0, la2=0, lo1=0, lo2=0):
        self.GB_lonlatbox = {"lat_min":la1, "lat_max":la2, "lon_min":lo1, "lon_max":lo2}

    def define_description(self, model_sim, model_name, model_sc ):
        self.GB_model_detail = {"model-simulation":model_sim,"model-name":model_name,"scenario":model_sc}
        
    def topo(self):
        topology_control.load_topology(self)  
        
        
    def convert2dssat(self,output_path="",start_year="",end_year=""):
        t1_start = process_time() 

        self.GB_output_path = output_path        
        getFileProcess.get_file_list(self,start_year,end_year) 
            
        netcdf2dataframe.dataframe_format(self)
        
        t1_stop = process_time()
        #print("\n\nElapsed time:", t1_stop, t1_start) 
        print("\nProcessed in seconds:",t1_stop-t1_start) 

    
        