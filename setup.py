from setuptools import setup, find_packages

setup(
    name='wNetCDF2DSSAT',
    version='1.3',
    packages=['wNetCDF2DSSAT'],
    package_data={'wNetCDF2DSSAT': ['topology/*.nc']},
    include_package_data=True,
    description='The wNetCDF2DSSAT is a Python library for converting Networked Common Data Form (NetCDF) to DSSAT input data format.',
    author='RU-CORE',
    author_email='rucore.center@gmail.com, nick.ratchanan@gmail.com',
    url='https://github.com/RU-CORE-LAB/wNetCDF2DSSAT',
    install_requires=[
        #'numpy',
        'cftime',
        'xarray',
        'tqdm',
        'matplotlib',
        'pyshp',
        'geopandas',
        'python-cmethods',
        'h5netcdf',
        #'netCDF4',
        'h5py',
        #'pandas',
        
        #'numpy==1.26.4',
        #'cftime>=1.6.3',
        #'h5netcdf>=1.3.0',
        #'geopandas>=0.14.3',
        #'netCDF4>=1.6.5',
        #'pandas>=2.2.2',
        #'pyshp>=2.3.1',
        #'xarray>=2024.3.0',
        #'tqdm>=4.66.2'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: MIT License',
    ],
)

