from setuptools import setup, find_packages

setup(
    name='wNetCDF2DSSAT',
    version='1.3',
    packages=find_packages(),
    description='Description of my library',
    author='RU-CORE',
    author_email='rucore.center@gmail.com, nick.ratchanan@gmail.com',
    url='https://github.com/RU-CORE-LAB/wNetCDF2DSSAT',
    install_requires=[
        'cftime',
        'h5netcdf',
        'geopandas',
        'netCDF4',
        'pandas',
        'pyshp',
        'xarray',
        'tqdm'

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

