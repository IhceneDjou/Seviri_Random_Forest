# Seviri_Random_Forest
Precipitation Estimation from SEVIRI using Random Forest

This code is to estimate precipitation intensity over Trentino (Italy) using SEVIRI satellite data and ground-based TAASDAAR observations from January 2014.

**Overview :
  Input: 6 selected SEVIRI satellite channels (15-minute resolution) ["IR_108", "IR_120", "WV_062", "WV_073", "VIS008", "IR_134"]
  Target: Gridded precipitation maps from TAASDAAR .ascii.gz archives
  Method: Random Forest classifier for pixel-wise rainfall intensity classification


**Steps
   Preprocess SEVIRI .zarr file to extract selected channels
   Extract and align TAASDAAR ASCII grids from .tar files
   Match timestamps on 15-minute intervals (00, 15, 30, 45)
   Train a Random Forest model
   Evaluate and visualize predictions

Results

