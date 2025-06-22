# Seviri_Random_Forest
Precipitation Estimation from SEVIRI using Random Forest

This code is to estimate precipitation intensity over Trentino (Italy) using SEVIRI satellite data and ground-based TAASDAAR observations from January 2014.

**Overview :
  Input: 6 selected SEVIRI satellite channels (15-minute resolution) ["IR_108", "IR_120", "WV_062", "WV_073", "VIS008", "IR_134"]
  
  Target: Gridded precipitation maps from TAASDAAR .ascii.gz archives
  
  Method: Random Forest classifier for pixel-wise rainfall intensity classification


**Steps

   1-Preprocess SEVIRI .zarr file to extract selected channels
   
   2-Extract and align TAASDAAR ASCII grids from .tar files
   
   3-Match timestamps on 15-minute intervals (00, 15, 30, 45)
   
   4-Train a Random Forest model
   
   5-Evaluate and visualize predictions
   

**Results
Test RMSE: 3.30

Test MAE: 2.21

Test R²: 0.67


POD: 0.9999903256368644

FAR: 0.9990195534343378

BIAS: 1.0118642871544465

Precision: 0.9882652627745426

Recall: 0.9999903256368644

tn, fp, fn, tp 

Confusion Matrix: [[53, 54004], [44, 4548059]]

![line_true_vs_pred](https://github.com/user-attachments/assets/7c877323-307e-4360-804c-dcc5f612092f)
![line_true_vs_pred](https://github.com/user-attachments/assets/4a6ed3aa-91eb-4f23-b250-a8533bebf7d6)
![classification_rain_intensity](https://github.com/user-attachments/assets/ef09c8ae-fdbf-4f9d-8ea7-1596df22d8d8)
