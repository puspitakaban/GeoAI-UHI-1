---
license: mit
task_categories:
- image-to-image
---
# ThermalGen Dataset - STGL

This is the repository for ThermalGen - Satellite-Aerial datasets.

Paper link: https://www.arxiv.org/abs/2509.24878

Put the following dataset files in ThermalGen/dataset_preprocess/STGL/

File structure is like:

```
.
├── folder_config.yml
├── maps
│   ├── Boson
│   │   ├── boson_region1_flight1_night_test.png
│   │   ├── boson_region1_flight2_night_train.png
│   │   ├── boson_region1_flight2_night_val.png -> boson_region1_flight2_night_train.png
│   │   ├── boson_region1_flight3_night_test.png
│   │   ├── boson_region1_flight4_night_train.png
│   │   ├── boson_region1_flight4_night_val.png -> boson_region1_flight4_night_train.png
│   │   ├── boson_region1_flight5_night_test.png
│   │   ├── boson_region1_flight6_night_train.png
│   │   └── boson_region1_flight6_night_val.png -> boson_region1_flight6_night_train.png
│   ├── Bosonplus
│   │   ├── bosonplus_region1_day_train_val.png
│   │   ├── bosonplus_region1_night_train.png
│   │   ├── bosonplus_region2_day_train.png
│   │   └── bosonplus_region2_night_train.png
│   ├── DJI
│   │   └── DJI_region1_day_train.png
│   └── satellite
│       ├── 20201117_BingSatellite.png
│       ├── 20201117_ESRI_Satellite.png
│       ├── ESRI_region1_night.png
│       ├── ESRI_region1.png
│       ├── ESRI_similar_region2.png
│       └── ESRI_region2.png
└── README.md
```

Note that our dataset is based on the ESRI basemap and Bing satellite imagery. For detailed copyright information, please refer to https://www.esri.com/en-us/legal/copyright-proprietary-rights and [Bing Maps Print Rights](https://www.microsoft.com/en-us/maps/bing-maps/product/print-rights).