# src/dataset_aa.py
#
# ============================================================
# Ann Arbor Dataset — placeholder for future implementation
# ============================================================
#
# PURPOSE:
#   This file will load local Ann Arbor geospatial/thermal data
#   and pair it with AlphaEarth embeddings for the same region.
#
# WHEN TO USE:
#   Use this dataset instead of STGLDataset (dataset.py) when
#   training or fine-tuning on Ann Arbor specific imagery.
#
# DATA YOU WILL NEED (collect later):
#   - RGB satellite imagery of Ann Arbor region
#     → source: Google Earth Engine, Sentinel-2, or Bing Maps
#   - Paired thermal imagery of the same region
#     → source: Landsat thermal band (Band 10), ECOSTRESS, or
#               local drone/aerial thermal captures
#   - AlphaEarth embeddings for Ann Arbor bounding box
#     → lat/lon bounding box approx:
#         north: 42.3230, south: 42.2230
#         east: -83.6880, west: -83.8010
#     → pull from Google Earth Engine:
#         ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
#         .filterBounds(ee.Geometry.Rectangle([-83.801, 42.223,
#                                              -83.688, 42.323]))
#
# FILE STRUCTURE TO CREATE (when ready):
#   data/
#   ├── raw/
#   │   └── AnnArbor/
#   │       ├── rgb/
#   │       │   ├── annarbor_2022_train.png
#   │       │   ├── annarbor_2022_val.png
#   │       │   └── annarbor_2022_test.png
#   │       └── thermal/
#   │           ├── annarbor_2022_train_thermal.png
#   │           ├── annarbor_2022_val_thermal.png
#   │           └── annarbor_2022_test_thermal.png
#   └── embeddings/
#       └── alphaearth/
#           └── annarbor_2022.tif   ← AlphaEarth 64-dim .tif
#
# CONFIG TO ADD (in configs/config.yml) when ready:
#   dataset: ann_arbor              # switch from STGL to this
#   ann_arbor_data_root: data/raw/AnnArbor/
#   ann_arbor_train_rgb:     annarbor_2022_train.png
#   ann_arbor_val_rgb:       annarbor_2022_val.png
#   ann_arbor_test_rgb:      annarbor_2022_test.png
#   ann_arbor_train_thermal: annarbor_2022_train_thermal.png
#   ann_arbor_val_thermal:   annarbor_2022_val_thermal.png
#   ann_arbor_test_thermal:  annarbor_2022_test_thermal.png
#   alphaearth_dir: data/embeddings/alphaearth/
#   alphaearth_year: 2022
#
# ============================================================

# TODO: implement this class when Ann Arbor data is ready
# Follow the same structure as STGLDataset in dataset.py
# Key difference: this dataset has REAL thermal pairs
# (not placeholders) since you will collect matched RGB+thermal

class AnnArborDataset:
    """
    Placeholder — not yet implemented.

    Will load paired RGB + thermal imagery for Ann Arbor, MI
    with AlphaEarth geospatial embeddings per patch.

    To implement:
      1. Collect Ann Arbor RGB + thermal image pairs
      2. Export AlphaEarth embeddings for Ann Arbor bbox from GEE
      3. Copy STGLDataset structure from dataset.py
      4. Replace placeholder thermal with real paired thermal maps
      5. Update config.yml with ann_arbor_* keys above
      6. In train.py, swap STGLDataset → AnnArborDataset
    """
    def __init__(self, config_path, split="train"):
        raise NotImplementedError(
            "AnnArborDataset is not yet implemented.\n"
            "See the comments at the top of this file for instructions.\n"
            "Use STGLDataset (dataset.py) for now."
        )

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError