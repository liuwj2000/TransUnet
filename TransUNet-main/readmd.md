1) make dir for the following files:

```
.
├──datasets
│       └── dataset_synapse.py
├──train.py
├──test.py
├──utils.py
├──trainer.py
├── model
│   └── vit_checkpoint
│       └── imagenet21k
│           ├── R50+ViT-B_16.npz
├── data
|    └──Synapse
|        ├── test_vol_h5
|        │   ├── 
|        └── train_npz
|            ├── 
├──lists
|    └──lists_Synapse
|         ├──  test_vol.txt 
|         └──  train.txt
├── networks
|     ├── vit_seg_configs.py
|     ├── vit_seg_modeling.py
|     └── vit_seg_modeling_resnet_skip.py
```

2) unrar raw data

3) run png2npz.py, create_txt.py separately.

4) download  R50+ViT-B_16.npz in imagenet21k/ from https://console.cloud.google.com/storage/browser/vit_models;tab=objects?pli=1&prefix=&forceOnObjectsSortingFiltering=false

