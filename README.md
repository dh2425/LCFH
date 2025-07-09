# LCFH  The code is being organized


### Dataset  
The experimental data used in the paper can be downloaded at:  
**Baidu Netdisk**  
- Link: [https://pan.baidu.com/s/12sl3cOsUS_ucyweWqrsi8Q](https://pan.baidu.com/s/12sl3cOsUS_ucyweWqrsi8Q)  
- Extraction code: `nmsm`  

#### Data Placement  
After downloading, please extract the data to the `data/` folder in the project root directory. The final structure should be as follows:  

```
data/                    # Root data directory  
└── data_name/           # Specific dataset name (e.g., coco/ or custom name)  
    └── raw/             # Raw data directory  
        ├── all/         # Global files  
        │   └── image.txt  # Image path list file (one path per line)  
        └── images/      # Directory for all image files  
            ├── 0001.jpg  
            ├── 0002.jpg  
            └── ...  
```  

#### Notes  
1. If the directories do not exist, please create them manually.  
2. Raw data must be placed in `raw/`, and other directories will be generated automatically by the script.
