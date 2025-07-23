# Datasets Directory

This directory should contain your medical imaging datasets after download.

## Download Instructions

Download the pre-processed datasets from our Google Drive:
- **Google Drive**: [Download All Datasets](https://drive.google.com/drive/folders/1SjzYE_dD5IimiiBYIgd8AO-85F9u3LEj?usp=sharing)

Extract the downloaded datasets here to get this structure:

```
datasets/
├── monuseg_2/
│   ├── Train_Folder/
│   │   ├── img/
│   │   ├── labelcol/
│   │   └── Train_text.xlsx
│   ├── Val_Folder/
│   │   ├── img/
│   │   ├── labelcol/
│   │   └── Val_text.xlsx
│   └── Test_Folder/
│       ├── img/
│       ├── labelcol/
│       └── Test_text.xlsx
├── MosMedDataPlus/
│   └── (same structure)
└── qata_cov19_v2_2/
    └── (same structure)
```

## Alternative: Original Sources

You can also download from original sources:
- **MoNuSeg**: [Multi-Organ Nuclei Segmentation Challenge](https://monuseg.grand-challenge.org/)
- **MosMedData+**: [COVID-19 CT Scan Dataset](https://mosmed.ai/en/)
- **QATA COVID-19**: [COVID-19 Chest X-ray Dataset](https://www.kaggle.com/datasets/aysendegerli/qatacov19-dataset)

**Note**: If using original sources, you'll need to preprocess the data to match our expected structure.
