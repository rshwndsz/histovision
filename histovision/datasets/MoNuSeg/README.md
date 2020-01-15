# MoNuSeg

Paper: N. Kumar, R. Verma, S. Sharma, S. Bhargava, A. Vahadane and A. Sethi, ["A Dataset and a Technique for Generalized Nuclear Segmentation for Computational Pathology," in IEEE Transactions on Medical Imaging](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7872382), vol. 36, no. 7, pp. 1550-1560, July 2017

Dataset: https://monuseg.grand-challenge.org/Data/

## Instructions

Move the training data into `histovision/datasets/MoNuSeg/MoNuSegTrainData`  
Move the test data into `histovision/datasets/MoNuSeg/MoNuSegTestData`

Final structure looks something like this:  
.  
└── MoNuSeg  
   ├── MoNuSegTestData  
   │  ├── TCGA-2Z-A9J9-01A-01-TS1.tif  
   │  ├── TCGA-2Z-A9J9-01A-01-TS1.xml  
   │  ├── TCGA-44-2665-01B-06-BS6.tif  
   │  ├── TCGA-44-2665-01B-06-BS6.xml  
   │  ├── TCGA-69-7764-01A-01-TS1.tif  
   │  ├── .   
   │  ├── .  
   │  └── .  
   ├── MoNuSegTrainData  
   │  ├── Annotations  
   │  └── Tissue Images  
   └── README.md  

