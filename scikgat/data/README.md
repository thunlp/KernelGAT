# Data for SCIKGAT

* We provide all data in SCIKGAT, and you can download from this [link](https://thunlp.oss-cn-qingdao.aliyuncs.com/KernelGAT/SCIFACT/data.zip).
* The data preprocessing is the same as KGAT. We add golden evidence and choose ranked top negatives to form the evidence set for training. We directly mix FEVER and SCIFACT data with the proportion 1:50 (duplicate SCIFACT several times in the training set).