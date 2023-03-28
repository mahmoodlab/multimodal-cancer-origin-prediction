# Multimodal Primary Site of Cancer Origin Prediction
<img src="/docs/Flow_Diagram.jpeg" height="400px"/>

© This code is made available only for manuscript reviewers. 

### Pre-requisites:
* Linux (Tested on Ubuntu 20.04.4 LTS)
* NVIDIA GPU (Tested on Nvidia GeForce RTX 3090)
* Python (3.9.12), h5py (3.6.0), numpy (1.21.5), pandas (1.4.2), PyTorch (1.11.0), scikit-learn (1.1.1)

### Dataset
We have provided the data of publicaly available cases (TCGA and CPTAC) in our dataset to seamlessly run and validate the code.

### Training
``` shell
python train_valid_test.py --classification_type='MM' --exec_mode='train' --exp_name='exp_01'
```

### Evaluation 
``` shell
python train_valid_test.py --classification_type='MM' --exec_mode='eval' --exp_name='exp_01' --split_name='test'
```
