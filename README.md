# TLab Summer 2026 Project - Vikram Ganesan

## 3DAPT: Adapting VFMs to understand 3-D Medical Images
## [Slide Deck](https://docs.google.com/presentation/d/18PJwfVF5cvbKKnwOClTv-HdnZAa4KZG5_w2PB-fB1DE/edit?slide=id.g3f99e3cc3d0_0_168#slide=id.g3f99e3cc3d0_0_168)

### Setup
To run code in this repo, just git clone the repository, as well as a copy of the dinov3 repository in a separate directory. You can install dependencies by running

`conda create -f environment.yml`


### 3-D Aware Post Training Method (3DAPT):

#### Overview: 
The code for the 3-D Aware Post Training Method is contained in the ssl_finetuning module. This code is adapted
from the original DINOv3 training code. Instead of processing 1 image per forward pass (with corresponding global and local crops), it processes 2 adjacent 2-D slices (1 global crop for each slice, and then local crops for each). It then adds an extra term to the DINO loss for the CLS tokens between the two global crops (termed UWSD loss). This UWSD loss is weighted by the entropy of the teacher distribution, in order to encourage more uncertain teacher outputs (inspired by VESSA paper). Then, we also utilize the gram loss from dinov3, however we add spatial windowing, to restrict it to terms of the matrix within a certain window. We apply the gram loss on the teacher gram matrix of slice 1 and a student gram matrix of slice 2, and vice versa. The cross_slice_gram_loss is implemented on a delay, by default 1.0 epochs, because applying it immediately degrades features. The training loop occurs on a model initialized with LoRA, and you may specify the rank in the configuration. A default configuration is given in /common/ganesanv/tlab/src/ssl_finetuning/config.yaml. 

#### How to Run: 
In order to run 3DAPT, first set up a configuration similar to /common/ganesanv/tlab/src/ssl_finetuning/config.yaml, as well as an output directory. You can run 3DAPT on a slurm cluster using an adapted version of the DINOv3 submitit script. Add your config path and write path to src/run_scripts/run_ssl_finetuning.sh, then run: 

`bash src/run_scripts/run_ssl_finetuning.sh` on your login node. 

### ExPLoRA (continued_pretraining)
#### Overview: 
The code for the ExPLoRA extended pretraining is contained in the continued_pretraining module. This code contains the dinov3 pre-training code, except edited to accept the datasets that we have created here as well as to freeze the teacher and student backbones, and add LoRA adapters to both. 

#### How to Run: 
In order to run ExPLoRA, first set up a configuration similar to /common/ganesanv/tlab/src/continued_pretraining/config.yaml, as well as an output directory. You can run ExPloRA on a slurm cluster using an adapted version of the DINOv3 submitit script. Add your config path and write path to src/run_scripts/run_ssl_finetuning.sh, as well as how many GPUs you want to run it on, then run: 

`bash src/run_scripts/run_continued_pretraining.sh` on your login node. 

### Classification
The classification is designed to use a 2-D VFM backbone as well as a small classfication head to classify 3-D medical images. Given a 3-D volume as input, the classification module separates this into 2-D slices, and passes the 2-D slices through the 2-D VFM, to obtain global and local features. By default, we will select the CLS token from each of these slices and have n_slices CLS tokens. (Could also be attention-pooled patch features). These n_slices CLS tokens will be fed into an aggregator module (either just meanpooling, or a TransformerEncoderBlock), and then a small classification head to produce the final logits. An example configuration file for this is in

### Segmentation
The segmentation part of this module is unfinished, but it uses the nnUNet framework for segmentation. nnUNet automatically preprocesses the dataset and configures hyperparameters. For evaluation of the 2-D backbones, we extract local features from multiple layers using the frozen 2-D VFM encoder, and then pass these local features into a Primus segmentation head. Here would be an example command to run segmentation using MedDINOv3 encoder. 

`nnUNetv2_train dataset_id 2d 0 -tr meddinov3_base_primus_multiscale_Trainer`. 
Note that you must setup the dataset in nnUNet as well as environment variables before running this. 

### Utils
Utils has a whole bunch of quick scripts that I made to test things out, but there are a couple of useful ones
- fold_cv.py: This script is used in the classification pipeline to split datasets into n_folds, stratified by label
- pca_dino_backbones.py: This script is used to visualize the pca of patch features created by the DINO backbone, either the default or a fine-tuned version. If you have run ExPLoRA or 3DAPT and want to visualize how features change over training, you can run: 
`python src/utils/pca_dino_backbones.py evolution --checkpoint-parent [your ckpt directory] --output [where you want image to go] --dataset [dataset to use] --image-size [image size (multiple of 16)] --n-images [number of images (default 5)]`



### Datasets
#### Overview: 
In datasets, we have several classification datasets as well as a couple segmentation datasets, which are used for 3DAPT fine-tuning, ExPLoRA fine-tuning, as well as classification. Each of these dataset modules exports three different PyTorch dataset objects: 
- Single Slice Dataset (samples single slices from the 3-D volumes contained in the dataset)
- Paired Slice Dataset (samples pairs of slices that are within max_dist of each other from the 3-D volumes in the dataset)
- Multi-Slice Dataset (samples full 3-D volumes - resized to n_slices, 3, image_size, image_size)

The single slice dataset is used for ExPLoRA fine-tuning, the paired slice dataset is used for the 3DAPT fine-tuning, and then the multi-slice dataset is used for the downstream classification. The specifics of how these are implemented may differ slightly across different datasets. 

#### ADNI
Alzheimer's Disease Neuroimaging Initiative, NiFTI files downloaded directly from their data repository. Has three different tasks: cn_ad, cn_mci, and cn_mci_ad. These tasks will exclude scans from patients with the excluded condition (ex. cn_ad only has control and alzheimer's scans). 

#### CQ500
CQ500 dataset has around 500 CT scans of the head area, and labels for intracranial hemorrhage. There are two tasks: ich, and subtype. ICH task contains all scans, and it involves classifying whether a scan has a ICH or not. Subtype task is restricted to only tasks with ICH, and it involves classifying the specific ICH subtype (IPH, IVH, SDH, EDH, SAH). The data came as DICOM files, so had to use dicom2nifti python package to convert it (cq500/convert.py)

#### OrganMNIST3D
OrganMNIST3D dataset has 3-D CT scans of 11 different organs, and labels for this organ classification. Data came as a .npz, and this .npz

#### LLDMMRI
LLDMMRI has around 500 MRI scans of liver lesions, with 7 different weightings/modalities per scan. Only one task, classifying which type of liver lesion it is. You can specify the type of scan to use with the `scan` argument, or you can pass `scan: all` to use all types of scans.

#### AMOS
AMOS is a segmentation dataset, and the base data comes from the nnUNet preprocessed data. It has abdominal CT scans, and the task is to segment out all of the different organs. To reproduce this, you would have to download dataset, preprocess with nnUNet, and then point this dataset to the path of the processed data. Because it is for segmentation, it only exposes the single slice and paired slice dataset. 

#### BraTS-MEN
BraTS-MEN is a segmentation dataset, and the base data also comes from the nnUNet preprocessed data. It has MRIs of patients with brain tumors (meningioma), and the task is to segment these tumors. Again, because it is for segmentation, it only exposes the single slice and paired slice dataset. 

#### The Rest
Any other datasets in the datasets module (BreastDM, Duke, etc. ) weren't really too heavily experimented with, kinda just played around with them.


#### Run Scripts
In run scripts there are a lot of bash scripts to run the classification or 3DAPT/ExPLoRA on SLURM clusters. You do not have to run it on slurm cluster, and you could just extract the python commands to replicate this performance. 




#### Citations
[1]	O. Siméoni et al., “DINOv3,” 2025, arXiv. doi: 10.48550/ARXIV.2508.10104. 
[2]	Y. Li, Y. Wu, Y. Lai, M. Hu, and X. Yang, “MedDINOv3: How to adapt vision foundation models for medical image segmentation?,” Oct. 15, 2025, arXiv: arXiv:2509.02379. doi: 10.48550/arXiv.2509.02379. 
[3]	Y. Wu et al., “BrainDINO: A Brain MRI Foundation Model for Generalizable Clinical Representation Learning,” Jun. 11, 2026, arXiv: arXiv:2604.27277. doi: 10.48550/arXiv.2604.27277. 
[4]	S. Khanna, M. Irgau, D. B. Lobell, and S. Ermon, “ExPLoRA: Parameter-Efficient Extended Pre-Training to Adapt Vision Transformers under Domain Shifts,” Dec. 30, 2025, arXiv: arXiv:2406.10973. doi: 10.48550/arXiv.2406.10973. 
[5]	R. C. Petersen et al., “Alzheimer’s Disease Neuroimaging Initiative (ADNI): Clinical characterization,” Neurology, vol. 74, no. 3, pp. 201–209, Jan. 2010, doi: 10.1212/WNL.0b013e3181cb3e25. 
[6] Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021).
nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.
Nature Methods, 18(2), 203-211.`

