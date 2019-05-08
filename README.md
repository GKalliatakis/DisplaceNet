
<p align="center">
  <img src="https://github.com/GKalliatakis/DisplaceNet/blob/master/logo_v2.png?raw=true" width="300" />

[![GitHub license](https://img.shields.io/github/license/GKalliatakis/DisplaceNet.svg)](https://github.com/GKalliatakis/DisplaceNet/blob/master/LICENSE)
![GitHub issues](https://img.shields.io/github/issues/GKalliatakis/DisplaceNet.svg)
![GitHub release](https://img.shields.io/github/release/GKalliatakis/DisplaceNet.svg)
[![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=DisplaceNet:%20Recognising%20Displaced%20People%20from%20Images%20by%20Exploiting%20Dominance%20Level&url=https://github.com/GKalliatakis/DisplaceNet&hashtags=ML,DeepLearning,CNNs,HumanRights,HumanRightsViolations,ComputerVisionForHumanRights)
</p>


--------------------------------------------------------------------------------
### Introduction
<p align="justify">To reduce the amount of manual labour required for human-rights-related image analysis, 
we introduce <i>DisplaceNet</i>, a novel model which infers potential displaced people from images 
by integrating the dominance level of the situation and a CNN classifier into one framework.</p>

<p align="center">
  <img src="https://github.com/GKalliatakis/DisplaceNet/blob/master/DisplaceNet.png?raw=true" width="700" />
</p>

<p align="center">
  <a href="https://scholar.google.com/citations?user=LMY5lhwAAAAJ&hl=en&oi=ao" target="_blank">Grigorios Kalliatakis</a> &nbsp;&nbsp;&nbsp;
  <a href="https://scholar.google.com/citations?user=40KlWugAAAAJ&hl=en" target="_blank">Shoaib Ehsan</a> &nbsp;&nbsp;&nbsp;
  <a href="https://scholar.google.com/citations?user=Hg2osmAAAAAJ&hl=en" target="_blank">Maria Fasli</a> &nbsp;&nbsp;&nbsp;
  <a href="https://scholar.google.com/citations?user=xYARJTQAAAAJ&hl=en" target="_blank">Klaus McDonald-Maier</a> &nbsp;&nbsp;&nbsp;
</p>

<p align="center">
<i>To appear in 1<sup>st</sup> CVPR Workshop on <br> <a href="https://www.cv4gc.org/" target="_blank">Computer Vision for Global Challenges (CV4GC)</a> &nbsp;&nbsp;&nbsp;
</i>
<br>
<a href="https://arxiv.org/pdf/1905.02025.pdf" target="_blank">[arXiv preprint]</a>
 &nbsp;&nbsp;&nbsp;
<a href="https://arxiv.org/pdf/1905.02025.pdf" target="_blank">[poster coming soon...]</a>
</p>



### Dependencies
* Python 2.7+
* Keras 2.1.5+
* TensorFlow 1.6.0+

### Usage

Clone the repository:

    $ git clone https://github.com/GKalliatakis/DisplaceNet.git


#### Inference with pretrained models
To make a single image inference using DisplaceNet, run the script below. See [run_DisplaceNet.py](https://github.com/GKalliatakis/DisplaceNet/blob/master/run_DisplaceNet.py) for a list of selectable parameters.

   ```bash
   $ python run_DisplaceNet.py --img_path test_image.jpg \
                               --hra_model_backend_name VGG16 \
                               --emotic_model_backend_name VGG16 \
                               --nb_of_conv_layers_to_fine_tune 1
   ``` 
   
#### Inference results DisplaceNet vs vanilla CNNs
Make a single image inference using DisplaceNet and display the results against vanilla CNNs (as shown in the paper). 
For example to reproduce image below, run the following script.
See [displacenet_vs_vanilla.py](https://github.com/GKalliatakis/DisplaceNet/blob/master/displacenet_vs_vanilla.py) for a list of selectable parameters.

   ```bash
   $ python displacenet_vs_vanilla.py --img_path test_image.jpg \
                                      --hra_model_backend_name VGG16 \
                                      --emotic_model_backend_name VGG16 \
                                      --nb_of_conv_layers_to_fine_tune 1
   ``` 
   
   <p align="center">
    <img src="https://github.com/GKalliatakis/DisplaceNet/blob/master/evaluation/results_VGG16_1L__VGG16.png?raw=true" width="700" />
   </p>


#### Training DisplaceNet's branches from scratch

1. To train _displaced people_ branch on the HRA subset, run the training script below. See [train_emotic_unified.py](https://github.com/GKalliatakis/DisplaceNet/blob/master/train_emotic_unified.py) for a list of selectable parameters.
    
    ```bash
    $ python train_hra_2class_unified.py --pre_trained_model vgg16 \
                                	     --nb_of_conv_layers_to_fine_tune 1 \
                                	     --nb_of_epochs 50
    ```
1. To train _human-centric_ branch on the EMOTIC subset, run the training script below. See [train_emotic_unified.py](https://github.com/GKalliatakis/DisplaceNet/blob/master/train_emotic_unified.py) for a list of selectable parameters.
    
    ```bash
    $ python train_emotic_unified.py --body_backbone_CNN VGG16 \
                                     --image_backbone_CNN VGG16_Places365 \
                                     --modelCheckpoint_quantity val_loss \
                                     --earlyStopping_quantity val_loss \
                                     --nb_of_epochs 100 \
    ```   
    _Please note that for training the human-centric branch yourself, the HDF5 file containing the preprocessed images and their respective annotations is required (10.4GB)._
    
### Data of DisplaceNet

Here we release the data for training DisplaceNet to the public.

[Human Rights Archive](https://github.com/GKalliatakis/Human-Rights-Archive-CNNs) is the core set of our dataset, which has been used to train DisplaceNet.

The constructed dataset contains 609 images of displaced people and the same number of non displaced
people counterparts for training, as well as 100 images collected from the web for testing and validation.

* [Train images](https://github.com/GKalliatakis/DisplaceNet/releases/download/v1.0/train.zip)
* [Validation images](https://github.com/GKalliatakis/DisplaceNet/releases/download/v1.0/val.zip)
* [Test images](https://github.com/GKalliatakis/DisplaceNet/releases/download/v1.0/test.zip)


---

### Results
<p align="center">
  <img src="https://github.com/GKalliatakis/DisplaceNet/blob/master/inference/results/results_1.jpg" width="275" />
  <img src="https://github.com/GKalliatakis/DisplaceNet/blob/master/inference/results/results_2.jpg" width="275" />
  <img src="https://github.com/GKalliatakis/DisplaceNet/blob/master/inference/results/results_3.jpg" width="275" />
  <img src="https://github.com/GKalliatakis/DisplaceNet/blob/master/inference/results/results_4.jpg" width="275" />
  <img src="https://github.com/GKalliatakis/DisplaceNet/blob/master/inference/results/results_5.jpg" width="275" />
  <img src="https://github.com/GKalliatakis/DisplaceNet/blob/master/inference/results/results_6.jpg" width="275" />
</p>

---

### Performance of AbuseNet

<p align="justify">The performance of displaced people recognition using DisplaceNet is listed below. 
As comparison, we list the performance of various vanilla CNNs trained with various network backbones, 
for recognising displaced people. We report comparisons in both accuracy and coverage-the proportion of a data set for which a classifier is able to produce a prediction- metrics</p>

<p align="center">
  <img src="https://github.com/GKalliatakis/DisplaceNet/blob/master/evaluation/performance_table.png?raw=true" width="700" />
</p>


### Citing DisplaceNet
If you use our code in your research or wish to refer to the baseline results, please use the following BibTeX entry:

    @article{kalliatakis2019displacenet,
    title={DisplaceNet: Recognising Displaced People from Images by Exploiting Dominance Level},
    author={Kalliatakis, Grigorios and Ehsan, Shoaib and Fasli, Maria and McDonald-Maier, Klaus D},
    journal={arXiv preprint arXiv:1905.02025},
    year={2019}
    }

<p align="center">
  :octocat:  <br>
  <i>Repo will be updated with more details soon!</i><br>
  <i>Make sure you have starred and forked this repository before moving on!</i></b>
  
</p>
