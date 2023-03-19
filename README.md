# Video Classification
In this repository, I conduct three experiments to train video classifiers on [Kinetics-400](https://github.com/cvdfoundation/kinetics-dataset) and [MSR DailyActivity3D](https://sites.google.com/view/wanqingli/data-sets/msr-dailyactivity3d).

To prepare  **Kinetics-400**, follow the notebook `download-kinetics-400.ipynb`.
The other three notebooks already contain cells to prepare **MSR DailyActivity3D**.

## 1. Experiments
### 1.1. ViViT - `ViViT.ipynb`
In this experiment, I try to replicate the original paper [ViViT: A Video Vision Transformer](https://arxiv.org/abs/2103.15691) using PyTorch, based on the official [repository](https://github.com/google-research/scenic/tree/main/scenic/projects/vivit) and the unofficial PyTorch [implementation](https://github.com/rishikksh20/ViViT-pytorch).
Key implementations for this experiment:  
- **Tubelet embedding**: Conv3d is used to embed a tuple of size *(t, h, w)* into $R^d$ space. The weight of Conv3d corresponding to the central frame of a tuple is initialized with the Conv2d embedding of a pre-trained [ViT model](https://huggingface.co/google/vit-base-patch16-224), while weights at other positions are set to zero - O. &rarr; This embedding method can fuse the spatio-temporal information. The weight initialization makes the 3D convolutional filter behaves like 'Uniform frame sampling'.
- **Factorized Encoders**: there are 2 separate stacks of Transformer Encoder blocks, one is for learning the spatial features, one is for aggregating the temporal information from the learned spatial features via CLS-spatial tokens. This is a late fusion architecture and requires few floating-point operations (FLOPs) than the unfactorized Encoder blocks.

The experiment has data-preparation cells for both **Kinetics-400** and **MSR DailyActivity3D**.

### 1.2. Mini-ViViT - `ViViT_tiny.ipynb`
Implement the same ideas from **1.1**, but using a light-weight version of pre-trained [Encoder block](https://github.com/microsoft/Cream/tree/main/MiniViT/Mini-DeiT)
The experiment has data-preparation cells for both **Kinetics-400** and **MSR DailyActivity3D**.

### 1.3 ResNet & Temporal ViT
In this experiment, in order to create a light-weight classifier, a pre-trained [ResNet-50](https://huggingface.co/microsoft/resnet-50) is used to extract spatial features, then a stack of Transformer Encoder blocks (**Temporal ViT**) is used to aggregate the temporal information from the extracted features. The loss is only backproped on the **Temporal ViT**.

## 2. Results & Discussion
### 2.1. On Kinetics-400
Because Kinetics-400 is a huge dataset, I do not have enough resources to train models. The training times for 1 epoch of **ViViT** and **Mini-ViViT** are 15 hours and 10 hours respectively.
Data processing (including normalization, affine augmentation) accounts for a large portion of training time (about 80%).
### 2.2. On MSR DailyActivity3D
|         | ViViT | Mini-ViViT | ResNet & Temporal ViT |
|---------|-------|------------|-----------------------|
| Val Acc | 9.375% | 7.813%     | 50.0%                |

The above table shows the best results of 20-epoch training for each experiment. 
- It is clear to see that the third experiment (**1.3**) can achieve the best result because ResNet-50 can extract spatial effectively, while the loss is only backward on the Temporal ViT, which can avoid gradient vanishing problem. 
- In contrast, loss of **ViViT** needs to be backprop through both spatial and temporal encoders, which may lead to weak gradient signal update. In addition, despite the fact that spatial encoder's weights are initialized from a pre-trained **ViT**, the temporal encoder has its weights initialized randomly, and as shown from the origin [ViT paper](https://arxiv.org/abs/2010.11929), ViT should be trained with a huge number of data to get good performance, while **MSR DailyActivity3D** has only 320 videos, and 80% of them are used for training. Moreover, training a large ViT requires a large batch size and many epochs, the experiment **1.1** is trained with a batch size of 8 and 20 epochs only due to the GPU memory and time-constrained of Colab. These three reasons are contributed to the low validation accuracy.
- For the experiment **1.2**, because the model have much smaller capacity than model in **1.1**, the result obtained is the worst.

## 3. Improvement Directions
- Preprocess then save the video data first, then during training, there is no data processing overhead, and the training time can reduce significantly.
- Need a strong GPU with high memory capacity, unlimited time to train models with **Kinetics-400** and more epochs.
- Tune hyperparameters to get the best training setting.

## 4. Citation
```
@misc{arnab2021vivit,
      title={ViViT: A Video Vision Transformer}, 
      author={Anurag Arnab and Mostafa Dehghani and Georg Heigold and Chen Sun and Mario Lučić and Cordelia Schmid},
      year={2021},
      eprint={2103.15691},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
```
@InProceedings{MiniViT,
    title     = {MiniViT: Compressing Vision Transformers With Weight Multiplexing},
    author    = {Zhang, Jinnian and Peng, Houwen and Wu, Kan and Liu, Mengchen and Xiao, Bin and Fu, Jianlong and Yuan, Lu},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {12145-12154}
}
```
```
@article{dosovitskiy2020vit,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and  Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  journal={ICLR},
  year={2021}
}
```
