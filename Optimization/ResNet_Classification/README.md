

# ResNet_Optimization
This repository is for the implementation of different layers based Deep ResNet models on the CIFAR Datasets.

This repository contains the pytorch code for different layers based Deep ResNet networks and improve methods based on the following papers

1. Deep Residual Learning for Image Recognition https://arxiv.org/abs/1512.03385

2. Identity Mappings in Deep Residual Networks  https://arxiv.org/abs/1603.05027


# ResNet Models
1. ResNet-20
2. ResNet-32
3. ResNet-44
4. ResNet-110
5. ResNet-164

# Datasets
1. CIFAR-10
2. CIFAR-100

![cf10](https://user-images.githubusercontent.com/47654834/114135781-b6e1a880-9944-11eb-8310-436d10c870c8.png)

# Software Requirements 
1. Python (>=3.6)
2. PyTorch (>=1.1.0)
3. Tensorboard(>=1.4.0) (for visualization)
4. Other dependencies (pyyaml, easydict)

# Hardware Requirements
	For the training of Deep ResNet models, 1 or 2 (~11G of memory) GPUs are enough because the torch is memory efficient.

# Usage
1. Clone/Download the repository

This repository contains the data, model, and result folders. These are main.py, gen_mean_std.py, requirements.txt, and the run.sh files.

The script "gen_mean_std.py" si used to calculate the mean and standar deviation value of the dataset.

2. Main.py file

There is need to select the specific model and dataset for the training and the directory for the result like

	model = **resnet20_cifar**(num_classes=**10**)

	fdir = 'result/**resnet20**__cifar_**10**'

3. run.sh file

	We are using the manual optimization approach so there is need to specify the parameters with differnt values to trained and optimize the network. For the complex networks epochs must be 300. 

	Modify the line as required

	"CUDA_VISIBLE_DEVICES=0 python main.py --epoch **300** --batch-size 128 --**lr 0.1** --momentum 0.9 --wd **1e-4** -ct **10**"

# Results
Note: In our work result are obtained after multi-time training with the combinations of different values of hyperparameters and the average results obtained are mentioned below which are better then the originals.

**Network|CIFAR-10 (Error %)|CIFAR-100 (Error%)**


ResNet-20|7.73|30.31

ResNet-32|6.98|27.95

ResNet-44|6.67|27.02

ResNet-110|5.68|23.86

ResNet-164|5.20|23.01

# References

[1]. K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.

[2]. K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.

