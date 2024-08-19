# NeuronMapper

NeuronMapper is a comprehensive brain-wide neuron detection and mapping approach powered by Python. 
This method features two key components: a classification module that identifies regions containing somata within volumetric images, and a detection module that precisely localizes these somata in 3D space. 
Once the somata are accurately identified and localized, their coordinates are registered with the Allen Brain Atlas, facilitating the mapping and understanding of brain functions.


## Technologies

List the technologies, frameworks, and libraries used in the project:
- Python 3.6
- PyTorch 1.10.2
- Pandas 0.20.3
- Opencv-python 4.6.0
- Matplotlib 3.3.1
- SimpleITK


## Installation

### Prerequisites

Specify any prerequisites needed before installing your project:
- Operating system: Windows/Linux
- Python version: 3.6+
- Other dependencies: CUDA, CUDNN, Anaconda

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/XZH-James/NeuronMapper.git
   cd NeuronMapper
   ```

2. **Create a virtual environment:**
   ```bash
   conda create --name NeuronMapper python=3.6
   conda activate  NeuronMapper 
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```


## Usage

### **Dataset preparation** 

**ClassificationModule**
* In this folder, [Input](https://github.com/XZH-James/NeuronMapper/tree/main/ClassificationModule-main/input) stores the sub block(512*512*512) maximum intensity projection images.  
* [make_dataset.py](https://github.com/XZH-James/NeuronMapper/blob/main/ClassificationModule-main/input/make_dataset.py) is used to randomly create data sets for classification network training/validation/testing.  

**SegmentModule**
* In this folder, [dataset](https://github.com/XZH-James/NeuronMapper/tree/main/SegmentModule-main/dataset) stores the sub block(512*512*512) images for training/validation/testing. 

### **training or testing** 

**ClassificationModule**
* After obtaining the dataset, you can run [train.py](https://github.com/XZH-James/NeuronMapper/blob/main/ClassificationModule-main/train.py) to train the classification network. 
* The model is saved in [output](https://github.com/XZH-James/NeuronMapper/tree/main/ClassificationModule-main/output), 
and [test.py](https://github.com/XZH-James/NeuronMapper/blob/main/ClassificationModule-main/test.py) can be run for multi-indicator detection.


**SegmentModule**
* After obtaining the dataset, you can run [train.py](https://github.com/XZH-James/NeuronMapper/blob/main/SegmentModule-main/train.py) to train the classification network. 
* The model is saved in [experiments](https://github.com/XZH-James/NeuronMapper/blob/main/SegmentModule-main/experiments), 
and [predict.py](https://github.com/XZH-James/NeuronMapper/blob/main/SegmentModule-main/predict.py) can be run for for whole brain neuron segmentation.


## Acknowledgements

- [SegNet](https://github.com/vinceecws/SegNet_PyTorch)
- [3D U-Net](https://github.com/lee-zq/3DUNet-Pytorch)
- [TransBTS](https://github.com/Rubics-Xuan/TransBTS)

## FAQ

Include any frequently asked questions and their answers.

### Question 1  Environment configuration problem

The environment profile we provide may not work on all servers.
The code runs depending on the CUDA and Cudnn versions supported by the current server graphics card. 
Before installing the environment, install CUDA and Cudnn compatible with the graphics card version, and install pytorch and related environment dependencies based on the current version.

### Question 2  Dataset format problem

Classification network: *_*_*-GT.tif   Such as 5_10_2-1.tif  means whole brain coordinates X:5 Y:10 Z:2 contain projected images of neurons.
0_9_3-0.tif means whole brain coordinates X:0 Y:9 Z:3 do not contain projected images of neurons.

Segmentation network: image -   volume-1.tif  label - label-1.tif

## Contact

If you have any questions about this project, please feel free to contact us.
- **Email:** zhehao_xu@qq.com



