# Image Classification

  

This project implements and compares different CNN architectures for image classification on the CIFAR-10 dataset using PyTorch. The project includes implementations of LeNet, a custom CNN, and ResNet18, along with tools for training, evaluation, and visualization.

  

## Project Structure

  

```

image-classifier/

├── configs/ # Configuration files for training

│ ├── lenet.yaml # LeNet training config

│ ├── custom_cnn.yaml # Custom CNN training config

│ └── resnet.yaml # ResNet training config

├── experiments/ # Training outputs and model checkpoints

│ ├── lenet_baseline/ # LeNet experiment results

│ ├── custom_cnn_baseline/ # Custom CNN experiment results

│ └── resnet_baseline/ # ResNet experiment results

├── src/

│ ├── data/ # Data loading and preprocessing

│ ├── models/ # Model architectures

│ │ ├── lenet.py

│ │ ├── custom_cnn.py

│ │ └── resnet.py

│ ├── viz/ # Visualization tools

│ │ ├── eval_viz.py # Model evaluation and comparison

│ │ └── plot_history.py # Training history plotting

│ └── train.py # Training script

└── tests/ # Unit tests

```

  

## Models

  

The project implements three different CNN architectures:

  

1.  **LeNet**: A classic CNN architecture adapted for CIFAR-10

2.  **Custom CNN**: A custom-designed CNN with modern architecture elements

3.  **ResNet18**: A residual network implementation with 18 layers

  

## Results

  

Our experiments show the following test accuracies:

- LeNet: 68.97%

- Custom CNN: 79.71%

- ResNet18: 81.36%

  

Detailed evaluation results, including confusion matrices and classification reports, can be found in the `experiments/_model_comparison` directory.

  

## Setup

  

1. Clone the repository:

```bash

git  clone  https://github.com/yourusername/image-classifier.git

cd  image-classifier

```

  

2. Create and activate a virtual environment:

```bash

python  -m  venv  venv

# On Windows:

venv\Scripts\activate

# On Unix/MacOS:

source  venv/bin/activate

```

  

3. Install dependencies:

```bash

pip  install  -r  requirements.txt

```

  

4. Download the CIFAR-10 dataset:

The dataset will be automatically downloaded when you first run the training script.

  

## Usage

  

### Training

  

To train a model, use the training script with a configuration file:

  

```bash

python  -m  src.train  --config  configs/lenet.yaml

```

  

Available configuration files:

-  `configs/lenet.yaml` - LeNet training config

-  `configs/custom_cnn.yaml` - Custom CNN training config

-  `configs/resnet.yaml` - ResNet training config

  

You can also override the model specified in the config:

```bash

python  -m  src.train  --config  configs/lenet.yaml  --model  resnet

```

  

### Evaluation

  

To evaluate and compare all models:

```bash

python  src/viz/eval_viz.py

```

  

This will:

- Load the best checkpoint for each model

- Evaluate them on the test set

- Generate comparison plots and metrics

- Save results in `experiments/_model_comparison/`
