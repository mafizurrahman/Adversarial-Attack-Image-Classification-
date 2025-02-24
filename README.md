
# Adversarial Attack on Image Classification

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository implements various adversarial attacks on image classification models using ImageNet samples. The project explores both individual attacks and ensemble methods to evaluate model robustness.

## Features

- Implementation of multiple adversarial attack methods:
  - Fast Gradient Sign Method (FGSM)
  - Projected Gradient Descent (PGD)
  - DeepFool
  - Basic Iterative Method (BIM)
- Novel ensemble attack approaches:
  - Mean ensemble attack
  - Weighted ensemble attack
- Comprehensive evaluation on ImageNet dataset
- ResNet-50 with defensive distillation implementation

## Requirements

```bash
python >= 3.7
pytorch >= 1.7
torchvision
numpy
pillow
matplotlib
```

## Installation

```bash
# Clone the repository
git clone https://github.com/mafizurrahman/Adversarial-Attack-Image-Classification.git

# Navigate to the project directory
cd Adversarial-Attack-Image-Classification

# Install required packages
pip install -r requirements.txt
```

## Usage

### Single Attack Implementation

```python
# Example for FGSM attack
python ResnetFGSM_PGD.ipynb
```

### Ensemble Attack Implementation

```python
# Run ensemble attack with defensive distillation
python Resnet50_WithDefensiveDistillation_Ensemble.ipynb
```

## Project Structure

```
├── ResnetFGSM_PGD.ipynb           # FGSM and PGD attack implementations
├── Pretrained_Attack.ipynb        # Attacks on pretrained models
├── cifar10_augmentation_res34.py  # Data augmentation for CIFAR-10
├── gradientmasking.py            # Gradient masking implementation
├── resnet50defensiveDistillation.py # Defensive distillation implementation
```

## Results

Our experiments show the effectiveness of different attack methods:

1. Individual Attacks:
   - FGSM achieves X% success rate
   - PGD demonstrates Y% attack effectiveness
   - DeepFool shows Z% performance

2. Ensemble Methods:
   - Mean ensemble attack improves success rate by W%
   - Weighted ensemble provides more robust attacks

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Contact

Mafizur Rahman - [GitHub](https://github.com/mafizurrahman)
