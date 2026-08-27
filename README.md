# LitUNet-DLU: CPU-Optimized Medical Image Segmentation

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.x-orange)

Official implementation of **"CPU-Optimized LitUNet for Medical Image Segmentation with a Trainable Dynamic Linear Unit (DLU)"**

## Overview

This repository presents a lightweight neural network architecture designed for efficient medical image segmentation on CPU-constrained environments. The proposed LitUNet model is enhanced with a **Trainable Dynamic Linear Unit (DLU)** activation mechanism that improves segmentation performance while maintaining computational efficiency.

The framework includes comprehensive tools for:
- Model training and evaluation
- Runtime benchmarking (CPU/GPU comparison)
- Memory consumption analysis
- Reproducible experiments on multiple medical imaging datasets

## Key Features

✨ **Lightweight Architecture**
- CPU-optimized LitUNet model for edge deployment
- Reduced parameter count without compromising accuracy

⚡ **Trainable Dynamic Linear Unit (DLU)**
- Novel activation function with learnable parameters
- Improved gradient flow and model expressiveness

📊 **Comprehensive Benchmarking**
- Runtime performance metrics (CPU vs GPU)
- Memory footprint analysis
- Model efficiency reports

🏥 **Multi-Dataset Evaluation**
- Wound Segmentation Dataset
- Kvasir-SEG Polyp Segmentation Dataset
- Lung CT Lesion Segmentation Dataset

🔬 **Production-Ready**
- TensorFlow/Keras implementation
- Reproducible experimental pipelines
- Well-documented code and notebooks

## Supported Datasets

### 1. Wound Segmentation Dataset
- Medical wound images with ground truth masks
- Applications: wound care assessment and monitoring
- Location: `./Wound Dataset/`

### 2. Kvasir-SEG Polyp Segmentation Dataset
- Endoscopic polyp images from gastrointestinal examinations
- Applications: cancer screening and prevention
- Location: `./Kevasir dataset/`

### 3. Lung CT Lesion Segmentation Dataset
- Computed tomography lung lesion images
- Applications: pulmonary disease detection
- Location: `./Lung_CT/`

## Getting Started

### Requirements

- Python 3.8 or higher
- TensorFlow 2.x
- NumPy
- Matplotlib
- OpenCV (cv2)
- Jupyter Notebook

### Installation

```bash
# Clone the repository
git clone https://github.com/prestigentech/LitUNet-DLU-CPU-Optimized-Medical-Image-Segmentation.git
cd LitUNet-DLU-CPU-Optimized-Medical-Image-Segmentation

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

1. **Data Preparation**: Place datasets in respective directories (`Wound Dataset/`, `Kevasir dataset/`, `Lung_CT/`)

2. **Training**: Execute the training notebook
   ```bash
   jupyter notebook training.ipynb
   ```

3. **Evaluation**: Run evaluation on test set
   ```bash
   jupyter notebook evaluation.ipynb
   ```

4. **Benchmarking**: Compare performance metrics
   ```bash
   jupyter notebook benchmarking.ipynb
   ```

## Architecture

### LitUNet Model

The LitUNet architecture is a lightweight encoder-decoder network designed for medical image segmentation:

```
Input Image
    ↓
[Encoder Blocks] → Progressively downsamples and extracts features
    ↓
[Bottleneck] → Captures semantic information
    ↓
[Decoder Blocks] → Progressively upsamples and refines segmentation
    ↓
Output Segmentation Mask
```

### Dynamic Linear Unit (DLU)

The DLU activation function is defined as:
```
DLU(x) = max(0, min(1, α·x + β))
```

Where α and β are learnable parameters optimized during training.

**Advantages:**
- Adaptive activation based on data characteristics
- Improved gradient flow during backpropagation
- Better generalization across different medical imaging modalities

## Usage Examples

### Training a Model

```python
from models import LitUNetDLU
from utils import load_dataset

# Load dataset
train_images, train_masks = load_dataset('Wound Dataset')

# Initialize model
model = LitUNetDLU(input_shape=(256, 256, 3), num_classes=1)

# Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['dice_coefficient'])

# Train
model.fit(train_images, train_masks, epochs=50, batch_size=16, validation_split=0.2)
```

### Inference

```python
import cv2
import numpy as np

# Load trained model
model = LitUNetDLU.load('trained_model.h5')

# Predict on new image
image = cv2.imread('sample_image.jpg')
image = cv2.resize(image, (256, 256))
image = image / 255.0

# Get segmentation mask
mask = model.predict(np.expand_dims(image, axis=0))[0]
```

## Performance Metrics

The model is evaluated using standard segmentation metrics:

- **Dice Coefficient**: Measures overlap between predicted and ground truth masks
- **Intersection over Union (IoU)**: Jaccard similarity coefficient
- **Sensitivity (Recall)**: True positive rate
- **Specificity**: True negative rate
- **F1-Score**: Harmonic mean of precision and recall

## Benchmarking Results

Results comparing CPU vs GPU performance:

| Dataset | CPU Time (ms) | GPU Time (ms) | Memory (MB) | Accuracy |
|---------|---------------|---------------|-------------|----------|
| Wound | 45.2 | 12.1 | 128 | 94.3% |
| Kvasir-SEG | 48.5 | 13.8 | 145 | 92.7% |
| Lung CT | 50.1 | 14.5 | 156 | 91.2% |

*Note: Results are illustrative. Run benchmarking notebooks for actual performance on your hardware.*

## Project Structure

```
├── README.md
├── requirements.txt
├── models/
│   ├── litunet_dlu.py
│   └── activation_functions.py
├── utils/
│   ├── data_loading.py
│   ├── preprocessing.py
│   └── metrics.py
├── training.ipynb
├── evaluation.ipynb
├── benchmarking.ipynb
├── Wound Dataset/
├── Kevasir dataset/
└── Lung_CT/
```

## Citation

If you use this implementation in your research, please cite:

```bibtex
@inproceedings{litunet_dlu_2024,
  title={CPU-Optimized LitUNet for Medical Image Segmentation with a Trainable Dynamic Linear Unit (DLU)},
  author={Prestigentech},
  year={2024}
}
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact & Support

For questions, issues, or collaboration opportunities:
- Open an issue on GitHub
- Contact: [prestigentech]

## Acknowledgments

- TensorFlow/Keras community for excellent deep learning framework
- Medical imaging research community for inspiration and datasets
- Contributors and collaborators

---

**Last Updated**: August 2024

For the latest updates and developments, please star ⭐ this repository and follow for notifications.
