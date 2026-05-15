# Vietnamese Sign Language Recognition

A deep learning-based project for recognizing Vietnamese Sign Language (VSL) from videos or webcam input.

---

## Overview

This project focuses on the task of Vietnamese Sign Language Recognition (VSLR). The system uses deep learning models to extract both spatial and temporal features from video sequences and predict the corresponding sign language gesture.

The project idea and dataset were provided by the Vietnamese Olympic AI 2025 Organizing Committee.

The project supports:

* Training deep learning models on video or frame-sequence datasets
* Model evaluation using common metrics
* Video inference
* Webcam inference
* Demo deployment with Gradio / Hugging Face Spaces
* Training logs and visualization tools

---

## Dataset

The given sign language datasets are organized as follows:

```text
dataset/
├── train/
│   ├── Ăn/
│   ├── Ăn mừng/
│   └── ...
├── public_test/
├── private_test/
└── label_mapping.pkl
```

Each subfolder in `train/` represents a specific sign.

The `label_mapping.pkl` file contains mapping from string label to integer.

---

## Project Structure

```text
project/
├── dataset/
├── fig/
├── logs/
├── models/
├── notebooks/
├── src/
├── train_progress/
├── validation_results/
├── app.py
└── requirements.txt
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/lucvantien1211/Vietnamese_Sign_Language_Recognition
cd Vietnamese_Sign_Language_Recognition
```

### 2. Create a Virtual Environment

#### Conda

```bash
conda create -n vslr python=3.10
conda activate vslr
```

#### venv

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---
## Model Architecture

The architectures that were experimented in the project include:

### CNN Backbone (Spatial Features Extraction)

* ResNet (ResNet 18)
* ConvNeXt (Tiny)

### Temporal Modeling

* Transformer
* LSTM / GRU
* Temporal pooling

### General Pipeline

```text
Video / Webcam
      ↓
Frame Extraction
      ↓
CNN Feature Extractor
      ↓
Temporal Module
      ↓
Classifier
      ↓
Predicted Sign
```

---

## Training

### Prerequisites
Before training, you must:
- Create `video_metadata.csv` file by running the script:
```bash
python -m src.generate_video_metadata
```
- Convert `label_mapping.pkl` to `label_mapping.json` by running the script:
```bash
python -m src.convert_label_mapping_json
```
(Add arguments if needed)

### Train the Model

```bash
python -m src.train \
    --arg_1 arg_1_value \
    --arg_2 arg_2_value \
    ...
```

### Example

```bash
python -m src.train \
    --batch_size 16 \
    --epochs 50 \
    --lr 1e-4
```

---

## Evaluation
Supported metrics (Macro Average):

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

## Gradio Demo
### Local

Run the application:

```bash
python app.py
```

Then open the generated local URL in your browser.

### Hugging Face Spaces Deployment

A Hugging Face Space demo for this project can be accessed here: [Hugging Face Space demo](https://huggingface.co/spaces/lucvantien1211/Vietnamese_Sign_Language_Recognition)

---

## Reproducibility

To improve reproducibility:

* Fix random seeds
* Save training configurations
* Log hyperparameters
* Save checkpoints regularly
* Version control datasets and source code

---

## Results

| Model                  | Macro F1 | Notes              |
| ---------------------- | -------- | ------------------ |
| Baseline CRNN          | 85.69%   | Initial baseline   |
| ConvNeXt + Transformer | 92.30%   | Current best model |

---

## Consideration for future improvements:

* [ ] Sentence-level sign recognition
* [ ] Real-time optimization
* [ ] Hand keypoint integration
* [ ] Multimodal fusion
* [ ] ONNX / TensorRT deployment
* [ ] Mobile deployment
* [ ] Continuous sign recognition

---

## Technologies Used

* Python
* PyTorch
* Torchvision
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Gradio

---

## References and Credits

Many of the code (for model architectures, custom dataset class and video augmentation, ...) was inspired/given by [AI VIET NAM](https://aivietnam.edu.vn).
Shout out for their hard works on building and sharing DS/AI knowledge for Vietnamese students.

---

## License

```text
This project is licensed under the MIT License.
```

---

## Author

* Nguyen Ngoc Lan
* GitHub: [lucvantien1211](https://github.com/lucvantien1211)

---

## Acknowledgements

Special thanks to:

* AI VIET NAM
* The open-source community
* PyTorch
* Hugging Face
* Contributors of sign language datasets and research resources