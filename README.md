# ADDS: Open-Vocabulary Multi-Label Classification with CLIP and Pyramid-Forwarding

## Overview

This project implements a novel approach to open-vocabulary multi-label classification using a combination of CLIP-based text and image encoders, Pyramid-Forwarding, and a dual-modal decoder (DM-Decoder). The architecture is inspired by the paper Open Vocabulary Multi-Label Classification with
Dual-Modal Decoder on Aligned Visual-Textual Features, aiming to improve performance on high-resolution images without retraining pre-trained models on low-resolution images.

### Key Features

Pyramid-Forwarding: Enables resolution compatibility by processing higher-resolution images in multiple levels, reducing computational costs.
Dual-Modal Decoder (DM-Decoder): Aligns visual and textual embeddings for multi-label classification.
Selective Language Supervision: Efficiently handles large label sets by balancing positive and negative samples during training.
Validation and Checkpoints: Supports model validation and checkpoint saving during training.
## Installation

Clone the repository: ```git clone https://github.com/Thomas2419/ADDS cd ADDS ```

Install dependencies:

Ensure you have Python 3.9+ installed.
Install required Python packages: ```pip install -r requirements.txt ```

## Project Structure

main.py: The main entry point for training the model.
model.py: Contains the architecture definitions for Pyramid-Forwarding, DM-Decoder, and the ADDS model.
data.py: Handles data preprocessing and dataset management.
training_utils.py: Provides utilities for model training, including functions for saving/loading models, calculating accuracy, and managing learning rate schedules.
## Usage

### Running the Training Process To start training the model, run the following command:
```python main.py --data_dir /path/to/your/data --tags_path /path/to/organized_Tags.json --model_name ViT-L-14 --pretrained laion2b_s32b_b82k --device cuda --num_epochs 2000 ```

### Parameters

--data_dir: Path to the directory containing your images and corresponding JSON label files.
--tags_path: Path to the organized_Tags.json file that contains all possible labels.
--model_name: The name of the CLIP model to use (e.g., ViT-L-14).
--pretrained: Specifies the pre-trained weights to use (e.g., laion2b_s32b_b82k).
--device: The device to use for training (cuda or cpu).
--num_epochs: Number of epochs for training.
### Expected File Formats

Images:

Supported formats: .png, .jpg, .jpeg, .webp.
Images should be stored in the directory specified by --data_dir.
JSON Labels:

Each image should have a corresponding JSON file with the same name (except for the extension).
Example structure of a JSON label file: ``` { "encoded_labels": [1, 0, 0, 1, ...] } ```
The organized_Tags.json file should contain all possible labels, formatted as: ``` { "label1": ["sub_label1", "sub_label2"], "label2": [], ... } ```
## Training Output

During training, the following outputs are generated:

Logs: TensorBoard logs are saved in the main_logs directory.
Checkpoints: Model checkpoints are saved periodically in the checkpoints directory. The best-performing model at each threshold is saved separately.
Final Model: After training, the final model is saved as final_fine_tuned_open_clip_adds.pth.
## Evaluation and Inference

You can load the trained model using the load_model function in training_utils.py to perform inference or further evaluation.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to improve the project.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
