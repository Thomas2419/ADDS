## ADDS: Open-Vocabulary Multi-Label Classification with CLIP and Pyramid-Forwarding

## Overview

This project implements a novel approach to open-vocabulary multi-label classification using a combination of CLIP-based text and image encoders, Pyramid-Forwarding, and a dual-modal decoder (DM-Decoder). The architecture is inspired by the paper *Open Vocabulary Multi-Label Classification with Dual-Modal Decoder on Aligned Visual-Textual Features*, aiming to improve performance on high-resolution images without retraining pre-trained models on low-resolution images.

### Key Features

- **Pyramid-Forwarding**: Enables resolution compatibility by processing higher-resolution images in multiple levels, reducing computational costs. You can configure the number of pyramid levels and whether overlapping is allowed in the `model.py` file.
- **Dual-Modal Decoder (DM-Decoder)**: Aligns visual and textual embeddings for multi-label classification.
- **Asymmetric Loss**: Implements a custom loss function optimized for handling imbalanced datasets, focusing on improving performance for both positive and negative samples.
- **Validation and Checkpoints**: Supports model validation and checkpoint saving during training, with checkpoints saved at regular intervals and the best-performing model saved based on validation accuracy.
- **TensorBoard Logging**: Logs training and validation metrics for easy visualization and tracking.

### Planned Features

- **Selective Language Supervision**: Future updates will include the implementation of selective language supervision to efficiently handle large label sets by balancing positive and negative samples during training.
- **Refactoring and Optimization**: The project will undergo refactoring for better modularity and potential optimizations, particularly to address memory usage challenges.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Thomas2419/ADDS
   cd ADDS
   
2. Install dependencies:

   - Ensure you have Python 3.9+ installed.
   - Install required Python packages:

     ```bash
     pip install -r requirements.txt
     ```

## Project Structure

- **main.py**: The main entry point for training the model, including configuration settings.
- **model.py**: Contains the architecture definitions for Pyramid-Forwarding, DM-Decoder, and the ADDS model.
- **data.py**: Handles data preprocessing and dataset management, including JSON label processing and image loading.
- **training_utils.py**: Provides utilities for model training, including functions for saving/loading models, calculating accuracy, and managing learning rate schedules.

## Usage

### Running the Training Process

To start training the model, run the following command:

```bash
python main.py --data_dir /path/to/your/data --tags_path /path/to/organized_Tags.json --model_name ViT-L-14 --pretrained laion2b_s32b_b82k --device cuda --num_epochs 2000
````

### Parameters

- --data_dir: Path to the directory containing your images and corresponding JSON label files.
- --tags_path: Path to the organized_Tags.json file that contains all possible labels.
- --model_name: The name of the CLIP model to use (e.g., ViT-L-14).
- --pretrained: Specifies the pre-trained weights to use (e.g., laion2b_s32b_b82k).
- --device: The device to use for training (cuda or cpu).
- --num_epochs: Number of epochs for training.

### Customizing the Model
Adjusting the Number of Labels
To change the number of labels your model can predict, you need to adjust the tags_path in main.py to point to a JSON file that contains all possible labels. The number of labels is dynamically determined based on the contents of this file:

```tags_path = "/path/to/your/organized_Tags.json"```
Ensure the JSON file contains all possible labels in the correct format:

```
{
    "label1": ["sub_label1", "sub_label2"],
    "label2": [],
    ...
}
```
The model will automatically set the number of labels based on this file.

### Configuring Pyramid Levels
The Pyramid-Forwarding mechanism can be adjusted in the model.py file. To change the number of pyramid levels modify the PyramidForwarding instantiation:

```
self.pyramid_forwarding = PyramidForwarding(
    clip_model.visual, 
    patch_size=224, 
    pyramid_levels=3,  # Change this value to adjust the number of pyramid levels
    allow_overlap=True  # Set to False to disallow overlapping
)
```
pyramid_levels: Controls the number of resolution levels used. Increasing this number allows the model to process images at multiple scales but increases computational cost.

### Changing the Learning Rate and Optimizer Settings
In the main.py file, the learning rates for different parts of the model are set when defining the optimizer:

```
optimizer = torch.optim.AdamW([
    {'params': model.decoder.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4},
    {'params': model.final_fc.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4},
])
```
You can adjust these values to fine-tune the learning rates for different parts of the model:

lr: Learning rate for the specific parameter group.
weight_decay: Regularization parameter to prevent overfitting.

### Expected File Formats

#### Images:

Supported formats: .png, .jpg, .jpeg, .webp.
Images should be stored in the directory specified by --data_dir.
#### JSON Labels:

- Each image should have a corresponding JSON file with the same name (except for the extension).
- Example structure of a JSON label file: ``` { "encoded_labels": [1, 0, 0, 1, ...] } ```
- The organized_Tags.json file should contain all possible labels, formatted as: ``` { "label1": ["sub_label1", "sub_label2"], "label2": [], ... } ```
## Training Output

During training, the following outputs are generated:

- Logs: TensorBoard logs are saved in the main_logs directory.
- Checkpoints: Model checkpoints are saved periodically in the checkpoints directory. The best-performing model at each threshold is saved separately.
- Final Model: After training, the final model is saved as final_fine_tuned_open_clip_adds.pth.
## Evaluation and Inference

You can load the trained model using the load_model function in training_utils.py to perform inference or further evaluation.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to improve the project.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
