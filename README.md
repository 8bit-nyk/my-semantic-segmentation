# Semantic Scene Understanding in Autonomous Robotics

## Project Overview
This project develops a semantic segmentation model to identify objects in industrial scenes using deep learning techniques. 

The model employs a **U-Net** architecture with an **EfficientNet** backbone, trained on a synthetic dataset consisting of annotated images with corresponding segmentation masks and labels.

The datset used is a subset of the Synthetic Object Recognition Dataset for Industries [SORDI](https://sordi.ai/) dataset.

__Disclaimer__: This project was implemented for educational purposes and submitted for a graduate level course concerned with data-driven modelling and machine learning applications [ML4SCIENCE.com](https://www.ml4science.com/)


## Key Features
- **Semantic Segmentation**: Automatically segment and identify object classes within images of industrial settings.
- **Model Evaluation**: Rigorous training, validation, and testing to ensure robustness and accuracy.
- **Hyperparameter Tuning**: Search for optimal hyperparametrs to deliver the best performance.
- **Inference Capabilities**: Use pre-trained models to perform segmentation on new images.

## Dataset Description
For each object class, the dataset includes 1000 datapoints:
- **RGB Images**: Named `rgb_0000.png` to `rgb_0999.png`.
- **Segmentation Masks**: Corresponding masks for object segmentation named `semantic_segmentation_0000.png` to `semantic_segmentation_0999.png`.
- **Labels**: JSON files detailing class and color code information for each image, named `semantic_segmentation_label_0000.json` to `semantic_segmentation_label_0999.json`.

## Installation

Clone the repository to your local machine:
```
git clone https://github.com/your-username/my-semantic-segmentation.git
cd my-semantic-segmentation
```

Install the required dependencies:
```
pip install -r requirements.txt
```

## Usage

### Training the Model
To train the model from scratch, navigate to the project directory and run:
```
python train.py
```
This script will train the model using the training dataset, validate it using the validation set, and save the best-performing model.

### Hyperparameter Tuning
To find the optimal model settings, run:
```
python hyperparameter_tuning.py
```
This will iterate over a predefined set of hyperparameters to find the combination that yields the best validation performance.

### Running Inference
To segment new images using the trained model, run:
```
python inference.py --image path_to_your_image.jpg
```

## Results
Results include segmented outputs demonstrating the model's accuracy and effectiveness. For detailed performance metrics, refer to the included project report.

## Contributing
Contributions to the project are welcome. Please fork the repository and submit a pull request with your enhancements.

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Authors
[8bit-nyk](https://github.com/8bit-nyk)



