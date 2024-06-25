<div align="center">
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch">
  </a>
  <a href="https://encord.com/blog/guide-to-contrastive-learning/">
    <img src="https://img.shields.io/badge/Contrastive%20Learning-0066CC?style=for-the-badge" alt="Contrastive Learning">
  </a>
  <a href="https://en.wikipedia.org/wiki/Residual_neural_network">
    <img src="https://img.shields.io/badge/ResNet-FFB266?style=for-the-badge" alt="ResNet">
  </a>
</div>

<hr/>

# Contrastive ResNet50

## Abstract
The project focuses on developing an optical character recognition (OCR) model for car plate recognition. The workflow includes two stages: model training and transfer learning. In the model training stage, a balanced EMNIST dataset is used to train a ResNet50 model with data augmentation techniques. Transfer learning is then applied in the second stage, where a custom dataset resembling car plate characters is used to fine-tune the pre-trained model. The model parameters and settings are carefully chosen to optimize performance. The trained OCR model is capable of recognizing car plate characters with high accuracy. Preprocessing steps are applied to input images, and the model outputs recognized characters and confidence levels. The model can be easily reused and deployed for future tasks or inference scenarios.

<hr/>

## Dependencies
You can install all the packages via
```
pip install -r requirements.txt
```

<hr/>

## Instructions
Run the following command to split the dataset into train, validation, and test sets.  
```
python3 data_split.py
```  
Then run the following command to train and infer the models.
```
python3 main.py
```

<hr/>

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<hr/>


