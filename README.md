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

# Contrastive ResNet-50

## Abstract
Contrastive learning is a widely adopted technique for training models to encode representations by maximizing the dissimilarity between differently augmented views of the same data point, while minimizing the similarity between representations of different data points. This approach aims to leverage the inherent structure within the data and is particularly effective in scenarios with limited labeled data. In this study, we utilize SimCLR, a prominent framework in the field of contrastive learning, as a pre-training step to acquire meaningful representations from unlabeled skin lesion images. Through experimental evaluations conducted on the ISIC dataset, we demonstrate significant enhancements in accuracy and robustness compared to traditional supervised learning approaches.

<hr/>

## References
SimCLR Paper: [SimCLR](https://proceedings.mlr.press/v119/chen20j.html)  
SupContrast Paper: [Supervised Contrastive Learning](https://proceedings.neurips.cc/paper_files/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html)  
Code: [SupContrast](https://github.com/HobbitLong/SupContrast)

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

## Results
<div align="center">

|Model|	Test accuracy|	Test AUC|
| ---------- | -----------|-----------|
|Baseline|	79.4	|73.1|
|SimCLR (100 epoch)|	**83.1**|	**78.5**|
|SimCLR (300 epoch)	|81.5|	76.6|
|SimCLR (500 epoch)|	80.7|	74.2|

</div>

<hr/>

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<hr/>


