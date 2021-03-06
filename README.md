# Kaggle Competition: Cassava Leaf Disease Classification

The competition can be found [here](https://www.kaggle.com/c/cassava-leaf-disease-classification/overview).

Notebooks:
- [training.ipynb](https://github.com/felixglush/kaggle-cassava-disease-classification/blob/master/training.ipynb)
- [inference.ipynb](https://github.com/felixglush/kaggle-cassava-disease-classification/blob/master/inference.ipynb)
- [scraps.ipynb](https://github.com/felixglush/kaggle-cassava-disease-classification/blob/master/scraps.ipynb): some snippets of code I played around with or tested
- [hyperparameter_tuning.ipynb](https://github.com/felixglush/kaggle-cassava-disease-classification/blob/master/hyperparameter_tuning.ipynb)

Resources:
- AdaBound [implementation](https://github.com/Luolc/AdaBound/).
- [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) for hyperparameter selection.
- EfficientNet-B4 Noisy Student implementation [here](https://rwightman.github.io/pytorch-image-models/).

Ideas to try:
1. Cross entropy loss, stratified CV, no fmix, cutmix, mixup, w gradient scaling & accumulation [done]
2. add hyperparam tuning with raytune [done]
5. emsemble of models - train a model for each fold and then average their predictions during inference [done]
13. AdaBound - "as good as SGD and as fast as Adam" [done]
2. smoothed cross entropy loss [done], bi tempered loss [done], focal loss, cosine focal loss
4. external data
8. Better ensemble prediction - majority vote [done], weighted average...?
10. resnet model
11. oversample classes 0,1,2,4 [done]
12. verify per class accuracy [done]
13. use the OOF prediction of the model trained with all data, and eliminate images where the predicted value is too small for the correct label. After eliminating a small quantity of training images, retrain from scratch with the remaining ones.
14. 1cycle policy [done]
15. Learning rate range plots for LR selection [done]

To do:
- assymetric loss, jsd loss
- Test time augmentation
- fmix, cutmix
- metric learning
- other models [wip]
- Average of model weight [done] 