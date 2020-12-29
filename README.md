# Kaggle Competition: Cassava Leaf Disease Classification

The competition can be found [here](https://www.kaggle.com/c/cassava-leaf-disease-classification/overview).

AdaBound [implementation](https://github.com/Luolc/AdaBound/).

[Ray Tune](https://docs.ray.io/en/latest/tune/index.html) for hyperparameter selection.

EfficientNet implementation from [here](https://rwightman.github.io/pytorch-image-models/).

Ideas to try:
1. Cross entropy loss, stratified CV, no fmix, cutmix, mixup, w gradient scaling & accumulation [done]
2. add hyperparam tuning with raytune [done]
5. emsemble of models - train a model for each fold and then average their predictions during inference [done]
13. AdaBound - "as good as SGD and as fast as Adam" [done]
2. smoothed cross entropy loss, bi tempered loss, focal loss, cosine focal loss
3. fmix, cutmix
4. external data
7. Test time augmentation
8. Better ensemble prediction - majority vote, other...?
10. resnet model
11. balanced classes instead of stratified?
12. verify per class accuracy
13. use the OOF prediction of the model trained with all data, and eliminate images where the predicted value is too small for the correct label. After eliminating a small quantity of training images, retrain from scratch with the remaining ones.
14. 1cycle policy
15. LR plots
