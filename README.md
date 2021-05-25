# Visualizer

![Visualizer](static/images/viser.png)

## TODO

- [x] Activations
- [x] Filters
- [x] `Maximizing the activation` - [Visualizing Higher-Layer Features of a Deep Network](http://www.iro.umontreal.ca/~lisa/publications2/index.php/publications/show/247), 2009
- [x] `Maximizing the class` - [Understanding Neural Networks Through Deep Visualization](https://arxiv.org/abs/1506.06579), 2015
- [ ] `Deconvolutional networks` - [Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901), 2013
- [x] `Saliency Maps` - [Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://arxiv.org/abs/1312.6034), 2013
- [x] `Guided backpropagation` - [Striving for Simplicity: The All Convolutional Net](https://arxiv.org/abs/1412.6806), 2014
- [x] `SmoothGrad` - [SmoothGrad: removing noise by adding noise](https://arxiv.org/abs/1706.03825), 2017
- [ ] `CAM(Class Activation Mapping)` - [Learning Deep Features for Discriminative Localization](https://arxiv.org/abs/1512.04150), 2015
- [x] `GradCAM` - [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391), 2016
- [ ] `IntergratedGradients` - [Axiomatic Attribution for Deep Networks](https://arxiv.org/abs/1703.01365), 2017

## usage

```shell
# build frontend
cd nn-viser/frontend
npm install
npm run build
```

```shell
# run server
cd nn-viser
python run.py
```

* Running on http://127.0.0.1:5000/
