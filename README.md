# CNN Visualization

|    | King Snake | Mastiff | Spider |
|:--:|:--:|:--:|:--:|
| Original Image | ![](/images/snake.jpg) | ![](/images/cat_dog.png) | ![](/images/spider.png) |
| Vanilla Backpropagation | ![](/images/grad_colorful_56.png) | ![](/images/grad_colorful_243.png) | ![](/images/grad_colorful_72.png) |
| Vanilla Backpropagation Saliency | ![](/images/grad_grayscale_56.png) | ![](/images/grad_grayscale_243.png) | ![](/images/grad_grayscale_72.png) |
| Vanilla Backpropagation Saliency<br> **X** <br> Image  | ![](/images/grad_x_image_56.png) | ![](/images/grad_x_image_243.png) | ![](/images/grad_x_image_72.png) |
| Guided Backpropagation | ![](/images/guided_grad_colorful_56.png) | ![](/images/guided_grad_colorful_243.png) |![](/images/guided_grad_colorful_72.png) |
| Guided Backpropagation Saliency | ![](/images/guided_grad_grayscale_56.png) | ![](/images/guided_grad_grayscale_243.png) |![](/images/guided_grad_grayscale_72.png) |
| Guided Backpropagation Saliency<br> **X** <br> Image | ![](/images/guided_grad_x_image_56.png) |![](/images/guided_grad_x_image_243.png) |![](/images/guided_grad_x_image_72.png) |

# Class Specific Image Generation
|   |  No Regularization |  L1 | L2 | Gaussian Blur |
|:--:|:--:|:--:|:--:|
| 130 |

## Usage

### server

```shell
python server.py
```

### Web

```shell
npm install
npm run serve
```
