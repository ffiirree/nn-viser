import abc
import torch
import torchvision.transforms as T
from torch import Tensor
from typing import List, Optional
import torchvision.transforms.functional as TF

__all__ = ['Augment', 'Original', 'Invert', 'Compose',
           'HorizontalFlip', 'VerticalFlip',
           'AdjustBrightness', 'AdjustContrast',
           'AdjustHue', 'AdjustSaturation', 'AdjustSharpness',
           'GaussianBlur', 'Noise', 'CroppedPad'
          ]

class Augment(abc.ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abc.abstractmethod
    def __call__(self, input: Tensor) -> Tensor:
        ...

    @abc.abstractmethod
    def reverse(self, input: Tensor) -> Tensor:
        ...
        
class Compose(Augment):
    def __init__(self, ops: list) -> None:
        self.ops = ops
    
    def __call__(self, input: Tensor) -> Tensor:
        for op in self.ops:
            input = op(input)
        return input
    
    def reverse(self, input: Tensor) -> Tensor:
        for op in self.ops:
            input = op.reverse(input)
        return input

class Original(Augment):
    def __init__(self) -> None:
        super().__init__()
        
    def __call__(self, input: Tensor) -> Tensor:
        return input
    
    def reverse(self, input: Tensor) -> Tensor:
        return input
    
class Invert(Augment):
    def __init__(self) -> None:
        super().__init__()
        
    def __call__(self, input: Tensor) -> Tensor:
        return TF.invert(input)
    
    def reverse(self, input: Tensor) -> Tensor:
        return input

class HorizontalFlip(Augment):
    def __init__(self) -> None:
        super().__init__()
        
    def __call__(self, input: Tensor) -> Tensor:
        return TF.hflip(input)
    
    def reverse(self, input: Tensor) -> Tensor:
        return TF.hflip(input)
    
class VerticalFlip(Augment):
    def __init__(self) -> None:
        super().__init__()
        
    def __call__(self, input: Tensor) -> Tensor:
        return TF.vflip(input)
    
    def reverse(self, input: Tensor) -> Tensor:
        return TF.vflip(input)

class AdjustBrightness(Augment):
    def __init__(self, brightness_factor: float) -> None:
        r"""
        Args:
            brightness_factor:
                0 -> black image
                1 -> original image
                2 -> increase the brightness by a factor of 2
        """
        super().__init__()
        
        self.brightness_factor = brightness_factor
    
    def __call__(self, input: Tensor) -> Tensor:
        return TF.adjust_brightness(input, self.brightness_factor)
    
    def reverse(self, input: Tensor) -> Tensor:
        return input
    
class AdjustContrast(Augment):
    def __init__(self, contrast_factor) -> None:
        r"""
        Args:
            contrast_factor:
                0 -> solid gray image
                1 -> original image
                2 -> increase the contrast by a factor of 2
        """
        super().__init__()
        
        self.contrast_factor = contrast_factor
        
    def __call__(self, input: Tensor) -> Tensor:
        return TF.adjust_contrast(input, self.contrast_factor)
    
    def reverse(self, input: Tensor) -> Tensor:
        return input
    
    
class AdjustHue(Augment):
    r"""The image hue is adjusted by converting the image to HSV and 
    cyclically shifting the intensities in the hue channel (H). The 
    image is then converted back to original image mode.
    """
    def __init__(self, hue_factor) -> None:
        r"""
        Args:
            hue_factor: [-0.5, 0.5]
                0.5 and -0.5 give complete reversal of hue channel in HSV space in positive and negative direction respectively
                0 -> no shift
        """
        super().__init__()
        
        self.hue_factor = hue_factor
        
    def __call__(self, input: Tensor) -> Tensor:
        return TF.adjust_hue(input, self.hue_factor)
    
    def reverse(self, input: Tensor) -> Tensor:
        return input
    
class AdjustSaturation(Augment):
    r"""Adjust color saturation of an image."""
    def __init__(self, saturation_factor) -> None:
        r"""
        Args:
            saturation_factor:
                0 -> a black and white image
                1 -> the original image
                2 -> enhance the saturation by a factor of 2.
        """
        super().__init__()
        
        self.saturation_factor = saturation_factor
        
    def __call__(self, input: Tensor) -> Tensor:
        return TF.adjust_saturation(input, self.saturation_factor)
    
    def reverse(self, input: Tensor) -> Tensor:
        return input
    
class AdjustSharpness(Augment):
    r"""Adjust color sharpness of an image."""
    def __init__(self, sharpness_factor) -> None:
        r"""
        Args:
            saturation_factor:
                0 -> a blurred image
                1 -> the original image
                2 -> increases the sharpness by a factor of 2
        """
        super().__init__()
        
        self.sharpness_factor = sharpness_factor
        
    def __call__(self, input: Tensor) -> Tensor:
        return TF.adjust_sharpness(input, self.sharpness_factor)
    
    def reverse(self, input: Tensor) -> Tensor:
        return input
    
class GaussianBlur(Augment):
    def __init__(self, kernel_size: List[int], sigma: Optional[List[float]] = None) -> None:
        super().__init__()
        
        self.kernel_size = kernel_size
        self.sigma = sigma
        
    def __call__(self, input: Tensor) -> Tensor:
        return TF.gaussian_blur(input, self.kernel_size, self.sigma)
    
    def reverse(self, input: Tensor) -> Tensor:
        return input
    
class Noise(Augment):
    def __init__(self, factor: float) -> None:
        super().__init__()
        
        self.factor = factor
        
    def __call__(self, input: Tensor) -> Tensor:
        noise = torch.randn(input.shape) / self.factor
        output = input.detach().clone()
        output.data = output.data + noise.data
            
        return output
    
    def reverse(self, input: Tensor) -> Tensor:
        return input
    
class CroppedPad(Augment):
    def __init__(self, size: List[int], padding: List[int], fill=0) -> None:
        r"""
        Args:
            size: [H, W]
            padding: [top, left]
        """
        super().__init__()
        self.size = size
        self.padding = padding
        self.fill = fill
        
    def __call__(self, input: Tensor) -> Tensor:
        padding = [self.padding[1], self.padding[0], input.shape[3] - self.size[1] - self.padding[1], input.shape[2] - self.size[0] - self.padding[0]]

        # H, W
        x = TF.center_crop(input, self.size)
        # left top right bottom
        x = TF.pad(x, padding, self.fill)
        assert input.shape == x.shape, 'Error: '
        
        return x
    
    def reverse(self, input: Tensor) -> Tensor:
        # top, left, height, width
        x = TF.crop(
            input,
            self.padding[0],
            self.padding[1],
            input.shape[2] - (self.padding[0]),
            input.shape[3] - (self.padding[1]),
        )
        
        padding_h = int((input.shape[2] - x.shape[2]) / 2)
        padding_w = int((input.shape[3] - x.shape[3]) / 2)
        x = TF.pad(x, [padding_w, padding_h, padding_w, padding_h])
        assert input.shape == x.shape, f"Error: input.shape != x.shape, {input.shape}/{x.shape}"

        return x