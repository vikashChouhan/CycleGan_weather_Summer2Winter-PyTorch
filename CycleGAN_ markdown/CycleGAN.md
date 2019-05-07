
# CycleGAN, Image-to-Image Translation

In this notebook, we're going to define and train a CycleGAN to read in an image from a set $X$ and transform it so that it looks as if it belongs in set $Y$. Specifically, we'll look at a set of images of [Yosemite national park](https://en.wikipedia.org/wiki/Yosemite_National_Park) taken either during the summer of winter. The seasons are our two domains!

>The objective will be to train generators that learn to transform an image from domain $X$ into an image that looks like it came from domain $Y$ (and vice versa). 

Some examples of image data in both sets are pictured below.

<img src='notebook_images/XY_season_images.png' width=50% />

### Unpaired Training Data

These images do not come with labels, but CycleGANs give us a way to learn the mapping between one image domain and another using an **unsupervised** approach. A CycleGAN is designed for image-to-image translation and it learns from unpaired training data. This means that in order to train a generator to translate images from domain $X$ to domain $Y$, we do not have to have exact correspondences between individual images in those domains. For example, in [the paper that introduced CycleGANs](https://arxiv.org/abs/1703.10593), the authors are able to translate between images of horses and zebras, even though there are no images of a zebra in exactly the same position as a horse or with exactly the same background, etc. Thus, CycleGANs enable learning a mapping from one domain $X$ to another domain $Y$ without having to find perfectly-matched, training pairs!

<img src='notebook_images/horse2zebra.jpg' width=50% />

### CycleGAN and Notebook Structure

A CycleGAN is made of two types of networks: **discriminators, and generators**. In this example, the discriminators are responsible for classifying images as real or fake (for both $X$ and $Y$ kinds of images). The generators are responsible for generating convincing, fake images for both kinds of images. 

This notebook will detail the steps we should take to define and train such a CycleGAN. 

>1. We'll load in the image data using PyTorch's DataLoader class to efficiently read in images from a specified directory. 
2. Then, We'll be tasked with defining the CycleGAN architecture according to provided specifications. We'll define the discriminator and the generator models.
3.We'll complete the training cycle by calculating the adversarial and cycle consistency losses for the generator and discriminator network and completing a number of training epochs. *It's suggested that you enable GPU usage for training.*
4. Finally, we'll evaluate your model by looking at the loss over time and looking at sample, generated images.


---

## Load and Visualize the Data

We'll first load in and visualize the training data, importing the necessary libraries to do so.

> If you are working locally, you'll need to download the data as a zip file by [clicking here](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/November/5be66e78_summer2winter-yosemite/summer2winter-yosemite.zip).

It may be named `summer2winter-yosemite/` with a dash or an underscore, so take note, extract the data to your home directory and make sure the below `image_dir` matches. Then you can proceed with the following loading code.


```python
# loading in and transforming data
import os
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# visualizing data
import matplotlib.pyplot as plt
import numpy as np
import warnings

%matplotlib inline
```

### DataLoaders

The `get_data_loader` function returns training and test DataLoaders that can load data efficiently and in specified batches. The function has the following parameters:
* `image_type`: `summer` or `winter`,  the names of the directories where the X and Y images are stored
* `image_dir`: name of the main image directory, which holds all training and test images
* `image_size`: resized, square image dimension (all images will be resized to this dim)
* `batch_size`: number of images in one batch of data

The test data is strictly for feeding to our generators, later on, so we can visualize some generated samples on fixed, test data.

We can see that this function is also responsible for making sure our images are of the right, square size (128x128x3) and converted into Tensor image types.

**It's suggested that you use the default values of these parameters.**

Note: If we are trying this code on a different set of data, we may get better results with larger `image_size` and `batch_size` parameters. If we change the `batch_size`, make sure that you create complete batches in the training loop otherwise you may get an error when trying to save sample data. 


```python
def get_data_loader(image_type, image_dir='summer2winter_yosemite', 
                    image_size=128, batch_size=16, num_workers=0):
    """Returns training and test data loaders for a given image type, either 'summer' or 'winter'. 
       These images will be resized to 128x128x3, by default, converted into Tensors, and normalized.
    """
    
    # resize and normalize the images
    transform = transforms.Compose([transforms.Resize(image_size), # resize to 128x128
                                    transforms.ToTensor()])

    # get training and test directories
    image_path =  image_dir
    train_path = os.path.join(image_path, image_type)
    test_path = os.path.join(image_path, 'test_{}'.format(image_type))

    # define datasets using ImageFolder
    train_dataset = datasets.ImageFolder(train_path, transform)
    test_dataset = datasets.ImageFolder(test_path, transform)

    # create and return DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
```


```python
# Create train and test dataloaders for images from the two domains X and Y
# image_type = directory names for our data
dataloader_X, test_dataloader_X = get_data_loader(image_type='summer')
dataloader_Y, test_dataloader_Y = get_data_loader(image_type='winter')
```

## Display some Training Images

Below we provide a function `imshow` that reshape some given images and converts them to NumPy images so that they can be displayed by `plt`. This cell should display a grid that contains a batch of image data from set $X$.


```python
# helper imshow function
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    

# get some images from X
dataiter = iter(dataloader_X)
# the "_" is a placeholder for no labels
images, _ = dataiter.next()

# show images
fig = plt.figure(figsize=(12, 8))
imshow(torchvision.utils.make_grid(images))
```


![png](output_7_0.png)


Next, let's visualize a batch of images from set $Y$.


```python
# get some images from Y
dataiter = iter(dataloader_Y)
images, _ = dataiter.next()

# show images
fig = plt.figure(figsize=(12,8))
imshow(torchvision.utils.make_grid(images))
```


![png](output_9_0.png)


### Pre-processing: scaling from -1 to 1

We need to do a bit of pre-processing; we know that the output of our `tanh` activated generator will contain pixel values in a range from -1 to 1, and so, we need to rescale our training images to a range of -1 to 1. (Right now, they are in a range from 0-1.)


```python
# current range
img = images[0]

print('Min: ', img.min())
print('Max: ', img.max())
```

    Min:  tensor(0.0039)
    Max:  tensor(0.9922)
    


```python
# helper scale function
def scale(x, feature_range=(-1, 1)):
    ''' Scale takes in an image x and returns that image, scaled
       with a feature_range of pixel values from -1 to 1. 
       This function assumes that the input x is already scaled from 0-255.'''
    
    # scale from 0-1 to feature_range
    min, max = feature_range
    x = x * (max - min) + min
    return x
```


```python
# scaled range
scaled_img = scale(img)

print('Scaled min: ', scaled_img.min())
print('Scaled max: ', scaled_img.max())
```

    Scaled min:  tensor(-0.9922)
    Scaled max:  tensor(0.9843)
    

---
## Define the Model

A CycleGAN is made of two discriminator and two generator networks.

## Discriminators

The discriminators, $D_X$ and $D_Y$, in this CycleGAN are convolutional neural networks that see an image and attempt to classify it as real or fake. In this case, real is indicated by an output close to 1 and fake as close to 0. The discriminators have the following architecture:

<img src='notebook_images/discriminator_layers.png' width=80% />

This network sees a 128x128x3 image, and passes it through 5 convolutional layers that downsample the image by a factor of 2. The first four convolutional layers have a BatchNorm and ReLu activation function applied to their output, and the last acts as a classification layer that outputs one value.

### Convolutional Helper Function

To define the discriminators, you're expected to use the provided `conv` function, which creates a convolutional layer + an optional batch norm layer.


```python
import torch.nn as nn
import torch.nn.functional as F

# helper conv function
def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                           kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)
```

### Define the Discriminator Architecture

Our task is to fill in the `__init__` function with the specified 5 layer conv net architecture. Both $D_X$ and $D_Y$ have the same architecture, so we only need to define one class, and later instantiate two discriminators. 
> It's recommended that we use a **kernel size of 4x4** and use that to determine the correct stride and padding size for each layer. [This Stanford resource](http://cs231n.github.io/convolutional-networks/#conv) may also help in determining stride and padding sizes.

* Define  convolutional layers in `__init__`
* Then fill in the forward behavior of the network

The `forward` function defines how an input image moves through the discriminator, and the most important thing is to pass it through your convolutional layers in order, with a **ReLu** activation function applied to all but the last layer.

We should **not** apply a sigmoid activation function to the output, here, and that is because we are planning on using a squared error loss for training. And you can read more about this loss function, later in the notebook.


```python
class Discriminator(nn.Module):
    
    def __init__(self, conv_dim=64):
        super(Discriminator, self).__init__()

        # Define all convolutional layers
        # Should accept an RGB image as input and output a single value
        self.conv1 = conv(3,64,4,batch_norm=False)
        self.conv2 = conv(64,128,4)
        self.conv3 = conv(128,256,4)
        self.conv4 = conv(256,512,4)
        
        self.conv5 = conv(512,1,4,1,batch_norm=False)
        

    def forward(self, x):
        # define feedforward behavior
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        
        return x
      
print(Discriminator())
```

    Discriminator(
      (conv1): Sequential(
        (0): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
      )
      (conv2): Sequential(
        (0): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv3): Sequential(
        (0): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv4): Sequential(
        (0): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv5): Sequential(
        (0): Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1), bias=False)
      )
    )
    

## Generators

The generators, `G_XtoY` and `G_YtoX` (sometimes called F), are made of an **encoder**, a conv net that is responsible for turning an image into a smaller feature representation, and a **decoder**, a *transpose_conv* net that is responsible for turning that representation into an transformed image. These generators, one from XtoY and one from YtoX, have the following architecture:

<img src='notebook_images/cyclegan_generator_ex.png' width=90% />

This network sees a 128x128x3 image, compresses it into a feature representation as it goes through three convolutional layers and reaches a series of residual blocks. It goes through a few (typically 6 or more) of these residual blocks, then it goes through three transpose convolutional layers (sometimes called *de-conv* layers) which upsample the output of the resnet blocks and create a new image!

Note that most of the convolutional and transpose-convolutional layers have BatchNorm and ReLu functions applied to their outputs with the exception of the final transpose convolutional layer, which has a `tanh` activation function applied to the output. Also, the residual blocks are made of convolutional and batch normalization layers, which we'll go over in more detail, next.

---
### Residual Block Class

To define the generators, we're expected to define a `ResidualBlock` class which will help you connect the encoder and decoder portions of the generators. You might be wondering, what exactly is a Resnet block? It may sound familiar from something like ResNet50 for image classification, pictured below.

<img src='notebook_images/resnet_50.png' width=90%/>

ResNet blocks rely on connecting the output of one layer with the input of an earlier layer. The motivation for this structure is as follows: very deep neural networks can be difficult to train. Deeper networks are more likely to have vanishing or exploding gradients and, therefore, have trouble reaching convergence; batch normalization helps with this a bit. However, during training, we often see that deep networks respond with a kind of training degradation. Essentially, the training accuracy stops improving and gets saturated at some point during training. In the worst cases, deep models would see their training accuracy actually worsen over time!

One solution to this problem is to use **Resnet blocks** that allow us to learn so-called *residual functions* as they are applied to layer inputs. You can read more about this proposed architecture in the paper, [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf) by Kaiming He et. al, and the below image is from that paper.

<img src='notebook_images/resnet_block.png' width=40%/>

### Residual Functions

Usually, when we create a deep learning model, the model (several layers with activations applied) is responsible for learning a mapping, `M`, from an input `x` to an output `y`.
>`M(x) = y` (Equation 1)

Instead of learning a direct mapping from `x` to `y`, we can instead define a **residual function**
> `F(x) = M(x) - x`

This looks at the difference between a mapping applied to x and the original input, x. `F(x)` is, typically, two convolutional layers + normalization layer and a ReLu in between. These convolutional layers should have the same number of inputs as outputs. This mapping can then be written as the following; a function of the residual function and the input x. The addition step creates a kind of loop that connects the input x to the output, y:
>`M(x) = F(x) + x` (Equation 2) or

>`y = F(x) + x` (Equation 3)

#### Optimizing a Residual Function

The idea is that it is easier to optimize this residual function `F(x)` than it is to optimize the original mapping `M(x)`. Consider an example; what if we want `y = x`?

From our first, direct mapping equation, **Equation 1**, we could set `M(x) = x` but it is easier to solve the residual equation `F(x) = 0`, which, when plugged in to **Equation 3**, yields `y = x`.


### Defining the `ResidualBlock` Class

To define the `ResidualBlock` class, we'll define residual functions (a series of layers), apply them to an input x and add them to that same input. This is defined just like any other neural network, with an `__init__` function and the addition step in the `forward` function. 

In our case, you'll want to define the residual block as:
* Two convolutional layers with the same size input and output
* Batch normalization applied to the outputs of the convolutional layers
* A ReLu function on the output of the *first* convolutional layer

Then, in the `forward` function, add the input x to this residual block. Feel free to use the helper `conv` function from above to create this block.


```python
# residual block class
class ResidualBlock(nn.Module):
    """Defines a residual block.
       This adds an input x to a convolutional layer (applied to x) with the same size input and output.
       These blocks allow a model to learn an effective transformation from one domain to another.
    """
    def __init__(self, conv_dim):
        super(ResidualBlock, self).__init__()
        # conv_dim = number of inputs  
        
        # define two convolutional layers + batch normalization that will act as our residual function, F(x)
        # layers should have the same shape input as output; I suggest a kernel_size of 3
        self.conv1 = conv(conv_dim,conv_dim,3,1)
        self.conv2 = conv(conv_dim,conv_dim,3,1)
        
    def forward(self, x):
        # apply a ReLu activation the outputs of the first layer
        # return a summed output, x + resnet_block(x)
        input_x = x
        x = F.relu(self.conv1(x))
        x = input_x + self.conv2(x)
        return x
      
      
print(ResidualBlock(128))
```

    ResidualBlock(
      (conv1): Sequential(
        (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv2): Sequential(
        (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    

### Transpose Convolutional Helper Function

To define the generators, we're expected to use the above `conv` function, `ResidualBlock` class, and the below `deconv` helper function, which creates a transpose convolutional layer + an optional batchnorm layer.


```python
# helper deconv function
def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a transpose convolutional layer, with optional batch normalization.
    """
    layers = []
    # append transpose conv layer
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
    # optional batch norm layer
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)
```

---
## Define the Generator Architecture

* Complete the `__init__` function with the specified 3 layer **encoder** convolutional net, a series of residual blocks (the number of which is given by `n_res_blocks`), and then a 3 layer **decoder** transpose convolutional net.
* Then complete the `forward` function to define the forward behavior of the generators. Recall that the last layer has a `tanh` activation function.

Both $G_{XtoY}$ and $G_{YtoX}$ have the same architecture, so we only need to define one class, and later instantiate two generators.


```python
class CycleGenerator(nn.Module):
    
    def __init__(self, conv_dim=64, n_res_blocks=6):
        super(CycleGenerator, self).__init__()

        # 1. Define the encoder part of the generator
        self.enc_conv1 = conv(3,conv_dim,4,2)
        self.enc_conv2 = conv(conv_dim,conv_dim*2,4,2)
        self.enc_conv3 = conv(conv_dim*2,conv_dim*4,4,2)
        

        # 2. Define the resnet part of the generator
        l = [ResidualBlock(conv_dim*4) for i in range(n_res_blocks)]
        self.resBlock = nn.Sequential(*l)

        # 3. Define the decoder part of the generator
        self.dec_conv1 = deconv(conv_dim*4,conv_dim*2,4,2)
        self.dec_conv2 = deconv(conv_dim*2,conv_dim,4,2)
        self.dec_conv3 = deconv(conv_dim,3,4,2)
       

    def forward(self, x):
        """Given an image x, returns a transformed image."""
        # define feedforward behavior, applying activations as necessary
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        
        x = self.resBlock(x)
        
        x = F.relu(self.dec_conv1(x))
        x = F.relu(self.dec_conv2(x))
        x = F.tanh(self.dec_conv3(x))
        return x
      
print(CycleGenerator())
```

    CycleGenerator(
      (enc_conv1): Sequential(
        (0): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (enc_conv2): Sequential(
        (0): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (enc_conv3): Sequential(
        (0): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (resBlock): Sequential(
        (0): ResidualBlock(
          (conv1): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (conv2): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): ResidualBlock(
          (conv1): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (conv2): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (2): ResidualBlock(
          (conv1): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (conv2): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (3): ResidualBlock(
          (conv1): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (conv2): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (4): ResidualBlock(
          (conv1): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (conv2): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (5): ResidualBlock(
          (conv1): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (conv2): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (dec_conv1): Sequential(
        (0): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (dec_conv2): Sequential(
        (0): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (dec_conv3): Sequential(
        (0): ConvTranspose2d(64, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    

---
## Create the complete network

Using the classes you defined earlier, you can define the discriminators and generators necessary to create a complete CycleGAN. The given parameters should work for training.

First, create two discriminators, one for checking if $X$ sample images are real, and one for checking if $Y$ sample images are real. Then the generators. Instantiate two of them, one for transforming a painting into a realistic photo and one for transforming a photo into  into a painting.


```python
def create_model(g_conv_dim=64, d_conv_dim=64, n_res_blocks=6):
    """Builds the generators and discriminators."""
    
    # Instantiate generators
    G_XtoY = CycleGenerator(g_conv_dim,n_res_blocks)
    G_YtoX = CycleGenerator(g_conv_dim,n_res_blocks)
    # Instantiate discriminators
    D_X = Discriminator(d_conv_dim)
    D_Y = Discriminator(d_conv_dim)

    # move models to GPU, if available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        G_XtoY.to(device)
        G_YtoX.to(device)
        D_X.to(device)
        D_Y.to(device)
        print('Models moved to GPU.')
    else:
        print('Only CPU available.')

    return G_XtoY, G_YtoX, D_X, D_Y
```


```python
# call the function to get models
G_XtoY, G_YtoX, D_X, D_Y = create_model()
```

    Models moved to GPU.
    

## Check that you've implemented this correctly

The function `create_model` should return the two generator and two discriminator networks. After we've defined these discriminator and generator components, it's good practice to check our work. The easiest way to do this is to print out our model architecture and read through it to make sure the parameters are what you expected. The next cell will print out their architectures.


```python
# helper function for printing the model architecture
def print_models(G_XtoY, G_YtoX, D_X, D_Y):
    """Prints model information for the generators and discriminators.
    """
    print("                     G_XtoY                    ")
    print("-----------------------------------------------")
    print(G_XtoY)
    print()

    print("                     G_YtoX                    ")
    print("-----------------------------------------------")
    print(G_YtoX)
    print()

    print("                      D_X                      ")
    print("-----------------------------------------------")
    print(D_X)
    print()

    print("                      D_Y                      ")
    print("-----------------------------------------------")
    print(D_Y)
    print()
    

# print all of the models
print_models(G_XtoY, G_YtoX, D_X, D_Y)
```

                         G_XtoY                    
    -----------------------------------------------
    CycleGenerator(
      (enc_conv1): Sequential(
        (0): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (enc_conv2): Sequential(
        (0): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (enc_conv3): Sequential(
        (0): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (resBlock): Sequential(
        (0): ResidualBlock(
          (conv1): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (conv2): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): ResidualBlock(
          (conv1): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (conv2): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (2): ResidualBlock(
          (conv1): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (conv2): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (3): ResidualBlock(
          (conv1): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (conv2): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (4): ResidualBlock(
          (conv1): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (conv2): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (5): ResidualBlock(
          (conv1): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (conv2): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (dec_conv1): Sequential(
        (0): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (dec_conv2): Sequential(
        (0): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (dec_conv3): Sequential(
        (0): ConvTranspose2d(64, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    
                         G_YtoX                    
    -----------------------------------------------
    CycleGenerator(
      (enc_conv1): Sequential(
        (0): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (enc_conv2): Sequential(
        (0): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (enc_conv3): Sequential(
        (0): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (resBlock): Sequential(
        (0): ResidualBlock(
          (conv1): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (conv2): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): ResidualBlock(
          (conv1): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (conv2): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (2): ResidualBlock(
          (conv1): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (conv2): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (3): ResidualBlock(
          (conv1): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (conv2): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (4): ResidualBlock(
          (conv1): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (conv2): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (5): ResidualBlock(
          (conv1): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (conv2): Sequential(
            (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (dec_conv1): Sequential(
        (0): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (dec_conv2): Sequential(
        (0): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (dec_conv3): Sequential(
        (0): ConvTranspose2d(64, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    
                          D_X                      
    -----------------------------------------------
    Discriminator(
      (conv1): Sequential(
        (0): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
      )
      (conv2): Sequential(
        (0): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv3): Sequential(
        (0): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv4): Sequential(
        (0): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv5): Sequential(
        (0): Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1), bias=False)
      )
    )
    
                          D_Y                      
    -----------------------------------------------
    Discriminator(
      (conv1): Sequential(
        (0): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
      )
      (conv2): Sequential(
        (0): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv3): Sequential(
        (0): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv4): Sequential(
        (0): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv5): Sequential(
        (0): Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1), bias=False)
      )
    )
    
    

## Discriminator and Generator Losses

Computing the discriminator and the generator losses are key to getting a CycleGAN to train.

<img src='notebook_images/CycleGAN_loss.png' width=90% height=90% />

**Image from [original paper](https://arxiv.org/abs/1703.10593) by Jun-Yan Zhu et. al.**

* The CycleGAN contains two mapping functions $G: X \rightarrow Y$ and $F: Y \rightarrow X$, and associated adversarial discriminators $D_Y$ and $D_X$. **(a)** $D_Y$ encourages $G$ to translate $X$ into outputs indistinguishable from domain $Y$, and vice versa for $D_X$ and $F$.

* To further regularize the mappings, we introduce two cycle consistency losses that capture the intuition that if
we translate from one domain to the other and back again we should arrive at where we started. **(b)** Forward cycle-consistency loss and **(c)** backward cycle-consistency loss.

## Least Squares GANs

We've seen that regular GANs treat the discriminator as a classifier with the sigmoid cross entropy loss function. However, this loss function may lead to the vanishing gradients problem during the learning process. To overcome such a problem, we'll use a least squares loss function for the discriminator. This structure is also referred to as a least squares GAN or LSGAN, and you can [read the original paper on LSGANs, here](https://arxiv.org/pdf/1611.04076.pdf). The authors show that LSGANs are able to generate higher quality images than regular GANs and that this loss type is a bit more stable during training! 

### Discriminator Losses

The discriminator losses will be mean squared errors between the output of the discriminator, given an image, and the target value, 0 or 1, depending on whether it should classify that image as fake or real. For example, for a *real* image, `x`, we can train $D_X$ by looking at how close it is to recognizing and image `x` as real using the mean squared error:

```
out_x = D_X(x)
real_err = torch.mean((out_x-1)**2)
```

### Generator Losses

Calculating the generator losses will look somewhat similar to calculating the discriminator loss; there will still be steps in which you generate fake images that look like they belong to the set of $X$ images but are based on real images in set $Y$, and vice versa. You'll compute the "real loss" on those generated images by looking at the output of the discriminator as it's applied to these _fake_ images; this time, your generator aims to make the discriminator classify these fake images as *real* images. 

#### Cycle Consistency Loss

In addition to the adversarial losses, the generator loss terms will also include the **cycle consistency loss**. This loss is a measure of how good a reconstructed image is, when compared to an original image. 

Say you have a fake, generated image, `x_hat`, and a real image, `y`. We can get a reconstructed `y_hat` by applying `G_XtoY(x_hat) = y_hat` and then check to see if this reconstruction `y_hat` and the orginal image `y` match. For this, we recommed calculating the L1 loss, which is an absolute difference, between reconstructed and real images. We may also choose to multiply this loss by some weight value `lambda_weight` to convey its importance.

<img src='notebook_images/reconstruction_error.png' width=40% height=40% />

The total generator loss will be the sum of the generator losses and the forward and backward cycle consistency losses.

---
### Define Loss Functions

To help us calculate the discriminator and gnerator losses during training, let's define some helpful loss functions. Here, we'll define three.
1. `real_mse_loss` that looks at the output of a discriminator and returns the error based on how close that output is to being classified as real. This should be a mean squared error.
2. `fake_mse_loss` that looks at the output of a discriminator and returns the error based on how close that output is to being classified as fake. This should be a mean squared error.
3. `cycle_consistency_loss` that looks at a set of real image and a set of reconstructed/generated images, and returns the mean absolute error between them. This has a `lambda_weight` parameter that will weight the mean absolute error in a batch.

It's recommended to take a [look at the original, CycleGAN paper](https://arxiv.org/pdf/1703.10593.pdf) to get a starting value for `lambda_weight`.




```python
def real_mse_loss(D_out):
    # how close is the produced output from being "real"?
    return torch.mean((D_out-1)**2)

def fake_mse_loss(D_out):
    # how close is the produced output from being "false"?
    return torch.mean(D_out**2)

def cycle_consistency_loss(real_im, reconstructed_im, lambda_weight):
    # calculate reconstruction loss 
    # return weighted loss
    return torch.mean(torch.abs(real_im-reconstructed_im)) * lambda_weight
    

```

### Define the Optimizers

Next, let's define how this model will update its weights. This uses [Adam](https://pytorch.org/docs/stable/optim.html#algorithms) optimizers for the discriminator and generator. It's again recommended that you take a [look at the original, CycleGAN paper](https://arxiv.org/pdf/1703.10593.pdf) to get starting hyperparameter values.



```python
import torch.optim as optim

# hyperparams for Adam optimizers
lr= 0.0002
beta1= 0.5
beta2= 0.999

g_params = list(G_XtoY.parameters()) + list(G_YtoX.parameters())  # Get generator parameters

# Create optimizers for the generators and discriminators
g_optimizer = optim.Adam(g_params, lr, [beta1, beta2])
d_x_optimizer = optim.Adam(D_X.parameters(), lr, [beta1, beta2])
d_y_optimizer = optim.Adam(D_Y.parameters(), lr, [beta1, beta2])
```

---

## Training a CycleGAN

When a CycleGAN trains, and sees one batch of real images from set $X$ and $Y$, it trains by performing the following steps:

**Training the Discriminators**
1. Compute the discriminator $D_X$ loss on real images
2. Generate fake images that look like domain $X$ based on real images in domain $Y$
3. Compute the fake loss for $D_X$
4. Compute the total loss and perform backpropagation and $D_X$ optimization
5. Repeat steps 1-4 only with $D_Y$ and your domains switched!


**Training the Generators**
1. Generate fake images that look like domain $X$ based on real images in domain $Y$
2. Compute the generator loss based on how $D_X$ responds to fake $X$
3. Generate *reconstructed* $\hat{Y}$ images based on the fake $X$ images generated in step 1
4. Compute the cycle consistency loss by comparing the reconstructions with real $Y$ images
5. Repeat steps 1-4 only swapping domains
6. Add up all the generator and reconstruction losses and perform backpropagation + optimization

<img src='notebook_images/cycle_consistency_ex.png' width=70% />


### Saving Your Progress

A CycleGAN repeats its training process, alternating between training the discriminators and the generators, for a specified number of training iterations. Here is given code that will save some example generated images that the CycleGAN has learned to generate after a certain number of training iterations. Along with looking at the losses, these example generations should give you an idea of how well your network has trained.

Below, we may choose to keep all default parameters; our only task is to calculate the appropriate losses and complete the training cycle.


```python
# import save code
from helpers import save_samples, checkpoint
```


```python
# train the network
def training_loop(dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y, 
                  n_epochs=1000):
    
    print_every=10
    
    # keep track of losses over time
    losses = []

    test_iter_X = iter(test_dataloader_X)
    test_iter_Y = iter(test_dataloader_Y)

    # Get some fixed data from domains X and Y for sampling. These are images that are held
    # constant throughout training, that allow us to inspect the model's performance.
    fixed_X = test_iter_X.next()[0]
    fixed_Y = test_iter_Y.next()[0]
    fixed_X = scale(fixed_X) # make sure to scale to a range -1 to 1
    fixed_Y = scale(fixed_Y)

    # batches per epoch
    iter_X = iter(dataloader_X)
    iter_Y = iter(dataloader_Y)
    batches_per_epoch = min(len(iter_X), len(iter_Y))

    for epoch in range(1, n_epochs+1):

        # Reset iterators for each epoch
        if epoch % batches_per_epoch == 0:
            iter_X = iter(dataloader_X)
            iter_Y = iter(dataloader_Y)

        images_X, _ = iter_X.next()
        images_X = scale(images_X) # make sure to scale to a range -1 to 1

        images_Y, _ = iter_Y.next()
        images_Y = scale(images_Y)
        
        # move images to GPU if available (otherwise stay on CPU)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        images_X = images_X.to(device)
        images_Y = images_Y.to(device)


        # ============================================
        #            TRAIN THE DISCRIMINATORS
        # ============================================

        ##   First: D_X, real and fake loss components   ##
        
        d_x_optimizer.zero_grad()
        
        # 1. Compute the discriminator losses on real images
        D_X_real_loss = real_mse_loss(D_X(images_X))
        
        # 2. Generate fake images that look like domain X based on real images in domain Y
        G_Y2X_fake_image = G_YtoX(images_Y) 
        
        # 3. Compute the fake loss for D_X
        D_X_fake_loss = fake_mse_loss(D_X(G_Y2X_fake_image))
        
        # 4. Compute the total loss and perform backprop
        d_x_loss = D_X_real_loss + D_X_fake_loss
        d_x_loss.backward()
        d_x_optimizer.step()
        
        ##   Second: D_Y, real and fake loss components   ##
        D_Y_real_loss = real_mse_loss(D_Y(images_Y))
        G_X2Y_fake_image = G_XtoY(images_X) 
        D_Y_fake_loss = fake_mse_loss(D_Y(G_X2Y_fake_image))
        
        d_y_loss = D_Y_real_loss + D_Y_fake_loss
        
        d_y_loss.backward()
        d_y_optimizer.step()


        # =========================================
        #            TRAIN THE GENERATORS
        # =========================================

        ##    First: generate fake X images and reconstructed Y images    ##
        g_optimizer.zero_grad()
        
        # 1. Generate fake images that look like domain X based on real images in domain Y
        G_X_img = G_YtoX(images_Y)
        
        # 2. Compute the generator loss based on domain X
        G_X_real_loss = real_mse_loss(D_X(G_X_img))
        
        # 3. Create a reconstructed y
        G_Y_reconstructed = G_XtoY(G_X_img)
        
        # 4. Compute the cycle consistency loss (the reconstruction loss)
        G_Y_consistency_loss = cycle_consistency_loss(images_Y,G_Y_reconstructed,10)

        ##    Second: generate fake Y images and reconstructed X images    ##
        G_Y_img = G_XtoY(images_X)
        G_Y_real_loss = real_mse_loss(D_Y(G_Y_img))
        G_X_reconstructed = G_YtoX(G_Y_img)
        G_X_consistency_loss = cycle_consistency_loss(images_X,G_X_reconstructed,10)
        
        # 5. Add up all generator and reconstructed losses and perform backprop
        g_total_loss = G_X_real_loss + G_Y_real_loss + G_Y_consistency_loss + G_X_consistency_loss
        g_total_loss.backward()
        g_optimizer.step()
        
        # Print the log info
        if epoch % print_every == 0:
            # append real and fake discriminator losses and the generator loss
            losses.append((d_x_loss.item(), d_y_loss.item(), g_total_loss.item()))
            print('Epoch [{:5d}/{:5d}] | d_X_loss: {:6.4f} | d_Y_loss: {:6.4f} | g_total_loss: {:6.4f}'.format(
                    epoch, n_epochs, d_x_loss.item(), d_y_loss.item(), g_total_loss.item()))

            
        sample_every=100
        # Save the generated samples
        if epoch % sample_every == 0:
            G_YtoX.eval() # set generators to eval mode for sample generation
            G_XtoY.eval()
            save_samples(epoch, fixed_Y, fixed_X, G_YtoX, G_XtoY, batch_size=16)
            G_YtoX.train()
            G_XtoY.train()

        # uncomment these lines, if you want to save your model
#         checkpoint_every=1000
#         # Save the model parameters
#         if epoch % checkpoint_every == 0:
#             checkpoint(epoch, G_XtoY, G_YtoX, D_X, D_Y)

    return losses

```


```python
n_epochs = 2000 # keep this small when testing if a model first works, then increase it to >=1000

losses = training_loop(dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y, n_epochs=n_epochs)
```

    Epoch [   10/ 2000] | d_X_loss: 0.3908 | d_Y_loss: 0.5196 | g_total_loss: 4.0003
    Epoch [   20/ 2000] | d_X_loss: 0.2827 | d_Y_loss: 0.5251 | g_total_loss: 3.9263
    Epoch [   30/ 2000] | d_X_loss: 0.5576 | d_Y_loss: 0.5676 | g_total_loss: 3.3420
    Epoch [   40/ 2000] | d_X_loss: 0.4097 | d_Y_loss: 0.6527 | g_total_loss: 3.6992
    Epoch [   50/ 2000] | d_X_loss: 0.4269 | d_Y_loss: 0.7527 | g_total_loss: 3.6087
    Epoch [   60/ 2000] | d_X_loss: 0.4510 | d_Y_loss: 0.6422 | g_total_loss: 3.7853
    Epoch [   70/ 2000] | d_X_loss: 0.3271 | d_Y_loss: 0.6786 | g_total_loss: 3.2795
    Epoch [   80/ 2000] | d_X_loss: 0.3700 | d_Y_loss: 0.7087 | g_total_loss: 3.6273
    Epoch [   90/ 2000] | d_X_loss: 0.4506 | d_Y_loss: 0.7077 | g_total_loss: 3.9180
    Epoch [  100/ 2000] | d_X_loss: 0.4182 | d_Y_loss: 0.7122 | g_total_loss: 3.5773
    Saved samples_cyclegan/sample-000100-X-Y.png
    Saved samples_cyclegan/sample-000100-Y-X.png
    Epoch [  110/ 2000] | d_X_loss: 0.5340 | d_Y_loss: 0.6516 | g_total_loss: 3.3455
    Epoch [  120/ 2000] | d_X_loss: 0.4625 | d_Y_loss: 0.6549 | g_total_loss: 3.4522
    Epoch [  130/ 2000] | d_X_loss: 0.4469 | d_Y_loss: 0.6477 | g_total_loss: 3.6395
    Epoch [  140/ 2000] | d_X_loss: 0.3371 | d_Y_loss: 0.5929 | g_total_loss: 3.3198
    Epoch [  150/ 2000] | d_X_loss: 0.3949 | d_Y_loss: 0.5577 | g_total_loss: 3.5048
    Epoch [  160/ 2000] | d_X_loss: 0.3579 | d_Y_loss: 0.6077 | g_total_loss: 3.5555
    Epoch [  170/ 2000] | d_X_loss: 0.3587 | d_Y_loss: 0.5370 | g_total_loss: 3.3549
    Epoch [  180/ 2000] | d_X_loss: 0.3563 | d_Y_loss: 0.5639 | g_total_loss: 4.0384
    Epoch [  190/ 2000] | d_X_loss: 0.1877 | d_Y_loss: 0.5956 | g_total_loss: 3.8218
    Epoch [  200/ 2000] | d_X_loss: 0.3870 | d_Y_loss: 0.5500 | g_total_loss: 3.1688
    Saved samples_cyclegan/sample-000200-X-Y.png
    Saved samples_cyclegan/sample-000200-Y-X.png
    Epoch [  210/ 2000] | d_X_loss: 0.2716 | d_Y_loss: 0.6405 | g_total_loss: 3.5022
    Epoch [  220/ 2000] | d_X_loss: 0.4254 | d_Y_loss: 0.6726 | g_total_loss: 3.5711
    Epoch [  230/ 2000] | d_X_loss: 0.2825 | d_Y_loss: 0.7194 | g_total_loss: 3.6237
    Epoch [  240/ 2000] | d_X_loss: 0.6414 | d_Y_loss: 0.7057 | g_total_loss: 3.0326
    Epoch [  250/ 2000] | d_X_loss: 0.6502 | d_Y_loss: 0.7216 | g_total_loss: 3.1187
    Epoch [  260/ 2000] | d_X_loss: 0.4791 | d_Y_loss: 0.6529 | g_total_loss: 2.7702
    Epoch [  270/ 2000] | d_X_loss: 0.3793 | d_Y_loss: 0.6291 | g_total_loss: 3.4453
    Epoch [  280/ 2000] | d_X_loss: 0.3563 | d_Y_loss: 0.6173 | g_total_loss: 3.8744
    Epoch [  290/ 2000] | d_X_loss: 0.3522 | d_Y_loss: 0.5801 | g_total_loss: 4.0950
    Epoch [  300/ 2000] | d_X_loss: 0.4863 | d_Y_loss: 0.5770 | g_total_loss: 3.3658
    Saved samples_cyclegan/sample-000300-X-Y.png
    Saved samples_cyclegan/sample-000300-Y-X.png
    Epoch [  310/ 2000] | d_X_loss: 0.4254 | d_Y_loss: 0.5728 | g_total_loss: 3.5712
    Epoch [  320/ 2000] | d_X_loss: 0.6564 | d_Y_loss: 0.5183 | g_total_loss: 3.1784
    Epoch [  330/ 2000] | d_X_loss: 0.5352 | d_Y_loss: 0.5597 | g_total_loss: 3.7186
    Epoch [  340/ 2000] | d_X_loss: 0.2827 | d_Y_loss: 0.5971 | g_total_loss: 3.5514
    Epoch [  350/ 2000] | d_X_loss: 0.3822 | d_Y_loss: 0.6012 | g_total_loss: 3.2769
    Epoch [  360/ 2000] | d_X_loss: 0.3774 | d_Y_loss: 0.5863 | g_total_loss: 3.5751
    Epoch [  370/ 2000] | d_X_loss: 0.2850 | d_Y_loss: 0.5929 | g_total_loss: 3.3102
    Epoch [  380/ 2000] | d_X_loss: 0.3408 | d_Y_loss: 0.6189 | g_total_loss: 3.5951
    Epoch [  390/ 2000] | d_X_loss: 0.4451 | d_Y_loss: 0.6232 | g_total_loss: 3.6746
    Epoch [  400/ 2000] | d_X_loss: 0.4249 | d_Y_loss: 0.6579 | g_total_loss: 3.2274
    Saved samples_cyclegan/sample-000400-X-Y.png
    Saved samples_cyclegan/sample-000400-Y-X.png
    Epoch [  410/ 2000] | d_X_loss: 0.3497 | d_Y_loss: 0.6717 | g_total_loss: 3.3629
    Epoch [  420/ 2000] | d_X_loss: 0.5953 | d_Y_loss: 0.6763 | g_total_loss: 2.7858
    Epoch [  430/ 2000] | d_X_loss: 0.3288 | d_Y_loss: 0.6360 | g_total_loss: 3.3495
    Epoch [  440/ 2000] | d_X_loss: 0.4111 | d_Y_loss: 0.6224 | g_total_loss: 3.5843
    Epoch [  450/ 2000] | d_X_loss: 0.3906 | d_Y_loss: 0.5601 | g_total_loss: 3.3582
    Epoch [  460/ 2000] | d_X_loss: 0.5266 | d_Y_loss: 0.5552 | g_total_loss: 3.5802
    Epoch [  470/ 2000] | d_X_loss: 0.5742 | d_Y_loss: 0.5359 | g_total_loss: 2.9097
    Epoch [  480/ 2000] | d_X_loss: 0.3398 | d_Y_loss: 0.5370 | g_total_loss: 3.4772
    Epoch [  490/ 2000] | d_X_loss: 0.3376 | d_Y_loss: 0.5585 | g_total_loss: 3.1877
    Epoch [  500/ 2000] | d_X_loss: 0.3926 | d_Y_loss: 0.5434 | g_total_loss: 3.4803
    Saved samples_cyclegan/sample-000500-X-Y.png
    Saved samples_cyclegan/sample-000500-Y-X.png
    Epoch [  510/ 2000] | d_X_loss: 0.4915 | d_Y_loss: 0.5876 | g_total_loss: 3.3370
    Epoch [  520/ 2000] | d_X_loss: 0.3914 | d_Y_loss: 0.6241 | g_total_loss: 3.0653
    Epoch [  530/ 2000] | d_X_loss: 0.3848 | d_Y_loss: 0.6060 | g_total_loss: 3.7220
    Epoch [  540/ 2000] | d_X_loss: 0.3907 | d_Y_loss: 0.6376 | g_total_loss: 3.3617
    Epoch [  550/ 2000] | d_X_loss: 0.3620 | d_Y_loss: 0.6534 | g_total_loss: 3.1231
    Epoch [  560/ 2000] | d_X_loss: 0.3575 | d_Y_loss: 0.6642 | g_total_loss: 3.3502
    Epoch [  570/ 2000] | d_X_loss: 0.3208 | d_Y_loss: 0.6525 | g_total_loss: 3.7238
    Epoch [  580/ 2000] | d_X_loss: 0.3582 | d_Y_loss: 0.6228 | g_total_loss: 2.7846
    Epoch [  590/ 2000] | d_X_loss: 0.3943 | d_Y_loss: 0.6138 | g_total_loss: 3.3536
    Epoch [  600/ 2000] | d_X_loss: 0.2638 | d_Y_loss: 0.5842 | g_total_loss: 3.5995
    Saved samples_cyclegan/sample-000600-X-Y.png
    Saved samples_cyclegan/sample-000600-Y-X.png
    Epoch [  610/ 2000] | d_X_loss: 0.4190 | d_Y_loss: 0.5549 | g_total_loss: 4.2880
    Epoch [  620/ 2000] | d_X_loss: 0.4819 | d_Y_loss: 0.5455 | g_total_loss: 3.3035
    Epoch [  630/ 2000] | d_X_loss: 0.5202 | d_Y_loss: 0.5349 | g_total_loss: 3.7227
    Epoch [  640/ 2000] | d_X_loss: 0.3845 | d_Y_loss: 0.5240 | g_total_loss: 3.2742
    Epoch [  650/ 2000] | d_X_loss: 0.3111 | d_Y_loss: 0.5265 | g_total_loss: 3.1096
    Epoch [  660/ 2000] | d_X_loss: 0.3773 | d_Y_loss: 0.5381 | g_total_loss: 3.8817
    Epoch [  670/ 2000] | d_X_loss: 0.2647 | d_Y_loss: 0.5510 | g_total_loss: 3.6321
    Epoch [  680/ 2000] | d_X_loss: 0.3079 | d_Y_loss: 0.5802 | g_total_loss: 3.3153
    Epoch [  690/ 2000] | d_X_loss: 0.3563 | d_Y_loss: 0.6245 | g_total_loss: 3.1325
    Epoch [  700/ 2000] | d_X_loss: 0.3303 | d_Y_loss: 0.6595 | g_total_loss: 3.3316
    Saved samples_cyclegan/sample-000700-X-Y.png
    Saved samples_cyclegan/sample-000700-Y-X.png
    Epoch [  710/ 2000] | d_X_loss: 0.3769 | d_Y_loss: 0.6771 | g_total_loss: 3.0968
    Epoch [  720/ 2000] | d_X_loss: 0.4348 | d_Y_loss: 0.6675 | g_total_loss: 3.3088
    Epoch [  730/ 2000] | d_X_loss: 0.4485 | d_Y_loss: 0.6535 | g_total_loss: 3.6647
    Epoch [  740/ 2000] | d_X_loss: 0.3319 | d_Y_loss: 0.6285 | g_total_loss: 3.1984
    Epoch [  750/ 2000] | d_X_loss: 0.4564 | d_Y_loss: 0.5698 | g_total_loss: 3.9921
    Epoch [  760/ 2000] | d_X_loss: 0.3267 | d_Y_loss: 0.5650 | g_total_loss: 3.2505
    Epoch [  770/ 2000] | d_X_loss: 0.3483 | d_Y_loss: 0.5433 | g_total_loss: 3.7722
    Epoch [  780/ 2000] | d_X_loss: 0.3365 | d_Y_loss: 0.5263 | g_total_loss: 3.6107
    Epoch [  790/ 2000] | d_X_loss: 0.3581 | d_Y_loss: 0.5278 | g_total_loss: 2.7848
    Epoch [  800/ 2000] | d_X_loss: 0.3783 | d_Y_loss: 0.5332 | g_total_loss: 3.6428
    Saved samples_cyclegan/sample-000800-X-Y.png
    Saved samples_cyclegan/sample-000800-Y-X.png
    Epoch [  810/ 2000] | d_X_loss: 0.4013 | d_Y_loss: 0.5229 | g_total_loss: 3.7397
    Epoch [  820/ 2000] | d_X_loss: 0.3328 | d_Y_loss: 0.5745 | g_total_loss: 3.4196
    Epoch [  830/ 2000] | d_X_loss: 0.3444 | d_Y_loss: 0.5744 | g_total_loss: 3.7772
    Epoch [  840/ 2000] | d_X_loss: 0.3196 | d_Y_loss: 0.5770 | g_total_loss: 3.2350
    Epoch [  850/ 2000] | d_X_loss: 0.4231 | d_Y_loss: 0.5978 | g_total_loss: 2.9469
    Epoch [  860/ 2000] | d_X_loss: 0.3279 | d_Y_loss: 0.6488 | g_total_loss: 2.8604
    Epoch [  870/ 2000] | d_X_loss: 0.3611 | d_Y_loss: 0.6526 | g_total_loss: 2.7246
    Epoch [  880/ 2000] | d_X_loss: 0.3529 | d_Y_loss: 0.6420 | g_total_loss: 3.3650
    Epoch [  890/ 2000] | d_X_loss: 0.3843 | d_Y_loss: 0.6632 | g_total_loss: 3.2594
    Epoch [  900/ 2000] | d_X_loss: 0.4000 | d_Y_loss: 0.6361 | g_total_loss: 3.3979
    Saved samples_cyclegan/sample-000900-X-Y.png
    Saved samples_cyclegan/sample-000900-Y-X.png
    Epoch [  910/ 2000] | d_X_loss: 0.4880 | d_Y_loss: 0.5932 | g_total_loss: 3.4026
    Epoch [  920/ 2000] | d_X_loss: 0.3413 | d_Y_loss: 0.5672 | g_total_loss: 2.8978
    Epoch [  930/ 2000] | d_X_loss: 0.2817 | d_Y_loss: 0.5508 | g_total_loss: 3.4321
    Epoch [  940/ 2000] | d_X_loss: 0.3829 | d_Y_loss: 0.5307 | g_total_loss: 3.5434
    Epoch [  950/ 2000] | d_X_loss: 0.3624 | d_Y_loss: 0.5368 | g_total_loss: 2.7115
    Epoch [  960/ 2000] | d_X_loss: 0.3951 | d_Y_loss: 0.5244 | g_total_loss: 3.0112
    Epoch [  970/ 2000] | d_X_loss: 0.2854 | d_Y_loss: 0.5381 | g_total_loss: 3.1927
    Epoch [  980/ 2000] | d_X_loss: 0.3577 | d_Y_loss: 0.5491 | g_total_loss: 3.1398
    Epoch [  990/ 2000] | d_X_loss: 0.2341 | d_Y_loss: 0.5320 | g_total_loss: 3.8045
    Epoch [ 1000/ 2000] | d_X_loss: 0.3105 | d_Y_loss: 0.5675 | g_total_loss: 3.0063
    Saved samples_cyclegan/sample-001000-X-Y.png
    Saved samples_cyclegan/sample-001000-Y-X.png
    Epoch [ 1010/ 2000] | d_X_loss: 0.5026 | d_Y_loss: 0.5942 | g_total_loss: 3.0799
    Epoch [ 1020/ 2000] | d_X_loss: 0.2879 | d_Y_loss: 0.6016 | g_total_loss: 2.7306
    Epoch [ 1030/ 2000] | d_X_loss: 0.5224 | d_Y_loss: 0.6521 | g_total_loss: 2.8705
    Epoch [ 1040/ 2000] | d_X_loss: 0.3426 | d_Y_loss: 0.6395 | g_total_loss: 3.0361
    Epoch [ 1050/ 2000] | d_X_loss: 0.3851 | d_Y_loss: 0.6632 | g_total_loss: 3.1317
    Epoch [ 1060/ 2000] | d_X_loss: 0.3403 | d_Y_loss: 0.6449 | g_total_loss: 2.9362
    Epoch [ 1070/ 2000] | d_X_loss: 0.3223 | d_Y_loss: 0.6227 | g_total_loss: 2.8064
    Epoch [ 1080/ 2000] | d_X_loss: 0.4410 | d_Y_loss: 0.5629 | g_total_loss: 3.0934
    Epoch [ 1090/ 2000] | d_X_loss: 0.3121 | d_Y_loss: 0.5547 | g_total_loss: 3.0403
    Epoch [ 1100/ 2000] | d_X_loss: 0.2577 | d_Y_loss: 0.5405 | g_total_loss: 3.1456
    Saved samples_cyclegan/sample-001100-X-Y.png
    Saved samples_cyclegan/sample-001100-Y-X.png
    Epoch [ 1110/ 2000] | d_X_loss: 0.2825 | d_Y_loss: 0.5496 | g_total_loss: 3.8128
    Epoch [ 1120/ 2000] | d_X_loss: 0.4006 | d_Y_loss: 0.5220 | g_total_loss: 3.0756
    Epoch [ 1130/ 2000] | d_X_loss: 0.3171 | d_Y_loss: 0.5219 | g_total_loss: 3.1386
    Epoch [ 1140/ 2000] | d_X_loss: 0.3263 | d_Y_loss: 0.5195 | g_total_loss: 3.2513
    Epoch [ 1150/ 2000] | d_X_loss: 0.2996 | d_Y_loss: 0.5421 | g_total_loss: 3.5828
    Epoch [ 1160/ 2000] | d_X_loss: 0.1717 | d_Y_loss: 0.5525 | g_total_loss: 3.7188
    Epoch [ 1170/ 2000] | d_X_loss: 0.3989 | d_Y_loss: 0.5813 | g_total_loss: 2.9517
    Epoch [ 1180/ 2000] | d_X_loss: 0.3532 | d_Y_loss: 0.6055 | g_total_loss: 2.7426
    Epoch [ 1190/ 2000] | d_X_loss: 0.2934 | d_Y_loss: 0.6481 | g_total_loss: 3.7147
    Epoch [ 1200/ 2000] | d_X_loss: 0.3409 | d_Y_loss: 0.6601 | g_total_loss: 2.6008
    Saved samples_cyclegan/sample-001200-X-Y.png
    Saved samples_cyclegan/sample-001200-Y-X.png
    Epoch [ 1210/ 2000] | d_X_loss: 0.2754 | d_Y_loss: 0.6828 | g_total_loss: 2.8676
    Epoch [ 1220/ 2000] | d_X_loss: 0.3950 | d_Y_loss: 0.6492 | g_total_loss: 3.4421
    Epoch [ 1230/ 2000] | d_X_loss: 0.2573 | d_Y_loss: 0.6451 | g_total_loss: 3.1399
    Epoch [ 1240/ 2000] | d_X_loss: 0.2974 | d_Y_loss: 0.6109 | g_total_loss: 3.2178
    Epoch [ 1250/ 2000] | d_X_loss: 0.2577 | d_Y_loss: 0.5667 | g_total_loss: 2.9512
    Epoch [ 1260/ 2000] | d_X_loss: 0.5163 | d_Y_loss: 0.5517 | g_total_loss: 2.9901
    Epoch [ 1270/ 2000] | d_X_loss: 0.2966 | d_Y_loss: 0.5484 | g_total_loss: 3.0696
    Epoch [ 1280/ 2000] | d_X_loss: 0.3151 | d_Y_loss: 0.5619 | g_total_loss: 3.7764
    Epoch [ 1290/ 2000] | d_X_loss: 0.3286 | d_Y_loss: 0.5190 | g_total_loss: 3.5412
    Epoch [ 1300/ 2000] | d_X_loss: 0.3440 | d_Y_loss: 0.5859 | g_total_loss: 3.2503
    Saved samples_cyclegan/sample-001300-X-Y.png
    Saved samples_cyclegan/sample-001300-Y-X.png
    Epoch [ 1310/ 2000] | d_X_loss: 0.1885 | d_Y_loss: 0.5294 | g_total_loss: 3.3743
    Epoch [ 1320/ 2000] | d_X_loss: 0.2085 | d_Y_loss: 0.5600 | g_total_loss: 3.6843
    Epoch [ 1330/ 2000] | d_X_loss: 0.5225 | d_Y_loss: 0.5532 | g_total_loss: 2.9469
    Epoch [ 1340/ 2000] | d_X_loss: 0.4493 | d_Y_loss: 0.5641 | g_total_loss: 2.6944
    Epoch [ 1350/ 2000] | d_X_loss: 0.2634 | d_Y_loss: 0.6020 | g_total_loss: 3.2118
    Epoch [ 1360/ 2000] | d_X_loss: 0.3202 | d_Y_loss: 0.6345 | g_total_loss: 2.6288
    Epoch [ 1370/ 2000] | d_X_loss: 0.1991 | d_Y_loss: 0.6539 | g_total_loss: 3.3036
    Epoch [ 1380/ 2000] | d_X_loss: 0.3958 | d_Y_loss: 0.6623 | g_total_loss: 2.3905
    Epoch [ 1390/ 2000] | d_X_loss: 0.3416 | d_Y_loss: 0.7336 | g_total_loss: 2.9088
    Epoch [ 1400/ 2000] | d_X_loss: 0.3774 | d_Y_loss: 0.6455 | g_total_loss: 3.0907
    Saved samples_cyclegan/sample-001400-X-Y.png
    Saved samples_cyclegan/sample-001400-Y-X.png
    Epoch [ 1410/ 2000] | d_X_loss: 0.2243 | d_Y_loss: 0.6260 | g_total_loss: 3.2325
    Epoch [ 1420/ 2000] | d_X_loss: 0.2560 | d_Y_loss: 0.5883 | g_total_loss: 2.9026
    Epoch [ 1430/ 2000] | d_X_loss: 0.2537 | d_Y_loss: 0.5556 | g_total_loss: 3.5076
    Epoch [ 1440/ 2000] | d_X_loss: 0.3475 | d_Y_loss: 0.5611 | g_total_loss: 2.6592
    Epoch [ 1450/ 2000] | d_X_loss: 0.2012 | d_Y_loss: 0.5353 | g_total_loss: 3.4860
    Epoch [ 1460/ 2000] | d_X_loss: 0.3499 | d_Y_loss: 0.5241 | g_total_loss: 3.4864
    Epoch [ 1470/ 2000] | d_X_loss: 0.2392 | d_Y_loss: 0.5957 | g_total_loss: 3.3200
    Epoch [ 1480/ 2000] | d_X_loss: 0.2098 | d_Y_loss: 0.5110 | g_total_loss: 3.2373
    Epoch [ 1490/ 2000] | d_X_loss: 0.2326 | d_Y_loss: 0.5303 | g_total_loss: 3.1584
    Epoch [ 1500/ 2000] | d_X_loss: 0.2272 | d_Y_loss: 0.5523 | g_total_loss: 3.0208
    Saved samples_cyclegan/sample-001500-X-Y.png
    Saved samples_cyclegan/sample-001500-Y-X.png
    Epoch [ 1510/ 2000] | d_X_loss: 0.2693 | d_Y_loss: 0.5447 | g_total_loss: 4.0187
    Epoch [ 1520/ 2000] | d_X_loss: 0.2910 | d_Y_loss: 0.5813 | g_total_loss: 3.2861
    Epoch [ 1530/ 2000] | d_X_loss: 0.2467 | d_Y_loss: 0.5629 | g_total_loss: 3.1717
    Epoch [ 1540/ 2000] | d_X_loss: 0.2533 | d_Y_loss: 0.6179 | g_total_loss: 2.8999
    Epoch [ 1550/ 2000] | d_X_loss: 0.1559 | d_Y_loss: 0.6143 | g_total_loss: 2.7798
    Epoch [ 1560/ 2000] | d_X_loss: 0.1632 | d_Y_loss: 0.6329 | g_total_loss: 3.2525
    Epoch [ 1570/ 2000] | d_X_loss: 0.2399 | d_Y_loss: 0.6542 | g_total_loss: 3.2447
    Epoch [ 1580/ 2000] | d_X_loss: 0.2615 | d_Y_loss: 0.6431 | g_total_loss: 3.0553
    Epoch [ 1590/ 2000] | d_X_loss: 0.2581 | d_Y_loss: 0.6343 | g_total_loss: 2.5153
    Epoch [ 1600/ 2000] | d_X_loss: 0.2920 | d_Y_loss: 0.6075 | g_total_loss: 3.7326
    Saved samples_cyclegan/sample-001600-X-Y.png
    Saved samples_cyclegan/sample-001600-Y-X.png
    Epoch [ 1610/ 2000] | d_X_loss: 0.2090 | d_Y_loss: 0.5699 | g_total_loss: 3.8113
    Epoch [ 1620/ 2000] | d_X_loss: 0.3796 | d_Y_loss: 0.5798 | g_total_loss: 2.6243
    Epoch [ 1630/ 2000] | d_X_loss: 0.3584 | d_Y_loss: 0.5520 | g_total_loss: 2.5868
    Epoch [ 1640/ 2000] | d_X_loss: 0.2667 | d_Y_loss: 0.5428 | g_total_loss: 3.3110
    Epoch [ 1650/ 2000] | d_X_loss: 0.3014 | d_Y_loss: 0.5467 | g_total_loss: 3.2347
    Epoch [ 1660/ 2000] | d_X_loss: 0.1559 | d_Y_loss: 0.5316 | g_total_loss: 3.0785
    Epoch [ 1670/ 2000] | d_X_loss: 0.1558 | d_Y_loss: 0.5398 | g_total_loss: 3.6034
    Epoch [ 1680/ 2000] | d_X_loss: 0.3410 | d_Y_loss: 0.5436 | g_total_loss: 2.5295
    Epoch [ 1690/ 2000] | d_X_loss: 0.2328 | d_Y_loss: 0.5492 | g_total_loss: 3.1536
    Epoch [ 1700/ 2000] | d_X_loss: 0.4133 | d_Y_loss: 0.5445 | g_total_loss: 3.4235
    Saved samples_cyclegan/sample-001700-X-Y.png
    Saved samples_cyclegan/sample-001700-Y-X.png
    Epoch [ 1710/ 2000] | d_X_loss: 0.1472 | d_Y_loss: 0.5767 | g_total_loss: 3.0886
    Epoch [ 1720/ 2000] | d_X_loss: 0.2064 | d_Y_loss: 0.5939 | g_total_loss: 3.4704
    Epoch [ 1730/ 2000] | d_X_loss: 0.2911 | d_Y_loss: 0.6099 | g_total_loss: 2.8320
    Epoch [ 1740/ 2000] | d_X_loss: 0.3589 | d_Y_loss: 0.6071 | g_total_loss: 2.9319
    Epoch [ 1750/ 2000] | d_X_loss: 0.2897 | d_Y_loss: 0.6338 | g_total_loss: 3.1887
    Epoch [ 1760/ 2000] | d_X_loss: 0.3062 | d_Y_loss: 0.6385 | g_total_loss: 3.0443
    Epoch [ 1770/ 2000] | d_X_loss: 0.2407 | d_Y_loss: 0.6575 | g_total_loss: 2.8730
    Epoch [ 1780/ 2000] | d_X_loss: 0.3163 | d_Y_loss: 0.6364 | g_total_loss: 3.4078
    Epoch [ 1790/ 2000] | d_X_loss: 0.1573 | d_Y_loss: 0.6292 | g_total_loss: 3.4823
    Epoch [ 1800/ 2000] | d_X_loss: 0.3282 | d_Y_loss: 0.6218 | g_total_loss: 3.6113
    Saved samples_cyclegan/sample-001800-X-Y.png
    Saved samples_cyclegan/sample-001800-Y-X.png
    Epoch [ 1810/ 2000] | d_X_loss: 0.1112 | d_Y_loss: 0.6056 | g_total_loss: 3.6526
    Epoch [ 1820/ 2000] | d_X_loss: 0.1959 | d_Y_loss: 0.5741 | g_total_loss: 3.1193
    Epoch [ 1830/ 2000] | d_X_loss: 0.6359 | d_Y_loss: 0.5509 | g_total_loss: 4.1466
    Epoch [ 1840/ 2000] | d_X_loss: 0.1912 | d_Y_loss: 0.5502 | g_total_loss: 3.0787
    Epoch [ 1850/ 2000] | d_X_loss: 0.2226 | d_Y_loss: 0.5464 | g_total_loss: 2.8494
    Epoch [ 1860/ 2000] | d_X_loss: 0.2934 | d_Y_loss: 0.5240 | g_total_loss: 4.0390
    Epoch [ 1870/ 2000] | d_X_loss: 0.7198 | d_Y_loss: 0.5278 | g_total_loss: 2.1073
    Epoch [ 1880/ 2000] | d_X_loss: 0.2727 | d_Y_loss: 0.5268 | g_total_loss: 3.3954
    Epoch [ 1890/ 2000] | d_X_loss: 0.2501 | d_Y_loss: 0.5408 | g_total_loss: 7.3264
    Epoch [ 1900/ 2000] | d_X_loss: 0.0733 | d_Y_loss: 0.5606 | g_total_loss: 3.2965
    Saved samples_cyclegan/sample-001900-X-Y.png
    Saved samples_cyclegan/sample-001900-Y-X.png
    Epoch [ 1910/ 2000] | d_X_loss: 0.1343 | d_Y_loss: 0.5353 | g_total_loss: 3.3947
    Epoch [ 1920/ 2000] | d_X_loss: 1.2580 | d_Y_loss: 0.5396 | g_total_loss: 2.6008
    Epoch [ 1930/ 2000] | d_X_loss: 0.1094 | d_Y_loss: 0.5480 | g_total_loss: 2.9541
    Epoch [ 1940/ 2000] | d_X_loss: 0.1626 | d_Y_loss: 0.5710 | g_total_loss: 3.0697
    Epoch [ 1950/ 2000] | d_X_loss: 0.2281 | d_Y_loss: 0.5828 | g_total_loss: 3.1899
    Epoch [ 1960/ 2000] | d_X_loss: 0.1745 | d_Y_loss: 0.6007 | g_total_loss: 2.5259
    Epoch [ 1970/ 2000] | d_X_loss: 0.2401 | d_Y_loss: 0.6094 | g_total_loss: 3.1087
    Epoch [ 1980/ 2000] | d_X_loss: 0.3266 | d_Y_loss: 0.6195 | g_total_loss: 3.2503
    Epoch [ 1990/ 2000] | d_X_loss: 0.2728 | d_Y_loss: 0.6320 | g_total_loss: 3.2241
    Epoch [ 2000/ 2000] | d_X_loss: 0.2502 | d_Y_loss: 0.6163 | g_total_loss: 3.5034
    Saved samples_cyclegan/sample-002000-X-Y.png
    Saved samples_cyclegan/sample-002000-Y-X.png
    

## Tips on Training and Loss Patterns

A lot of experimentation goes into finding the best hyperparameters such that the generators and discriminators don't overpower each other. It's often a good starting point to look at existing papers to find what has worked in previous experiments, I'd recommend this [DCGAN paper](https://arxiv.org/pdf/1511.06434.pdf) in addition to the original [CycleGAN paper](https://arxiv.org/pdf/1703.10593.pdf) to see what worked for them. Then, you can try your own experiments based off of a good foundation.

#### Discriminator Losses

When you display the generator and discriminator losses you should see that there is always some discriminator loss; recall that we are trying to design a model that can generate good "fake" images. So, the ideal discriminator will not be able to tell the difference between real and fake images and, as such, will always have some loss. You should also see that $D_X$ and $D_Y$ are roughly at the same loss levels; if they are not, this indicates that your training is favoring one type of discriminator over the and you may need to look at biases in your models or data.

#### Generator Loss

The generator's loss should start significantly higher than the discriminator losses because it is accounting for the loss of both generators *and* weighted reconstruction errors. You should see this loss decrease a lot at the start of training because initial, generated images are often far-off from being good fakes. After some time it may level off; this is normal since the generator and discriminator are both improving as they train. If you see that the loss is jumping around a lot, over time, you may want to try decreasing your learning rates or changing your cycle consistency loss to be a little more/less weighted.



```python
fig, ax = plt.subplots(figsize=(12,8))
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator, X', alpha=0.5)
plt.plot(losses.T[1], label='Discriminator, Y', alpha=0.5)
plt.plot(losses.T[2], label='Generators', alpha=0.5)
plt.title("Training Losses")
plt.legend()
```




    <matplotlib.legend.Legend at 0x7fac800cfe80>




![png](output_40_1.png)


---
## Evaluate the Result!

As you trained this model, you may have chosen to sample and save the results of your generated images after a certain number of training iterations. This gives you a way to see whether or not your Generators are creating *good* fake images. For example, the image below depicts real images in the $Y$ set, and the corresponding generated images during different points in the training process. You can see that the generator starts out creating very noisy, fake images, but begins to converge to better representations as it trains (though, not perfect).

<img src='notebook_images/sample-004000-summer2winter.png' width=50% />

Below, you've been given a helper function for displaying generated samples based on the passed in training iteration.


```python
import matplotlib.image as mpimg

# helper visualization code
def view_samples(iteration, sample_dir='samples_cyclegan'):
    
    # samples are named by iteration
    path_XtoY = os.path.join(sample_dir, 'sample-{:06d}-X-Y.png'.format(iteration))
    path_YtoX = os.path.join(sample_dir, 'sample-{:06d}-Y-X.png'.format(iteration))
    
    # read in those samples
    try: 
        x2y = mpimg.imread(path_XtoY)
        y2x = mpimg.imread(path_YtoX)
    except:
        print('Invalid number of iterations.')
    
    fig, (ax1, ax2) = plt.subplots(figsize=(18,20), nrows=2, ncols=1, sharey=True, sharex=True)
    ax1.imshow(x2y)
    ax1.set_title('X to Y')
    ax2.imshow(y2x)
    ax2.set_title('Y to X')

```


```python
# view samples at iteration 100
view_samples(100, 'samples_cyclegan')
```


![png](output_43_0.png)



```python
# view samples at iteration 1000
view_samples(1000, 'samples_cyclegan')
```


![png](output_44_0.png)

