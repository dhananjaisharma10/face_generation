# Face Generation from Binary Facial Features using Generative Adversarial Networks

## Introduction
This project focuses on generating faces corresponding to certain facial features. We use Generative Adversarial Networks (GANs) to pursue this task. Our GAN is an extension of the DCGAN with addition of deep residual learning to learn effectively. Such a model can assist police sketch artists and even automate the sketch generation task. Here the descriptive facial features are the inputs to the GAN which generates a face corresponding to those features. We are able to control cetain facial features individually, which is demonstrated by tweaking a particular feature in the face generated.  

##
The project code was written using the ```PyTorch``` library.

The directory ```code``` contains files used to build and run the model. A brief description is as follows:
1. ```dataset.py``` defines the dataloader.

2. ```runner.py``` defines the training and testing functions.

3. ```config.py``` defines the parameters.

4. ```model.py``` defines the generator and the discriminator networks.

5. In order to run the model, type the following command from within the ```code``` directory:
```
python3 main.py
```

## References
- ada
- aa
