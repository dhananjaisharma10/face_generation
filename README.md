# Face Generation from Binary Facial Features using Generative Adversarial Networks

In this project, we focus on generating faces corresponding to certain facial features. We use Generative Adversarial Networks (GANs) to pursue this task. Our GAN is an extension of the DCGAN with addition of deep residual learning to learn effectively. Such a model can assist police sketch artists and even automate the sketch generation task. Here the descriptive facial features are the inputs to the GAN which generates a face corresponding to those features. We also try to control the individual facial features to tweak the changes in the face generated.  

The directory ```code``` contains files used to build and run the model. A brief description is as follows:
1. dataset.py has dataloader defined to get mini batches of data.

2. runner.py has the training and testing functions defined.

3. config.py has the configrations/values for all the variables.

4. model.py file has generator and discriminator models defined.

5. In order to run the model, run the following command:
```
python3 main.py
```

For further details, please refer to the project report and poster.
