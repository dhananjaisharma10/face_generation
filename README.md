# Deep-Learning-Project

Please read the project report and poster for better understading of the project. 

In this project, we focus on generating faces corresponding to certain facial features. We use Generative Adversarial Network (GAN) to pursue our task. Our GAN is an extension of the DCGAN with addition of deep residual learning to learn effectively. We intended to develop a GAN to assist police sketch artists, or to automate the sketch generation task. Here the descriptive facial features are the inputs to the GAN which generates a face corresponding to those features. We also try to control the individual facial features to tweak the changes in the face generated.  

Run: python3 main.py.

Dataset.py has dataloader defined to get mini batches of data.

Runner.py has training and testing functions defined.

Config.py finles has the configrations/values for all the variables.

Model.py file has generator and discriminator models defined.
