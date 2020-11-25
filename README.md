## Matrix Factorizer

This repository contains the scripts for performing a matrix factorization from an user-ratings like matrix in the form `A = U*V`. It minimizes the loss function (squared error plus independent regularization for each embedding matrix) using ALS. Data is not provided.

---
### Contents:

The repo contains factorizer.py, train.py and app.py

##### factorizer.py

Contains the main class. It inherits from scikit-learn to adopt the set of tools it provides and implements the usual fit/predict methods. 

---

##### train.py

This script runs the training for the model. The function ``prepare_data`` needs to be adapted to load the desired data and perform the appropriate transformations.

It can be run using

`python train.py --path <path_to_data> --p <embedding_dim>
 --alpha <X_regularization> --beta <Y_regularization> 
 --n_iter <number_of_iterations>`
 
---

##### app.py

Simple API developed using fastAPI to run the inference for the model. It takes a user_id and returns the 30 most likable items for it.

---