# MOVIE-GENRE-CLASSIFICATION



Data Preparation:

The script starts by importing necessary libraries and setting up some configurations.
It downloads a dataset from a Google Drive link provided in DATA_SOURCE_MAPPING. This dataset contains movie descriptions and their corresponding genres.
The dataset is then extracted into the /kaggle/input directory.
The script imports required libraries such as pandas, transformers, torch, and tensorflow.
Tokenization:

The movie descriptions are tokenized using the BERT tokenizer (BertTokenizerFast from the Hugging Face transformers library).
Embedding:

The BERT model (bert-base-uncased) is used to encode the tokenized movie descriptions into fixed-size vectors.
Training Neural Network:

A simple neural network class (NN) is defined with fully connected layers.
The training data is prepared, combining the encoded movie descriptions with their corresponding labels.
The train function is defined to train the neural network using the provided data loader, optimizer, and loss function.
The get_accuracy function calculates the accuracy of the model on the test dataset.
The get_prediction function gets the model's predictions on the test dataset.
The print_pred function is used to visualize predictions made by the model.
