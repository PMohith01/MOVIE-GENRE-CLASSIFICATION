# MOVIE-GENRE-CLASSIFICATION





Data Preparation:

After importing necessary libraries and setting configurations, the script downloads a dataset from a Google Drive link provided in DATA_SOURCE_MAPPING. This dataset contains movie descriptions and their corresponding genres. The dataset is compressed (either as a zip or tar file).
Once downloaded, the script extracts the dataset into the /kaggle/input directory.
Libraries such as pandas, transformers, torch, and tensorflow are imported to handle data processing, model building, and training.
Tokenization:

In this section, the BERT tokenizer (BertTokenizerFast) from the Hugging Face transformers library is utilized to tokenize the movie descriptions. Tokenization involves splitting the text into individual tokens (words or subwords) and converting them into numerical representations that the model can understand.
Each tokenized description is converted into an encoded dictionary containing the token IDs, attention mask, and other necessary information for model input.
Embedding:

The BERT model (bert-base-uncased) is employed to encode the tokenized movie descriptions into fixed-size vectors. These vectors represent the semantic meaning of the descriptions in a continuous vector space.
The encoded vectors are then used as input features for training the neural network classifier.
Training Neural Network:

A simple neural network class (NN) is defined with fully connected layers (nn.Linear). The architecture consists of multiple hidden layers followed by an output layer.
The training data is prepared by combining the encoded movie descriptions with their corresponding labels (genres).
The train function is responsible for training the neural network using the provided data loader, optimizer, and loss function. It iterates over the dataset in mini-batches, computes the loss, and updates the model parameters through backpropagation.
The get_accuracy function calculates the accuracy of the trained model on the test dataset. It compares the model's predictions with the actual labels and computes the accuracy metric.
The get_prediction function generates predictions for the test dataset using the trained model. It returns the predicted labels and actual labels.
The print_pred function visualizes a random sample of predictions made by the model. It displays the movie images along with their predicted genres.
