# Seq2Seq-Chatbot

## Introduction
This Chatbot is a TensorFlow implementation of Seq2Seq Mode. It make use of a seq2seq model RNN for sentence predictions. The chatbot is trained on Cornell Movie Dialogs Corpus on Conversation of lenghth between 5 to 2 words

## Installation
To run this chabtot you will need:
* Python 3.5 
* Tensorflow (1.13.1)
* Numpy
* Flask
* [Dataset](http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip)

## Running
### To chat with the trained model from Web Chat:
1. First install the required depedencies from `Chat_UI_Web` and run 
`pip install -r requirment.txt`
2. From the Chat_UI_Web directory run run.py
`python run.py`
3. Open a browser to the URL indicated by the server console, This is typically: http://localhost:5000/

### To chat with the trained model from Python Console:
1. First install the required depedencies from root directory and run 
`pip install -r requirment.txt`
2. From the root directory run 
`python chat.py`

## List of Hyperparameters
* Batch_size to 128
* Rnn_size to 128
* Embed_size to 128
* Learning_rate to 0.0.1
* Epochs to 500
* Keep_prob to 0.75

## For Training Model 
To Train Model from Python Console:
1. Configure the `config.py` file for desired training hyperparameters
2. Run train.py
`python train.py`
3. Make sure the datasets should be present in the `datasets` folder

## Results
Model was able to train to `Accuracy of 62%` and `Loss of 19%` and after that it was overfitting

### Accuracy
<img src="https://github.com/karma9874/Seq2Seq-Chatbot/blob/master/Images/Accuracy.JPG" alt="alt text" width="100" height="100">

### Loss
<img src="https://github.com/karma9874/Seq2Seq-Chatbot/Images/Loss.jpg" alt="alt text" width="100" height="100">

### Chat Results
```
Human: who are you
Bot: girlfriend
Human: who am i
Bot: doctor livingston
Human: where are you from
Bot: southern california
Human: how old are you
Bot: nineteen or
Human: are you a boy
Bot: i am trying
Human: are you a girl
Bot: something like that
Human: do you drink
Bot: no thanks
```
Some Funny Result
<img src="https://github.com/karma9874/Seq2Seq-Chatbot/Images/Result.jpg" alt="alt text" width="850" height="500">



