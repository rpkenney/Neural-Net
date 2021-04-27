# Neural-Net

Over the course of my enrollment of CSC 499, I began to develop a Neural Network to develop my skills in the machine learning field.

Included in this repository are two files which are the results of my efforts.

Neural Network.py - This is my first success at developing a neural network. It is an expandable, and object-oriented neural net that I initially tested by teaching basic logic gates. As shown in the program it is successful at learning logic gates.

sentiment_analysis.py - This was my first attempt at teaching my neural network how to handle information that would actually be of some use. As a first step I decided to go with sentiment analysis, and have my neural net analyze a sentence for being positive, or negative. In order to train the neural network, I used a library which provided a multitude of tweets from an online database, each labeled as positive or negative. Once run through the neural net the user has the opportunity to enter their own sentence for analysis. Though it is not perfect, I gained a greater understanding of how to go about, and the intricacies of training a neural network.

I accomplished this by analyzing the tweets, and creating a vocabulary of the most common words that were in the positive, and negative tweets. I filtered out words such as "and", "the", and "as", along with made all words into the same tense, to remove noise. The presence of these "vocabulary" words in novel input would be represented as one input node into the neural network. 

One drawback to this is if the user's sentence does not have any of the "vocabulary" words in it, the neural network cannot make an accurate prediction. In the future, I may go with a different strategy of text analysis that I read up on, which would be to analyze "trigrams" or sequences of three characters in the sentences, rather than words.

After typing in a sentence, the output layer will be printed to the user. The idea is that a value closer to one is positive, and a value closer to zero is a negatively charged sentence. I settled on 5 output nodes to increase the accuracy of the output.

To run these programs, download and run in any python environment.
