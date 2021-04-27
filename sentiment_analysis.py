from nltk.corpus import twitter_samples
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk import FreqDist
from nltk.stem.wordnet import WordNetLemmatizer
import re, string, random
import numpy as np
import fileinput

class Perceptron:

    def __init__(self, inputs, bias = 1):
        self.weights = (np.random.rand(inputs + 1) * 2) - 1
        self.bias = bias

    def set_weights(self, w_init):
        self.weights = np.array(w_init)

    def run(self, input):
        input = np.append(input, self.bias)
        return self.sigmoid(np.dot(input, self.weights))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


class NeuralNetwork:

    def __init__(self, layers, bias=1, rate = 1):
        self.layers = np.array(layers, dtype=object)
        self.network = []
        self.values = []
        self.errorTerms = []

        self.bias = bias
        self.rate = rate

        for i in range(len(self.layers)):
            self.values.append([])
            self.network.append([])
            self.errorTerms.append([])
            self.values[i] = [0.0 for j in range(self.layers[i])]
            self.errorTerms[i] = [0.0 for j in range(self.layers[i])]
            if i > 0:
                for j in range(self.layers[i]):
                    self.network[i].append(Perceptron(self.layers[i - 1], self.bias))

        self.network = np.array([np.array(x) for x in self.network], dtype= object)
        self.values = np.array([np.array(x) for x in self.values], dtype= object)
        self.errorTerms = np.array([np.array(x) for x in self.errorTerms], dtype= object)

    def set_weights(self, w_init):
        for i in range(len(w_init)):
            for j in range(len(w_init[i])):
                self.network[i + 1][j].set_weights(w_init[i][j])

    def run(self, input):
        input = np.array(input, dtype= object)
        self.values[0] = input
        for i in range(1, len(self.network)):
            for j in range(len(self.network[i])):
                self.values[i][j] = self.network[i][j].run(self.values[i - 1])

        return self.values[-1]

    def backpropagation(self, input, expected):
        input = np.array(input)
        expected = np.array(expected)

        output = self.run(input)

        error = (expected - output)

        MSE = sum(error**2) / len(output)

        self.errorTerms[-1] = output * (1 - output) * error

        for i in reversed(range(1, len(self.network) - 1)):
            for j in range(self.layers[i]):
                fwd_error = 0.0
                for k in range(self.layers[i + 1]):
                    fwd_error += self.network[i + 1][k].weights[j] * self.errorTerms[i + 1][k]
                self.errorTerms[i][j] = self.values[i][j] * (1 - self.values[i][j]) * fwd_error

        for i in range(1, len(self.network)):
            for j in range(self.layers[i]):
                for k in range(self.layers[i - 1] + 1):
                    if k == self.layers[i - 1]:
                        change = self.rate * self.errorTerms[i][j] * self.bias
                    else:
                        change = self.rate * self.errorTerms[i][j] * self.values[i - 1][k]
                    self.network[i][j].weights[k] += change

        return MSE       
        
stop_words = stopwords.words('english')

def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence

def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token


def get_most_common_words(cleaned_tokens_list, count):
    freq_dist_pos = FreqDist(get_all_words(cleaned_tokens_list))
    cleaned_tokens_list = freq_dist_pos.most_common(count)

    words = []
    for tokens in cleaned_tokens_list:
        for token in tokens:
            if not token in words and type(token) is str:
                words.append(token)
    return words

positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')
text = twitter_samples.strings('tweets.20150430-223406.json')
positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

positive_cleaned_tokens_list = []
negative_cleaned_tokens_list = []

for tokens in positive_tweet_tokens:
    positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

for tokens in negative_tweet_tokens:
    negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))


positive_dataset = [(tweet, [1, 1, 1, 1, 1])
                     for tweet in positive_cleaned_tokens_list]

negative_dataset = [(tweet, [0, 0, 0, 0, 0])
                     for tweet in negative_cleaned_tokens_list]

dataset = positive_dataset + negative_dataset

random.shuffle(dataset)

train_data = dataset[:7000]
test_data = dataset[7000:]


vocabulary = get_most_common_words(positive_cleaned_tokens_list, 30) + get_most_common_words(negative_cleaned_tokens_list, 30)


nn = NeuralNetwork([len(vocabulary), 30, 15, 5])

num = 0
for tweet in train_data:
    num += 1
    print(num)
    inputs = []
    for i in range(len(vocabulary)):
        if vocabulary[i] in tweet[0]:
            inputs.append(1)
        else:
            inputs.append(0)
    nn.backpropagation(inputs, tweet[1])

"""
for tweet in test_data:
    inputs = []
    for i in range(len(vocabulary)):
        if vocabulary[i] in tweet[0]:
            inputs.append(1)
        else:
            inputs.append(0)

"""

while True:
    print("Enter a sentence:")
    inputs = input().split(" ")
    inputs = remove_noise(inputs, stop_words)
    nninput = []
    for i in range(len(vocabulary)):
        if vocabulary[i] in inputs:
            nninput.append(1)
        else:
            nninput.append(0)
    print(nn.run(nninput))