from mrjob.job import MRJob
from mrjob.step import MRStep
from collections import Counter
import csv
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer


tweet = 'loovvee kindle not cool fantastic right'
tweet_word_array = tweet.split(" ")
tweet_dict = Counter(tweet_word_array)#{"loves":1,"twitter":1}
#print(tweet_dict)


emoticons_str = r"""
    (?:
    [:=;] #Eyes
    [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

regex_str = [
    emoticons_str,
     r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
 
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]

tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)

class Sentient(MRJob):

    def mapper_init(self):
        self.positive_count = 0
        self.negative_count = 0

    def mapper_final(self):
        yield "counts", [self.positive_count,self.negative_count]

    def mapper(self, _, line):
        values = line.split(",")
        tweet = ' '.join(values[5:])
        label = values[0]

        if (label == "0"):
            self.negative_count+=1
        else:
            self.positive_count+=1

        procesed_text = self.preprocess(tweet)
        #print(procesed_text)
        for p in procesed_text:
            if (label == "0"):
                yield p+".negative", 1
            else:
                yield p+".positive", 1

    def preprocess(self, tweet):
        processed_tweet = self.processText(tweet)

        return processed_tweet

    def processText(self,unprocessed_tweet):
        
        # remove URLs (www.* ou https?://*)
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',unprocessed_tweet)

        # remove @username
        tweet = re.sub('@[^\s]+','AT_USER',tweet)

        # remove multiple spaces
        tweet = re.sub('[\s]+', ' ', tweet)

        # substitute #work for work
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)

        # trim
        tweet = tweet.strip('\'"')

        tweet = self.removeRepeats(tweet)

        tokenized_tweet = self.tokenize(tweet)
        tweet = self.removeStopWords(tokenized_tweet)

        return tweet

    def removeRepeats(self,tweet):
        pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
        
        tweet = pattern.sub(r"\1\1", tweet)
        
        return tweet

    def tokenize(self, tweet):
        return tokens_re.findall(tweet)

    def removeStopWords(self, token):

        stop_words = set(stopwords.words('english'))
        new=set(('AT_USER','URL'))
        new_stopwords = stop_words.union(new)
        
        content_new = [w for w in token if not w in new_stopwords and len(w)>2 and re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w) is not None]        

        token = " ".join(content_new)
        if emoticon_re.search(token):
            toke = token 
        else:
            toke = token.lower()
      
        return toke.split(" ")

        
    def reducer(self, key, values):
        
        if key != "counts":
            
            yield key, sum(values)
        else:
            
            yield key, list(values)

    def second_mapper(self,key,values):
       
        yield key,values

    def second_reducer_init(self):
        self.positive_words = 0
        self.negative_words = 0
        self.positive_tweets = 0
        self.negeative_tweets = 0

    def second_reducer(self,key,values):
        

        val_list = list(values)

        if (key == "counts"):
            for val in val_list[0]:
                self.positive_tweets += val[0]
                self.negeative_tweets += val[1]

        else:
            key_list = key.split(".")
            
            if(key_list[1] == "positive"):
                self.positive_words += 1
            else:
                self.negative_words += 1
                
            # word.positive | word.negative , number of time it appears
            if key_list[0] in tweet_word_array:
                
                yield key, val_list[0]
        

    def second_reducer_final(self):
        yield "word_counts", (self.positive_words, self.negative_words, self.positive_tweets, self.negeative_tweets)

    def third_reducer(self, key, value):
        list_values = list(value)
        #print(key,list_values)
        # positive_words = 0
        # negative_words = 0
        # positive_tweets = 0
        # negeative_tweets = 0

        if key == "word_counts":
            for value in list_values:
                self.positive_words += value[0]
                self.negative_words += value[1]
                self.positive_tweets += value[2]
                self.negeative_tweets += value[3]

            # yield "final" , (positive_words, negative_words, positive_tweets, negeative_tweets)
        else:
            self.count_dictionary[key] = list_values[0]
            # yield key, list_values

    def third_reducer_init(self):
        self.positive_words = 0
        self.negative_words = 0
        self.positive_tweets = 0
        self.negeative_tweets = 0
        self.count_dictionary = {}

    def third_reducer_final(self):
        yield "final", (self.count_dictionary, self.positive_words, self.negative_words, self.positive_tweets, self.negeative_tweets)


    def fourth_mapper(self,key,value):
        print(key, value)

    def fourth_reducer(self,key,value):
        positive_words = 0
        negative_words = 0
        positive_tweets = 0
        negeative_tweets = 0
        count_dictionary = {}

        list_values = list(value)
        #print(list_values)
        for value in list_values:
            if(value[0]):
                count_dictionary = value[0]
               
            if value[1]:
                positive_words = value[1]

            if value[2]:
                negative_words = value[2]

            if value[3]:
                positive_tweets = value[3]

            if value[4]:
                negeative_tweets = value[4]


       
        positive_prediction = self.make_prediction(positive_words, negative_words, positive_tweets, negeative_tweets, count_dictionary,"positive")
        negative_prediction = self.make_prediction(positive_words, negative_words, positive_tweets, negeative_tweets, count_dictionary,"negative")
        
        print('positive_prediction: ',positive_prediction)
        print('negative_prediction: ',negative_prediction)
        
        if(negative_prediction > positive_prediction):
             yield tweet, 0
        else:
            yield tweet, 4
            
            
       # print(' # test with test data set and append all predictions to the predictions array for accuracy calculations #')
            
       

    def make_prediction(self,positive_words, negative_words, positive_tweets, negeative_tweets, count_dictionary, label = "positive"):
#         tweet_word_array = self.preprocess(tweet)
#         tweet_dict = Counter(tweet_word_array)
        global tweet_word_array
        prediction = 1
        numerator = 1
        denom = 1
        

        for word in tweet_word_array:
            
            if(label == "positive"):
                try:
                    numerator = count_dictionary[word+".positive"]
                    
                except:
                    numerator = 0
                denom = positive_words
                class_counts = positive_tweets
                class_prob = positive_tweets/(positive_tweets+negeative_tweets)
            else:
                try:
                    numerator = count_dictionary[word+".negative"]
                except:
                    numerator = 0
                denom = negative_words
                class_counts = negeative_tweets
                class_prob = negeative_tweets/(positive_tweets+negeative_tweets)

            #print((word,label, tweet_dict[word],numerator,denom,class_counts,class_prob))
            prediction *= tweet_dict[word] * (numerator+1/denom+class_counts)
            

        return prediction*class_prob
    
    
    
    
    
    #         test_tweet = test_data['tokens']
#         predictions = []
#         for i in range(0,len(test_tweet)):
#             conten = make_decision(test_tweet[i], make_class_prediction)
#             predictions.append(conten)
            
   
    
    #checking accuracy
#     actual = list(test_data['label'])

#     from sklearn import metrics

#     # Generate the roc curve using scikit-learn.
#     fpr, tpr, thresholds = metrics.roc_curve(actual, predictions, pos_label=4)

#     # Measure the area under the curve.  The closer to 1, the "better" the predictions.
#     print("AUC of the predictions: {0}".format(metrics.auc(fpr, tpr)))




    def steps(self):
        return [
                MRStep(
                    mapper_init=self.mapper_init,
                    mapper_final=self.mapper_final,
                    mapper=self.mapper,
                    # combiner = self.combiner,
                    reducer=self.reducer,
                ),
                MRStep(
                    mapper = self.second_mapper,
                    reducer_init=self.second_reducer_init,
                    reducer=self.second_reducer,
                    reducer_final=self.second_reducer_final,
                ),
                MRStep(
                    reducer_init=self.third_reducer_init,
                    reducer=self.third_reducer,
                    reducer_final=self.third_reducer_final,
                ),
                MRStep(
                    # mapper=self.fourth_mapper,
                    reducer=self.fourth_reducer,
                ),
            ]


if __name__ == '__main__':
    Sentient.run()
