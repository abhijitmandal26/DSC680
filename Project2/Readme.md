### Sentiment Analysis
<img width="683" alt="image" src="https://user-images.githubusercontent.com/13950516/167273732-fbad11cf-777d-4efc-b4bd-a57de08a4144.png">


<img width="598" alt="image" src="https://user-images.githubusercontent.com/13950516/167273813-90ca05ae-133c-400c-a496-242998eb5af8.png">


Natural Language Processing (NLP) is a hotbed of research in data science these days and one of the most common applications of NLP is sentiment analysis. From opinion polls to creating entire marketing strategies, this domain has completely reshaped the way businesses work, which is why this is an area every data scientist must be familiar with.
Thousands of text documents can be processed for sentiment in seconds, compared to the hours it would take a team of people to manually complete the same task.

We will do so by following a sequence of steps needed to solve a general sentiment analysis problem. We will start with preprocessing and cleaning of the raw text of the tweets. Then we will explore the cleaned text and try to get some intuition about the context of the tweets. After that, we will extract numerical features from the data and
finally use these feature sets to train models and identify the sentiments of the tweets.


### Business Problem

<img width="332" alt="image" src="https://user-images.githubusercontent.com/13950516/167273870-2ebedc04-62b2-42a8-abbb-eb7078b9f968.png">

Sentiment analysis can be defined as a process that automates mining of attitudes, opinions, views and emotions from text, speech, tweets and database sources through Natural Language Processing (NLP). Sentiment analysis involves classifying opinions in text into categories like "positive" or "negative”. It's also referred as subjectivity analysis, opinion mining, and appraisal extraction.
The words opinion, sentiment, view and belief are used interchangeably but there are differences between them.

- **Opinion** : A conclusion open to dispute (because different experts have different opinions)
- **View** : subjective opinion
- **Belief** : deliberate acceptance and intellectual assent 
- **Sentiment** : opinion representing one’s feelings, Sentiment analysis and Natural Language processing are very important area nowadays. There is a massive amount of information being uploaded to the internet daily on social media websites and blogs that computers cannot understand. Traditionally it was not possible to process such large amounts of data, but with computer performance following the projections of Moore’s law and the introduction of distributed computing like Hadoop or Apache Spark, large data sets can now be processed with relative ease. With further research and investment into this area, computers will soon be able to gain an understanding from text which will greatly improve data analytics and search engines.

A good use case is to identify a customer’s perception for a product, this is an extremely valuable data to some companies. From the knowledge gained from an analysis such as this a company can identify issues with their products, spot trends before their competitors, create improved communications with their target audience, and gain valuable insight into how effective their marketing campaigns were. Through this knowledge companies gain valuable feedback which allows them to further develop the next generation of their product.


### Approach

<img width="612" alt="image" src="https://user-images.githubusercontent.com/13950516/167273974-1dd1b7ff-a32f-443f-bdbe-850137869ccf.png">


The dataset provided is the Sentiment140 Dataset which consists of 1,600,000 tweets that have been extracted using the Twitter API. The various columns present in the dataset are: 
- **target** : the polarity of the tweet (positive or negative) 
- **ids** : Unique id of the tweet 
- **date** : the date of the tweet 
- **flag** : It refers to the query. If no such query exist’s then it is NO QUERY. 
- **user** : It refers to the name of the user that tweeted 
- **text** : It refers to the text of the tweet 

![image](https://user-images.githubusercontent.com/13950516/167274017-6ac43147-db31-4f6c-b35a-bd3b9995cbb8.png)


### Data Cleansing
A tweet contains a lot of opinions about the data which are expressed in different ways by different users. The twitter dataset used in this project work is already labeled into two classes viz. negative and positive polarity and thus the sentiment analysis of the data becomes easy to observe the effect of various features. The raw data having polarity is highly susceptible to inconsistency and redundancy. Preprocessing of tweet include following points,
- **Remove all URLs (e.g. www.xyz.com), hash tags (e.g. #topic), targets (@username)** 
- **Remove Stop words** 
- **Replace Repeated Characters** 
- **Remove all punctuations, symbols, numbers** 
The second phase of the system will be to cleanse the data collected, this will involve removing any punctuations and making everything lower case. This will help in the next stage of the project especially in the “Bag of Words” approach. Removing lower case words will decrease the redundancy in the database that will be used to store the words.

### Data Classification

To reach the ultimate goal, there was a need to clean up the individual tweets. I used a concept known as "Tokenization" in NLP. It is a method of splitting a sentence into smaller units called "tokens" to remove unnecessary elements. Another technique worthy of mention is "Lemmatization". This is a process of returning words to their "base" form. A simple illustration is shown below.

![image](https://user-images.githubusercontent.com/13950516/167274243-9a3d0644-9ea0-43ae-ade1-61b560f0e50d.png)


Machine learning techniques require representing the key features of text or documents for processing. These key features are considered as feature vectors which are used for the classification task. Some examples feature that have been reported in literature are: 

- **Words and Their Frequencies**:  Unigrams, bigrams and n-gram models with their frequency counts are considered as features. 

- **Parts of Speech Tags**: Parts of speech like adjectives, adverbs and some groups of verbs and nouns are good indicators of subjectivity and sentiment. We can generate syntactic dependency patterns by parsing or dependency trees. 

- **Opinion Words and Phrases**: Apart from specific words, some phrases and idioms which convey sentiments can be used as features. e.g. cost someone an arm and leg. 

- **Position of Terms**: The position of a term within a text can affect how much the term makes difference in overall sentiment of the text. 

- **Negation**: Negation is an important but difficult feature to interpret. The presence of a negation usually changes the polarity of the opinion. 
- **Syntax**: Syntactic patterns like collocations are used as features to learn subjectivity patterns by many of the researchers. 
This is expected to be the most difficult part of the project; it will entail looking at individual words or groups of words in a tweet and attempting to assign a sentiment to them. This is no easy task as it is very difficult for a computer to “understand” slang words and sarcasm. 
“Bag of Words” Model The bag of words approach will involve building databases of positive, negative, and neutral words. Each tweet will be broken up into individual words and then compared to the words in the databases. When there is a match, a counter will be incremented or decremented by a fixed amount depending on a weighting assigned. When this process is complete the counter will be used to classify the sentiment for example if the words in the tweet are largely positive the counter should be high.

To get the most common words used, I made use of the POS-tag (Parts of Speech tagging) module in the NLTK library. Using the WordCloud library, one can generate a Word Cloud based on word frequency and superimpose these words on any image. In this case, I used the Twitter logo and Matplotlib to display the image. The Word Cloud shows the words with higher frequency in bigger text size while the "not-so" common words are in smaller text sizes. 			


![image](https://user-images.githubusercontent.com/13950516/167274294-432be8f1-72df-433b-8456-b994f0f0781b.png)


### Data Analysis

When the data is classified, there will have to be analysis performed on it. This may include simple percentages of customer satisfaction, or a more complex analysis could be performed such as comparing the customer sentiment on two similar products with the aim of finding a correlation between good sentiment and high sales of those products.

#### Setting up the Classification Model

After training the model we then apply the evaluation measures to check how the model is performing. Accordingly, we use the following evaluation parameters to check the performance of the models respectively :
- **Accuracy Score**
- **Confusion Matrix with Plot**
- **ROC-AUC Curve**


![image](https://user-images.githubusercontent.com/13950516/167274366-63d73cde-6fd4-4c5f-ba0d-d6fd8370bd7b.png)



![image](https://user-images.githubusercontent.com/13950516/167274403-94211279-1daf-4646-a297-355a1e4edc49.png)

### Conclusion
Overall, we found that Logistic Regression is the best model for analyzing Sentiments on the dataset.
Logistic Regression is following the principle of Occam’s Razor which defines that for a particular problem statement if the data has no assumption, then the simplest model works the best. Since our dataset does not have any assumptions and Logistic Regression is a simple model, therefore the concept holds true for the above-mentioned dataset.

