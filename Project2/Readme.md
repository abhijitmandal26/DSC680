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
