import json
import sys
import re
import time
import subprocess

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils, OffsetRange, TopicAndPartition

from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import HashingTF, IDF
from pyspark.mllib.util import MLUtils
from pyspark.sql import Row
from pyspark.ml.feature import StopWordsRemover, Tokenizer, RegexTokenizer

from nltk.stem import PorterStemmer


if __name__ == "__main__":
	#get parameter
	brokers, topic, model_name = sys.argv[1:]
	#number of features to be used in tf idf calculation
	numFeatures = 1000	
	
	#Spark configuration
	conf = SparkConf()
	conf.setMaster('spark://VM10-1-0-20:7077')
	conf.setAppName('sentiment_stream')
	sc = SparkContext(conf=conf)
	spark = SparkSession(sc)

	#sliding window configuration (in seconds)
	ssc = StreamingContext(sc, 5)
	
	#set kafka connection with list of broker and set offset to be read to the smallest offset (read the whole data in Kafka)
	kafkaParams = {"metadata.broker.list": brokers, 'auto.offset.reset':'smallest'}
	
	#stemming words
	def porter_stem(words):
    		stem = [PorterStemmer().stem(x) for x in words]
		return stem 
	
	#sentiment classification function
	def process(reviews):
		if(reviews.isEmpty()):
			pass
		else:
			start = time.time()
			#get reviews with overall rating > 3 and overall rating < 3
			pos_reviews = reviews.filter(lambda x: x[0] > 3.0)
			neg_reviews = reviews.filter(lambda x: x[0] < 3.0)
			#set label for each class. 0.0 is positive - 1.0 is negative
			review_labels = reviews.map(lambda x: 0.0 if x[0] > 3.0 else 1.0)
			
			Words = Row('label', 'words')
			words = reviews.map(lambda r: Words(*r))
			words_df = spark.createDataFrame(words)

			#reviews tokenization
			token = RegexTokenizer(minTokenLength=2, pattern="[^A-Za-z]+", inputCol="words", outputCol="token", toLowercase=True)
			token_filtered = token.transform(words_df)			
			
			#stopwords elimination
			remover = StopWordsRemover(inputCol="token", outputCol="stopwords", caseSensitive=False)
			stopwords_filtered = remover.transform(token_filtered)

			prep_filtered = (stopwords_filtered.select('stopwords').rdd).map(lambda x: x[0])

			#tf-idf calculation
			tf = HashingTF(numFeatures=numFeatures).transform(prep_filtered.map(porter_stem, preservesPartitioning=True))
			idf = IDF().fit(tf)
			train_tfidf = idf.transform(tf)

			#set training dataset with label
			training = review_labels.zip(train_tfidf).map(lambda x: LabeledPoint(x[0], x[1]))
	
			#train the model classifier
			model = DecisionTree.trainClassifier(training, numClasses=2, categoricalFeaturesInfo={},impurity='gini', maxDepth=10, maxBins=32)
			#save model classifier to HDFS
			output_dir = "hdfs://VM10-1-0-14:9000/classifier/"+model_name
			model.save(sc, output_dir)

			end = time.time()
			
			print("Total Reviews : ", reviews.count(), "Processing Time : ", (end-start))
			
			ssc.stop()
	
	#create stream initiation to Kafka
	kvs = KafkaUtils.createDirectStream(ssc, [topic], kafkaParams)
	parsed = kvs.map(lambda v: json.loads(v[1]))
	reviews = parsed.map(lambda r: [r['overall'], r['reviewText']])	
	
	reviews.foreachRDD(process)	

	ssc.start()
	ssc.awaitTermination()
