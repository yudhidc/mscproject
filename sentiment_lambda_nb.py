from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import HashingTF, IDF
from pyspark.ml.feature import StopWordsRemover, Tokenizer, RegexTokenizer
from pyspark.mllib.util import MLUtils
from pyspark.sql import Row
from pyspark.mllib.linalg import Vectors

import sys
import re
import json
import subprocess
import time

from nltk.stem import PorterStemmer

if __name__ == "__main__":
	#get parameters
	model_name = sys.argv[1]
	
	#Spark configuration
	conf = SparkConf()
	conf.setMaster('spark://VM10-1-0-20:7077')
	conf.setAppName('sentiment')
	sc = SparkContext(conf=conf)

	#number of features to be used in tf idf calculation
	numFeatures = 1000

	start = time.time()
	json_file = sc.textFile("hdfs://VM10-1-0-14:9000/review/*.review")
	data = json_file.map(json.loads)

	#fetch overall column and reviewText and filter out other columns
	reviews = data.map(lambda r: [r['overall'], r['reviewText']])
	#separate reviews into two classes: positive and negative
	pos_reviews = reviews.filter(lambda x: x[0] > 3.0)
	neg_reviews = reviews.filter(lambda x: x[0] < 3.0)

	#set label: 0.0 is positive and 1.0 is negative
	review_labels = reviews.map(lambda x: 0.0 if x[0] > 3.0 else 1.0)

	#words stemming
	def porter_stem(words):
		stem = [PorterStemmer().stem(x) for x in words]
		return stem

	#check if the model file is exist or not
	def patherror(path):
		try:
			testRDD = sc.textFile(path)
			testRDD.take(1)
			return False
		except Py4JJavaError as e:
			return True

	spark = SparkSession(sc)

	Words = Row('label', 'words')
	words = reviews.map(lambda r: Words(*r))
	words_df = spark.createDataFrame(words)

	#review tokenization
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

	#set training data with label
	training = review_labels.zip(train_tfidf).map(lambda x: LabeledPoint(x[0], x[1]))

	#train model classifier
	model = NaiveBayes.train(training, 1.0)
	
	#save model to HDFS
	output_dir = "hdfs://VM10-1-0-14:9000/classifier/"+model_name
	model.save(sc, output_dir)

	end = time.time()
	print("Total Records : ", reviews.count(), "  , Processing Time : ", (end - start))
