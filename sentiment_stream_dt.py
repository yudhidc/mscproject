import os
import json
import sys
import re
import time
import subprocess
from datetime import datetime
from py4j.protocol import Py4JJavaError

from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.sql import SparkSession
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils

from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import HashingTF, IDF
from pyspark.mllib.util import MLUtils
from pyspark.sql import Row
from pyspark.ml.feature import StopWordsRemover, Tokenizer, RegexTokenizer
from pyspark.mllib.evaluation import MulticlassMetrics

from nltk.stem import PorterStemmer

ctr = 0
if __name__ == "__main__":
	#get parameter
	brokers, topic = sys.argv[1:]
	#number of features used in tf-idf calculation
	numFeatures = 1000	
	
	#configuration for Spark
	conf = SparkConf()
	conf.setMaster('spark://VM10-1-0-28:7077')
	conf.setAppName('sentiment_stream')
	sc = SparkContext(conf=conf)
	spark = SparkSession(sc)
	
	#sliding window to read Kafka (in seconds)
	ssc = StreamingContext(sc, 5)
	kvs = KafkaUtils.createDirectStream(ssc, [topic], {"metadata.broker.list":brokers})
	
	parsed = kvs.map(lambda v: json.loads(v[1]))
	
	#save reviews to HDFS
	def dump_to_hdfs(reviews):
		if(reviews.isEmpty()):
			pass
		else:
			filename = "hdfs://VM10-1-0-14:9000/review/" + re.sub('[^0-9]','',str(datetime.now())) + ".review"
			reviews.saveAsTextFile(filename)
	
	#stemming the words
	def porter_stem(words):
                stem = [PorterStemmer().stem(x) for x in words]
                return stem
	
	#checking whether file is exist or not
	def patherror(path):
		try:
			testRDD = sc.textFile(path)
			testRDD.take(1)
			return False
		except Py4JJavaError as e:
			return True

	str_result = ""
	def process(reviews):
		if(reviews.isEmpty()):
			pass
		else:
			model_name = "dt"
			updated_model = "dt0"
			model_path, data_path, metadata_path = '','',''
			
			#performing looping process to check the availability of new model classifier
			for i in range(25,-1,-1):
				model_path = "hdfs://VM10-1-0-14:9000/classifier/"+model_name+str(i)
				updated_model = model_name+str(i)
				data_path = model_path+"/data/part-r*"
				metadata_path = model_path+"/metadata/part-00000"
				if(patherror(data_path) == False and patherror(metadata_path) == False):
					break
			
			#load model classifier
			model = DecisionTreeModel.load(sc, model_path)

			start = time.time()
			reviews_label = reviews.map(lambda x: 0.0 if x[0] > 3.0 else 1.0)
			
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
			tfidf = idf.transform(tf)
			
			prediction = model.predict(tfidf)
			
			labeled_prediction = reviews_label.zip(prediction).map(lambda x: (float(x[1]), x[0]))
			
			metrics = MulticlassMetrics(labeled_prediction)	
			
			output = reviews.zip(prediction)
				
			filename = "hdfs://VM10-1-0-14:9000/output/" + re.sub('[^0-9]','',str(datetime.now())) + ".out"
			output.saveAsTextFile(filename)
			
			end = time.time()	
			print(updated_model,';',reviews.count(),';',metrics.accuracy,';',metrics.precision(0.0),';',metrics.precision(1.0),';',metrics.recall(0.0),';',metrics.recall(1.0),';',metrics.fMeasure(0.0),';',metrics.fMeasure(1.0),';',(end-start))
	
	parsetest = kvs.map(lambda x: x[1])
	parsetest.foreachRDD(dump_to_hdfs)

	reviews = parsed.map(lambda r: [r['overall'], r['reviewText']])
	reviews.foreachRDD(process)
	
	ssc.start()
	ssc.awaitTermination()
