import json
import os
import time
import glob
import random
from datetime import datetime
from kafka import KafkaProducer
from kafka.errors import KafkaError

#initialization kafka connection to deliver JSON file
producer = KafkaProducer(bootstrap_servers=['10.1.0.23:9092','10.1.0.24:9092','10.1.0.25:9092'],
		retries=3,
		value_serializer=lambda v: json.dumps(v).encode('utf-8'))

#set the name of topic which the reviews will be published
topic = 'sentiment'

#read JSON file source and send to Kafka
def production():
	ctr, flag = 0, 0
	path = 'pool_1000_shuffle/*.review'
	#path = 'init_pool/*.review'
	files = glob.glob(path)

	for fl in files:
		with open(fl) as ofile:
			for line in ofile:
				j = json.loads(line.replace('\n', ''))
	               		producer.send(topic, {'reviewText':j['reviewText'],'overall':j['overall']})
				ctr += 1
		print(str(datetime.now())," : ",ctr)
		time.sleep(5)
	return ctr
		
ctr = production()
end = time.time()
print("Record : ", (ctr), " Time : ", (end-start))