from pymongo import MongoClient
import multiprocessing as mp
import time
from pyvi import ViTokenizer
import os

input_file = "/Users/tannp3.aic/Transformer-MGK/test/data_2_3.txt"

def chunkify(fname,size=1024*1024):
    fileEnd = os.path.getsize(fname)
    with open(fname,'rb') as f:
        chunkEnd = f.tell()
        while True:
            chunkStart = chunkEnd
            f.seek(size, 1)
            f.readline()
            chunkEnd = f.tell()
            yield chunkStart, chunkEnd - chunkStart
            if chunkEnd > fileEnd:
                break

def process_wrapper(chunkStart, chunkSize):
    client = MongoClient('localhost', 27017)
    db = client['test']
    collection = db['test_sample']
    with open(input_file,'rb') as f:
        f.seek(chunkStart)
        lines = f.read(chunkSize).splitlines()
        for line in lines:
            line = line.decode('utf-8')
            line = line.lower()
            line = line.replace("\n", "")
            document = {'content': ViTokenizer.tokenize(line)}
            collection.insert_one(document)
            # process(line)

cores = mp.cpu_count()
pool = mp.Pool(cores)
jobs = []

#create jobs
for chunkStart,chunkSize in chunkify(input_file):
    print(f"chunk start at:{chunkStart}, end at {chunkStart + chunkSize}")
    jobs.append(pool.apply_async(process_wrapper,(chunkStart,chunkSize)) )

#wait for all jobs to finish
for job in jobs:
    job.get()

#clean up
pool.close()

