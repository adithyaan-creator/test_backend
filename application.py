import time
import json
import datetime
import requests
import pickle
import pandas as pd

from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import TfidfRetriever
from haystack.pipeline import DocumentSearchPipeline

from flask import Flask, request, Response, render_template

from config import *
from classifiers.bert_subject import bert_subject_classifier
#from flask_cors import CORS
#from flask_ngrok import run_with_ngrok


application = Flask(__name__)
#CORS(application)
#run_with_ngrok(application)

### Loading data to Haystack document store
all_dicts = []

print(str(datetime.datetime.now()) + " :: Loading data from csv file for IR started")
questions_df = pd.read_csv('data/questions_tags4k.csv')
print(str(datetime.datetime.now()) + " :: Loading data from csv file for IR completed")

for idx, row in questions_df.iterrows():
    temp_dict = {}
    temp_dict = {
        "content": str(row['question']),
        "meta":{
            "chapter":str(row['CHAPTER']),
            "class":str(row['CLASS']),
            "custom":str(row['CUSTOM']),
            "topic":str(row['TOPIC']),
            "subject":str(row['SUBJECT']),
            "bloom_taxonomy":str(row['BLOOM_TAXONOMY']),
            "curriculum":str(row['CURRICULUM'])
        }
    }
    all_dicts.append(temp_dict)

document_store = InMemoryDocumentStore()
print(str(datetime.datetime.now()) + "InMemory Document Store initialized")
document_store.write_documents(all_dicts)
print(str(datetime.datetime.now()) + "InMemory Document Store data loading complete")

retriever = TfidfRetriever(document_store)
question_search_pipeline = DocumentSearchPipeline(retriever)
print(str(datetime.datetime.now()) + "InMemory document search pipeline loaded")


### Loading subject classifier LR model
print(str(datetime.datetime.now()) + " :: LR Model loading started")
with open('models/tfidf_lr_2062.pkl', 'rb') as file:
    subject_tagger_model = pickle.load(file)
print(str(datetime.datetime.now()) + " :: LR Model loading completed")


## Loading bert subject classifier model
print(str(datetime.datetime.now()) + " :: Bert Model loading started")
bert_subject_classifier_instance = bert_subject_classifier(MODEL_PATH)
print(str(datetime.datetime.now()) + " :: Bert Model loading completed")

@application.route("/")
def home():
   return "Jackett api"
   

@application.route("/subject", methods=['POST'])
def subject():
    if request.method == 'POST':
        
        print(str(datetime.datetime.now()) + "  " + str(request))

        request_data = json.loads(request.data)

        if 'questions' in request_data:
            questions = request_data['questions']
        else:
            return Response(response=({'error' : "No questions provided"}), 
                            status=401, mimetype="application/json")
        
        try:
            preds = subject_tagger_model.predict(questions)
        except :
            return Response(response=({'error' : "Error during model prediction"}), 
                            status=401, mimetype="application/json")

        print("Number of samples to predict :: " + str(len(questions)))

        return Response(response= json.dumps({'message' : "Done", 'preds' : list(preds)}),
            status=200, mimetype="application/json")
        

@application.route("/test/bert_subject", methods=['POST'])
def bert_subject():
    if request.method == 'POST':
        
        print(str(datetime.datetime.now()) + "  " + str(request))

        request_data = json.loads(request.data)

        if 'questions' in request_data:
            questions = request_data['questions']
        else:
            return Response(response=({'error' : "No questions provided"}), 
                            status=401, mimetype="application/json")
        print("Questions added from request body")
        print(questions)
        
        #preds = bert_subject_classifier_instance.classify(questions)
        
        try:
            preds = bert_subject_classifier_instance.classify(questions)
        except :
            return Response(response=({'error' : "Error during model prediction"}), 
                            status=401, mimetype="application/json")

        print("Number of samples to predict :: " + str(len(questions)))

        return Response(response= json.dumps({'message' : "Done", 'preds' : list(preds)}),
            status=200, mimetype="application/json")


@application.route("/test/recommend_subject", methods=['POST'])
def recommend_subject():
    if request.method == 'POST':

        request_data = json.loads(request.data)

        if 'question' in request_data:
            question = request_data['question']
        else:
            return Response(response=({'error' : "No question provided"}), 
                            status=401, mimetype="application/json")
        
        res = question_search_pipeline.run(query=question, params={"Retriever": {"top_k": 10}})
        print(" :::: Retrieval done")

        out = []
        for i in res['documents']:
            temp_out_dict = {
                "question":i.content,
                "chapter":i.meta['chapter'],
                "class":i.meta['class'],
                "custom":i.meta['custom'],
                "topic":i.meta['topic'],
                "subject":i.meta['subject'],
                "bloom_taxonomy":i.meta['bloom_taxonomy'],
                "curriculum":i.meta['curriculum']

            }
            out.append(temp_out_dict)

        return Response(response= json.dumps({'message' : 'Done', 'input_question':question, 'data' : out}),
            status=200, mimetype="application/json")
        


if __name__ == "__main__":
    application.run()
