from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

## {'Biology': 0, 'Physics': 1, 'Chemistry': 2, 'Maths': 3}

class bert_subject_classifier:

    def __init__(self, model_path):

        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.subject_pipeline = pipeline(task='sentiment-analysis',tokenizer=self.tokenizer, model=self.model, padding=True, truncation=True)
        self.label_mapping = {'LABEL_0':"Biology", 'LABEL_1':"Physics", 'LABEL_2':"Chemistry", 'LABEL_3':"Maths"}

        
    def clean_text(self, texts):
        '''
        
        '''
        
        input_texts = list(texts)
        cleaned_texts = []
        for i in texts:
            print("clean")

            cleaned_texts.append(i)

        return cleaned_texts
    
    def classify(self, text):
        '''
        
        '''

        pred_out = self.subject_pipeline([text])
        predictions = [i['label'] for i in pred_out]

        out = {
            'input_text' : text,
            'prediction' : self.label_mapping[predictions[0]]
        }

        return out
