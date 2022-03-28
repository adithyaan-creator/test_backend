from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

## {'Biology': 0, 'Physics': 1, 'Chemistry': 2, 'Maths': 3}

class bert_subject_classifier:

    def __init__(self, model_path):

        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.subject_pipeline = pipeline(task='sentiment-analysis',tokenizer=self.tokenizer, model=self.model, padding=True, truncation=True)
        
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

        pred_out = self.subject_pipeline(text)
        predictions = [i['label'] for i in pred_out]

        out = {
            'input_text' : text,
            'prediction' : predictions
        }

        return out