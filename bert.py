from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
from warnings import filterwarnings 
filterwarnings(action = 'ignore')
model_name = "finiteautomata/bertweet-base-sentiment-analysis"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
torch.save(model.state_dict(), "model.pth")
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)


result = classifier("This is Rubbish")
print(result)
