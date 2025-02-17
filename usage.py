import torch
from transformers import RobertaConfig, RobertaForSequenceClassification, AutoTokenizer, logging
import torch.nn.functional as F
from configurations import USERNAME , MODEL_REPO

logging.set_verbosity_error()


tokenizer = AutoTokenizer.from_pretrained(f"{USERNAME}/{MODEL_REPO}")


config = RobertaConfig(
    vocab_size=64001,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=130,
    type_vocab_size=1,
    num_labels=3,  
    model_type="roberta",
    id2label={0: "negative", 1: "neutral", 2: "positive"},
    label2id={"negative": 0, "neutral": 1, "positive": 2},
)

model = RobertaForSequenceClassification(config)
state_dict = torch.load("model.pth", map_location="cpu", weights_only=True)
model.load_state_dict(state_dict)
model.eval()

def predict_label(input_text):
    encoded_input = tokenizer(
        input_text,
        return_tensors="pt",      
        padding=True,
        truncation=True  
    )
    with torch.no_grad():
        outputs = model(**encoded_input)
        logits = outputs.logits 


    probs = F.softmax(logits, dim=-1)  
    pred_label_id = torch.argmax(probs, dim=-1).item() 
    confidence = probs[0, pred_label_id].item()

    if hasattr(model.config, "id2label") and model.config.id2label:
        pred_label_str = model.config.id2label.get(pred_label_id, str(pred_label_id))
    else:
        pred_label_str = str(pred_label_id)

    return pred_label_str

