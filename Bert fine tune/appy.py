from flask import Flask, request, render_template
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)

# Load model and tokenizer
model_path = "./my_bert_model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
model.eval()

# Label mapping
label_map = {
    0: "pants-fire",
    1: "false",
    2: "barely-true",
    3: "half-true",
    4: "mostly-true",
    5: "true"
}

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        text = request.form['text']
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
            prediction = label_map[pred]
    return render_template('index.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
