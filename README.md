# 🔍 Fake News Detection with BERT + Flask UI

This project fine-tunes a pre-trained BERT model (`bert-base-uncased`) on the [LIAR dataset](https://huggingface.co/datasets/liar) to classify political statements based on their truthfulness. It also features a simple web interface using Flask, allowing users to input a statement and get a real-time prediction.

---

## 📁 Project Structure

.
├── appy.py # Flask app for the UI  

│ └── my_bert_model/ # Saved BERT model and tokenizer  

├── templates/  

│ └── index.html # HTML file for the frontend  

├── save_model.py # Script to save the fine-tuned model  

└── README.md # This file

---
## 🔧 Training  
To train this model, run the script on your browser. I have set the script to detect Nvidia GPU for faster training. The current parameters for training are set to:  
epoch = 5, training rate = 2e-5, training on the entire dataset (liar)

## 🔧 Installation

🚀 Run the Application
Make sure the my_bert_model/ folder exists and contains the saved model and tokenizer files. Then run:
py appy.py


🧪 How It Works
User inputs a political statement on the web form.
The Flask backend sends the input through the fine-tuned BERT model.
The model predicts the label (e.g., true, false, pants-fire).
The result is displayed back on the same page.

📌 Notes
This is a multi-class classification task (6 classes).

Model: BertForSequenceClassification with 6 output labels.

Input is tokenized to a max length of 128.

You must use the same tokenizer (bert-base-uncased) during inference.

