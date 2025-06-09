from transformers import BertTokenizer, BertForSequenceClassification

# Load the trained model checkpoint
model = BertForSequenceClassification.from_pretrained("./results/checkpoint-1000") 

# Load the tokenizer (same one used during training)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Save the model and tokenizer
model.save_pretrained("./my_bert_model")
tokenizer.save_pretrained("./my_bert_model")

print(" Model and tokenizer saved to ./my_bert_model/")
