from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
import torch 


def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='weighted')
    }


dataset = load_dataset("liar", trust_remote_code=True)

#1------load the tokenizer 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#2------tokenize the date 
def tokenize_function(examples):
    return tokenizer(examples['statement'], padding = 'max_length', truncation = True, max_length = 128 )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

#3------convert data to torch and set correct format 
tokenized_dataset = tokenized_dataset.rename_column('label', 'labels') #the spam vs not spam are store in label in the dataset but BERT expects it to be in labels so we rename it 
#properly format the PyTorch tensors
tokenized_dataset.set_format("torch", columns = ['input_ids', 'attention_mask', 'labels'])
#define the training data
train_dataset = tokenized_dataset["train"].shuffle(seed=42)
#define the testing data 
test_dataset = tokenized_dataset["test"].shuffle(seed=42).select(range(1000))

#load BERT
model =BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = 6)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = model.to(device)


#define trainign args 
training_args = TrainingArguments(
output_dir="./results",
do_eval=True,
learning_rate=2e-5,
per_device_train_batch_size=16,
per_device_eval_batch_size=16,
num_train_epochs=5,
weight_decay=0.01,
logging_dir='./logs',
logging_steps=10,
save_steps=10,
)

# Define a Trainer instance
trainer = Trainer(
model=model,
args=training_args,
train_dataset=train_dataset,
eval_dataset=test_dataset,
compute_metrics=compute_metrics

)

# Train the model
trainer.train()

# Evaluate the model
eval_result = trainer.evaluate()
print(f"Evaluation results: {eval_result}")

model.save_pretrained("./my_bert_model")
tokenizer.save_pretrained("./my_bert_model")
