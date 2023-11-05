from flask import Flask, request, jsonify,render_template
from transformers import BartTokenizer, TFBartForSequenceClassification
from tensorflow.keras.layers import Dense
import tensorflow as tf
# import numpy as np
app = Flask(__name__)

def format_labels(labels):
    labels_list = []
    label = ''
    for i in labels:
        if i == ',':
            labels_list.append(label)
            label = ''
        elif i == ' ':
            continue
        else:
            label = str(label) + str(i)

    labels_list.append(label)
    return labels_list

@app.route('/')
def index():
    return render_template('index.html')

# to display the output in html do this => <p>{{output}}</p>
@app.route('/dataset1', methods=['GET', 'POST'])
def dataset1():
    if request.method=='POST':
        description=request.form['description']
        labels=request.form['label']
        zeroshot_model=request.form['zeroshot_model']

        input = []
        input.append(description)
        if labels == '':
            labels_list = None
        else:
            labels_list = format_labels(labels)
        input.append(labels_list)

        if zeroshot_model == 'roberta':
            netflix_model = 2
        else:
            netflix_model = 1

        output = predict_netflix(input,labels_list, netflix_model)
        return render_template('dataset1_output.html', output = output ,description = description, labels = labels)
    return render_template('dataset1.html')

@app.route('/dataset2', methods=['GET', 'POST'])
def dataset2():
    if request.method=='POST':
        purpose=request.form['purpose']
        desc=request.form['desc']
        zeroshot_model=request.form['zeroshot_model']

        print(purpose)
        print(desc)
        print(zeroshot_model)

        input = []
        input.append(purpose)
        input.append(desc)

        if zeroshot_model == 'deberta':
            dataset_model = 2
        else:
            dataset_model = 1

        output, confidence = predict(input, dataset_model)
        return render_template('dataset2_output.html', output = output, purpose = purpose, desc = desc, confidence = confidence)
    return render_template('dataset2.html')

@app.route('/dataset3', methods=['GET', 'POST'])
def dataset3():
    if request.method=='POST':
        essay=request.form['essay']
        zeroshot_model=request.form['zeroshot_model']

        print(essay)
        print(zeroshot_model)

        input = []
        input.append(essay)

        if zeroshot_model == 'deberta':
            dataset_model = 2
        else:
            dataset_model = 1

        output, confidence = predict(input, dataset_model)
        return render_template('dataset3_output.html', output = output, essay = essay)
    return render_template('dataset3.html')

def predict(input, dataset_model):
    model = TFBartForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
    model.config.dropout = 0.5
    model.classification_head.dense = Dense(model.config.d_model, activation='linear', use_bias=True)
    model.classification_head.out_proj = Dense(5, activation='linear', use_bias=True)

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-mnli",  return_tensors="tf")
    prediction = 'output here'

    if dataset_model == 1:
        loaded_model = model.from_pretrained('C:/Users/aryan/Actual-Coding/CDAC/patents-output/bart') # type: ignore
        loaded_tokenizer = tokenizer.from_pretrained('C:/Users/aryan/Actual-Coding/CDAC/patents-output/bart')
        premise = input[0]
        hypothesis = input[1]

        input_ids = tokenizer(premise, hypothesis, truncation=True, padding=True, return_tensors="tf")
        outputs = loaded_model(input_ids)
        logits = outputs.logits

        probabilities = tf.nn.softmax(logits, axis=-1)
        scores = tf.linspace(1.0, 3.0, num=3)
        expected_score = tf.reduce_sum(probabilities * scores, axis=-1)

        max_score = 3.0
        normalized_score = expected_score / max_score
        rounded_score = tf.round(normalized_score * 4) / 4

        score_to_label_mapping = {
            0.00: "Very close match",
            0.25: "Close synonym",
            0.50: "Synonyms which don’t have the same meaning (same function, same properties)",
            0.75: "Somewhat related",
            1.00: "Unrelated"
        }

        rounded_score_value = float(rounded_score.numpy()[0])
        label = score_to_label_mapping.get(rounded_score_value, "Label not found")
        print(label)

        confidence = float(normalized_score.numpy()[0])
        print(confidence)
        return(label, confidence)

    if dataset_model == 2:
        loaded_model = model.from_pretrained('C:/Users/aryan/Actual-Coding/CDAC/patents-output/deberta') # type: ignore
        loaded_tokenizer = tokenizer.from_pretrained('C:/Users/aryan/Actual-Coding/CDAC/patents-output/deberta')
        premise = input[0]
        hypothesis = input[1]

        input_ids = tokenizer(premise, hypothesis, truncation=True, padding=True, return_tensors="tf")
        outputs = loaded_model(input_ids)
        logits = outputs.logits

        probabilities = tf.nn.softmax(logits, axis=-1)
        scores = tf.linspace(1.0, 3.0, num=3)
        expected_score = tf.reduce_sum(probabilities * scores, axis=-1)

        max_score = 3.0
        normalized_score = expected_score / max_score
        rounded_score = tf.round(normalized_score * 4) / 4

        score_to_label_mapping = {
            0.00: "Very close match",
            0.25: "Close synonym",
            0.50: "Synonyms which don’t have the same meaning (same function, same properties)",
            0.75: "Somewhat related",
            1.00: "Unrelated"
        }

        rounded_score_value = float(rounded_score.numpy()[0])
        label = score_to_label_mapping.get(rounded_score_value, "Label not found")
        print(label)

        confidence = float(normalized_score.numpy()[0])
        print(confidence)
        return(label, confidence)
    
    labels_list = input[1]
    outputs = loaded_model(**input_text) # type: ignore
    logits = outputs.logits
    predicted_class_idx = tf.argmax(logits, axis=-1).numpy()[0]
    prediction = labels_list[predicted_class_idx]

    return prediction

def predict_netflix(input,labels_list, netflix_model):
    model = TFBartForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-mnli")
    prediction = 'output here'

    if netflix_model == 1:
        loaded_model = model.from_pretrained('C:/Users/aryan/Actual-Coding/CDAC/netflix-output/bart') # type: ignore
        loaded_tokenizer = tokenizer.from_pretrained('C:/Users/aryan/Actual-Coding/CDAC/netflix-output/bart')
        input_text = input[0]
        input_text = loaded_tokenizer(input_text, return_tensors="tf", padding=True, truncation=True)

    if netflix_model == 2:
        loaded_model = model.from_pretrained('C:/Users/aryan/Actual-Coding/CDAC/netflix-output/roberta') # type: ignore
        loaded_tokenizer = tokenizer.from_pretrained('C:/Users/aryan/Actual-Coding/CDAC/netflix-output/roberta')
        input_text = input[0]
        input_text = loaded_tokenizer(input_text, return_tensors="tf", padding=True, truncation=True)
    
    labels_list = input[1]
    outputs = loaded_model(**input_text) # type: ignore
    logits = outputs.logits
    predicted_class_idx = tf.argmax(logits, axis=-1).numpy()[0]
    prediction = labels_list[predicted_class_idx]

    return prediction

if __name__ == "__main__":
    app.run(debug=True)