import os
import csv
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, send_from_directory
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

import torch

app = Flask(__name__)

# Load the BERT model and tokenizer
model_path = 'model/'  # Path to your model directory
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)


# Ensure the model is on the correct device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# File to store predictions
output_file = 'detailed_feedback.csv'

@app.route('/download_feedback')
def download_feedback():
    return send_from_directory(directory='.', path='detailed_feedback.csv', as_attachment=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    user_type = request.args.get('user_type', 'student')  # Default to 'student' if not provided
    sentiment = None
    feedback = None
    success = False  # Success flag for submission alert

    if request.method == 'POST':
        feedback = request.form.get('feedback')

        # Process the feedback and determine sentiment (using the BERT model)
        if feedback:
            inputs = tokenizer(feedback, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs = {key: val.to(device) for key, val in inputs.items()}

            # Make prediction
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_class = torch.argmax(logits, dim=1).item()

            sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
            sentiment = sentiment_map.get(predicted_class, 'Neutral')
        else:
            sentiment = 'Neutral'

        # Save feedback data using the correct function name
        save_feedback_to_csv(request.form, sentiment)
        generate_graph()
        success = True  # Set success flag to True after successful submission
        return redirect(url_for('index', success=success))
    if user_type == 'student':
        return render_template('index_student.html', sentiment=sentiment, feedback=feedback)
    else:
        feedback_data = load_feedback_data()  # Load feedback data for librarian view
        return render_template('index_librarian.html', feedback_data=feedback_data)

@app.route('/predict', methods=['POST'])
def predict():
    feedback = request.form.get('feedback')

    if feedback:
        inputs = tokenizer(feedback, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()

        sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        sentiment = sentiment_map.get(predicted_class, 'Neutral')
    else:
        sentiment = 'Neutral'

    save_feedback_to_csv(request.form, sentiment)
    generate_graph()

    user_type = request.form.get('user_type', 'student')
    feedback_data = load_feedback_data()

    if user_type == 'librarian':
        return render_template('index_librarian.html', feedback_data=feedback_data)
    else:
        return render_template('index_student.html', sentiment=sentiment, feedback=feedback)

def save_feedback_to_csv(form_data, sentiment):
    feedback_data = [
        form_data.get('name'),
        form_data.get('reg_no'),
        form_data.get('department'),
        form_data.get('email'),
        form_data.get('mobile'),
        form_data.get('visit_frequency'),
        form_data.get('books_collected'),
        form_data.get('reading_hours'),
        form_data.get('resource_purpose'),
        form_data.get('library_facility'),
        form_data.get('staff_support'),
        form_data.get('resource_quality'),
        form_data.get('learning_needs'),
        form_data.get('suggestions'),
        form_data.get('feedback'),
        sentiment
    ]

    if not os.path.exists(output_file):
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Name', 'Register No/Emp No', 'Department', 'Email ID', 'Mobile No',
                'Library Visit Frequency', 'Books Collected Weekly', 'Hours Spent Reading',
                'Library Usage Purpose', 'Library Facility Rating', 'Staff Support Rating',
                'Book Collection Rating', 'Learning Needs Fulfillment', 'Suggestions', 'Feedback', 'Sentiment'
            ])

    with open(output_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(feedback_data)

def load_feedback_data():
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
        return df.to_dict(orient='records')
    return []

def generate_graph():
    # Generate the sentiment distribution graph
    df = pd.read_csv(output_file)

    sentiment_counts = df['Sentiment'].value_counts()
    plt.figure(figsize=(10, 6))
    sentiment_counts.plot(kind='bar', color=['red', 'blue', 'green'])
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.savefig('static/feedback_analysis.png')  # Save graph to static folder
    plt.close()

if __name__ == '__main__':
    app.run(debug=True)
