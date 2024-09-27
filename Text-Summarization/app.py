from flask import Flask, render_template, request, session, redirect, url_for, flash
from pyngrok import ngrok
from transformers import BartForConditionalGeneration, BartTokenizer
import torch

app = Flask(__name__)
app.secret_key = 'insert ngrok secret key here'

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

def generate_summary(text):
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)

    summary_ids = model.generate(
        inputs,
        max_length=200,
        min_length=100,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary.strip()

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'summary_history' not in session:
        session['summary_history'] = []

    if request.method == 'POST':
        input_text = request.form['text']
        summary = generate_summary(input_text)
        session['summary_history'].append(summary)
        session.modified = True
        flash('Summary generated successfully!', 'success')

    return render_template('index.html', summaries=session.get('summary_history', []))

@app.route('/clear_history')
def clear_history():
    session.pop('summary_history', None)
    flash('Summary history cleared!', 'info')
    return redirect(url_for('index'))

if __name__ == '__main__':
    public_url = ngrok.connect(5000)
    print(f"Public URL: {public_url}")
    app.run()
