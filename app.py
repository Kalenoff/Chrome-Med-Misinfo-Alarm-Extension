from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
from transformers import BertTokenizer, BertForSequenceClassification
from mistralai import Mistral, UserMessage
import torch, ast

app = Flask(__name__)
CORS(app)

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

tokenizer = BertTokenizer.from_pretrained('Fine-tuned BERT')
model = BertForSequenceClassification.from_pretrained('Fine-tuned BERT')

device = torch.device("cpu")
model.to(device)

api_key = "1vefZPJDg38cD3tCn2XdKmxkNTC0c2fm"
client = Mistral(api_key=api_key)

backend_options = ["GPT", "BERT", "Mistral"]
backend = backend_options[0]


@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Flask server!"}), 200


# Endpoint to receive the parsed text
@app.route('/receive_text', methods=['POST'])
def receive_text():
    data = request.get_json()  # Get the JSON data from the request
    if not data:
        return jsonify({"error": "Invalid request, no JSON data found"}), 400

    if 'pageText' in data:
        page_text = data['pageText']
        formatt = {"probabilities": {"credible": int, "misleading": int}, "commentary": str}

        message_content = f"""Classify the following health-related text into one of two categories: 'Credible', 'Misleading'. 
            Always provide the answer formatted as {formatt}. 
            Don't quote the statement in the commentary. 
            You must enclose the commentary string in single quotation marks, but never use any quotation marks inside the commentary string.
            Probabilities must be given in range 0-100.
            The commentary must be not longer than 30 words.
            If the statement is not health-related, return 0 for both classes and the following commentary:
            "The text on the page is not health-related."
            Statement = {page_text}"""

        if backend == backend_options[0]:
            chat_completion = openai_client.chat.completions.create(
                messages=[{"role": "user", "content": message_content[:3000]}],
                model="gpt-4"
            )
            generated_text = chat_completion.choices[0].message.content

        elif backend == backend_options[1]:
            encoded_dict = tokenizer.encode_plus(
                page_text,
                add_special_tokens=True,
                max_length=64,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )

            input_ids = encoded_dict['input_ids'].to(device)
            attention_mask = encoded_dict['attention_mask'].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
            predicted_label = torch.argmax(logits, dim=1).cpu().numpy()[0]

            result_dict = {
                "probabilities": {
                    "credible": int(probabilities[0] * 100),
                    "misleading": int(probabilities[1] * 100)  # Adjusted to use only misleading
                },
                "commentary": "No commentary is available for the chosen backend."
            }

            generated_text = str(result_dict)

        elif backend == backend_options[2]:
            messages = [UserMessage(content=message_content)]
            chat_response = client.chat.complete(model=model, messages=messages)
            generated_text = chat_response.choices[0].message.content

        # Parse the generated text to extract classification data
        classification_result = _extract_classification(generated_text)

        return jsonify({
            "summarized_topic": classification_result.get('commentary', ''),
            "probabilities": classification_result['probabilities'],
            "background": classification_result['background']
        })
    else:
        return jsonify({"error": "No text provided"}), 400


def _extract_classification(generated_text):
    # Convert the string representation of the dictionary to a Python dictionary
    try:
        result_dict = ast.literal_eval(generated_text)  # Safe evaluation of string to dict
    except (ValueError, SyntaxError):
        return {
            "probabilities": {"credible": 0, "misleading": 0},
            "commentary": "Error in parsing the response.",
            "background": "unknown"
        }

    # Extract values from the parsed dictionary
    probabilities = result_dict.get('probabilities', {})
    commentary = result_dict.get('commentary', '')

    # Determine the scores
    credible_score = probabilities.get('credible', 0)
    misleading_score = probabilities.get('misleading', 0)

    # Calculate total score for classification
    total_score = credible_score + misleading_score

    # Determine background color based on defined ranges
    if total_score == 0:
        bg_color = 'unknown'  # Handle the case where both scores are 0
    else:
        # Normalize scores to percentages
        credible_percentage = credible_score / total_score

        if credible_percentage > 0.75:
            bg_color = 'green'
        elif credible_percentage > 0.5:
            bg_color = 'orange'
        else:
            bg_color = 'red'  # Changed to 'red' for misleading

    return {
        "probabilities": {
            "credible": credible_score,
            "misleading": misleading_score
        },
        "commentary": commentary,
        "background": bg_color
    }


if __name__ == '__main__':
    app.run(debug=True)
