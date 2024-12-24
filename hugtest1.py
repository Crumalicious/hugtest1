from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging


# Load GPT-2 model and tokenizer
def load_model():
    """
    Load GPT-2 model and tokenizer and return them.
    """
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32
    )
    return tokenizer, model


# Initialize Flask app
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Load model and tokenizer
tokenizer, model = load_model()
device = torch.device("cpu")


@app.after_request
def add_no_cache_headers(response):
    """
    Add headers to prevent browser caching.
    """
    response.headers['Cache-Control'] = (
        'no-store, no-cache, must-revalidate, max-age=0'
    )
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/')
def home():
    """
    Render the home page.
    """
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate_text():
    """
    Generate text using the GPT-2 model.
    """
    try:
        data = request.json
        prompt = data.get('prompt', '').strip()
        max_length = data.get('max_length', 50)
        temperature = data.get('temperature', 1.0)
        top_k = data.get('top_k', 50)
        top_p = data.get('top_p', 0.95)

        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        if max_length > 1000:
            return jsonify({"error": "Max length cannot exceed 1000"}), 400

        logging.info(f"Generating text for prompt: {prompt}")

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=1
        )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        return jsonify({
            "prompt": prompt,
            "generated_text": generated_text,
            "settings": {
                "max_length": max_length,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p
            }
        })
    except Exception as error:
        logging.error(f"Error during text generation: {error}")
        return jsonify({"error": str(error)}), 500


@app.errorhandler(404)
def page_not_found(error):
    """
    Handle 404 errors with a JSON response.
    """
    return jsonify({"error": "Page not found"}), 404


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
