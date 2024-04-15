from flask import Flask, request, jsonify, render_template
from transformers import GPT2LMHeadModel, AutoTokenizer

app = Flask(__name__)

# path to the saved model
model_path = "C:\\Users\\irosh\\fine-tuned-gpt2-large"

# Loading the model
model = GPT2LMHeadModel.from_pretrained(model_path)

# Initializing tokenizer using AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# generation parameters
generation_params = {
    "max_length": 300,  
    "temperature": 0.90,  
    "top_p": 0.5,
    "top_k": 40,
    "num_return_sequences": 1,
    "pad_token_id": tokenizer.eos_token_id,  
    "eos_token_id": tokenizer.eos_token_id,  
    "do_sample": True, 
    "early_stopping": False  
}

# route to html page
@app.route("/")
def chat():
    return render_template("chat.html")

# Endpoint to handle user input and generate responses
@app.route("/get-response", methods=["POST"])
def get_response():
    try:
        data = request.json
        prompt = data["prompt"]
        
        if not prompt:
            return jsonify({"error": "Empty prompt"}), 400

        # Tokenizing the input prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        # Generating response
        output_sequences = model.generate(input_ids=input_ids, **generation_params)
        generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

        return jsonify({"response": generated_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
