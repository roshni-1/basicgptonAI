# Chat Application with GPT-2 Large Model
# Application: DemoGPT-1.0

![DemoGPT-1.0](https://github.com/roshni-1/basicgptonAI/blob/main/Screenshot%202024-04-15%20102542.png)

## Overview
This project aims to develop a chat application that uses the GPT-2 Large model to generate responses based on a small dataset. The application's frontend is developed using Flask, HTML, CSS, and JavaScript, while the backend is implemented in Python using the transformers library (part of the Hugging Face library) with TensorFlow.

## Project Steps
1. **Dataset Gathering**: Collecting a dataset of informational text realted to AI, Data Science, Machine Learning etc. This dataset will be used to fine-tune the GPT-2 Large model.
2. **Preprocessing and Tokenization**: Preprocessing the dataset to clean the text and tokenize it for model training.
3. **Model Training**: Training the GPT-2 Large model on the preprocessed dataset. This step involves fine-tuning the pre-trained model to adapt it to the specific domain of the dataset.
4. **Frontend Development**: Developing the frontend of the chat application using Flask, HTML, CSS, and JavaScript. The frontend will provide a user interface for entering prompts and displaying generated responses.
5. **Integration with Model**: Integrating the trained GPT-2 Large model with the frontend of the chat application. This involves setting up an endpoint in the backend to receive user prompts, generating responses using the model, and sending the generated responses back to the frontend.
6. **Testing and Evaluation**: Testing the chat application to ensure that it generates coherent and relevant responses. Evaluating its performance and user experience(based on generated response).
7. **Iterative Improvement**: Iterate on the project by fine-tuning the model on more datasets, improving the frontend interface, and enhancing the overall functionality and accuracy of the chat application.

![DemoGPT-1.0](https://github.com/roshni-1/basicgptonAI/blob/main/Screenshot%202024-04-15%20100720.png)

## Current Status
The project is currently in development. The chat application is able to generate responses using the GPT-2 Large model trained on a small dataset. However, the responses are sometimes mixed and incomplete, indicating the need for further fine-tuning and improvement. The next phase of the project will focus on training the model on more datasets and fine-tuning it to generate more accurate and complete responses.

## Usage
To run the chat application locally:
1. Download the dataset.
2. Run the notebook and python codes related to model training and fine-tuning (Change the path as per your directory path). 
1. Run the Flask app using `python app.py`.
2. Access the chat application in your web browser at the specified address (e.g., `http://localhost:5000`).

## Contributor
- Roshni Yadav

## Acknowledgments
- This project is inspired by the work of OpenAI and the Hugging Face team in developing state-of-the-art natural language processing models.
