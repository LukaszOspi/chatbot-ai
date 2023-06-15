# AI Chatbot with LLM and Gradio

This is an AI chatbot using the LLM (Language Model with Memory) system with OpenAI's GPT-3.5-turbo model, implemented in Gradio interface for easy testing and deployment.

## Setup and Installation

1. Clone the repository to your local machine.
2. Install the necessary libraries by running `pip install -r requirements.txt` or `pip3 install -r requirements.txt`, depending on your system.
3. Declare the OpenAI API key in your system environment: `export OPENAI_API_KEY='key'`.

## Project Structure

This project is structured as follows:

- `llama_index.py`: This file contains various classes and functions for loading and searching documents in the AI model's memory.
- `langchain.chat_models.py`: This file defines the `ChatOpenAI` model which is used to communicate with OpenAI's API.
- `docs/`: This directory should contain the PDF documents that will be indexed by the chatbot. It should be at the same level as the main script.

## Index Construction

Before the chatbot can be used, an index must be built from the documents that should be searchable by the chatbot. This is done by calling the `construct_index` function with the directory path to your 'docs' folder.

This function reads the documents, builds an index, and then saves the index to disk for later use.

Please note: Comment out the line `index = construct_index("docs")` after the first run if the training documents are not changing.

## Chatbot Functionality

The `chatbot` function handles querying the index and generating a response based on the input text. It first checks if an index and query engine have already been initialized, if not, it initializes them. It then submits the input text as a query to the query engine and returns the generated response.

## Gradio Interface

Gradio is used to provide an interface for easy interaction with the chatbot. The `gr.Interface` function is used to set up the interface, defining the `chatbot` function as the function to be run when text is input. The interface is then launched with the `launch` method.

The `share` parameter of the `launch` function determines whether a publicly accessible link to the interface is created. If you set `share=True`, a link will be provided in the console which can be shared with others to use the chatbot.

## Running the Program

Depending on your OS, run the Python script with `python filename.py` or `python3 filename.py`.

## Note

The `temperature` parameter in the `ChatOpenAI` model constructor controls the creativity of the responses. The lower the value (between 0.1 and 1.0), the more focused and less random the output will be.

## License

This project is licensed under the terms of the MIT license.
