from llama_index import SimpleDirectoryReader, LLMPredictor, GPTVectorStoreIndex, PromptHelper, ServiceContext, load_index_from_storage, StorageContext
from langchain.chat_models import ChatOpenAI

import gradio as gr
import os

os.environ["OPENAI_API_KEY"]


# Define initial parameters
init_max_input_size = 4096
init_num_output = 512  # number of output tokens
init_max_chunk_overlap = 0.1
init_temperature = 0.7


def construct_index(directory_path, service_context):
    # load in the documents
    docs = SimpleDirectoryReader(directory_path).load_data()

    index = GPTVectorStoreIndex.from_documents(
        docs, service_context=service_context)

    # save index to disk
    index.set_index_id("vector_index")
    index.storage_context.persist(persist_dir="./gpt_store")

    return index


def chatbot(input_text, max_tokens, temperature, num_output, max_chunk_overlap):
    # Now max_tokens, temperature, num_output, and max_chunk_overlap are arguments to the chat function.
    # You need to use these arguments when calling your model.

 # Print parameters to the console:
    print(f"input_text: {input_text}")
    print(f"max_tokens: {max_tokens}")
    print(f"temperature: {temperature}")
    print(f"num_output: {num_output}")
    print(f"max_chunk_overlap: {max_chunk_overlap}")

# Here additional instructions can be put to enhance the quality of the answer.
    input_text = input_text + \
        ' Additional instructions: 1. Recognise the language at the beginning of this prompt and answer in that language 2. Give reference to the place(s) where these information(s) were found in the document.'

    # Sanity checks:
    assert max_tokens > 0, "Max Tokens must be a positive value!"
    assert 0.0 <= temperature <= 1.0, "Temperature must be between 0 and 1!"
    assert num_output > 0, "Num Tokens Answer Output must be a positive value!"
    assert 0.0 <= max_chunk_overlap < 1.0, "Max Chunk Overlap must be between 0 and 1!"

    prompt_helper = PromptHelper(
        init_max_input_size, num_output, max_chunk_overlap)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(
        temperature=temperature, model_name="gpt-3.5-turbo-16k", max_tokens=max_tokens))
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    # If not already done, initialize 'index' and 'query_engine'
    if not hasattr(chatbot, "index"):
        # rebuild storage context and load index
        storage_context = StorageContext.from_defaults(
            persist_dir="./gpt_store")
        chatbot.index = load_index_from_storage(
            service_context=service_context, storage_context=storage_context, index_id="vector_index")

        # Initialize query engine
        chatbot.query_engine = chatbot.index.as_query_engine()

    # Submit query
    response = chatbot.query_engine.query(input_text)

    return response.response


iface = gr.Interface(fn=chatbot,
                     inputs=[
                         gr.components.Textbox(
                             lines=7, label="Enter your text"),
                         gr.components.Slider(
                             minimum=512, maximum=2048, step=128,  label="Max Tokens"),
                         gr.components.Slider(
                             minimum=0.1, maximum=1, step=0.1,  label="Temperature"),
                         gr.components.Slider(
                             minimum=512, maximum=2048, step=128,  label="Num Tokens Answer Output"),
                         gr.components.Slider(
                             minimum=0.1, maximum=1, step=0.1,  label="Max Chunk Overlap")
                     ],
                     outputs="text",
                     title="Custom-trained AI Chatbot")

# comment out after 1st run if training docs aren't changing
service_context_init = ServiceContext.from_defaults(
    llm_predictor=LLMPredictor(llm=ChatOpenAI(
        temperature=init_temperature, model_name="gpt-3.5-turbo", max_tokens=init_num_output)),
    prompt_helper=PromptHelper(
        init_max_input_size, init_num_output, init_max_chunk_overlap)
)
# index = construct_index("docs", service_context_init)
iface.launch(share=False)
