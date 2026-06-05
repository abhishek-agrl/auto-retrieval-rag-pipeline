# Autonomous Information Retrieval & RAG Pipeline
A simple project to understand RAG systems and their integration with LLMs.

## Setup
You need to have access to an LLM for this project to work. I used [LM-Studio](https://lmstudio.ai) to locally host [`llama-3.2-3b-instruct 4bit`](https://huggingface.co/mlx-community/Llama-3.2-3B-Instruct-4bit) on my `Mac Air M2 8GB` and it struggles but runs nonetheless.
LM-Studio provides [OpenAI like endpoints](https://lmstudio.ai/docs/app/api/endpoints/openai) which work nicely with [LangChain's OpenAI Integration Interface](https://python.langchain.com/docs/integrations/llms/openai/). 
As my choosen LLM doesn't provide an embedding model, I load [`nomic-embed-text-v1.5 - GGUF`](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF) to use as the embedding model for my vectorstore(ChromaDB).
If you choose to use the same config then you wont really need to change the code, but in case you do something different then please modify `config.py` accordingly. \
\
Steps - 
1. Clone the repo.
2. (Optional) Setup a virtual/conda environment.
3. Run `pip install -r requirements.txt`
4. (local LLM) Make sure LM-Studio is running and hosting the LLM and embedding model.
5. Run `python3 app.py` and start generating! (Visible at `http://localhost:5000/`)

## Architecture
* The application has a simple UI written in vanilla JS and HTML/CSS. It is served by a Flask server which also contains an endpoint to trigger the RAG pipeline.
* On user prompt the following flow takes place -
  1. The query is first used to search through the local vectorstore and get 5 relevant documents, in case the information is already available locally. This prevents going to Wikipedia for repeated as well as similar prompts. Obviously, this also boosts overall response times.
  2. If 5 relevant documents are found (relevancy determined by the ChromaDB retriever parameter [score_threshold](https://python.langchain.com/api_reference/chroma/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.as_retriever)), We jump to step 7.
  3. If not, then the query is sent to the LLM with a hard-coded prompt to generate a list of 5 relevant topics to search on Wikipedia.
  4. The generated topics are parallelly searched on Wikipedia using [LangChain RunnableParallel](https://python.langchain.com/docs/how_to/parallel/) and [LangChain Wikipedia Retriever](https://python.langchain.com/docs/integrations/retrievers/wikipedia/) to get 2 documents per topic.
  5. The documents (still parallelly) are splitted into chunks according to the token size of embedding model and stored in the vectorstore after embedding.
  6. The vectorstore is again searched for 5 relevant documents, and with the updated vectorstore the criteria is usually met. If still not found then there is a great chance that the user prompt may be gibberish and hence an error is returned and displayed in UI.
  7. The found documents are plugged in a hard-coded prompt with the user query and inputed to the LLM for article generation.
  8. The LLM output is then passed on to the UI for display.

## Summary
Learnt a lot about RAG and its implementation in Python using LangChain. Although it looked like prompt engineering at first, the semantic document search using vectorstore was a tricky problem to solve. Nonetheless, the project works as intended! One way I used to test the system was to ask to generate an article about the latest "Oscars" and its winners.
The response was satisfactory with up to date information about the winners, host and the event in general. The thing to note is that my choosen LLM (Llama 3.2) had a [training cutoff of December 2023](https://www.prompthub.us/models/llama-3-2-90b#:~:text=The%20knowledge%20cut%2Doff%20date,90B%20is%20December%201%2C%202023.) and couldn't have been pre-trained on this information.
