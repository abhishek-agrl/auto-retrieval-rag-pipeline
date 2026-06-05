def get_system_prompt(content_dict: dict) -> str:
    system_prompt = ""
    for i, (source, content) in enumerate(content_dict.items()):
        system_prompt += f"Source {i+1}: {source}\nContext {i+1}: {content}"

    return system_prompt

def get_wiki_topic_template():
    template = \
    """
<|start_header_id|>system<|end_header_id|>
Context: You are a Wikipedia researcher and a helpful assistant. You list at most 5 relevant topics, including basics, to a query asked. 
Only concise searchable topic headings at most 4 words, nothing else. If you don't know the answer, just say "<unretrievableerror>".
Strictly structure the output as comma seperated values.

<|start_header_id|>user<|end_header_id|>
Query: {query}

<|start_header_id|>assistant<|end_header_id|>
    """
    return template

def get_generation_template():
    template = \
    """
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant and smart and funny journalist. You summarize contexts very well and present information as asked.
Use the following context when responding:

{context}


<|start_header_id|>user<|end_header_id|>
Write a brief article with a catchy headline according to User Query. Strictly output 1 headline and 3 other paragraphs.
List your sources.
User Query: {query}


<|start_header_id|>assistant<|end_header_id|>
    """
    return template


from langchain_google_genai import ChatGoogleGenerativeAI

class PatchedChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    """
    A patched version of ChatGoogleGenerativeAI that resolves a TypeError in langchain-google-genai.
    The library's _chat_with_retry function fails to strip out `max_retries` when the model name
    does not contain the substring 'gemini' (e.g., 'gemma-4-26b-a4b-it'), causing it to pass
    `max_retries` to the underlying client's `generate_content()` method which does not accept it.
    """
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        original_generate_content = self.client.generate_content
        def patched_generate_content(*args, **gen_kwargs):
            gen_kwargs.pop("max_retries", None)
            return original_generate_content(*args, **gen_kwargs)
        self.client.generate_content = patched_generate_content
        try:
            return super()._generate(messages, stop, run_manager, **kwargs)
        finally:
            self.client.generate_content = original_generate_content