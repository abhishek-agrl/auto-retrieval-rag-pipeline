from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from wikipedia_retriever import get_all_wiki_docs
from helper import get_system_prompt, get_generation_template, PatchedChatGoogleGenerativeAI
import time
from config import GEMINI_API_KEY, GEMINI_MODEL, EMBEDDING_MODEL, LLM_TEMPERATURE, \
                   VECTORSTORE_CONFIDENCE_THRESHOLD, VECTORSTORE_DOCUMENT_LIMIT
class ArticleGenerator:
    def __init__(self, 
                 api_key=GEMINI_API_KEY, 
                 model_name=GEMINI_MODEL,
                 embedding_model=EMBEDDING_MODEL,
                 temperature = LLM_TEMPERATURE,
                 ):
        
        self.model = PatchedChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model_name,
            temperature=temperature,
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=api_key,
            model=embedding_model,
        )
        self.vector_store = Chroma(
            embedding_function=self.embeddings,
            persist_directory='./chroma_langchain_db'
        )
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                'score_threshold': VECTORSTORE_CONFIDENCE_THRESHOLD, 
                'k': VECTORSTORE_DOCUMENT_LIMIT
            }
        )
    
    def load_vector_store(self, query: str):
        wiki_docs = get_all_wiki_docs(model=self.model, search_query=query)
        if wiki_docs and len(wiki_docs)>0:
            self.vector_store.add_documents(wiki_docs)

    def get_relevant_content(self, search_query: str) -> str:
        docs = self.retriever.invoke("search_query: "+search_query)
        if len(docs)<5:
            self.load_vector_store(search_query)
            time.sleep(2)
            docs = self.retriever.invoke("search_query: "+search_query)
        
        if len(docs)<1:
            print("Couldn't find valid documents")
            return None
        
        # Strip "search_document: " from the start of each document
        pg_content = {doc.metadata['source']: doc.page_content[17:] for doc in docs}
        return pg_content
    
    def get_context(self, search_query: str) -> str:
        content_dict = self.get_relevant_content(search_query)
        if not content_dict:
            return None
        
        context = get_system_prompt(content_dict)
        return context

    def generate(self, search_query: str) -> dict:
        if search_query.strip()=="":
            return None
        generation_prompt = ChatPromptTemplate.from_template(get_generation_template())
        
        generation_chain = generation_prompt | self.model | StrOutputParser()
        context = self.get_context(search_query=search_query)
        if not context:
            return None
        
        return generation_chain.invoke({'context': context, 'query':search_query})
        
        
# ag = ArticleGenerator()
# print(ag.generate("Oscars 2025 results"))
