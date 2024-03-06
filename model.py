from langchain.chains import RetrievalQA 
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.chat_models import ChatOllama 
from llamaapi import LlamaAPI

# Replace 'Your_API_Token' with your actual API token
llama = LlamaAPI("pip install –upgrade –quiet llamaapi")

class QA_chat:
    def __init__(self, retriever, model_name='zephyr'):
        # Retriever
        self.retriever = retriever        

        # Model
        self.chat_model = ChatOllama(
            model=model_name,
            verbose=True,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
        )

    def answer(self, query, num_sites, qa_template):
        # QA prompt
        QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=['context', 'question'],
            template=qa_template
        )

        # Retriever 
        added_retriver = self.retriever.retrieve(query, num_sites=num_sites)

        # QA chain
        qa_chain = RetrievalQA.from_chain_type(
            self.chat_model,
            chain_type='stuff',
            retriever=added_retriver,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
            return_source_documents=True
        )

        # result = qa_chain({"query": query})
        result = qa_chain.invoke({"query": query})
        
        return result