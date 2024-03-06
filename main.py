from model import QA_chat
from template import template
from key_params import reading_hf_key, writing_hf_key
from retrieval import ParentDocumentRetriver
from langchain_community.llms import HuggingFaceEndpoint


retriever = ParentDocumentRetriver(
    child_chunk_size=400,
    parent_chunk_size=2000,
    hf_key=reading_hf_key
)

"""repo_id = 'meta-llama/Llama-2-7b-chat-hf'
llm = HuggingFaceEndpoint(
    repo_id=repo_id, token=writing_hf_key
)"""

model = QA_chat(retriever)
query = 'Facebook error'
print(model.answer(query, num_sites=4, qa_template=template[0]))