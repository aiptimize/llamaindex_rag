from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

embed_model = HuggingFaceEmbedding(
    model_name="llamaindex_rag/embedding/maple77/xiaobu-embedding-v2"
)


Settings.embed_model = embed_model

llm = HuggingFaceLLM(
    model_name="llamaindex_rag/model/Qwen/Qwen2.5-7B-Instruct",
    tokenizer_name="llamaindex_rag/model/Qwen/Qwen2.5-7B-Instruct",
    model_kwargs={"trust_remote_code":True},
    tokenizer_kwargs={"trust_remote_code":True}
)

Settings.llm = llm

documents = SimpleDirectoryReader("./data").load_data()

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("总结一下可行性研究报告的内容")

print(response)