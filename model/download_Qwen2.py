from modelscope import snapshot_download

model_dir = snapshot_download('Qwen/Qwen2.5-7B-Instruct',
                              cache_dir='llamaindex_rag/model')