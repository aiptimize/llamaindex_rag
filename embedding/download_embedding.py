from modelscope import snapshot_download

model_dir = snapshot_download('maple77/xiaobu-embedding-v2',
                              cache_dir='llamaindex_rag/embedding')
