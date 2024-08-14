import os
import sys
import logging
from pip._vendor import tomli
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from mmengine.config import read_base

CURRENT_PATH = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(CURRENT_PATH, '../../configs', 'ragas_config.toml')

logger = logging.getLogger(__name__)


def load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        logger.error(f'Config file {CONFIG_PATH} does not exist')
        sys.exit(1)

    with open(CONFIG_PATH, 'r') as f:
        cfg = tomli.loads(f.read())
    
    return cfg


config = load_config()


def load_llm(ragas_config: dict) -> BaseLanguageModel:
    models_config = config.get('models')
    llm_type = models_config.get('llm_type', 'openai')
    if llm_type == 'openai':
        openai_config = config.get('openai')

        os.environ["OPENAI_API_KEY"] = openai_config.get('openai_api_key', '')
        openai_api_base_ip = openai_config.get('openai_api_base_ip', 'localhost')
        openai_api_base_port = openai_config.get('openai_api_base_port', 8000)

        if 'ragas_id' in ragas_config:
            ragas_id = ragas_config['ragas_id']
            assert isinstance(ragas_id, int), f"ragas_id should be an integer, got {type(ragas_id).__name__}"
            openai_api_base_port += ragas_id
        os.environ["OPENAI_API_BASE"] = f"http://{openai_api_base_ip}:{openai_api_base_port}/v1"

        from langchain_openai.chat_models import ChatOpenAI

        return ChatOpenAI(model=openai_config.get('openai_model', 'gpt-3.5-turbo-16k'))
    
    elif llm_type == 'tongyi':
        os.environ["DASHSCOPE_API_KEY"] = models_config.get('dashscope_api_key', '')

        from langchain_community.chat_models.tongyi import ChatTongyi

        return ChatTongyi(model=models_config.get('llm_model', 'qwen1.5-72b-chat'))
    
    elif llm_type == 'vllm':
        from langchain_community.llms import VLLM
        vllm_config = config.get('vllm')
        llm = VLLM(model=vllm_config.get('vllm_model'),
                   trust_remote_code=True,
                   tensor_parallel_size=vllm_config.get('vllm_tp', 4),
                   vllm_kwargs={
                       "gpu_memory_utilization": vllm_config.get('vllm_gpu_memory_utilization', 0.8),
                       "max_model_len": vllm_config.get('vllm_max_model_len', 2048),
                    #    "enforce_eager": True,
                   } 
                   )
        
        return llm

    elif llm_type == 'lmdeploy':
        raise NotImplementedError
    elif llm_type == 'hfmodel':
        from langchain_community.llms import HuggingFaceModel
        huggingface_config = config.get('huggingface')
        llm = HuggingFaceModel(model=huggingface_config.get('hf_model'),
                               trust_remote_code=True,
                                
                               )
        
        return llm

    logger.error(f'Unsupported LLM model: {llm_type}')
    sys.exit(1)


def load_embeddings(ragas_config: dict) -> Embeddings:
    models_config = config.get('models')
    emb_type = models_config.get('emb_type', 'openai')
    if emb_type == 'openai':
        os.environ["OPENAI_API_BASE"] = models_config.get('openai_api_base', '')
        os.environ["OPENAI_API_KEY"] = models_config.get('openai_api_key', '')

        from langchain_openai.embeddings import OpenAIEmbeddings
        
        return OpenAIEmbeddings(model=models_config.get('embeddings_model', 'text-embedding-ada-002'))
    
    elif emb_type == 'dashscope':
        os.environ["DASHSCOPE_API_KEY"] = models_config.get('dashscope_api_key', '')

        from langchain_community.embeddings.dashscope import DashScopeEmbeddings

        return DashScopeEmbeddings(model=models_config.get('embeddings_model', 'text-embedding-v2'))
    elif emb_type == 'huggingface':

        from langchain_community.embeddings import HuggingFaceBgeEmbeddings

        return HuggingFaceBgeEmbeddings(
            model_name=models_config.get('embeddings_model', 'BAAI/bge-large-zh-v1.5'),
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
            )
    logger.error(f'Unsupported Embeddings model: {emb_type}')
    sys.exit(1)