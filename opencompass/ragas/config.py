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
        os.environ["OPENAI_API_BASE"] = models_config.get('openai_api_base', '')
        os.environ["OPENAI_API_KEY"] = models_config.get('openai_api_key', '')

        if 'api_ip' in ragas_config and 'api_port' in ragas_config:
            api_ip = ragas_config['api_ip']
            api_port = ragas_config['api_port']
            if 'ragas_port' in ragas_config:
                api_port = ragas_config['ragas_port']
            os.environ["OPENAI_API_BASE"] = f"http://{api_ip}:{api_port}/v1"

        from langchain_openai.chat_models import ChatOpenAI

        return ChatOpenAI(model=models_config.get('llm_model', 'gpt-3.5-turbo-16k'))
    
    elif llm_type == 'tongyi':
        os.environ["DASHSCOPE_API_KEY"] = models_config.get('dashscope_api_key', '')

        from langchain_community.chat_models.tongyi import ChatTongyi

        return ChatTongyi(model=models_config.get('llm_model', 'qwen1.5-72b-chat'))
    
    elif llm_type == 'vllm':
        # TODO

        from langchain_community.llms import VLLM

        llm = VLLM(model="/home/junetheriver/models/qwen/Qwen1.5-32B-Chat",
                   trust_remote_code=True,
                   tensor_parallel_size=4,
                   vllm_kwargs={
                       "gpu_memory_utilization": 0.7,
                       "max_model_len": 1024,
                    #    "enforce_eager": True,
                   } 
                   )
        
        return llm

    elif llm_type == 'vllm':

        from langchain_community.llms import VLLM

        llm = VLLM(model="/mnt/tenant-home_speed/gaozhengwei/projects/LLM/models/Qwen/Qwen1.5-72B-Chat",
                   trust_remote_code=True,
                   vllm_kwargs={
                    #    "tensor_parallel_size": 4,
                       "gpu_memory_utilization": 0.8,
                       "max_model_len": 2048,
                   }
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