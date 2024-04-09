import os
import sys
import logging
from pip._vendor import tomli
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings

CONFIG_PATH = os.getenv('CONFIG_PATH', 'config/config.toml')

logger = logging.getLogger(__name__)


def load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        logger.error('Config file does not exist')
        sys.exit(1)

    with open(CONFIG_PATH, 'r') as f:
        cfg = tomli.loads(f.read())
    
    return cfg


config = load_config()


def load_llm() -> BaseLanguageModel:
    models_config = config.get('models')
    llm_type = models_config.get('llm_type', 'openai')
    if llm_type == 'openai':
        os.environ["OPENAI_API_BASE"] = models_config.get('openai_api_base', '')
        os.environ["OPENAI_API_KEY"] = models_config.get('openai_api_key', '')

        from langchain_openai.chat_models import ChatOpenAI

        return ChatOpenAI(model=models_config.get('llm_model', 'gpt-3.5-turbo-16k'))
    
    elif llm_type == 'tongyi':
        os.environ["DASHSCOPE_API_KEY"] = models_config.get('dashscope_api_key', '')

        from langchain_community.chat_models.tongyi import ChatTongyi

        return ChatTongyi(model=models_config.get('llm_model', 'qwen1.5-72b-chat'))


    logger.error(f'Unsupported LLM model: {llm_type}')
    sys.exit(1)


def load_embeddings() -> Embeddings:
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