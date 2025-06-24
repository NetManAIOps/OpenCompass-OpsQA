from opencompass.models import OpenAIAzure
from mmengine import read_base
with read_base():
    from ...paths import ROOT_DIR
    from ...api_key import azure_gpt4o, azure_gpt4


models = [
    dict(abbr='GPT-4',
            type=OpenAIAzure,
            api_key=azure_gpt4['api_key'],
            api_version=azure_gpt4['api_version'],
            endpoint=azure_gpt4['endpoint'],
            deployment=azure_gpt4['deployment'],
         max_out_len=100, max_seq_len=2048, batch_size=1),
    dict(abbr='GPT-4o',
            type=OpenAIAzure,
            api_key=azure_gpt4o['api_key'],
            api_version=azure_gpt4o['api_version'],
            endpoint=azure_gpt4o['endpoint'],
            deployment=azure_gpt4o['deployment'],
            max_out_len=100, max_seq_len=2048, batch_size=1)
]