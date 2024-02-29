from opencompass.models import HuggingFace, HuggingFaceCausalLM
from mmengine.config import read_base
with read_base():
    from ..paths import ROOT_DIR

nm_qwen_14b_lora_3gpp_general_sft_owlinstruct = dict(
    type=HuggingFaceCausalLM,
    abbr='nm_qwen_14b_lora_3gpp_general_sft_owlinstruct',
    path=f"{ROOT_DIR}models/xll_sft_models/owl_instruct/lora_16_pt/lora_3gpp_general/merged_model",
    tokenizer_path=f"{ROOT_DIR}models/xll_sft_models/owl_instruct/lora_16_pt/lora_3gpp_general/merged_model",
    tokenizer_kwargs=dict(padding_side='left',
                            truncation_side='left',
                            trust_remote_code=True,
                            use_fast=False,),
    max_out_len=40,
    max_seq_len=2048,
    batch_size=8,
    model_kwargs=dict(device_map='auto', trust_remote_code=True),
    run_cfg=dict(num_gpus=1, num_procs=1),
)

nm_qwen_14b_freeze_3gpp_general_sft_owlinstruct = dict(
    type=HuggingFaceCausalLM, 
    abbr='nm_qwen_14b_freeze_3gpp_general_sft_owlinstruct',
    path=f"{ROOT_DIR}models/xll_sft_models/owl_instruct/freeze_pt/lora_3gpp_general/merged_model",
    tokenizer_path=f"{ROOT_DIR}models/xll_sft_models/owl_instruct/freeze_pt/lora_3gpp_general/merged_model",
    tokenizer_kwargs=dict(padding_side='left',
                            truncation_side='left',
                            trust_remote_code=True,
                            use_fast=False,),
    max_out_len=40,
    max_seq_len=2048,
    batch_size=8,
    model_kwargs=dict(device_map='auto', trust_remote_code=True),
    run_cfg=dict(num_gpus=1, num_procs=1),
)

nm_qwen_14b_lora_3gpp_general = dict(
    type=HuggingFaceCausalLM,
    abbr='nm_qwen_14b_lora_3gpp_general',
    path=f"{ROOT_DIR}models/xll_models/Qwen-14B/lora_16/3gpp_general/merged_model",
    tokenizer_path=f"{ROOT_DIR}models/xll_models/Qwen-14B/lora_16/3gpp_general/merged_model",
    tokenizer_kwargs=dict(padding_side='left',
                            truncation_side='left',
                            trust_remote_code=True,
                            use_fast=False,),
    max_out_len=40,
    max_seq_len=2048,
    batch_size=8,
    model_kwargs=dict(device_map='auto', trust_remote_code=True),
    run_cfg=dict(num_gpus=1, num_procs=1),
)

nm_qwen_14b_freeze_3gpp_general = dict(
    type=HuggingFaceCausalLM,
    abbr='nm_qwen_14b_freeze_3gpp_general',
    path=f"{ROOT_DIR}models/xll_models/Qwen-14B/freeze/3gpp_general",
    tokenizer_path=f"{ROOT_DIR}models/xll_models/Qwen-14B/freeze/3gpp_general",
    tokenizer_kwargs=dict(padding_side='left',
                            truncation_side='left',
                            trust_remote_code=True,
                            use_fast=False,),
    max_out_len=40,
    max_seq_len=2048,
    batch_size=8,
    model_kwargs=dict(device_map='auto', trust_remote_code=True),
    run_cfg=dict(num_gpus=1, num_procs=1),
)

nm_qwen_14b_full_mixture_sft_owlbench_instruct = dict(
    type=HuggingFaceCausalLM,
    abbr='nm_qwen_14b_full_mixture_sft_owlbench_instruct',
    path=f"{ROOT_DIR}models/xll_sft_models/owlbench_instruct/lora_mix_dataset/merged_model",
    tokenizer_path=f"{ROOT_DIR}models/xll_sft_models/owlbench_instruct/lora_mix_dataset/merged_model",
    tokenizer_kwargs=dict(padding_side='left',
                            truncation_side='left',
                            trust_remote_code=True,
                            use_fast=False,),
    max_out_len=40,
    max_seq_len=2048,
    batch_size=8,
    model_kwargs=dict(device_map='auto', trust_remote_code=True),
    run_cfg=dict(num_gpus=1, num_procs=1),
)

nm_qwen_14b_full_concat_sft_owlbench_instruct = dict(
    type=HuggingFaceCausalLM,
    abbr='nm_qwen_14b_full_concat_sft_owlbench_instruct',
    path=f"{ROOT_DIR}models/xll_sft_models/owlbench_instruct/lora_concat_dataset/merged_model",
    tokenizer_path=f"{ROOT_DIR}models/xll_sft_models/owlbench_instruct/lora_concat_dataset/merged_model",
    tokenizer_kwargs=dict(padding_side='left',
                            truncation_side='left',
                            trust_remote_code=True,
                            use_fast=False,),
    max_out_len=40,
    max_seq_len=2048,
    batch_size=8,
    model_kwargs=dict(device_map='auto', trust_remote_code=True),
    run_cfg=dict(num_gpus=1, num_procs=1),
)

nm_qwen_14b_full_mixture_sft_owlbench = dict(
    type=HuggingFaceCausalLM,
    abbr='nm_qwen_14b_full_mixture_sft_owlbench',
    path=f"{ROOT_DIR}models/xll_sft_models/owlbench/lora_mix_dataset/merged_model",
    tokenizer_path=f"{ROOT_DIR}models/xll_sft_models/owlbench/lora_mix_dataset/merged_model",
    tokenizer_kwargs=dict(padding_side='left',
                            truncation_side='left',
                            trust_remote_code=True,
                            use_fast=False,),
    max_out_len=40,
    max_seq_len=2048,
    batch_size=8,
    model_kwargs=dict(device_map='auto', trust_remote_code=True),
    run_cfg=dict(num_gpus=1, num_procs=1),
)

nm_qwen_14b_full_concat_sft_owlbench = dict(
    type=HuggingFaceCausalLM,
    abbr='nm_qwen_14b_full_concat_sft_owlbench',
    path=f"{ROOT_DIR}models/xll_sft_models/owl_bench/lora_concat_dataset/merged_model",
    tokenizer_path=f"{ROOT_DIR}models/xll_sft_models/owl_bench/lora_concat_dataset/merged_model",
    tokenizer_kwargs=dict(padding_side='left',
                            truncation_side='left',
                            trust_remote_code=True,
                            use_fast=False,),
    max_out_len=40,
    max_seq_len=2048,
    batch_size=8,
    model_kwargs=dict(device_map='auto', trust_remote_code=True),
    run_cfg=dict(num_gpus=1, num_procs=1),
)

nm_qwen_14b_full_mixture_sft_owlinstruct = dict(
    type=HuggingFaceCausalLM,
    abbr='nm_qwen_14b_full_mixture_sft_owlinstruct',
    path=f"{ROOT_DIR}models/xll_sft_models/owl_instruct/lora_mix_dataset/merged_model",
    tokenizer_path=f"{ROOT_DIR}models/xll_sft_models/owl_instruct/lora_mix_dataset/merged_model",
    tokenizer_kwargs=dict(padding_side='left',
                            truncation_side='left',
                            trust_remote_code=True,
                            use_fast=False,),
    max_out_len=40,
    max_seq_len=2048,
    batch_size=8,
    model_kwargs=dict(device_map='auto', trust_remote_code=True),
    run_cfg=dict(num_gpus=1, num_procs=1),
)

nm_qwen_14b_full_concat_sft_owlinstruct = dict(
    type=HuggingFaceCausalLM,
    abbr='nm_qwen_14b_full_concat_sft_owlinstruct',
    path=f"{ROOT_DIR}models/xll_sft_models/owl_instruct/lora_concat_dataset/merged_model",
    tokenizer_path=f"{ROOT_DIR}models/xll_sft_models/owl_instruct/lora_concat_dataset/merged_model",
    tokenizer_kwargs=dict(padding_side='left',
                            truncation_side='left',
                            trust_remote_code=True,
                            use_fast=False,),
    max_out_len=40,
    max_seq_len=2048,
    batch_size=8,
    model_kwargs=dict(device_map='auto', trust_remote_code=True),
    run_cfg=dict(num_gpus=1, num_procs=1),
)

nm_qwen_14b_freeze = dict(
    type=HuggingFaceCausalLM,
    abbr='nm_qwen_14b_freeze',
    path=f"{ROOT_DIR}models/xll_models/Qwen-14B/freeze",
    tokenizer_path=f"{ROOT_DIR}models/xll_models/Qwen-14B/freeze",
    tokenizer_kwargs=dict(padding_side='left',
                            truncation_side='left',
                            trust_remote_code=True,
                            use_fast=False,),
    max_out_len=40,
    max_seq_len=2048,
    batch_size=8,
    model_kwargs=dict(device_map='auto', trust_remote_code=True),
    run_cfg=dict(num_gpus=1, num_procs=1),
)

nm_qwen_14b_full_mixture = dict(
    type=HuggingFaceCausalLM,
    abbr='nm_qwen_14b_full_mixture',
    path=f"{ROOT_DIR}models/xll_models/Qwen-14B/full/mix_dataset_bak",
    tokenizer_path=f"{ROOT_DIR}models/xll_models/Qwen-14B/full/mix_dataset_bak",
    tokenizer_kwargs=dict(padding_side='left',
                            truncation_side='left',
                            trust_remote_code=True,
                            use_fast=False,),
    max_out_len=40,
    max_seq_len=2048,
    batch_size=8,
    model_kwargs=dict(device_map='auto', trust_remote_code=True),
    run_cfg=dict(num_gpus=1, num_procs=1),
)

nm_qwen_14b_full_concat = dict(
    type=HuggingFaceCausalLM,
    abbr='nm_qwen_14b_full_concat',
    path=f"{ROOT_DIR}models/xll_models/Qwen-14B/full/concat_dataset",
    tokenizer_path=f"{ROOT_DIR}models/xll_models/Qwen-14B/full/concat_dataset",
    tokenizer_kwargs=dict(padding_side='left',
                            truncation_side='left',
                            trust_remote_code=True,
                            use_fast=False,),
    max_out_len=40,
    max_seq_len=2048,
    batch_size=8,
    model_kwargs=dict(device_map='auto', trust_remote_code=True),
    run_cfg=dict(num_gpus=1, num_procs=1),
)

nm_qwen_1_8b_full_concat = dict(
    type=HuggingFaceCausalLM,
    abbr='nm_qwen_1_8b_full_concat',
    path=f"{ROOT_DIR}models/xll_models/Qwen-1_8B/full/concat_dataset",
    tokenizer_path=f"{ROOT_DIR}models/xll_models/Qwen-1_8B/full/concat_dataset",
    tokenizer_kwargs=dict(padding_side='left',
                            truncation_side='left',
                            trust_remote_code=True,
                            use_fast=False,),
    max_out_len=40,
    max_seq_len=2048,
    batch_size=8,
    model_kwargs=dict(device_map='auto', trust_remote_code=True),
    run_cfg=dict(num_gpus=1, num_procs=1),
)

nm_qwen_1_8b_full_mixture = dict(
    type=HuggingFaceCausalLM,
    abbr='nm_qwen_1_8b_full_mixture',
    path=f"{ROOT_DIR}models/xll_models/Qwen-1_8B/full/mix_dataset",
    tokenizer_path=f"{ROOT_DIR}models/xll_models/Qwen-1_8B/full/mix_dataset",
    tokenizer_kwargs=dict(padding_side='left',
                            truncation_side='left',
                            trust_remote_code=True,
                            use_fast=False,),
    max_out_len=40,
    max_seq_len=2048,
    batch_size=8,
    model_kwargs=dict(device_map='auto', trust_remote_code=True),
    run_cfg=dict(num_gpus=1, num_procs=1),
)

nm_qwen_14b_full_cp18000 = dict(
        type=HuggingFaceCausalLM,
        abbr='nm_qwen_14b_full_cp12000',
        path=f"{ROOT_DIR}models/xll_models/Qwen-14B/full/checkpoint-18000/final_model",
        tokenizer_path=f"{ROOT_DIR}models/xll_models/Qwen-14B/full/checkpoint-18000/final_model",
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              trust_remote_code=True,
                              use_fast=False,),
        max_out_len=40,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )

nm_qwen_14b_full_cp12000 = dict(
        type=HuggingFaceCausalLM,
        abbr='nm_qwen_14b_full_cp12000',
        path=f"{ROOT_DIR}models/xll_models/Qwen-14B/full/checkpoint-12000/final_model",
        tokenizer_path=f"{ROOT_DIR}models/xll_models/Qwen-14B/full/checkpoint-12000/final_model",
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              trust_remote_code=True,
                              use_fast=False,),
        max_out_len=40,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )

nm_qwen_14b_full_cp6000 = dict(
        type=HuggingFaceCausalLM,
        abbr='nm_qwen_14b_full_cp6000',
        path=f"{ROOT_DIR}models/xll_models/Qwen-14B/full/checkpoint-6000/final_model",
        tokenizer_path=f"{ROOT_DIR}models/xll_models/Qwen-14B/full/checkpoint-6000/final_model",
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              trust_remote_code=True,
                              use_fast=False,),
        max_out_len=40,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )

nm_qwen_14b_full_final = dict(
        type=HuggingFaceCausalLM,
        abbr='nm_qwen_14b_full_final',
        path=f"{ROOT_DIR}models/xll_models/Qwen-14B/full/final/final_model",
        tokenizer_path=f"{ROOT_DIR}models/xll_models/Qwen-14B/full/final/final_model",
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              trust_remote_code=True,
                              use_fast=False,),
        max_out_len=40,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )

nm_qwen_14b_full = dict(
        type=HuggingFaceCausalLM,
        abbr='nm_qwen_14b_full',
        path=f"{ROOT_DIR}models/xll_models/Qwen-14B/full",
        tokenizer_path=f"{ROOT_DIR}models/xll_models/Qwen-14B/full",
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              trust_remote_code=True,
                              use_fast=False,),
        max_out_len=400,
        max_seq_len=2048,
        batch_size=1,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )

nm_qwen_14b_lora = dict(
        type=HuggingFaceCausalLM,
        abbr='nm_qwen_14b_lora',
        path=f"{ROOT_DIR}models/xll_models/Qwen-14B/lora",
        tokenizer_path=f"{ROOT_DIR}models/xll_models/Qwen-14B/lora",
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              trust_remote_code=True,
                              use_fast=False,),
        max_out_len=400,
        max_seq_len=2048,
        batch_size=1,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )

nm_qwen_1_8b_books_all = dict(
        type=HuggingFaceCausalLM,
        abbr='nm_qwen_1_8b_books_all',
        path=f"{ROOT_DIR}models/xll_models/Qwen-1_8B/books_all",
        tokenizer_path=f"{ROOT_DIR}models/xll_models/Qwen-1_8B/books_all",
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              trust_remote_code=True,
                              use_fast=False,),
        max_out_len=400,
        max_seq_len=2048,
        batch_size=1,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )

nm_qwen_1_8b_books_all_freeze = dict(
        type=HuggingFaceCausalLM,
        abbr='nm_qwen_1_8b_books_all_freeze',
        path=f"{ROOT_DIR}models/xll_models/Qwen-1_8B/books_all-freeze",
        tokenizer_path=f"{ROOT_DIR}models/xll_models/Qwen-1_8B/books_all-freeze",
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              trust_remote_code=True,
                              use_fast=False,),
        max_out_len=400,
        max_seq_len=2048,
        batch_size=1,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )

nm_qwen_1_8b_books_all_lora = dict(
        type=HuggingFaceCausalLM,
        abbr='nm_qwen_1_8b_books_all_lora',
        path=f"{ROOT_DIR}models/xll_models/Qwen-1_8B/books_all-lora",
        tokenizer_path=f"{ROOT_DIR}models/xll_models/Qwen-1_8B/books_all-lora",
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              trust_remote_code=True,
                              use_fast=False,),
        max_out_len=400,
        max_seq_len=2048,
        batch_size=1,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )

nm_qwen_1_8b_mix5_1_sft_owl = dict(
        type=HuggingFaceCausalLM,
        abbr='nm_qwen_1_8b_mix5_1_sft_owl',
        path=f"{ROOT_DIR}models/xll_models/Qwen-1_8B/mixture/mix5_1-sft_owl-model",
        tokenizer_path=f"{ROOT_DIR}models/xll_models/Qwen-1_8B/mixture/mix5_1-sft_owl-model",
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              trust_remote_code=True,
                              use_fast=False,),
        max_out_len=400,
        max_seq_len=2048,
        batch_size=1,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )

nm_qwen_1_8b_mix5_1 = dict(
        type=HuggingFaceCausalLM,
        abbr='nm_qwen_1_8b_mix5_1',
        path=f"{ROOT_DIR}models/xll_models/Qwen-1_8B/mixture/mix5_1",
        tokenizer_path=f"{ROOT_DIR}models/xll_models/Qwen-1_8B/mixture/mix5_1",
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              trust_remote_code=True,
                              use_fast=False,),
        max_out_len=400,
        max_seq_len=2048,
        batch_size=1,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )

nm_qwen_1_8b_schedule= dict(
        type=HuggingFaceCausalLM,
        abbr='nm_qwen_1_8b_schedule',
        path=f"{ROOT_DIR}models/xll_models/Qwen-1_8B/lab_scheduler/Qwen-1_8B-schedule",
        tokenizer_path=f"{ROOT_DIR}models/xll_models/Qwen-1_8B/lab_scheduler/Qwen-1_8B-schedule",
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              trust_remote_code=True,
                              use_fast=False,),
        max_out_len=400,
        max_seq_len=2048,
        batch_size=1,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )

nm_qwen_1_8b_multiple_test= dict(
        type=HuggingFaceCausalLM,
        abbr='nm_qwen_1_8b_multiple_test',
        path=f"{ROOT_DIR}models/xll_models/Qwen-1_8B/lab_scheduler/Qwen-1_8B-schedule",
        tokenizer_path=f"{ROOT_DIR}models/xll_models/Qwen-1_8B/lab_scheduler/Qwen-1_8B-schedule",
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              trust_remote_code=True,
                              use_fast=False,),
        max_out_len=400,
        max_seq_len=2048,
        batch_size=1,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )

nm_qwen_14b_bookall_lora= dict(
        type=HuggingFaceCausalLM,
        abbr='nm_qwen_14b_bookall_lora',
        # /mnt/tenant-home_speed/lyh/OpenCompass-OpsQA-lyh/models/xll_models/Qwen-1_8B/lora/Qwen-14B-bookall-lora
        path="/mnt/tenant-home_speed/lyh/OpenCompass-OpsQA-lyh/models/xll_models/Qwen-1_8B/qwen14b-bookall-lora",
        tokenizer_path="/mnt/tenant-home_speed/lyh/OpenCompass-OpsQA-lyh/models/xll_models/Qwen-1_8B/qwen14b-bookall-lora",
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              trust_remote_code=True,
                              use_fast=False,),
        max_out_len=400,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
# /mnt/tenant-home_speed/lyh/OpenCompass-OpsQA-lyh/models/xll_models/Qwen-1_8B/14b/3gpp
nm_qwen_14b_3gpp_mix= dict(
        type=HuggingFaceCausalLM,
        abbr='nm_qwen_14b_3gpp_mix',
        # /mnt/tenant-home_speed/lyh/OpenCompass-OpsQA-lyh/models/xll_models/Qwen-1_8B/lora/Qwen-14B-bookall-lora
        path="/mnt/tenant-home_speed/lyh/OpenCompass-OpsQA-lyh/models/xll_models/Qwen-1_8B/14b/3gpp",
        tokenizer_path="/mnt/tenant-home_speed/lyh/OpenCompass-OpsQA-lyh/models/xll_models/Qwen-1_8B/14b/3gpp",
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              trust_remote_code=True,
                              use_fast=False,),
        max_out_len=400,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )

nm_qwen_14b_3gpp_mix_sft_owlb= dict(
        type=HuggingFaceCausalLM,
        abbr='nm_qwen_14b_3gpp_mix_sft_owlb',
        # /mnt/tenant-home_speed/lyh/OpenCompass-OpsQA-lyh/models/xll_models/Qwen-1_8B/lora/Qwen-14B-bookall-lora
        path="/mnt/tenant-home_speed/lyh/OpenCompass-OpsQA-lyh/models/xll_models/Qwen-1_8B/14b/qwen14b_pt_3gpp_sft_owlb",
        tokenizer_path="/mnt/tenant-home_speed/lyh/OpenCompass-OpsQA-lyh/models/xll_models/Qwen-1_8B/14b/qwen14b_pt_3gpp_sft_owlb",
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              trust_remote_code=True,
                              use_fast=False,),
        max_out_len=400,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
# /mnt/tenant-home_speed/lyh/OpenCompass-OpsQA-lyh/models/xll_models/Qwen-1_8B/14b/pt-lora/Qwen-14B-bookall-lora-16x2/checkpoint-5000
nm_qwen_14b_ptlora_book_16x2= dict(
        type=HuggingFaceCausalLM,
        abbr='nm_qwen_14b_ptlora_book_16x2',
        # /mnt/tenant-home_speed/lyh/OpenCompass-OpsQA-lyh/models/xll_models/Qwen-1_8B/lora/Qwen-14B-bookall-lora
        path="/mnt/tenant-home_speed/lyh/OpenCompass-OpsQA-lyh/models/xll_models/Qwen-1_8B/14b/pt-lora/Qwen-14B-bookall-lora-16x2/ck5000",
        tokenizer_path="/mnt/tenant-home_speed/lyh/OpenCompass-OpsQA-lyh/models/xll_models/Qwen-1_8B/14b/pt-lora/Qwen-14B-bookall-lora-16x2/ck5000",
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              trust_remote_code=True,
                              use_fast=False,),
        max_out_len=400,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
# /mnt/home/LLaMA-Factory-zzr/pt_checkpoint/remote-Qwen-1_8B/lora/Qwen-14B-bookall-lora/ck10000
nm_qwen_14b_ptlora_book_8x2= dict(
        type=HuggingFaceCausalLM,
        abbr='nm_qwen_14b_ptlora_book_8x2',
        # /mnt/tenant-home_speed/lyh/OpenCompass-OpsQA-lyh/models/xll_models/Qwen-1_8B/lora/Qwen-14B-bookall-lora
        path="/mnt/home/LLaMA-Factory-zzr/pt_checkpoint/remote-Qwen-1_8B/lora/Qwen-14B-bookall-lora/ck10000",
        tokenizer_path="/mnt/home/LLaMA-Factory-zzr/pt_checkpoint/remote-Qwen-1_8B/lora/Qwen-14B-bookall-lora/ck10000",
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              trust_remote_code=True,
                              use_fast=False,),
        max_out_len=400,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )

nm_qwen_14b_chat = dict(
        type=HuggingFaceCausalLM,
        abbr='nm_qwen_14b_chat',
        # /mnt/tenant-home_speed/lyh/OpenCompass-OpsQA-lyh/models/xll_models/Qwen-1_8B/lora/Qwen-14B-bookall-lora
        path="/mnt/tenant-home_speed/gaozhengwei/projects/LLM/models/Qwen/Qwen-14B-Chat",
        tokenizer_path="/mnt/tenant-home_speed/gaozhengwei/projects/LLM/models/Qwen/Qwen-14B-Chat",
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              trust_remote_code=True,
                              use_fast=False,),
        max_out_len=400,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
