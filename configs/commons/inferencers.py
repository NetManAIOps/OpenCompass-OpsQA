from opencompass.openicl.icl_inferencer import PPLInferencer, SCInferencer, CoTInferencer, GenInferencer, PPLQAInferencer


def get_ppl_inferencer(save_every=20):
    ppl_inferencer = dict(
                    type=PPLInferencer,
                    save_every=save_every,
                    infer_type='PPL',
                    # **fixidlist
                )
    return ppl_inferencer

def get_ppl_qa_inferencer():
    ppl_qa_inferencer = dict(
                type=PPLQAInferencer,
                save_every=20,
            )
    return ppl_qa_inferencer

def get_gen_inferencer(save_every=20, 
                       max_out_len=400, 
                       sc_size=1, 
                       generation_kwargs=dict(temperature=0.7),
                       sc=True,
                       ):
    if sc:
        inferencer = dict(
            type=SCInferencer,
            save_every=save_every,
            generation_kwargs=generation_kwargs,
            infer_type='SC',
            sc_size=sc_size,  
            max_out_len=max_out_len,
        )
    else:
        inferencer = dict(
            type=GenInferencer,
            save_every=save_every,
            generation_kwargs=generation_kwargs,
            infer_type='Gen',
            max_out_len=max_out_len,
        )
    return inferencer

def get_cot_inferencer(save_every=20, 
                       max_out_len=400, 
                       sc_size=1, 
                       generation_kwargs=dict(temperature=0.7), 
                       cot_prompts=None):
    inferencer = dict(
                    type=CoTInferencer,
                    save_every=save_every,
                    cot_prompts=cot_prompts,
                    generation_kwargs=generation_kwargs,
                    infer_type='SC',
                    sc_size=sc_size, 
                    max_out_len=max_out_len,
                )
    return inferencer