from opencompass.openicl.icl_inferencer import PPLInferencer, SCInferencer, CoTInferencer, GenInferencer


def get_ppl_inferencer(save_every=20, fixidlist=dict(fix_id_list=None)):
    ppl_inferencer = dict(
                    type=PPLInferencer,
                    save_every=save_every,
                    infer_type='PPL',
                    **fixidlist
                )
    return ppl_inferencer

def get_gen_inferencer(save_every=20, 
                       max_out_len=400, 
                       sc_size=1, 
                       fixidlist=dict(fix_id_list=None), 
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
            **fixidlist
        )
    else:
        inferencer = dict(
            type=GenInferencer,
            save_every=save_every,
            generation_kwargs=generation_kwargs,
            infer_type='Gen',
            max_out_len=max_out_len,
            **fixidlist
        )
    return inferencer

def get_cot_inferencer(save_every=20, max_out_len=400, sc_size=1, fixidlist=dict(fix_id_list=None), generation_kwargs=dict(temperature=0.7), cot_prompts=None):
    inferencer = dict(
                    type=CoTInferencer,
                    save_every=save_every,
                    cot_prompts=cot_prompts,
                    generation_kwargs=generation_kwargs,
                    infer_type='SC',
                    sc_size=sc_size, 
                    max_out_len=max_out_len,
                    **fixidlist
                )
    return inferencer