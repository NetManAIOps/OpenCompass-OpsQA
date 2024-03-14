from opencompass.openicl.icl_prompt_template import PromptTemplate


def mc_abcd_ppl_ice_template(prompt_hint, answer_hint):
    return dict(
        type=PromptTemplate,
        template={
            'A' : f'{prompt_hint}{{question}}\nA: {{A}}\nB: {{B}}\nC: {{C}}\nD: {{D}}\n{answer_hint} A',
            'B' : f'{prompt_hint}{{question}}\nA: {{A}}\nB: {{B}}\nC: {{C}}\nD: {{D}}\n{answer_hint} B',
            'C' : f'{prompt_hint}{{question}}\nA: {{A}}\nB: {{B}}\nC: {{C}}\nD: {{D}}\n{answer_hint} C',
            'D' : f'{prompt_hint}{{question}}\nA: {{A}}\nB: {{B}}\nC: {{C}}\nD: {{D}}\n{answer_hint} D'
        }
    )

def mc_abcd_ppl_prompt_template(prompt_hint, answer_hint):
    return dict(
        type=PromptTemplate,
        template={
            'A' : f'</E>{prompt_hint}{{question}}\nA: {{A}}\nB: {{B}}\nC: {{C}}\nD: {{D}}\n{answer_hint} A',
            'B' : f'</E>{prompt_hint}{{question}}\nA: {{A}}\nB: {{B}}\nC: {{C}}\nD: {{D}}\n{answer_hint} B',
            'C' : f'</E>{prompt_hint}{{question}}\nA: {{A}}\nB: {{B}}\nC: {{C}}\nD: {{D}}\n{answer_hint} C',
            'D' : f'</E>{prompt_hint}{{question}}\nA: {{A}}\nB: {{B}}\nC: {{C}}\nD: {{D}}\n{answer_hint} D'
        },
        ice_token="</E>",
    )

def mc_abcd_gen_ice_template(prompt_hint, answer_hint):
    return dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role="HUMAN",
                    prompt=f'{prompt_hint}{{question}}\nA: {{A}}\nB: {{B}}\nC: {{C}}\nD: {{D}}\n{answer_hint}'
                ),
                dict(role="BOT", prompt="{answer}")
            ]
        )
    )

def mc_abcd_gen_prompt_template(prompt_hint, answer_hint):
    return dict(
        type=PromptTemplate,
        template=dict(
            begin="</E>",
            round=[
                dict(
                    role="HUMAN",
                    prompt=f'{prompt_hint}{{question}}\nA: {{A}}\nB: {{B}}\nC: {{C}}\nD: {{D}}\n{answer_hint}'
                ),
                # dict(role="BOT", prompt="{answer}")
            ],
        ),
        ice_token="</E>",
    )

def mc_abcd_cot_ice_template(prompt_hint, cot_think_hint, cot_conclude_hint):
    return dict(
        type=PromptTemplate,
        template=dict( 
            round=[
                dict(
                    role="HUMAN",
                    prompt=f'{prompt_hint}{{question}}\nA: {{A}}\nB: {{B}}\nC: {{C}}\nD: {{D}}\n{cot_think_hint}'
                ),
                dict(role="BOT", prompt=f"{{solution}}"),
                dict(
                    role="HUMAN",
                    prompt=f'{cot_conclude_hint}'
                ),
                dict(
                    role="BOT", prompt=f"{{answer}}",
                )
            ]
        ),
    )

def mc_abcd_cot_prompt_template(prompt_hint, cot_think_hint):
    return dict(
        type=PromptTemplate,
        template=dict(
            begin="</E>", 
            round=[
                dict(
                    role="HUMAN",
                    prompt=f'{prompt_hint}{{question}}\nA: {{A}}\nB: {{B}}\nC: {{C}}\nD: {{D}}\n{cot_think_hint}'
                ),
                # dict(role="BOT", prompt="{answer}")
            ]
        ),
        ice_token="</E>",
    )


def qa_gen_ice_template(prompt_hint, answer_hint):
    return dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role="HUMAN",
                    prompt=f'{prompt_hint}{{question}}\n{answer_hint}'
                ),
                dict(role="BOT", prompt="{answer}")
            ]
        )
    )

def qa_gen_prompt_template(prompt_hint, answer_hint):
    return dict(
        type=PromptTemplate,
        template=dict(
            begin="</E>",
            round=[
                dict(
                    role="HUMAN",
                    prompt=f'{prompt_hint}{{question}}\n{answer_hint}'
                ),
                # dict(role="BOT", prompt="{answer}")
            ],
        ),
        ice_token="</E>",
    )

def qa_ppl_ice_template(prompt_hint, answer_hint):
    return dict(
        type=PromptTemplate,
        template=dict( 
            round=[
                dict(
                    role="HUMAN",
                    prompt=f'{prompt_hint}{{question}}'
                ),
                dict(role="BOT", prompt=f"{answer_hint}{{answer}}"),
            ]
        ),
    )

def qa_ppl_prompt_template(prompt_hint, answer_hint):
    return dict(
        type=PromptTemplate,
        template=dict(
            begin="</E>", 
            round=[
                dict(
                    role="HUMAN",
                    prompt=f'{prompt_hint}{{question}}'
                ),
                dict(role="BOT", prompt=f"{answer_hint}{{answer}}"),
            ]
        ),
        ice_token="</E>",
    )