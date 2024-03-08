from mmengine import read_base
with read_base():
    from .mc_gen import get_mc_gen_datasets
    from .mc_ppl import get_mc_ppl_datasets
    from .qa_gen import get_qa_gen_datasets
    from .qa_ppl import get_qa_ppl_datasets
    from ...paths import ROOT_DIR

def get_all_datasets(name, path, langs, qtypes):
    datasets_generators = [get_qa_gen_datasets, get_qa_ppl_datasets, get_mc_gen_datasets, get_mc_ppl_datasets]
    four_kinds = [func(name, path, langs, qtypes) for func in datasets_generators]
    all_datasets = []
    for d in four_kinds:
        all_datasets.extend(d)
    return all_datasets

def get_gen_datasets(name, path, langs, qtypes):
    datasets_generators = [get_qa_gen_datasets, get_mc_gen_datasets]
    four_kinds = [func(name, path, langs, qtypes) for func in datasets_generators]
    all_datasets = []
    for d in four_kinds:
        all_datasets.extend(d)
    return all_datasets

def get_ppl_datasets(name, path, langs, qtypes):
    datasets_generators = [get_qa_ppl_datasets, get_mc_ppl_datasets]
    four_kinds = [func(name, path, langs, qtypes) for func in datasets_generators]
    all_datasets = []
    for d in four_kinds:
        all_datasets.extend(d)
    return all_datasets

def get_selected_datasets(setting_func, name, path, langs, qtypes):
    datasets_generators = setting_func
    four_kinds = [func(name, path, langs, qtypes) for func in datasets_generators]
    all_datasets = []
    for d in four_kinds:
        all_datasets.extend(d)
    return all_datasets

zjyd = get_all_datasets('zjyd', f'{ROOT_DIR}data/opseval/zjyd/', langs=['zh'], qtypes=['single'])
zjyd_gen = get_gen_datasets('zjyd', f'{ROOT_DIR}data/opseval/zjyd/', langs=['zh'], qtypes=['single'])
zjyd_ppl = get_ppl_datasets('zjyd', f'{ROOT_DIR}data/opseval/zjyd/', langs=['zh'], qtypes=['single'])

zte_mc_ppl = get_selected_datasets([get_mc_ppl_datasets], 'zte', f'{ROOT_DIR}data/opseval/zte/splitted', langs=['zh', 'en'], qtypes=['single'])
zte_mc_gen = get_selected_datasets([get_mc_gen_datasets], 'zte', f'{ROOT_DIR}data/opseval/zte/splitted', langs=['zh', 'en'], qtypes=['single', 'multiple'])
zte_mc = zte_mc_ppl + zte_mc_gen

oracle_mc_ppl = get_selected_datasets([get_mc_ppl_datasets], 'oracle', f'{ROOT_DIR}data/opseval/oracle/splitted', langs=['zh', 'en'], qtypes=['single'])
oracle_mc_gen = get_selected_datasets([get_mc_gen_datasets], 'oracle', f'{ROOT_DIR}data/opseval/oracle/splitted', langs=['zh', 'en'], qtypes=['single', 'multiple'])
oracle_mc = oracle_mc_ppl + oracle_mc_gen

owl_mc = get_selected_datasets([get_mc_ppl_datasets, get_mc_gen_datasets], 'owl', f'{ROOT_DIR}data/opseval/owl', langs=['zh', 'en'], qtypes=['single'])
owl_qa = get_selected_datasets([get_qa_ppl_datasets, get_qa_gen_datasets], 'owl', f'{ROOT_DIR}data/opseval/owl', langs=['zh', 'en'], qtypes=None)

network_mc_ppl = get_selected_datasets([get_mc_ppl_datasets], 'network', f'{ROOT_DIR}data/opseval/network/processed', langs=['zh', 'en'], qtypes=['single'])
network_mc_gen = get_selected_datasets([get_mc_gen_datasets], 'network', f'{ROOT_DIR}data/opseval/network/processed', langs=['zh', 'en'], qtypes=['single', 'multiple'])
network_mc = network_mc_ppl + network_mc_gen

company_mc_list = [
    ('bosc', f'{ROOT_DIR}data/opseval/bosc/splitted', ['zh'], ['single']),
    # ('dfcdata', f'{ROOT_DIR}data/opseval/dfcdata/splitted', ['zh'], ['single']),
    ('gtja',  f'{ROOT_DIR}data/opseval/gtja/splitted', ['zh'], ['single']),
    ('huaweicloud', f'{ROOT_DIR}data/opseval/huaweicloud/splitted', ['zh'], ['single', 'multiple']),
    ('lenovo', f'{ROOT_DIR}data/opseval/lenovo/splitted', ['zh'], ['single', 'multiple']),
    ('pufa', f'{ROOT_DIR}data/opseval/pufa/splitted', ['zh'], ['single']),
    ('rzy', f'{ROOT_DIR}data/opseval/rzy/splitted', ['zh'], ['single']),
    ('zabbix', f'{ROOT_DIR}data/opseval/zabbix/splitted', ['zh'], ['single']),
    ('zjyd', f'{ROOT_DIR}data/opseval/zjyd/', ['zh'], ['single']),
]

company_mc = sum([
    get_selected_datasets([get_mc_ppl_datasets, get_mc_gen_datasets], name, path, langs, qtypes) for name, path, langs, qtypes in company_mc_list
], [])


rzy_qa = get_selected_datasets([get_qa_ppl_datasets, get_qa_gen_datasets], 'rzy', f'{ROOT_DIR}data/opseval/rzy/splitted', langs=['zh'], qtypes=None)