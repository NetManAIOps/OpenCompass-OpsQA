from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever, ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.utils.text_postprocessors import first_capital_postprocess_multi
from opencompass.datasets import OReillyChoiceDataset, OReillyEvaluator, oreilly_choice_postprocess
import pandas as pd

# book_df = pd.read_csv('/mnt/mfs/opsgpt/evaluation/ops-cert-eval/books.csv')

book_list = [
 {'name': 'CompTIA A+ Complete Practice Tests, 3rd Edition',
  'id': 9781119862642,
  'abbr': 'CTCPT-3E',
  'filename': '9781119862642.json'},
 {'name': 'CompTIA A+ Complete Study Guide, 5th Edition',
  'id': 9781119862918,
  'abbr': 'CTCSG-5E',
  'filename': '9781119862918.json'},
 {'name': 'CompTIA A+ Practice Questions Exam Cram Core 1 (220-1101) and Core 2 (220-1102)',
  'id': 9780137658237,
  'abbr': 'CTCPQEC1&2',
  'filename': '9780137658237.json'},
 {'name': 'CompTIA Network+ Practice Tests, 2nd Edition',
  'id': 9781119807308,
  'abbr': 'CTNPT-2E',
  'filename': '9781119807308.json'},
 {'name': 'CompTIA Network+ Study Guide, 5th Edition',
  'id': 9781119811633,
  'abbr': 'CTNSG-5E',
  'filename': '9781119811633.json'},
 {'name': 'CompTIA Linux+ Practice Tests, 3rd Edition',
  'id': 9781119879619,
  'abbr': 'CTLPT-3E',
  'filename': '9781119879619.json'},
 {'name': 'CompTIA Linux+ Study Guide, 5th Edition',
  'id': 9781119878940,
  'abbr': 'CTLSG-5E',
  'filename': '9781119878940.json'},
 {'name': 'CompTIA Security+ Practice Tests',
  'id': 9781119416920,
  'abbr': 'CTSPPT',
  'filename': '9781119416920.json'},
 {'name': 'CompTIA Security+ Study Guide, 7th Edition',
  'id': 9781119416876,
  'abbr': 'CTSSG-7E',
  'filename': '9781119416876.json'},
 {'name': 'CompTIA CySA+ Practice Tests, 3rd Edition',
  'id': 9781394182930,
  'abbr': 'CTCPT-3E',
  'filename': '9781394182930.json'},
 {'name': 'CompTIA CySA+ Study Guide, 3rd Edition',
  'id': 9781394182909,
  'abbr': 'CTCSG-3E',
  'filename': '9781394182909.json'},
 {'name': 'CompTIA Project+ Practice Tests, 2nd Edition',
  'id': 9781119892489,
  'abbr': 'CTPPT-2E',
  'filename': '9781119892489.json'},
 {'name': 'CompTIA Project+ Study Guide, 3rd Edition',
  'id': 9781119892458,
  'abbr': 'CTPSG-3E',
  'filename': '9781119892458.json'},
 {'name': 'CompTIA IT Fundamentals (ITF+) Study Guide, 2nd Edition',
  'id': 9781119513124,
  'abbr': 'CTFITF+SG-2E',
  'filename': '9781119513124.json'},
 {'name': 'CompTIA Server+ Study Guide, 2nd Edition',
  'id': 9781119891437,
  'abbr': 'CTSSG-2E',
  'filename': '9781119891437.json'},
 {'name': 'CompTIA Data+ Study Guide',
  'id': 9781119845256,
  'abbr': 'CTDSG',
  'filename': '9781119845256.json'},
 {'name': 'CompTIA Cloud Essentials+ Study Guide, 2nd Edition',
  'id': 9781119642220,
  'abbr': 'CTCE+SG-2E',
  'filename': '9781119642220.json'},
 {'name': 'CompTIA Cloud+ Study Guide, 3rd Edition',
  'id': 9781119810865,
  'abbr': 'CTCSG-3E',
  'filename': '9781119810865.json'},
 {'name': 'CompTIA PenTest+ Study Guide, 2nd Edition',
  'id': 9781119823810,
  'abbr': 'CTPSG-2E',
  'filename': '9781119823810.json'},
 {'name': 'CASP+ CompTIA Advanced Security Practitioner Study Guide, 4th Edition',
  'id': 9781119803164,
  'abbr': 'CACASPG-4E',
  'filename': '9781119803164.json'},
 {'name': '(ISC)2 CCSP Certified Cloud Security Professional Official Practice Tests, 3rd Edition',
  'id': 9781119909408,
  'abbr': 'IC2CCSPOT-3E',
  'filename': '9781119909408.json'},
 {'name': '(ISC)2 CCSP Certified Cloud Security Professional Official Study Guide, 3rd Edition',
  'id': 9781119909378,
  'abbr': 'IC2CCSPSG-3E',
  'filename': '9781119909378.json'},
 {'name': '(ISC)2 CISSP Certified Information Systems Security Professional Official Practice Tests, 3rd Edition',
  'id': 9781119787631,
  'abbr': 'IC2CISSPOT-3E',
  'filename': '9781119787631.json'},
 {'name': '(ISC)2 CISSP Certified Information Systems Security Professional Official Study Guide, 9th Edition',
  'id': 9781119786238,
  'abbr': 'IC2CISSPSG-9E',
  'filename': '9781119786238.json'},
 {'name': '(ISC)2 SSCP Systems Security Certified Practitioner Official Practice Tests, 2nd Edition',
  'id': 9781119852070,
  'abbr': 'IC2SSCPOT-2E',
  'filename': '9781119852070.json'},
 {'name': '(ISC)2 SSCP Systems Security Certified Practitioner Official Study Guide, 3rd Edition',
  'id': 9781119854982,
  'abbr': 'IC2SSCPSG-3E',
  'filename': '9781119854982.json'},
 {'name': 'CCNA 200-301 Official Cert Guide Library',
  'id': 9780136755562,
  'abbr': 'CCNAOCGL',
  'filename': '9780136755562.json'},
 {'name': 'CCNA Wireless 200-355 Official Cert Guide',
  'id': 9780134307183,
  'abbr': 'CCNAWOCG',
  'filename': '9780134307183.json'},
 {'name': 'CCNA Cyber Ops SECOPS 210-255 Official Cert Guide',
  'id': 9780134608938,
  'abbr': 'CCNASOOCG',
  'filename': '9780134608938.json'},
 {'name': 'CCNA Cyber Ops SECFND #210-250 Official Cert Guide',
  'id': 9780134609003,
  'abbr': 'CCNASEOCG',
  'filename': '9780134609003.json'},
 {'name': 'CCNP and CCIE Data Center Core DCCOR 350-601 Official Cert Guide',
  'id': 9780136555735,
  'abbr': 'CCNPCCDCCD',
  'filename': '9780136555735.json'},
 {'name': 'CCNP and CCIE Enterprise Core ENCOR 350-401 Official Cert Guide, 2nd Edition',
  'id': 9780138216993,
  'abbr': 'CCNPECECORE',
  'filename': '9780138216993.json'},
 {'name': 'CCNP Collaboration Call Control and Mobility CLACCM 300-815 Official Cert Guide',
  'id': 9780136575474,
  'abbr': 'CCNPCCCACCM',
  'filename': '9780136575474.json'},
 {'name': 'CCNP Collaboration Cloud and Edge Solutions CLCEI 300-820 Official Cert Guide',
  'id': 9780136733867,
  'abbr': 'CCNPCCCES',
  'filename': '9780136733867.json'},
 {'name': 'CCNP Data Center Application Centric Infrastructure 300-620 DCACI Official Cert Guide',
  'id': 9780136602804,
  'abbr': 'CCNPCCACI',
  'filename': '9780136602804.json'},
 {'name': 'CCNP Enterprise Advanced Routing ENARSI 300-410 Official Cert Guide, 2nd Edition',
  'id': 9780138217464,
  'abbr': 'CCNPEAR',
  'filename': '9780138217464.json'},
 {'name': 'CCNP Enterprise Design ENSLD 300-420 Official Cert Guide: Designing Cisco Enterprise Networks',
  'id': 9780136575160,
  'abbr': 'CCNPEDES',
  'filename': '9780136575160.json'},
 {'name': 'CCNP Enterprise Wireless Design ENWLSD 300-425 and Implementation ENWLSI 300-430 Official Cert Guide: Designing & Implementing Cisco Enterprise Wireless Networks',
  'id': 9780136600992,
  'abbr': 'CCNPEWD',
  'filename': '9780136600992.json'},
 {'name': 'CCNP Security Cisco Secure Firewall and Intrusion Prevention System Official Cert Guide',
  'id': 9780136589716,
  'abbr': 'CCNPSF&IPSOCG',
  'filename': '9780136589716.json'},
 {'name': 'CCNP Security Identity Management SISE 300-715 Official Cert Guide',
  'id': 9780136677710,
  'abbr': 'CCNPSIM',
  'filename': '9780136677710.json'},
 {'name': 'CCNP Security Virtual Private Networks SVPN 300-730 Official Cert Guide',
  'id': 9780136634829,
  'abbr': 'CCNPSVPN',
  'filename': '9780136634829.json'},
 {'name': 'Cisco Certified Design Expert (CCDE 400-007) Official Cert Guide',
  'id': 9780137601066,
  'abbr': 'CCDEOCG',
  'filename': '9780137601066.json'},
 {'name': 'Cisco Certified DevNet Associate DEVASC 200-901 Official Cert Guide',
  'id': 9780136677314,
  'abbr': 'CCNDADEVASC',
  'filename': '9780136677314.json'},
 {'name': 'Cisco Certified DevNet Professional DEVCOR 350-901 Official Cert Guide',
  'id': 9780137370337,
  'abbr': 'CCNDPDEVCOR',
  'filename': '9780137370337.json'},
 {'name': 'Cisco CyberOps Associate CBROPS 200-201 Official Cert Guide',
  'id': 9780136807964,
  'abbr': 'CCAACBROPS',
  'filename': '9780136807964.json'},
 {'name': 'CCDA 200-310 Official Cert Guide, Fifth Edition',
  'id': 9780134305653,
  'abbr': 'CCDAP',
  'filename': '9780134305653.json'},
 {'name': 'The KCNA Book',
  'id': 9781835080399,
  'abbr': 'TKCB',
  'filename': '9781835080399.json'},
 {'name': 'CWNA Certified Wireless Network Administrator Study Guide, 6th Edition',
  'id': 9781119734505,
  'abbr': 'CWNASG-6E',
  'filename': '9781119734505.json'},
 {'name': 'Docker Certified Associate (DCA): Exam Guide',
  'id': 9781839211898,
  'abbr': 'DCAEG',
  'filename': '9781839211898.json'},
 {'name': 'CEH v12 Certified Ethical Hacker Study Guide with 750 Practice Test Questions',
  'id': 9781394186921,
  'abbr': 'CEHV12CEHSG',
  'filename': '9781394186921.json'}
  ]

_hint = """Please answer with option uppercase letters or their combinations if it's a multiple choice problem. Some examples for your answers: 'A', 'B,C', 'A,C,E'.
    """
    
oreilly_datasets = []

for _row in book_list:
    _name = _row['name']
    _id = _row['id']
    _book_abbr = _row['abbr']
    _filename = _row['filename']


    oreilly_reader_cfg = dict(
        input_columns=['topic','question','choices','qtype'],
        output_column='answer')

    oreilly_infer_cfg = dict(
            ice_template=dict(
                type=PromptTemplate,
                template=dict(
                    round=[
                        dict(
                            role="HUMAN",
                            prompt=
                            f"Here's a {{qtype}} question about {{topic}}: {{question}}\n{{choices}}\n{_hint}\nAnswer: "
                        ),
                        dict(role="BOT", prompt="{answer}")
                    ]
                ),
                ice_token="</E>",
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer),
        )

    oreilly_eval_cfg = dict(
        evaluator=dict(type=AccEvaluator),
        pred_postprocessor=dict(type='oreilly-choice'))

    oreilly_datasets.append(
        dict(
            abbr=f"{_id}",
            type=OReillyChoiceDataset,
            path='/mnt/mfs/opsgpt/evaluation/ops-cert-eval/v2/splitted',
            filename=_filename,
            sample_setting=dict(
                seed=0,
                # load_list=None,
                sample_size=10,
                # sample_frac=None
            ),
            reader_cfg=oreilly_reader_cfg,
            infer_cfg=oreilly_infer_cfg,
            eval_cfg=oreilly_eval_cfg
        )
    )

del _hint


