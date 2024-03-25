prompts = [
    # Zero-shot
    [
        # Single
        ['Here is a single-answer question. You should provide the correct answer option directly.', '以下是单项选择题。请直接给出正确答案的选项。\n'],
        # Multiple
        ['Here is a multiple-answer question. You should select all appropriate option letters separated by commas to answer this question. Example of a possible answer: B,C.\n', 
         '以下是多项选择题。请直接给出所有正确答案的选项并用英文逗号分隔，例如：“B,C”。\n']
    ],
    # 3-shot
    [
        # Single
        ['', ''],
        ['', '']
    ]
]
