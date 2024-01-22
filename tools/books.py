import pandas as pd


def replace_bookname():
    """"""
    books_df = pd.read_csv(
        '/mnt/mfs/opsgpt/evaluation/ops-cert-eval/books.csv')
    isbn_to_title = dict(
        zip(books_df['id'].tolist(), books_df['name'].tolist()))
    summary_df = pd.read_csv(
        '/mnt/mfs/opsgpt/opencompass/outputs/gpt/20230815_095657/summary/summary_20230815_095657.csv'  # noqa
    )
    summary_df['dataset'] = summary_df['dataset'].replace(isbn_to_title)
    summary_df.to_csv('/mnt/mfs/opsgpt/opencompass/outputs/gpt_summary.csv')
