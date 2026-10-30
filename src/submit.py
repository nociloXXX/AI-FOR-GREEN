"""生成提交文件。开题后对照官方格式改列名与粒度。"""
from . import config as C


def write_submission(df, path=None):
    df = df.copy()
    if path is None:
        C.SUBMISSION_DIR.mkdir(exist_ok=True)
        path = C.SUBMISSION_DIR / "submission.csv"
    df.to_csv(path, index=False)
    return path
