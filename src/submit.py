"""生成提交文件，并对官方格式做显式校验。

官方评测器未公开前，格式契约集中在 ``config.SUBMISSION_COLUMNS``：
缺列直接报错，多出的列保留在末尾并打印提醒，避免静默写出错格式。
"""
from . import config as C


def validate_submission_format(df):
    """校验并返回按官方列顺序整理后的 DataFrame。"""
    df = df.copy()
    required = C.SUBMISSION_COLUMNS
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"提交文件缺少官方要求列: {missing}；"
            f"期望列顺序为 {required}"
        )
    extra = [col for col in df.columns if col not in required]
    if extra:
        print(f"提交格式提醒：额外列 {extra} 会保留在末尾")
    ordered = required + extra
    return df[ordered]


def write_submission(df, path=None):
    df = validate_submission_format(df)
    if path is None:
        C.SUBMISSION_DIR.mkdir(exist_ok=True)
        path = C.SUBMISSION_DIR / "submission.csv"
    df.to_csv(path, index=False)
    return path
