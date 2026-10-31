"""Build a hashed route manifest; never installs dependencies or executes models."""

# =============================================================================
# 提交包打包工具
#   用法: python make_submission.py --source . --out my_submission.zip \
#           --route ml [--environment-id cpu-ml-methods-v3] [--seed N] [--include 额外文件...]
#   流程: 解析参数 -> 收集并校验文件 -> 校验 submission.py 接口 ->
#         校验 explain.json -> 生成 SHA-256 清单 -> 写 ZIP 并检查大小
# =============================================================================
import argparse, ast, json, hashlib, zipfile
from pathlib import Path, PurePosixPath


# -----------------------------------------------------------------
# 1. 命令行参数
# -----------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='.')                 # 待打包目录
    p.add_argument('--out', default='my_submission.zip')    # 输出ZIP路径
    p.add_argument('--route', choices=['ml'], default='ml')  # 提交路线
    p.add_argument('--environment-id', default='cpu-ml-methods-v3')  # 评测环境ID,本赛事CPU环境为 cpu-ml-methods-v3
    p.add_argument('--seed', type=int, default=20261008)    # 写入清单的随机种子
    p.add_argument('--include', nargs='*', default=[])      # 额外打包的文件
    return p, p.parse_args()


# -----------------------------------------------------------------
# 2. 收集文件并做安全/类型校验,返回 {包内路径: 字节}
# -----------------------------------------------------------------
def collect_files(p, a, root):
    names = ['README.md', 'explain.json', 'submission.py'] + a.include
    if len(set(names)) != len(names):
        p.error('duplicate files')
    files = {}
    allowed = {'.py', '.json', '.txt', '.md', '.npz', '.npy', '.pkl', '.joblib'}  # ML路线允许的扩展名
    for name in names:
        rel = PurePosixPath(name)
        f = root / name
        # 拒绝:绝对路径、含 '..'、反斜杠、符号链接、非文件、解析后逃逸出待打包目录
        if rel.is_absolute() or '..' in rel.parts or '\\' in name or f.is_symlink() or not f.is_file() or not f.resolve().is_relative_to(root):
            p.error('unsafe/non-file path')
        if a.route == 'ml' and f.suffix not in allowed:
            p.error('unsupported ML extension')
        files[name] = f.read_bytes()
    return files


# -----------------------------------------------------------------
# 3. 校验 submission.py 必须导出 build_model(只做静态AST检查,不执行代码)
# -----------------------------------------------------------------
def check_submission_contract(p, a, files):
    if a.route == 'ml':
        tree = ast.parse(files['submission.py'].decode())
        if not any(isinstance(n, ast.FunctionDef) and n.name == 'build_model' for n in tree.body):
            p.error('submission.py must export build_model')


# -----------------------------------------------------------------
# 4. 校验 explain.json:method 为字符串,assumptions/limitations 为字符串列表
# -----------------------------------------------------------------
def check_explanation(p, files):
    explanation = json.loads(files['explain.json'])
    if not isinstance(explanation.get('method'), str) or any(
        not isinstance(explanation.get(k), list) or any(not isinstance(v, str) for v in explanation[k])
        for k in ('assumptions', 'limitations')
    ):
        p.error('invalid explain.json')


# -----------------------------------------------------------------
# 5. 生成带每个文件 SHA-256 哈希的清单,并加入待打包文件
# -----------------------------------------------------------------
def add_manifest(a, files):
    manifest = {
        'route': a.route,
        'contract_version': 'last-request-snapshot-1',
        'environment_id': a.environment_id,
        'seed': a.seed,
        'files': {k: hashlib.sha256(v).hexdigest() for k, v in files.items()},
    }
    files['manifest.json'] = json.dumps(manifest, ensure_ascii=False, indent=2).encode()


# -----------------------------------------------------------------
# 6. 写出 ZIP:先查解压后总大小,再查 ZIP 大小,超限即报错
# -----------------------------------------------------------------
def write_zip(p, a, files):
    if a.route == 'ml' and sum(map(len, files.values())) > 100 * 1024**2:
        p.error('expanded ML size exceeds limit')
    out = Path(a.out)
    if out.exists():
        p.error('output exists; use a new version filename')
    with zipfile.ZipFile(out, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    if out.stat().st_size > 20 * 1024**2:
        out.unlink()
        p.error('ZIP exceeds intake limit')
    print('Created', out, 'route', a.route, 'with verified SHA-256 manifest. No grading performed.')


def main():
    p, a = parse_args()
    root = Path(a.source).resolve()
    files = collect_files(p, a, root)       # 步骤1-2
    check_submission_contract(p, a, files)  # 步骤3
    check_explanation(p, files)             # 步骤4
    add_manifest(a, files)                  # 步骤5
    write_zip(p, a, files)                  # 步骤6


if __name__ == '__main__':
    main()
