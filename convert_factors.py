"""
因子格式双向转换脚本
=====================
回测框架 (py文件) ←→ 组合框架 (factor文件夹)

用法 (Linux/Mac):
  python3 convert_factors.py --to-folders factors.py     # py文件 → 因子文件夹
  python3 convert_factors.py --to-py ./factor             # 因子文件夹 → py文件
  python3 convert_factors.py                             # 交互模式

用法 (Windows):
  python convert_factors.py --to-folders factors.py
  python convert_factors.py --to-py ./factor
  python convert_factors.py                              # 交互模式 (推荐)
  也可直接双击运行, 跟随提示操作.

回测框架格式:
  factors.py 中每个因子是一个函数:
    def factor_YYYYMMDD_fanxinghang_Name(self, win1=X, win2=Y, win3=Z):
        ...
        return self._cs_zscore(raw).ewm(span=win3).mean()

组合框架格式:
  factor/<因子名>/factor.py  →  定义 def compute(gen, para_dic): ...
  factor/<因子名>/params.csv  →  参数默认值

变换规则:
  self.attr           →  gen.attr          (数据访问)
  self.method(args)   →  method(gen, args) (函数调用, gen前置)
  _helper(self, ...)  →  _helper(gen, ...)  (模块级helper, 同样gen前置)
"""

import ast, os, sys, re, copy, argparse
from pathlib import Path
import numpy as np, pandas as pd


# ============================================================
# 通用编码处理 — UTF-8 / GBK 自动适配
# ============================================================

# 写: 始终用 UTF-8
WRITE_ENCODING = 'utf-8'

# 读: 依次尝试编码列表
READ_ENCODINGS = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin-1']



def smart_read_text(filepath):
    """依次尝试多种编码读取文件全部文本."""
    for enc in READ_ENCODINGS:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        return f.read()


def smart_read_csv(filepath, **kwargs):
    """依次尝试多种编码读取 CSV."""
    for enc in READ_ENCODINGS:
        try:
            return pd.read_csv(filepath, encoding=enc, **kwargs)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return pd.read_csv(filepath, encoding='utf-8', errors='replace', **kwargs)


# ============================================================
# AST 变换核心
# ============================================================

# 已知的数据属性 (self.xxx 是 DataFrame, 不需要把调用拆出去)
DATA_ATTRS = {'close', 'open', 'high', 'low', 'volume', 'amount',
              'market_cap', 'total_turnover', 'turnoverrate'}


class SelfToGen(ast.NodeTransformer):
    """
    self → gen 变换:
      - 函数参数 self → gen
      - 所有 Name('self') → Name('gen') (包括嵌套闭包)
      - self.method(args)  由 MethodCallToFunction 进一步处理
    """

    def visit_arg(self, node):
        """函数参数名 self → gen."""
        if node.arg == 'self':
            node.arg = 'gen'
        return self.generic_visit(node)

    def visit_Name(self, node):
        if node.id == 'self':
            return ast.Name(id='gen', ctx=node.ctx)
        return node


class MethodCallToFunction(ast.NodeTransformer):
    """
    第二遍: self.method(args) → method(gen, args)
    条件: func是Attribute, 且func.value是Name('gen')
    排除: gen.DATA_ATTR.method() — pandas链式调用保持不变
    """

    def visit_Call(self, node):
        # 先递归处理子节点
        node = self.generic_visit(node)

        if isinstance(node.func, ast.Attribute):
            attr_node = node.func
            if isinstance(attr_node.value, ast.Name) and attr_node.value.id == 'gen':
                method = attr_node.attr
                # 如果 gen.xxx 是数据属性, 不动 (gen.close.rolling() — pandas调用)
                if method not in DATA_ATTRS:
                    # gen.method(args) → method(gen, args)
                    new_func = ast.Name(id=method, ctx=ast.Load())
                    new_args = [ast.Name(id='gen', ctx=ast.Load())] + node.args
                    return ast.Call(func=new_func, args=new_args, keywords=node.keywords)
        return node


class GenToSelf(ast.NodeTransformer):
    """
    逆变换: gen → self, method(gen, args) → self.method(args)
    """

    def visit_arg(self, node):
        """函数参数名 gen → self."""
        if node.arg == 'gen':
            node.arg = 'self'
        return self.generic_visit(node)

    def visit_Name(self, node):
        if node.id == 'gen':
            return ast.Name(id='self', ctx=node.ctx)
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)

        if isinstance(node.func, ast.Name):
            # method(self, args) → self.method(args)
            if node.args and isinstance(node.args[0], ast.Name) and node.args[0].id == 'self':
                method = node.func.id
                # 排除特定函数: getattr, isinstance, len, print 等内置
                if method not in ('getattr', 'isinstance', 'len', 'print', 'abs', 'max',
                                  'min', 'sum', 'round', 'int', 'float', 'str', 'bool',
                                  'np', 'pd', 'range', 'zip', 'enumerate', 'setattr'):
                    new_func = ast.Attribute(
                        value=ast.Name(id='self', ctx=ast.Load()),
                        attr=method,
                        ctx=ast.Load()
                    )
                    return ast.Call(func=new_func, args=node.args[1:], keywords=node.keywords)
        return node


# ============================================================
# 源文件解析
# ============================================================

def parse_factor_names(tree: ast.Module) -> set:
    """从 AST 中的 FACTOR_LIST 提取因子函数名集合."""
    names = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == 'FACTOR_LIST':
                    if isinstance(node.value, ast.List):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                # ['alpha_01', 'alpha_02']
                                names.add(elt.value)
                            elif isinstance(elt, ast.Tuple):
                                # [('label', alpha_01), ...]
                                for sub in elt.elts:
                                    if isinstance(sub, ast.Name):
                                        names.add(sub.id)
                            elif isinstance(elt, ast.Name):
                                # [alpha_01, alpha_02]
                                names.add(elt.id)
    return names


def parse_factor_file(filepath: str) -> dict:
    """
    解析因子py文件, 返回:
      factors: [(func_name, ast_node, params_dict), ...]
      helpers: [(func_name, ast_node), ...]
      imports: list[ast.stmt]
    """
    source = smart_read_text(filepath)
    tree = ast.parse(source)
    factor_names = parse_factor_names(tree)

    factors = []
    helpers = []
    imports = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
        elif isinstance(node, ast.FunctionDef):
            if node.name in factor_names:
                params = extract_params(node)
                factors.append((node.name, node, params))
            else:
                helpers.append((node.name, node))

    return {
        'factors': factors,
        'helpers': helpers,
        'imports': imports,
    }


def extract_params(func_node: ast.FunctionDef) -> dict:
    """从函数AST提取参数默认值."""
    params = {}
    args = func_node.args
    # 跳过 self/gen
    all_args = args.args[1:]  # 第一个是self
    defaults = args.defaults
    offset = len(all_args) - len(defaults)

    for i, arg in enumerate(all_args):
        name = arg.arg
        if i >= offset:
            default_node = defaults[i - offset]
            if isinstance(default_node, ast.Constant):
                params[name] = default_node.value
            elif isinstance(default_node, ast.UnaryOp) and isinstance(default_node.op, ast.USub):
                if isinstance(default_node.operand, ast.Constant):
                    params[name] = -default_node.operand.value
                else:
                    params[name] = None
            else:
                # 复杂表达式默认值 (如 Name, BinOp, Call), 不回测寻参
                params[name] = None
        else:
            params[name] = None
    return params


def _ast_has_name(node: ast.AST, name: str) -> bool:
    """检查 AST 节点树中是否包含对指定变量名的引用."""
    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.found = False
        def visit_Name(self, n):
            if n.id == name:
                self.found = True
            self.generic_visit(n)
    v = Visitor()
    v.visit(node)
    return v.found


# ============================================================
# Py → Folders 方向
# ============================================================

def transform_factor_for_gen(func_node: ast.FunctionDef, params: dict) -> str:
    """
    将一个factor函数AST变换为gen版.
    返回 gen 版源码字符串.
    """
    # 深拷贝AST
    new_node = copy.deepcopy(func_node)

    # 1. 第一遍: self → gen
    new_node = SelfToGen().visit(new_node)
    ast.fix_missing_locations(new_node)

    # 2. 第二遍: gen.method(args) → method(gen, args)
    new_node = MethodCallToFunction().visit(new_node)
    ast.fix_missing_locations(new_node)

    # 3. 修改函数签名: factor_...(self, win1=X, ...) → compute(gen, params)
    #    注意: 参数名 params 对应组合框架 onefactor 传入的 para_dic
    new_node.name = 'compute'
    new_node.args = ast.arguments(
        posonlyargs=[],
        args=[
            ast.arg(arg='gen', annotation=None, type_comment=None),
            ast.arg(arg='params', annotation=None, type_comment=None),
        ],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[],
    )

    # 4. 在AST body开头插入参数提取语句 (docstring之后)
    #    格式: win1 = int(params.get('win1', 20))
    param_stmts = []
    for k, v in params.items():
        # 根据默认值类型选择 cast 函数
        if isinstance(v, bool):
            cast = 'bool'
        elif isinstance(v, int):
            cast = 'int'
        elif isinstance(v, float):
            cast = 'float'
        else:
            cast = None  # 字符串或其他, 不cast

        if v is not None:
            # win1 = int(params.get('win1', 20))
            inner = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='params', ctx=ast.Load()),
                    attr='get',
                    ctx=ast.Load()
                ),
                args=[ast.Constant(value=k), ast.Constant(value=v)],
                keywords=[]
            )
        else:
            # win1 = int(params['win1'])
            inner = ast.Subscript(
                value=ast.Name(id='params', ctx=ast.Load()),
                slice=ast.Constant(value=k),
                ctx=ast.Load()
            )

        if cast:
            value_node = ast.Call(
                func=ast.Name(id=cast, ctx=ast.Load()),
                args=[inner],
                keywords=[]
            )
        else:
            value_node = inner

        param_stmts.append(
            ast.Assign(
                targets=[ast.Name(id=k, ctx=ast.Store())],
                value=value_node,
                type_comment=None
            )
        )

    # 插入位置: docstring之后 (如果有)
    insert_at = 1 if (
        new_node.body and
        isinstance(new_node.body[0], ast.Expr) and
        isinstance(new_node.body[0].value, ast.Constant) and
        isinstance(new_node.body[0].value.value, str)
    ) else 0

    new_body = list(new_node.body[:insert_at]) + param_stmts + list(new_node.body[insert_at:])
    new_node.body = new_body
    ast.fix_missing_locations(new_node)

    # 5. 展开 source
    new_source = ast.unparse(new_node)

    return new_source


def generate_params_csv(params: dict) -> str:
    """生成 params.csv 内容 (5列: param,min,max,step,default).

    min/max/step 从默认值反推, 避免无意义的超大搜索空间.
    """
    lines = ['param,min,max,step,default']
    for k, v in params.items():
        if v is None:
            # 无默认值 — 给宽范围
            lines.append(f'{k},1,500,1,1')
        elif isinstance(v, bool):
            lines.append(f'{k},0,1,1,{str(v).lower()}')
        elif isinstance(v, int):
            step = max(1, v // 5)
            min_v = max(1, v // 4)
            max_v = max(v * 4, v + 20)
            lines.append(f'{k},{min_v},{max_v},{step},{v}')
        elif isinstance(v, float):
            mag = abs(v)
            step = max(0.1, round(mag / 5, 2))
            min_v = max(0.01, round(mag / 4, 2))
            max_v = max(round(mag * 4, 1), mag + 5.0)
            lines.append(f'{k},{min_v},{max_v},{step},{v}')
        else:
            # 字符串等 — 给宽范围
            lines.append(f'{k},1,500,1,{v}')
    return '\n'.join(lines) + '\n'


def find_called_helpers(func_node: ast.FunctionDef, helper_names: set) -> set:
    """在 func_node 的 AST 中查找调用了哪些 helper 函数.

    匹配 self.helper(...) 或 helper(gen, ...) 两种形式.
    """
    used = set()

    class Finder(ast.NodeVisitor):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == 'self':
                    if node.func.attr in helper_names:
                        used.add(node.func.attr)
            elif isinstance(node.func, ast.Name):
                if node.func.id in helper_names:
                    used.add(node.func.id)
            self.generic_visit(node)

    Finder().visit(func_node)
    return used


def parse_external_helpers(filepath: str) -> list:
    """从外部 py 文件提取函数定义 (含类方法), 用于扩充 helper 池."""
    source = smart_read_text(filepath)
    tree = ast.parse(source)
    helpers = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    helpers.append((item.name, item))
        elif isinstance(node, ast.FunctionDef):
            helpers.append((node.name, node))
    return helpers


def py_to_folders(py_file: str, output_dir: str = './factor',
                  include_helpers: bool = True,
                  external_helpers: list = None):
    """
    将因子py文件转换为factor文件夹集合.
    """
    parsed = parse_factor_file(py_file)
    helpers = list(parsed['helpers'])

    # 合并外部 helper (如 FunctionPool 中的方法)
    # 同名时外部覆盖本地 — FunctionPool 是权威源, 参数更完整
    if external_helpers:
        existing = {hname for hname, _ in helpers}
        for ext_file in external_helpers:
            ext = parse_external_helpers(ext_file)
            n_added = 0
            n_overridden = 0
            for hname, hnode in ext:
                if hname not in existing:
                    helpers.append((hname, hnode))
                    existing.add(hname)
                    n_added += 1
                else:
                    for i, (n, _) in enumerate(helpers):
                        if n == hname:
                            helpers[i] = (hname, hnode)
                            n_overridden += 1
                            break
            print(f"  从 {ext_file} 加载 {n_added} 个新 helper, "
                  f"覆盖 {n_overridden} 个同名 (共 {len(ext)} 个函数)")

    helper_names = {hname for hname, _ in helpers}
    helper_asts = {hname: hnode for hname, hnode in helpers}

    # 构建 helper 间的依赖图 (传递依赖)
    dep_graph = {}
    for hname, hnode in helpers:
        dep_graph[hname] = find_called_helpers(hnode, helper_names)

    def transitive_closure(direct):
        result = set(direct)
        changed = True
        while changed:
            changed = False
            for h in list(result):
                for dep in dep_graph.get(h, set()):
                    if dep not in result:
                        result.add(dep)
                        changed = True
        return result

    os.makedirs(output_dir, exist_ok=True)
    created = []

    for fn_name, func_node, params in parsed['factors']:
        # 变换
        gen_source = transform_factor_for_gen(func_node, params)

        # 因子文件夹名 = 函数名 (保持一致)
        folder = os.path.join(output_dir, fn_name)
        os.makedirs(folder, exist_ok=True)

        # 分析该因子用到的辅助函数 (含传递依赖)
        if include_helpers and helpers:
            direct = find_called_helpers(func_node, helper_names)
            needed = transitive_closure(direct)
            # 按定义顺序输出
            needed_ordered = [h for h, _ in helpers if h in needed]
        else:
            needed_ordered = []

        # 构建 factor.py
        factor_py = 'import numpy as np\nimport pandas as pd\n\n'

        if needed_ordered:
            factor_py += '# ---- 辅助函数 ----\n'
            for hname in needed_ordered:
                new_node = copy.deepcopy(helper_asts[hname])
                new_node = SelfToGen().visit(new_node)
                ast.fix_missing_locations(new_node)
                new_node = MethodCallToFunction().visit(new_node)
                ast.fix_missing_locations(new_node)
                factor_py += ast.unparse(new_node) + '\n\n'
            factor_py += '# ---- 因子函数 ----\n'

        factor_py += gen_source + '\n'

        with open(os.path.join(folder, 'factor.py'), 'w', encoding=WRITE_ENCODING) as f:
            f.write(factor_py)

        # 生成 params.csv
        with open(os.path.join(folder, 'params.csv'), 'w', encoding=WRITE_ENCODING) as f:
            f.write(generate_params_csv(params))

        created.append(fn_name)
        helpers_info = f", helpers: {needed_ordered}" if needed_ordered else ""
        print(f"  ✓ {fn_name}/  (params: {list(params.keys())}{helpers_info})")

    print(f"\n创建了 {len(created)} 个因子文件夹 → {output_dir}/")
    return created


# ============================================================
# Folders → Py 方向
# ============================================================

def parse_factor_folder(folder: str) -> tuple:
    """
    解析单个因子文件夹, 返回 (factor_name, gen_ast, params_dict, helpers_source).
    """
    factor_py = os.path.join(folder, 'factor.py')
    params_csv = os.path.join(folder, 'params.csv')
    name = os.path.basename(folder)

    if not os.path.exists(factor_py):
        return None

    # 读取 params (5列格式: param,min,max,step,default)
    params = {}
    if os.path.exists(params_csv):
        df = smart_read_csv(params_csv)
        for _, row in df.iterrows():
            pname = str(row.iloc[0])
            v = row.iloc[4]  # default 列 (第5列, 0-indexed=4)
            if pd.isna(v):
                params[pname] = None
                continue
            # 类型推断: bool → int → float → str
            if isinstance(v, bool):
                params[pname] = v
            elif isinstance(v, (int, float)):
                params[pname] = int(v) if float(v) == int(v) else float(v)
            elif isinstance(v, str):
                vl = v.strip().lower()
                if vl in ('true', 'false'):
                    params[pname] = vl == 'true'
                else:
                    try:
                        params[pname] = int(v) if float(v) == int(float(v)) else float(v)
                    except (ValueError, TypeError):
                        params[pname] = v
            else:
                params[pname] = v

    # 读取 factor.py
    source = smart_read_text(factor_py)

    tree = ast.parse(source)

    # 找到 compute 函数
    compute_node = None
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'compute':
            compute_node = node
            break

    if compute_node is None:
        print(f"  [WARN] {name}: 未找到 compute 函数, 跳过")
        return None

    # 收集 helper 函数
    helpers = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and node.name != 'compute':
            helpers.append(node)

    return name, compute_node, params, helpers, source.split('\n')


def folders_to_py(factor_dir: str, output_file: str = 'factors_out.py',
                  date_str: str = None):
    """
    将factor文件夹集合转换回单个py文件.
    """
    if date_str is None:
        from datetime import date
        date_str = date.today().strftime('%Y%m%d')

    folders = sorted([
        d for d in os.listdir(factor_dir)
        if os.path.isdir(os.path.join(factor_dir, d))
        and os.path.exists(os.path.join(factor_dir, d, 'factor.py'))
    ])

    if not folders:
        print(f"未找到因子文件夹 → {factor_dir}/")
        return

    # 解析所有文件夹
    all_factors = []
    all_helpers = {}  # name → (source, count)

    for folder in folders:
        result = parse_factor_folder(os.path.join(factor_dir, folder))
        if result is None:
            continue
        name, compute_node, params, helpers, src_lines = result
        all_factors.append((name, compute_node, params))

        # 收集helper
        for h in helpers:
            h_src = ast.unparse(h)
            if h.name not in all_helpers:
                all_helpers[h.name] = h_src

    # 生成输出
    output = [
        '"""',
        f'Auto-generated from {len(all_factors)} factor folders.',
        f'Date: {date_str}',
        '"""',
        '',
        'import numpy as np',
        'import pandas as pd',
        '',
    ]

    # 写 helper 函数 (逆变换: gen→self, method(gen,args)→self.method(args))
    if all_helpers:
        output.append('# ============================================================')
        output.append('# 辅助函数')
        output.append('# ============================================================')
        output.append('')
        for h_name, h_src in sorted(all_helpers.items()):
            # 变换helper
            h_ast = ast.parse(h_src).body[0]
            h_ast = GenToSelf().visit(h_ast)
            ast.fix_missing_locations(h_ast)
            output.append(ast.unparse(h_ast))
            output.append('')
            output.append('')

    # 写因子函数
    output.append('# ============================================================')
    output.append(f'# 因子函数 ({len(all_factors)}个)')
    output.append('# ============================================================')
    output.append('')

    for folder_name, compute_node, params in all_factors:
        # 函数名 = 文件夹名 (保持一致)
        fn_name = folder_name

        # 逆变换AST
        new_node = copy.deepcopy(compute_node)
        new_node = GenToSelf().visit(new_node)
        ast.fix_missing_locations(new_node)

        # 恢复函数签名
        new_node.name = fn_name
        # 构建参数: (self, win1=X, win2=Y, ...)
        func_args = [ast.arg(arg='self', annotation=None, type_comment=None)]
        defaults = []
        for pname, pval in params.items():
            func_args.append(ast.arg(arg=pname, annotation=None, type_comment=None))
            if pval is not None:
                defaults.append(ast.Constant(value=pval))

        new_node.args = ast.arguments(
            posonlyargs=[],
            args=func_args,
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=defaults,
        )

        # 移除 para_dic 参数提取代码 (body 开头的 win1 = int(params.get(...)))
        new_body = []
        docstring_skipped = False
        for stmt in new_node.body:
            # 保留第一个 docstring
            if (isinstance(stmt, ast.Expr)
                    and isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, str)):
                if not docstring_skipped:
                    new_body.append(stmt)
                    docstring_skipped = True
                continue
            # 只跳过右值引用了 params 变量 且 左值是已知参数名的赋值
            if isinstance(stmt, ast.Assign):
                targets = stmt.targets
                if (len(targets) == 1
                        and isinstance(targets[0], ast.Name)
                        and targets[0].id in params
                        and _ast_has_name(stmt.value, 'params')):
                    continue
            new_body.append(stmt)

        new_node.body = new_body

        # 生成源码
        src = ast.unparse(new_node)
        output.append(src)
        output.append('')
        output.append('')

    # 生成 FACTOR_LIST (纯因子名)
    output.append('# ============================================================')
    output.append('# 因子注册')
    output.append('# ============================================================')
    output.append('')
    output.append('FACTOR_LIST = [')
    for folder_name, _, params in all_factors:
        output.append(f"    '{folder_name}',")
    output.append(']')
    output.append('')

    # 写入
    with open(output_file, 'w', encoding=WRITE_ENCODING) as f:
        f.write('\n'.join(output))

    print(f"✓ 生成 {output_file} ({len(all_factors)}个因子)")

    # 验证语法
    try:
        ast.parse('\n'.join(output))
        print(f"✓ 语法检查通过")
    except SyntaxError as e:
        print(f"[WARN] 语法错误: {e}")

    return output_file


# ============================================================
# 主入口
# ============================================================

def interactive():
    """交互模式 — 无命令行参数时自动进入."""
    print("=" * 60)
    print("  因子格式双向转换")
    print("=" * 60)
    print()
    print("选择方向:")
    print("  [1] Py文件 → 因子文件夹  (回测框架 → 组合框架)")
    print("  [2] 因子文件夹 → Py文件  (组合框架 → 回测框架)")
    print()

    choice = input("请输入 1 或 2: ").strip()
    if choice not in ('1', '2'):
        print("无效选择, 退出.")
        sys.exit(1)

    if choice == '1':
        default_py = 'factors.py' if os.path.exists('factors.py') else ''
        prompt = f"因子py文件路径 (默认 {default_py}): " if default_py else "因子py文件路径: "
        py_file = input(prompt).strip()
        if not py_file:
            py_file = default_py if default_py else 'factors.py'
        if not os.path.exists(py_file):
            print(f"文件不存在: {py_file}")
            sys.exit(1)
        output_dir = input("输出目录 (默认 ./factor, 直接回车): ").strip()
        if not output_dir:
            output_dir = './factor'
        no_helpers = input("是否跳过内嵌helper函数? (y/n, 默认n): ").strip().lower()
        inc_ext = input("是否引入外部 helper 文件? (Y/n, 默认Y): ").strip().lower()
        external = None
        if inc_ext != 'n':
            # 默认候选: FunctionPool.py
            default_ext = 'FunctionPool.py'
            if os.path.exists(default_ext):
                hint = f"外部helper文件路径, 空格分隔 (默认 {default_ext}, 直接回车): "
            else:
                hint = "外部helper文件路径, 空格分隔 (直接回车跳过): "
            ext_input = input(hint).strip()
            if not ext_input and os.path.exists(default_ext):
                ext_files = [default_ext]
            elif ext_input:
                ext_files = [x.strip() for x in ext_input.split() if x.strip()]
            else:
                ext_files = []
            # 验证文件存在
            valid = [f for f in ext_files if os.path.exists(f)]
            for f in ext_files:
                if f not in valid:
                    print(f"  [WARN] 文件不存在, 跳过: {f}")
            if valid:
                print(f"  将引入: {', '.join(valid)}")
                external = valid
            else:
                print(f"  未找到有效文件, 跳过外部 helper 引入")
        print()
        py_to_folders(py_file, output_dir, include_helpers=not no_helpers.startswith('y'),
                      external_helpers=external)

    elif choice == '2':
        default_dir = './factor' if os.path.isdir('./factor') else ''
        prompt = f"因子文件夹所在目录 (默认 ./factor, 直接回车): " if default_dir else "因子文件夹目录: "
        factor_dir = input(prompt).strip()
        if not factor_dir:
            factor_dir = default_dir if default_dir else './factor'
        if not os.path.isdir(factor_dir):
            print(f"目录不存在: {factor_dir}")
            sys.exit(1)
        output_file = input("输出py文件名 (默认 factors.py, 直接回车): ").strip()
        if not output_file:
            output_file = 'factors.py'
        date_str = input("日期 YYYYMMDD (默认今天, 直接回车): ").strip()
        if not date_str:
            from datetime import date
            date_str = date.today().strftime('%Y%m%d')
        print()
        folders_to_py(factor_dir, output_file, date_str)

    print("\n完成!")


def main():
    p = argparse.ArgumentParser(
        description='因子格式双向转换: py文件 ←→ factor文件夹\n'
                    '不带参数运行时自动检测环境或进入交互模式.'
    )
    p.add_argument('--to-folders', type=str, default=None, const='factors.py', nargs='?',
                   help='py文件路径 (默认 factors.py), 转换为factor文件夹')
    p.add_argument('--to-py', type=str, default=None, const='./factor', nargs='?',
                   help='factor目录路径 (默认 ./factor), 转换为py文件')
    p.add_argument('--output-dir', type=str, default='./factor',
                   help='文件夹输出目录 (默认 ./factor)')
    p.add_argument('--output-file', type=str, default='factors.py',
                   help='py输出文件 (默认 factors.py)')
    p.add_argument('--date', type=str, default=None,
                   help='日期 YYYYMMDD (用于生成因子名, 默认今天)')
    p.add_argument('--no-helpers', action='store_true',
                   help='Py→Folders时不内嵌helper函数')
    p.add_argument('--include-helpers-from', type=str, nargs='*', default=None,
                   help='额外的helper来源文件 (如 FunctionPool.py), '
                        '其中的函数会被纳入因子文件夹')
    args = p.parse_args()

    if args.to_folders is not None:
        py_file = args.to_folders if args.to_folders else 'factors.py'
        if not os.path.exists(py_file):
            print(f"文件不存在: {py_file}")
            sys.exit(1)
        if args.include_helpers_from:
            print(f"Py → Folders: {py_file} → {args.output_dir}/ "
                  f"(+ helpers from: {args.include_helpers_from})\n")
        else:
            print(f"Py → Folders: {py_file} → {args.output_dir}/\n")
        py_to_folders(py_file, args.output_dir,
                      include_helpers=not args.no_helpers,
                      external_helpers=args.include_helpers_from)

    elif args.to_py is not None:
        factor_dir = args.to_py if args.to_py else './factor'
        if not os.path.isdir(factor_dir):
            print(f"目录不存在: {factor_dir}")
            sys.exit(1)
        print(f"Folders → Py: {factor_dir} → {args.output_file}\n")
        folders_to_py(factor_dir, args.output_file, args.date)

    else:
        # 无参数: 自动检测环境
        has_py = os.path.exists('factors.py')
        has_dir = os.path.isdir('./factor')
        if has_py and not has_dir:
            fp = 'FunctionPool.py'
            ext = [fp] if os.path.exists(fp) else None
            info = f" (+ {fp})" if ext else ""
            print(f"检测到 factors.py → 自动执行: Py → Folders{info}\n")
            py_to_folders('factors.py', './factor', include_helpers=True,
                          external_helpers=ext)
        elif has_dir and not has_py:
            print(f"检测到 ./factor/ → 自动执行: Folders → Py\n")
            folders_to_py('./factor', 'factors.py', None)
        else:
            interactive()


if __name__ == '__main__':
    main()
