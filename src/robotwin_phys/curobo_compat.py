"""Guarded result ownership/schema repairs for pinned CuRobo v0.7.8."""
import ast
import copy
import hashlib
import importlib
import inspect
import json
import subprocess
import textwrap
from pathlib import Path

from .conditions import ROOT

PIN = 'd64c4b005459db10c5dd867d8b30a87d5bda9bdb'
SOURCE_SHA256 = '348ee2fa858b1d651a70fa2b2b5a5d4e240ea724b7817d7769473c3f7a7e46e8'
TRAJOPT_SHA256 = '56c755a521b06e0df5d23df52e21ca73d7be2076b993af9ab6f3104c9d2bc35b'
FIX = 'consistent-batch-results-v2'
METHOD_HASHES = {
    '_plan_from_solve_state': 'f6dd2027b9c917f08c77f75f0858500b2bad53eb69abc7943255cd2764cc2f5e',
    '_plan_from_solve_state_batch': '68e958e0fcd5a59fe11c7233df0627df4cb8955b82d53a69be1864e1db42da19',
}
RESULT_METHOD_HASHES = {
    'clone': '773fb89495af3f30dd0e240d86da887e858dd2eca53ca9a8a5f317490e1ddcac',
    'copy_idx': '9eb1c75f3ea6a40d417629dea64f51a138a6676009ff90fc60730b4efb6350f2',
}
ATTEMPT_METHOD_HASHES = {
    '_plan_batch_attempts': 'e8191faf5318793cd42956e17a17324aeed5b115cad1e089b205778604d23b91',
}
TRAJOPT_METHOD_HASHES = {'_get_result': '439b966f878e51915a24efab35287e80394503f259f2789daa8f4fa680a297ac'}


def method_checksum(node):
    canonical = copy.deepcopy(node)
    if (canonical.body and isinstance(canonical.body[0], ast.Expr) and
            isinstance(canonical.body[0].value, ast.Constant) and
            isinstance(canonical.body[0].value.value, str)):
        canonical.body = canonical.body[1:]

    def value(item):
        if isinstance(item, ast.AST):
            # Python 3.12 added type_params; the simulator uses Python 3.10.
            return {'type': type(item).__name__, 'fields': {
                key: value(child) for key, child in ast.iter_fields(item) if key != 'type_params'}}
        if isinstance(item, list):
            return [value(child) for child in item]
        if item is Ellipsis:
            return {'constant': 'ellipsis'}
        return item

    return hashlib.sha256(json.dumps(value(canonical), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def compatibility_metadata():
    repo = ROOT / 'vendor/curobo'
    revision = subprocess.check_output(['git', '-c', f'safe.directory={repo}', '-C', str(repo),
                                        'rev-parse', 'HEAD'], text=True).strip()
    path = repo / 'src/curobo/wrap/reacher/motion_gen.py'
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    trajopt_checksum = hashlib.sha256((path.parent / 'trajopt.py').read_bytes()).hexdigest()
    if revision != PIN or checksum != SOURCE_SHA256 or trajopt_checksum != TRAJOPT_SHA256:
        raise RuntimeError('CuRobo revision/source differs from the reviewed v0.7.8; do not apply this repair')
    return {'revision': revision, 'motion_gen_sha256': checksum,
            'trajopt_sha256': trajopt_checksum, 'compatibility_fix': FIX}


def repaired_function(function, name):
    function = inspect.unwrap(function)
    source, first_line = inspect.getsourcelines(function)
    node = ast.parse(textwrap.dedent(''.join(source))).body[0]
    checksum = method_checksum(node)
    reviewed = {**METHOD_HASHES, **RESULT_METHOD_HASHES, **ATTEMPT_METHOD_HASHES, **TRAJOPT_METHOD_HASHES}
    if checksum != reviewed[name]:
        raise RuntimeError(f'Unreviewed CuRobo method: {name}')
    def expression(source):
        return ast.parse(source, mode='eval').body

    def calls(class_name):
        return [n for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == class_name]

    if name in METHOD_HASHES:
        constructors = calls('MotionGenResult')
        if len(constructors) != 1:
            raise RuntimeError('Expected one CuRobo result constructor')
        # Graph/trajopt failures must not mutate the IK success mask.
        constructors[0].args[0] = ast.Call(
            func=ast.Attribute(value=constructors[0].args[0], attr='clone', ctx=ast.Load()),
            args=[], keywords=[])
    elif name == '_get_result':
        # The all-seed branch must return the same terminal goal index as best-select.
        # Keep the raw per-timestep metrics intact for debugging/optimization.
        constructors = calls('TrajOptResult')
        candidates = [kw for call in constructors for kw in call.keywords
                      if kw.arg == 'goalset_index' and isinstance(kw.value, ast.Attribute)]
        if len(candidates) != 1:
            raise RuntimeError('Expected one raw trajectory goal-index assignment')
        candidates[0].value = expression(
            'result.metrics.goalset_index[..., -1] if result.metrics.goalset_index is not None else None')
    elif name == 'clone':
        constructor, = calls('MotionGenResult')
        for keyword in constructor.keywords:
            if keyword.arg == 'path_buffer_last_tstep':
                keyword.value = expression(
                    'list(self.path_buffer_last_tstep) if self.path_buffer_last_tstep is not None else None')
        constructor.keywords.append(ast.keyword(arg='finetune_time', value=expression('self.finetune_time')))
    elif name == 'copy_idx':
        # The selected trajectory and its timestep must come from the same attempt.
        node.body.insert(2, ast.parse(
            'self.optimized_dt = self._check_none_and_copy_idx('
            'self.optimized_dt, source_result.optimized_dt, idx)').body[0])
    elif name == '_plan_batch_attempts':
        # Retime the accumulated result that is actually returned, not the last attempt.
        retime_blocks = [statement for statement in node.body if isinstance(statement, ast.If)
                         and any(isinstance(n, ast.Attribute) and n.attr == 'retime_trajectory'
                                 for n in ast.walk(statement))]
        if len(retime_blocks) != 1:
            raise RuntimeError('Expected one batch result retiming block')
        for item in ast.walk(retime_blocks[0]):
            if isinstance(item, ast.Name) and item.id == 'result':
                item.id = 'best_result'
        # This branch reports an invalid pose-cost query for every batch row.
        for item in ast.walk(node):
            if isinstance(item, ast.comprehension) and isinstance(item.iter, ast.Attribute):
                if ast.unparse(item.iter) == 'solve_state.batch_size':
                    item.iter = expression('range(solve_state.batch_size)')
            if isinstance(item, ast.Attribute) and ast.unparse(item) == 'self.motion_gen.tensor_args.device':
                item.value = expression('self.tensor_args')
    # Preserve the existing profiler wrapper instead of creating another one.
    node.decorator_list = []
    ast.increment_lineno(node, first_line - 1)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node],
                        type_ignores=[])
    scope = dict(function.__globals__)
    exec(compile(ast.fix_missing_locations(module), function.__code__.co_filename, 'exec'), scope)
    replacement = scope[name]
    if replacement.__code__.co_freevars != function.__code__.co_freevars:
        raise RuntimeError('CuRobo repair would change method closure')
    return replacement


def install():
    compatibility_metadata()
    pending = []
    targets = (
        ('motion_gen', 'MotionGen', {**METHOD_HASHES, **ATTEMPT_METHOD_HASHES}),
        ('motion_gen', 'MotionGenResult', RESULT_METHOD_HASHES),
        ('trajopt', 'TrajOptSolver', TRAJOPT_METHOD_HASHES),
    )
    for module_name, class_name, methods in targets:
        module = importlib.import_module('curobo.wrap.reacher.' + module_name)
        expected = (ROOT / f'vendor/curobo/src/curobo/wrap/reacher/{module_name}.py').resolve()
        if Path(module.__file__).resolve() != expected:
            raise RuntimeError('Imported CuRobo source is outside the reviewed repository')
        for name in methods:
            function = inspect.unwrap(getattr(getattr(module, class_name), name))
            if getattr(function, '_robotwin_mask_fix', None) == FIX:
                continue
            pending.append((function, repaired_function(function, name)))
    for function, replacement in pending:
        # Keep the original descriptor, defaults, globals and API identity.
        function.__code__ = replacement.__code__
        function._robotwin_mask_fix = FIX
