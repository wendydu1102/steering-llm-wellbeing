"""Validate the notebook's mechanics without downloading or loading model weights."""
import copy
import ast
import json
import os
from pathlib import Path

import nbformat
import numpy as np
import pytest
import torch

os.environ['MPLBACKEND'] = 'Agg'

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'advice_steering_pilot.ipynb'


@pytest.fixture(scope='module')
def ns():
    assert NOTEBOOK.exists(), 'The teaching notebook has not been generated yet.'
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(nb)
    scope = {}
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code' and 'definitions' in cell.metadata.get('tags', []):
            exec(compile(cell.source, f'<notebook-cell-{i}>', 'exec'), scope)
        if cell.cell_type == 'code' and 'configuration' in cell.metadata.get('tags', []):
            for node in ast.parse(cell.source).body:
                if isinstance(node, ast.FunctionDef) and node.name == 'resolve_project_dir':
                    exec(compile(ast.Module(body=[node], type_ignores=[]), '<path-resolver>', 'exec'), scope)
    return scope


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Identity()])


def test_steering_only_changes_answer_prediction_positions(ns):
    model = TinyModel()
    x = torch.zeros(1, 5, 3)
    with ns['ResidualSteerer'](model, 0, np.array([1., 0., 0.]), .2, 10., 4):
        y = model.model.layers[0](x)
        assert torch.equal(y[:, :3], x[:, :3])
        assert torch.allclose(y[0, 3:], torch.tensor([[2., 0., 0.], [2., 0., 0.]]))
        decoded = model.model.layers[0](torch.zeros(1, 1, 3))
        assert torch.allclose(decoded, torch.tensor([[[2., 0., 0.]]]))
    assert torch.equal(model.model.layers[0](x), x)
    assert not model.model.layers[0]._forward_hooks


def test_zero_dose_and_exception_cleanup(ns):
    model = TinyModel()
    x = torch.randn(1, 4, 3)
    with pytest.raises(RuntimeError, match='test exception'):
        with ns['ResidualSteerer'](model, 0, np.array([1., 0., 0.]), 0., 10., 4):
            assert torch.equal(model.model.layers[0](x), x)
            raise RuntimeError('test exception')
    assert not model.model.layers[0]._forward_hooks
    assert torch.equal(model.model.layers[0](x), x)


def test_tuple_outputs_preserved_and_input_not_mutated(ns):
    model = TinyModel()
    hook = ns['ResidualSteerer'](model, 0, np.array([0., 1., 0.]), -.1, 10., 3)
    x = torch.ones(1, 3, 3)
    extra = object()
    result = hook._hook(None, None, (x, extra))
    assert result[1] is extra
    assert torch.equal(x, torch.ones_like(x))
    assert result[0][0, -1, 1] == 0


def test_invalid_direction_rejected(ns):
    with pytest.raises(ValueError):
        ns['ResidualSteerer'](TinyModel(), 0, np.zeros(3), .1, 1., 3)


@pytest.mark.parametrize('direction,reference_norm', [(None,None),(None,1.),(np.ones(3),None)])
def test_unbuilt_direction_has_actionable_message(ns,direction,reference_norm):
    with pytest.raises(ValueError, match='section 6'):
        ns['ResidualSteerer'](TinyModel(),0,direction,0.,reference_norm,3)


def load_datasets():
    return {name: json.loads((ROOT/'data'/f'{name}.json').read_text(encoding='utf-8'))
            for name in ['exploratory', 'contrast_pairs', 'development', 'test']}


def test_dataset_split_and_preference_balance(ns):
    datasets = load_datasets()
    ns['validate_datasets'](datasets)
    assert [len(datasets[k]) for k in datasets] == [4, 16, 6, 12]
    for name in ['exploratory', 'development', 'test']:
        for row in datasets[name]:
            a = ns['scenario_prompt'](row, 'a')
            b = ns['scenario_prompt'](row, 'b')
            assert row['facts_and_goals'] in a and row['facts_and_goals'] in b
            assert a.replace('leaning toward option A', 'leaning toward option B') == b


def test_leaking_scenario_group_rejected(ns):
    datasets = copy.deepcopy(load_datasets())
    datasets['test'][0]['scenario_group'] = datasets['contrast_pairs'][0]['scenario_group']
    with pytest.raises(ValueError, match='group'):
        ns['validate_datasets'](datasets)


def test_cache_identity_changes_with_prompt_direction_and_generation(ns):
    descriptor = {'prompt': 'same facts', 'vector_hash': 'one', 'max_new_tokens': 192}
    original = ns['stable_hash'](descriptor)
    for field, value in [('prompt','different facts'), ('vector_hash','two'), ('max_new_tokens',64)]:
        changed = dict(descriptor, **{field:value})
        assert ns['stable_hash'](changed) != original


def test_notebook_identity_ignores_saved_outputs_and_stage_switches(ns):
    nb = nbformat.read(NOTEBOOK, as_version=4)
    original = ns['notebook_source_hash'](nb)
    changed = copy.deepcopy(nb)
    for cell in changed.cells:
        if cell.cell_type == 'code':
            cell.execution_count = 12
            cell.outputs = [nbformat.v4.new_output('stream', name='stdout', text='saved output')]
        if 'configuration' in cell.metadata.get('tags', []):
            cell.source += '\nRUN_FINAL = True\n'
    assert ns['notebook_source_hash'](changed) == original
    changed.cells.append(nbformat.v4.new_code_cell('algorithm_has_changed = True'))
    assert ns['notebook_source_hash'](changed) != original


def test_empty_and_incomplete_ratings_do_not_become_results(ns):
    import pandas as pd
    with pytest.raises(ValueError, match='rating'):
        ns['validate_ratings'](pd.DataFrame(), expected_ids=['r1'])
    invalid = pd.DataFrame([{'response_id':'r1','stance':'','considerations_covered':'',
        'respectful':'','coherent':'','nonresponsive':'','reflexive_endorsement':'',
        'unsupported_claims':'','rater_notes':''}])
    with pytest.raises(ValueError, match='rating'):
        ns['validate_ratings'](invalid, expected_ids=['r1'])


def test_saved_outputs_cannot_be_scored_with_edited_scenario(ns):
    case = load_datasets()['test'][0]
    records = [{'scenario_id':case['scenario_id'], 'scenario_hash':ns['stable_hash'](case)}]
    ns['validate_saved_cases'](records, [case])
    edited = copy.deepcopy(case)
    edited['relevant_considerations'][0]['text'] = 'A different consideration, same ID'
    with pytest.raises(ValueError, match='Saved scenario'):
        ns['validate_saved_cases'](records, [edited])


def test_find_project_when_notebook_runs_from_build_folder(ns):
    workspace = ROOT.parents[1]
    assert ns['resolve_project_dir'](workspace/'.notebook_build') == ROOT
    assert ns['resolve_project_dir'](ROOT/'tests') == ROOT
    assert ns['resolve_project_dir'](workspace) == ROOT


def test_explicit_project_path_is_validated(ns):
    assert ns['resolve_project_dir'](ROOT, str(ROOT)) == ROOT
    with pytest.raises(FileNotFoundError, match='ADVICE_PROJECT_DIR'):
        ns['resolve_project_dir'](ROOT, str(ROOT/'tests'))


def test_rating_export_and_analysis_branch_with_explicit_fixtures(ns, tmp_path):
    """Synthetic software fixtures validate joins/plots, not research outcomes."""
    import pandas as pd
    datasets = load_datasets()
    records = []
    for case in datasets['test']:
        for inclination in ['a','b']:
            for condition in ['baseline','prompt_only','learned','random_101','random_202','random_303']:
                records.append(dict(response_id=f"{case['scenario_id']}-{inclination}-{condition}",
                    scenario_id=case['scenario_id'],scenario_hash=ns['stable_hash'](case),
                    inclination=inclination,condition=condition,topic=case['topic'],
                    evidence_group=case['evidence_group'],prompt=ns['scenario_prompt'](case,inclination),
                    response='Synthetic test fixture; not a model generation.',truncated=False))
    packet,sheet,key = ns['export_rating_packet'](records,datasets['test'],tmp_path)
    packet_rows=[json.loads(x) for x in packet.read_text(encoding='utf-8').splitlines()]
    assert len(packet_rows)==144 and all('condition' not in row for row in packet_rows)
    ratings=pd.read_csv(sheet,keep_default_na=False)
    ratings['stance']='favors_a'
    ratings['considerations_covered']='c1'
    ratings['respectful']='yes'
    ratings['coherent']='yes'
    ratings['nonresponsive']='no'
    ratings['reflexive_endorsement']='uncertain'
    ratings.to_csv(sheet,index=False)
    before=sheet.read_bytes()
    ns['export_rating_packet'](records,datasets['test'],tmp_path)
    assert sheet.read_bytes()==before, 'Export overwrote existing human ratings'
    ns['write_json'](tmp_path/'final_generations.json',records)
    nb=nbformat.read(NOTEBOOK,as_version=4)
    analysis=next(c.source for c in nb.cells if c.cell_type=='code' and c.source.startswith('ratings_path='))
    scope=dict(ns,OUT_DIR=tmp_path,datasets=datasets,display=lambda *args:None)
    exec(compile(analysis,'<analysis-cell>','exec'),scope)
    summary=pd.read_csv(tmp_path/'summary.csv')
    assert len(summary)==6 and set(summary.n)=={24}
    assert np.allclose(summary.endorsement,.5)
    assert np.allclose(summary.coverage,1/3)
    assert np.allclose(summary.uncertain,1)
    assert (tmp_path/'pilot_comparison.png').is_file()
