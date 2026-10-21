"""Observed 2027 CSV contract plus a conservative local validation profile.

No metric implementation: syntactic validity is not a competition score.
IDs are opaque strings; the actual template supplies their set and row order.
"""
import csv
import hashlib
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path

MAX_TEMPLATE_BYTES = 4 * 1024 * 1024
MAX_ROWS = 10000


class ContractError(ValueError):
    pass


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContractError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def _constant(value):
    raise ContractError('Non-finite JSON constant: ' + value)


def strict_json(text):
    try:
        return json.loads(text, object_pairs_hook=_object, parse_constant=_constant)
    except (json.JSONDecodeError, TypeError, RecursionError) as exc:
        raise ContractError('Invalid JSON') from exc


def finite_number(value, name):
    if type(value) not in (int, float):
        raise ContractError(name + ' must be a finite number, not bool/string')
    try:
        valid = math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ContractError(name + ' must be finite')
    return value


def _keys(value, expected, name):
    # Exact keys are our conservative output profile, not a claim about
    # undocumented server tolerance of extra JSON fields.
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ContractError(name + ' has unexpected/missing fields')


def _localization_issues(action, *, prediction):
    issues = []
    expected = {'start', 'end', 'spatial', 'score'} if prediction else {'start', 'end', 'spatial', 'label'}
    _keys(action, expected, 'action')
    if prediction:
        finite_number(action['score'], 'score')
        if not 0 <= action['score'] <= 1:
            issues.append('score')
    elif not isinstance(action['label'], str) or not action['label']:
        raise ContractError('Ground-truth label must remain explicit')
    for key in ('start', 'end'):
        finite_number(action[key], key)
        if not 0 <= action[key] <= 1:
            issues.append(key)
    if action['start'] >= action['end']:
        issues.append('start_not_less_than_end')
    _keys(action['spatial'], {'action_cam'}, 'spatial')
    cameras = action['spatial']['action_cam']
    _keys(cameras, {'rank0', 'rank1'}, 'action_cam')
    for rank in ('rank0', 'rank1'):
        _keys(cameras[rank], {'x', 'y'}, rank)
        for axis in ('x', 'y'):
            field = rank + '.' + axis
            finite_number(cameras[rank][axis], field)
            if not 0 <= cameras[rank][axis] <= 1:
                issues.append(field)
    return issues


def validate_localization(action, *, prediction):
    issues = _localization_issues(action, prediction=prediction)
    if issues:
        raise ContractError('Invalid localization values: ' + ', '.join(issues))


def _prediction_issues(prediction):
    _keys(prediction, {'actions'}, 'prediction')
    if not isinstance(prediction['actions'], list):
        raise ContractError('actions must be a list')
    return [(index, field) for index, action in enumerate(prediction['actions'])
            for field in _localization_issues(action, prediction=True)]


def validate_prediction(prediction):
    issues = _prediction_issues(prediction)
    if issues:
        raise ContractError('Invalid prediction values: ' + str(issues[0]))
    return len(prediction['actions'])


@dataclass(frozen=True)
class TemplateContract:
    ids: tuple
    sha256: str
    byte_count: int
    total_actions: int
    actions_histogram: tuple
    placeholder_violations: tuple

    def summary(self):
        counts = {}
        for _, _, field in self.placeholder_violations:
            counts[field] = counts.get(field, 0) + 1
        return dict(columns=['id', 'prediction'], rows=len(self.ids),
                    unique_ids=len(set(self.ids)), sha256=self.sha256,
                    bytes=self.byte_count, total_actions=self.total_actions,
                    actions_histogram=dict(self.actions_histogram),
                    id_first=list(self.ids[:5]), id_last=list(self.ids[-5:]),
                    placeholder_value_validation='FAILED' if counts else 'PASS',
                    placeholder_violations_by_field=counts,
                    placeholder_affected_rows=len({v[0] for v in self.placeholder_violations}),
                    placeholder_affected_actions=len({v[:2] for v in self.placeholder_violations}))


def inspect_csv(payload, *, audit_template_placeholders=False):
    """Strict by default. Audit mode records sample value failures, never repairs them."""
    if not isinstance(payload, bytes) or len(payload) > MAX_TEMPLATE_BYTES:
        raise ContractError('CSV exceeds bounded byte input')
    try:
        reader = csv.reader(io.StringIO(payload.decode('utf-8-sig'), newline=''), strict=True)
        if next(reader, None) != ['id', 'prediction']:
            raise ContractError('Header must be exactly id,prediction')
        ids, seen, histogram, violations = [], set(), {}, []
        for row in reader:
            if len(row) != 2:
                raise ContractError('Each row needs exactly two columns')
            identity, encoded = row
            if not identity or identity != identity.strip() or identity in seen:
                raise ContractError('Empty, padded or duplicate ID')
            prediction = strict_json(encoded)
            if audit_template_placeholders:
                issues = _prediction_issues(prediction)
                violations.extend((identity, index, field) for index, field in issues)
                count = len(prediction['actions'])
            else:
                count = validate_prediction(prediction)
            ids.append(identity)
            seen.add(identity)
            histogram[count] = histogram.get(count, 0) + 1
            if len(ids) > MAX_ROWS:
                raise ContractError('Row limit exceeded')
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ContractError('Invalid CSV encoding or escaping') from exc
    if not ids:
        raise ContractError('Empty template cannot establish the ID contract')
    return TemplateContract(tuple(ids), hashlib.sha256(payload).hexdigest(), len(payload),
                            sum(k * v for k, v in histogram.items()), tuple(sorted(histogram.items())), tuple(violations))


def load_template(path):
    """Read template structure and expose every placeholder value violation explicitly."""
    with Path(path).open('rb') as stream:
        return inspect_csv(stream.read(MAX_TEMPLATE_BYTES + 1), audit_template_placeholders=True)


def serialize_predictions(template, predictions):
    """Return CSV bytes only; never write a file or submit to Kaggle."""
    if not isinstance(predictions, dict) or any(type(k) is not str for k in predictions):
        raise ContractError('Predictions must be keyed by exact string IDs')
    if set(predictions) != set(template.ids):
        raise ContractError('Prediction IDs differ from template IDs')
    output = io.StringIO(newline='')
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(['id', 'prediction'])
    for identity in template.ids:
        prediction = predictions[identity]
        validate_prediction(prediction)
        writer.writerow([identity, json.dumps(prediction, allow_nan=False, separators=(',', ':'))])
    payload = output.getvalue().encode('utf-8')
    if len(payload) > MAX_TEMPLATE_BYTES:
        raise ContractError('Output exceeds local 4 MiB review limit')
    return payload
