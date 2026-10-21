"""Compare two submission CSVs (id,prediction): byte identity first, then a semantic comparison.
usage: python -B compare.py <reference.csv> <candidate.csv> [--tol 1e-3]
Exit 0: ids, row order and action counts are identical and every score/start/end/coordinate differs by at most --tol (the maximum differences are printed; byte identity is reported separately).
Exit 2: any structural difference (ids, order, action count) or any value beyond --tol. Nothing is repaired or ignored."""
import argparse, csv, hashlib, io, json, sys
from pathlib import Path


def load(p):
    b = Path(p).read_bytes()
    rows = list(csv.reader(io.StringIO(b.decode('utf-8-sig'), newline='')))
    assert rows[0] == ['id', 'prediction'], 'HEADER'
    return b, [(i, json.loads(v)) for i, v in rows[1:]]


def flat(a):
    s = a['spatial']['action_cam']
    return [a['score'], a['start'], a['end'], s['rank0']['x'], s['rank0']['y'], s['rank1']['x'], s['rank1']['y']]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('reference'); ap.add_argument('candidate'); ap.add_argument('--tol', type=float, default=1e-3)
    a = ap.parse_args()
    rb, R = load(a.reference); cb, C = load(a.candidate)
    out = {'reference_sha256': hashlib.sha256(rb).hexdigest(), 'candidate_sha256': hashlib.sha256(cb).hexdigest(), 'byte_identical': rb == cb, 'rows': [len(R), len(C)]}
    ok = len(R) == len(C) and [i for i, _ in R] == [i for i, _ in C]
    out['ids_and_order_identical'] = ok
    names = ['score', 'start', 'end', 'rank0_x', 'rank0_y', 'rank1_x', 'rank1_y']
    mx = [0.0] * 7; bad_count = 0; bad_rows = []
    if ok:
        for (i, p), (_, q) in zip(R, C):
            pa, qa = p['actions'], q['actions']
            if len(pa) != len(qa):
                bad_count += 1; bad_rows.append(i); continue
            for x, y in zip(pa, qa):
                f, g = flat(x), flat(y)
                for k in range(7):
                    mx[k] = max(mx[k], abs(f[k] - g[k]))
    out['actions_count_mismatch_rows'] = bad_count
    out['max_abs_diff'] = dict(zip(names, mx))
    out['tolerance'] = a.tol
    out['semantically_equal_within_tolerance'] = bool(ok and bad_count == 0 and max(mx) <= a.tol)
    print(json.dumps(out, indent=1))
    sys.exit(0 if out['semantically_equal_within_tolerance'] else 2)


if __name__ == '__main__':
    main()
