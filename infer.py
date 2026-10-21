"""RetailAction Challenge inference: MViT-B (Kinetics-400 initialisation, fine-tuned) + DETR-style 4-slot localisation head. CPU only, no training, no network, no credentials.

Reads the official Challenge Test videos, writes a submission CSV (columns id,prediction) in the order of the official sample_submission.csv.
Frame sampling is the one used in training: the frame count of each view is obtained by a full decode, n_ref = max(n_rank0, n_rank1), and BOTH views are sampled at the same
16 normalised times  idx_k = min(floor((k + 0.5) / 16 * n_ref), n_view - 1);  frames are resized to 224x224 RGB, scaled to [0,1] and normalised with mean 0.45 / std 0.225.

usage: python -B infer.py --data-root <.../challenge_test_set/vam-dataset-challenge_test_no_labels/data> --template <sample_submission.csv> --output <new csv path> [--limit N]
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import MViTDETRv2                       # architecture (torchvision mvit_v1_b backbone + decoder), unchanged from the training code
from contract import load_template, inspect_csv, serialize_predictions, validate_prediction

CFG = {'slots': 4, 'minimum_predicted_interval': 0.001}
WEIGHTS = HERE / 'weights' / 'e3_model_state.pt'
WEIGHTS_SHA256 = '4b4c816268984dc86e9bd8f3aaaa44f0216d1ec8ba306f3a8fc60f7009dfce2e'
TEMPLATE_SHA256 = '826ef8a906dd435d812e06c54f2070e5a82455d087a0f12ac6e0e93d817de34c'
RK = ('rank0', 'rank1')


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def ref_indices(n_view, n_ref, T=16):
    idx = np.floor((np.arange(T) + 0.5) / T * n_ref).astype(np.int64)
    return np.minimum(idx, n_view - 1)


def count_frames(path):
    import av
    n = 0
    with av.open(str(path), mode='r') as c:
        if len(c.streams.video) != 1:
            raise ValueError('VIDEO_STREAM_COUNT')
        c.streams.video[0].codec_context.thread_count = 1
        for _ in c.decode(video=0):
            n += 1
            if n > 1024:
                raise ValueError('FRAME_SAFETY_LIMIT')
    if n < 1:
        raise ValueError('NO_FRAMES')
    return n


def decode_frames(path, idx, expect):
    import av
    need = {int(i) for i in idx}
    got, count = {}, 0
    with av.open(str(path), mode='r') as c:
        c.streams.video[0].codec_context.thread_count = 1
        for f in c.decode(video=0):
            if f.width != f.height:
                raise ValueError('NON_SQUARE_FRAME')
            if count in need:
                got[count] = f.reformat(width=224, height=224, format='rgb24').to_ndarray()
            count += 1
    if count != expect or len(got) != len(need):
        raise ValueError('FRAME_COUNT_CHANGED')
    return np.stack([got[int(i)] for i in idx])


class Clips(torch.utils.data.Dataset):
    def __init__(self, ids, root):
        self.ids, self.root = ids, Path(root)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        sid = self.ids[i]
        paths = {rk: self.root / sid / f'{rk}_video.mp4' for rk in RK}
        n = {rk: count_frames(paths[rk]) for rk in RK}
        nref = max(n.values())
        views = {rk: torch.from_numpy(decode_frames(paths[rk], ref_indices(n[rk], nref), n[rk])) for rk in RK}
        return {'id': sid, 'views': views, 'frames': (n['rank0'], n['rank1'])}


def collate(s):
    return {'ids': [x['id'] for x in s], 'views': {rk: torch.stack([x['views'][rk] for x in s]) for rk in RK}, 'frames': [x['frames'] for x in s]}


def to_input(u8):
    return ((u8.permute(0, 4, 1, 2, 3).float() / 255.0) - 0.45) / 0.225


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--template', required=True, help='official sample_submission.csv (only its ids and row order are used)')
    ap.add_argument('--output', required=True, help='csv path; must not exist')
    ap.add_argument('--limit', type=int, default=0, help='debug: only the first N ids; no csv is written')
    ap.add_argument('--threads', type=int, default=int(os.environ.get('RA_THREADS', '8')))
    ap.add_argument('--workers', type=int, default=int(os.environ.get('RA_WORKERS', '4')))
    ap.add_argument('--batch', type=int, default=2)
    a = ap.parse_args()
    out = Path(a.output)
    if out.exists():
        sys.exit('REFUSE: output exists: ' + str(out))
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, ''):
        print('note: CUDA_VISIBLE_DEVICES is set; this program never uses a GPU', file=sys.stderr)
    torch.set_grad_enabled(False)
    torch.set_num_threads(a.threads)
    h = sha256_file(WEIGHTS)
    if h != WEIGHTS_SHA256:
        sys.exit(f'WEIGHTS_SHA256_MISMATCH {h}')
    ts = sha256_file(a.template)
    if ts != TEMPLATE_SHA256:
        sys.exit(f'TEMPLATE_SHA256_MISMATCH {ts} (expected the official sample_submission.csv {TEMPLATE_SHA256})')
    tpl = load_template(a.template)
    ids = list(tpl.ids)[:a.limit] if a.limit else list(tpl.ids)
    for sid in ids:
        for rk in RK:
            if not (Path(a.data_root) / sid / f'{rk}_video.mp4').is_file():
                sys.exit(f'MISSING_VIDEO {sid}/{rk}')
    st = torch.load(WEIGHTS, map_location='cpu', weights_only=True)
    model = MViTDETRv2(CFG, backbone_state=None)
    model.load_state_dict(st['model_state'], strict=True)
    model.eval()
    print(json.dumps({'torch': torch.__version__, 'ids': len(ids), 'threads': a.threads, 'workers': a.workers, 'weights_sha256': h}), flush=True)
    ldr = torch.utils.data.DataLoader(Clips(ids, a.data_root), batch_size=a.batch, shuffle=False, num_workers=a.workers, collate_fn=collate,
                                      worker_init_fn=lambda _: torch.set_num_threads(1), prefetch_factor=2 if a.workers > 0 else None)
    preds, t0, done = {}, time.monotonic(), 0
    for b in ldr:
        o = model({rk: to_input(b['views'][rk]) for rk in RK})
        if not all(bool(torch.isfinite(v).all()) for v in o.values()):
            sys.exit('NONFINITE_MODEL_OUTPUT')
        conf = o['confidence_logits'].sigmoid().numpy()
        iv, pt = o['intervals'].numpy(), o['points'].numpy()
        for j, sid in enumerate(b['ids']):
            acts = [{'start': float(iv[j, k, 0]), 'end': float(iv[j, k, 1]), 'score': float(conf[j, k]),
                     'spatial': {'action_cam': {rk: {'x': float(pt[j, k, r, 0]), 'y': float(pt[j, k, r, 1])} for r, rk in enumerate(RK)}}}
                    for k in range(CFG['slots'])]
            acts.sort(key=lambda x: -x['score'])          # all four slots are kept; only the order (by confidence) is fixed
            v = {'actions': acts}
            validate_prediction(v)
            preds[sid] = v
        done += len(b['ids'])
        if done % 100 < a.batch or done == len(ids):
            print(json.dumps({'done': done, 'of': len(ids), 'seconds': round(time.monotonic() - t0, 1)}), flush=True)
    if a.limit:
        # partial run (debug / container smoke test): writes only the first N rows with the same row format; it is NOT a complete submission and must not be uploaded
        import csv, io
        buf = io.StringIO(newline='')
        w = csv.writer(buf, lineterminator='\n')
        w.writerow(['id', 'prediction'])
        for sid in ids:
            validate_prediction(preds[sid])
            w.writerow([sid, json.dumps(preds[sid], allow_nan=False, separators=(',', ':'))])
        payload = buf.getvalue().encode('utf-8')
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'xb') as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        print(json.dumps({'PARTIAL_RUN_NOT_A_SUBMISSION': True, 'output': str(out), 'sha256': sha256_file(out), 'rows': len(ids), 'seconds': round(time.monotonic() - t0, 1)}), flush=True)
        return
    payload = serialize_predictions(tpl, preds)
    ctr = inspect_csv(payload)
    assert ctr.ids == tpl.ids and not ctr.placeholder_violations
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'xb') as f:                                # exclusive create: never overwrites
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps({'output': str(out), 'sha256': sha256_file(out), 'bytes': out.stat().st_size, 'rows': len(ctr.ids), 'total_actions': ctr.total_actions,
                      'seconds': round(time.monotonic() - t0, 1)}), flush=True)


if __name__ == '__main__':
    main()
