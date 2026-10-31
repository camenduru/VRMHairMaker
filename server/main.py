"""VRM hair maker server (FastAPI): the HairDeck image pipeline behind a small JSON API, VRM bodies the hair can be
attached to, and the Blender VRM 0.x export. Serves the built web app from web/dist when present.

python server/main.py [--host 127.0.0.1] [--port 8300]
"""
import argparse
import json
import os
import shutil
import threading
import time
import traceback

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import bodies, config, pipeline
from .config import RESULTS, STATIC_DIR, WEB_DIST

config.ensure_dirs()
app = FastAPI(title='VRM hair maker')
_STATE = {'pipeline': 'loading', 'pipeline_error': None, 'ready': False, 'started': time.time()}


def log(*a):
    print(time.strftime('%F %T'), *a, flush=True)


def _warmup():
    try:
        p = pipeline.pipeline()
        _STATE['pipeline'] = 'ready'
        _STATE['ready'] = True
        log('pipeline loaded in %.1f s; chains', p.load_sec, 'on' if p.chain is not None else f'off ({p.chain_error})')
    except Exception:  # noqa: BLE001
        _STATE['pipeline'] = 'error'
        _STATE['pipeline_error'] = traceback.format_exc()[-2000:]
        log('PIPELINE LOAD FAILED', _STATE['pipeline_error'])


def _result_dir(out_id):
    d = os.path.abspath(os.path.join(RESULTS, out_id.replace('/', os.sep)))
    if not os.path.isdir(d) or not os.path.exists(os.path.join(d, 'info.json')):
        raise HTTPException(404, f'result {out_id} not found')
    return d


def _rel_url(path):
    """URL of a file under the results root (works for nested out dirs too)."""
    return '/results/' + os.path.relpath(os.path.abspath(path), RESULTS).replace('\\', '/')


def _sample_urls(d, s):
    """Download urls of one sample, exactly the files HairDeck writes per seed."""
    out = dict(seed=s['seed'], n_cards=s.get('n_cards'), n_meshed=s.get('n_meshed'), n_chains=s.get('n_chains'),
               label=f"Seed {s['seed']} · {s.get('n_cards')} cards" +
                     (f" · {s.get('n_chains')} chains" if s.get('n_chains') is not None else ''))
    for key in ('glb', 'glb_nohead', 'glb_chains', 'glb_nohead_chains', 'json', 'chains_json'):
        if s.get(key) and os.path.exists(s[key]):
            out[key] = _rel_url(s[key])
    return out


@app.get('/api/status')
def status():
    return dict(_STATE, bodies=bodies.all_bodies())


@app.post('/api/warmup')
def warmup():
    threading.Thread(target=_warmup, daemon=True).start()
    return {'ok': True}


@app.get('/api/models')
def models():
    return bodies.all_bodies()


@app.post('/api/models')
async def upload_model(file: UploadFile = File(...)):
    """Register an uploaded VRM as a body (its head fit is computed with Blender, ~30-60 s)."""
    if not file.filename or not file.filename.lower().endswith('.vrm'):
        raise HTTPException(400, 'Please upload a .vrm file.')
    data = await file.read()
    if len(data) < 1000:
        raise HTTPException(400, 'That file is too small to be a VRM model.')
    mid = bodies.new_upload_id()
    # detect the version first: VRM 1.0 files cannot go through the 0.x Blender export
    tmp = os.path.join(config.BODIES_DIR, f'_tmp_{mid}.vrm')
    os.makedirs(config.BODIES_DIR, exist_ok=True)
    with open(tmp, 'wb') as f:
        f.write(data)
    ver = bodies.vrm_version(tmp)
    os.remove(tmp)
    summary = bodies.build_uploaded(mid, file.filename, data)
    log('MODEL', mid, summary['file_tag'], summary['vrm_version'], summary['face_method'])
    return summary


@app.get('/api/models/{mid}')
def model_detail(mid: str):
    try:
        return bodies.summary(mid)
    except Exception:  # noqa: BLE001
        raise HTTPException(404, f'model {mid} not found')


@app.get('/api/examples')
def examples():
    d = config.EXAMPLES_SRC
    if not os.path.isdir(d):
        return []
    out = []
    for f in sorted(os.listdir(d)):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            out.append(f'/examples/{f}')
    return out


@app.post('/api/generate')
async def generate(image: UploadFile = File(...), vrm: str = Form('default'), seed: int = Form(0),
                   steps: int = Form(50), cfg: float = Form(2.0), n_samples: int = Form(1), chains: bool = Form(True)):
    if image.filename is None:
        raise HTTPException(400, 'Please upload an image first.')
    os.makedirs(os.path.join(RESULTS, 'uploads'), exist_ok=True)
    ext = os.path.splitext(image.filename or 'image.png')[1] or '.png'
    src = os.path.join(RESULTS, 'uploads', f'{int(time.time())}_{os.getpid()}_{abs(hash(image.filename)) % 99999}{ext}')
    with open(src, 'wb') as f:
        f.write(await image.read())
    try:
        r = pipeline.generate(src, seed=seed, steps=steps, cfg=cfg, n_samples=n_samples, chains=chains)
    except Exception:  # noqa: BLE001
        log('GENERATE FAILED', traceback.format_exc()[-2000:])
        raise HTTPException(500, 'Generation failed. Please try again or use another image.')
    finally:
        try:
            os.remove(src)
        except OSError:
            pass
    d = r['out_dir']
    out_id = os.path.relpath(d, RESULTS).replace('\\', '/')
    body = bodies.body(vrm) if os.path.exists(bodies.body_path(vrm)) else bodies.body('default')
    return dict(ok=r['ok'], message=r['message'], out_id=out_id, status=r['status'], notes=r['notes'],
                hairColor=r['hair_color'], timings=r['timings'],
                prepVis=_rel_url(f'{d}/prep.png') if os.path.exists(f'{d}/prep.png') else None,
                body=body.get('file_tag'), bodyId=body['key'],
                samples=[_sample_urls(d, s) for s in r['samples']])


@app.get('/api/result/{out_id}')
def result(out_id: str):
    d = _result_dir(out_id)
    info = json.load(open(os.path.join(d, 'info.json')))
    samples = []
    for s in info.get('samples', []):
        su = _sample_urls(d, s)
        su['glb_chains'] = su.get('glb_chains')
        if not su.get('glb_chains') and os.path.exists(f"{d}/chains_seed{s['seed']}.json"):
            su['chains_pending'] = True
        samples.append(su)
    return dict(ok=info.get('ok'), message=info.get('message'), out_id=out_id, notes=info.get('notes'),
                hairColor=info.get('hair_color'), timings=info.get('timings'), params=info.get('params'),
                prepVis=_rel_url(f'{d}/prep.png') if os.path.exists(f'{d}/prep.png') else None,
                samples=samples)


@app.post('/api/chains')
def add_chains(out_id: str = Form(...), seed: int = Form(...)):
    """Spring-bone chains of one saved sample (chain DiT, ~1 s), for a result generated without chains."""
    d = _result_dir(out_id)
    try:
        smp = pipeline.add_chains(d, seed)
    except Exception as e:  # noqa: BLE001
        log('CHAINS FAILED', traceback.format_exc()[-1500:])
        raise HTTPException(500, f'Chains could not be generated: {e}')
    return _sample_urls(d, smp)


@app.get('/api/hairrig/{out_id}/{seed}')
def hairrig(out_id: str, seed: int, vrm: str = 'default'):
    """Skinned hair + chains on a body, dataset frame, for the in-browser spring preview."""
    d = _result_dir(out_id)
    body = bodies.body(vrm)
    if body.get('k') is None:
        raise HTTPException(400, 'this model has no head fit')
    try:
        return pipeline.hair_rig(d, seed, body)
    except Exception:  # noqa: BLE001
        log('HAIRRIG FAILED', traceback.format_exc()[-2000:])
        raise HTTPException(500, 'The spring-rig data could not be built for this model.')


@app.post('/api/export_vrm')
def export_vrm(out_id: str = Form(...), seed: int = Form(...), vrm: str = Form('default')):
    """VRM 0.x of one sample on a body: hairdeck.vrm.export.export_vrm with the body file (render=False; the rest
    pose is previewed in the browser instead)."""
    d = _result_dir(out_id)
    body = bodies.body(vrm)
    if body.get('k') is None:
        raise HTTPException(400, 'this model has no head fit')
    info = json.load(open(os.path.join(d, 'info.json')))
    try:
        with pipeline._EXPORT_LOCK:
            vi = bodies_export(d, seed, info['hair_color'], body)
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001
        log('EXPORT FAILED', traceback.format_exc()[-3000:])
        raise HTTPException(500, 'The VRM export failed. Please try again.')
    log('EXPORT', out_id, seed, body['file_tag'], 'ok', vi.get('ok'), vi.get('size_mb'), 'MB')
    url = f'/results/{out_id}/{os.path.basename(vi["vrm"])}'
    return dict(vrm=url, ok=vi.get('ok'), problems=vi.get('problems', [])[:5], size_mb=vi.get('size_mb'),
                n_chains=vi.get('n_chains'), n_spring_groups=vi.get('n_spring_groups'),
                timings=vi.get('timings'), bodyId=body['key'])


def bodies_export(d, seed, colour, body):
    """hairdeck.vrm.export.export_vrm on one of our body files (worker when running, else one-shot Blender).

    VRM 1.0 bodies run one-shot (the worker's render-queue patch reads the 0.x re-export) with the 0.x-only file
    checks skipped; server.vrm1 adds the spring bones to the exported VRM 1.0 JSON afterwards."""
    from hairdeck.vrm import export as VX
    bp = bodies.body_path(body['key'])
    is_vrm1 = (body.get('meta') or {}).get('vrm_version') == '1.0'
    saved_check = None
    if is_vrm1:
        saved_check = VX.check
        VX.check = lambda *a, **k: {'ok': True, 'problems': ['VRM 1.0 body: the 0.x round-trip checks are skipped; '
                                                             'the spring bones are added to the VRM 1.0 file directly.']}
    try:
        if not is_vrm1 and VX.worker_pid(bp) is None:
            try:
                VX.start_worker(bp)
                for _ in range(150):  # wait for the worker's first cache build (import + re-export, ~30-60 s)
                    if VX.worker_pid(bp):
                        break
                    time.sleep(1.0)
            except Exception as e:  # noqa: BLE001
                log('worker start failed, one-shot fallback:', e)
        vi = VX.export_vrm(d, int(seed), colour, body=bp, render=False)
    finally:
        if saved_check is not None:
            VX.check = saved_check
    if is_vrm1:
        from . import vrm1
        rig = pipeline.hair_rig(d, seed, body)
        added = vrm1.inject_springs(vi['vrm'], rig)
        log('VRM1 springs injected:', added, 'chains')
        vi['n_spring_groups'] = added
        vi['n_chains'] = added
    return vi


@app.post('/api/turn')
def turn(out_id: str = Form(...), seed: int = Form(...), vrm: str = Form('default')):
    """Server-rendered 3 s head-turn video (hairdeck.vrm.export.turn_preview). The browser preview is the primary
    motion preview; this is the HairDeck-style mp4 download."""
    d = _result_dir(out_id)
    body = bodies.body(vrm)
    if body.get('k') is None:
        raise HTTPException(400, 'this model has no head fit')
    info = json.load(open(os.path.join(d, 'info.json')))
    try:
        with pipeline._EXPORT_LOCK:
            from hairdeck.vrm import export as VX
            res = VX.turn_preview(d, int(seed), info['hair_color'], show_chains=False, body=bodies.body_path(body['key']))
    except Exception:  # noqa: BLE001
        log('TURN FAILED', traceback.format_exc()[-3000:])
        raise HTTPException(500, 'The head-turn render failed (it needs an OpenGL offscreen context). '
                                 'Use the live 3D motion preview instead.')
    url = f'/results/{out_id}/{os.path.basename(res["out"])}'
    return dict(video=url, seconds=res.get('seconds'), max_motion_cm=res.get('max_motion_cm'))


@app.get('/api/assets')
def assets():
    """Static assets the page needs: the default model, the idle animation, example images."""
    return dict(defaultModel='/models/default.vrm', idleVrma='/models/idle_loop.vrma')


@app.get('/api/download_all/{out_id}')
def download_all(out_id: str):
    d = _result_dir(out_id)
    base = os.path.basename(d)
    tmp = os.path.join(RESULTS, f'{base}_assets')
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    for f in os.listdir(d):
        if os.path.isfile(os.path.join(d, f)) and not f.startswith('.'):
            shutil.copy2(os.path.join(d, f), os.path.join(tmp, f))
    shutil.make_archive(tmp, 'zip', tmp)
    shutil.rmtree(tmp)
    return FileResponse(tmp + '.zip', filename=f'{base}_assets.zip', media_type='application/zip')


# static: results, models (VRM files + vrma), bodies (uploaded + built-in avatars), examples, web app
app.mount('/results', StaticFiles(directory=RESULTS), name='results')
app.mount('/models', StaticFiles(directory=os.path.join(STATIC_DIR, 'models')), name='models')
app.mount('/bodies', StaticFiles(directory=config.BODIES_DIR), name='bodies')
app.mount('/examples', StaticFiles(directory=os.path.join(STATIC_DIR, 'examples')), name='examples')
if os.path.isdir(os.path.join(WEB_DIST, 'assets')):
    app.mount('/assets', StaticFiles(directory=os.path.join(WEB_DIST, 'assets')), name='assets')


@app.get('/')
def index():
    p = os.path.join(WEB_DIST, 'index.html')
    if os.path.exists(p):
        return FileResponse(p)
    return JSONResponse({'error': 'web app not built', 'hint': 'cd web && npm install && npm run build'}, 503)


@app.get('/{path:path}')
def spa(path: str):
    p = os.path.join(WEB_DIST, path)
    if os.path.isfile(p):
        return FileResponse(p)
    return index()


def seed_static():
    """Copy the default model, the idle animation and the example images into static/ once."""
    m = os.path.join(STATIC_DIR, 'models')
    if not os.path.exists(os.path.join(m, 'default.vrm')) and os.path.exists(config.DEFAULT_MODEL_SRC):
        shutil.copy2(config.DEFAULT_MODEL_SRC, os.path.join(m, 'default.vrm'))
    if not os.path.exists(os.path.join(m, 'idle_loop.vrma')) and os.path.exists(config.IDLE_VRMA_SRC):
        shutil.copy2(config.IDLE_VRMA_SRC, os.path.join(m, 'idle_loop.vrma'))
    ex = os.path.join(STATIC_DIR, 'examples')
    if os.path.isdir(config.EXAMPLES_SRC) and not os.listdir(ex):
        for f in sorted(os.listdir(config.EXAMPLES_SRC)):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                shutil.copy2(os.path.join(config.EXAMPLES_SRC, f), os.path.join(ex, f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8300)
    a = ap.parse_args()
    seed_static()
    bodies.add_default()
    log('bodies:', [(b['id'], b['file_tag']) for b in bodies.all_bodies()])
    threading.Thread(target=_warmup, daemon=True).start()
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port, log_level='warning')


if __name__ == '__main__':
    main()
