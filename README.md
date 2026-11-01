# VRM Hair Maker

https://github.com/user-attachments/assets/a2e3f0cd-4938-49a9-aead-bdf6fba6ba8a

Generate stylized 3D hair from a single portrait and wear it on a VRM avatar — in the browser, with live
spring-bone physics, and as a downloadable VRM file with the hair (and its spring bones) baked in.

Hair generation is [HairDeck](https://github.com/hyz317/HairDeck) (animation-ready hair cards + spring-bone chains
from one image); the VRM preview follows the technique of the local rtmlib web app (three.js + @pixiv/three-vrm,
MToon, VRMA idle animation).

## 🚀 Run

1. **Install Docker** — download [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows AMD64) and
   run it. Wait until the whale icon says Docker is running.

2. **Start VRM HairMaker**

    docker run -d --name vrm-hairmaker --restart unless-stopped --gpus all -p 8300:8300 -v vrm-hairmaker-data:/opt/HairDeck/hairmaker/results camenduru/vrmhairmaker

Open http://localhost:8300. That is the whole setup — the image is self-contained (HairDeck + its checkpoints,
Blender 4.2 LTS + the VRM Add-on, the built web app and the sample avatars), so nothing else has to be installed or
downloaded.

The hair model loads in the background (~30 s on first start); "Hair model ready." enables Generate. Generations and
exported VRMs are written to the `vrm-hairmaker-data` volume, so they survive `docker rm` / `docker pull` of a new
version. The image is ~17 GB and needs an NVIDIA GPU (the pipeline runs on CUDA through the WSL2 GPU support of
Docker Desktop).

## Run from source

    cd hairmaker
    venv\Scripts\python.exe -m server.main --port 8300

Open http://127.0.0.1:8300 — the page is served from `web/dist` (rebuild with `cd web && npm install && npm run build`).
The hair model loads in the background (~30 s); "Hair model ready." enables Generate.

## How it works

1. **Generate** — the uploaded portrait runs through HairDeck's pipeline (SAM 3 + BiRefNet matting → DINOv3 +
   image set-DiT → card VAE decode → chain DiT), producing hair-card geometry, spring-bone chains and GLB meshes
   (1–4 variations, seed / steps / guidance controls).
2. **Preview** — the viewer loads your VRM model and lays the generated hair onto its head with the body's
   head-frame transform (`p_dataset = (p_head − b) / k`, computed per model by a Blender head fit). The model's own
   hair meshes are hidden, mirroring what the export replaces.
3. **Spring bones** — with the checkbox on, the browser ports HairDeck's UniVRM verlet spring simulation
   (`hairdeck/spring/sim.py` → `web/src/spring.js`): the skinning is the exact geometric binding the Blender export
   uses (`hairdeck.vrm.prep`), driven by the VRM's real head motion (idle animation or the Turn / Nod / Jump / Sway
   motions).
4. **Export VRM** — Blender 4.2 LTS + the VRM Add-on removes the avatar's own hair, adds the generated hair mesh,
   `HairGen_*` chain bones and VRM spring groups, and exports a downloadable avatar:
   - VRM 0.x bodies: HairDeck's validated export path (`hairdeck.vrm.export`), validation included.
   - VRM 1.0 bodies: the same Blender edit, then `server/vrm1.py` injects the chains as `VRMC_springBone` springs
     directly into the exported glTF (the add-on keeps 1.0 files in 1.0 and does not map the 0.x spring groups).

## Layout

    hairmaker/
    ├─ server/            FastAPI backend (config, pipeline wrapper, bodies, VRM 1.0 spring injection)
    ├─ blender/           user_body_boxes.py — head/face fit for arbitrary VRM uploads (Blender script)
    ├─ web/               Vite + three.js + three-vrm frontend (src/main.js, src/spring.js)
    ├─ static/            served assets (default model copy, idle VRMA, example images)
    ├─ tools/blender/     Blender 4.2.23 LTS + the VRM Add-on (isolated user/extensions dirs)
    ├─ data/bodies/       one folder per avatar: model.vrm, meta.json, body.json (head-frame transform)
    ├─ results/           one folder per generation (HairDeck outputs + rig caches + exports)
    └─ venv/              Python 3.13 virtual environment (torch 2.8 cu128, HairDeck requirements)

HairDeck itself lives in the sibling `code/` tree and its checkpoints/data in `model/` (this is the HairDeck
workspace: `HAIRDECK_ROOT` points at `model/`, third-party weights in `model/third_party/weights`); the default
sample avatar and idle animation come from the sibling `rtmlib/vrm_models/`.

## API

    GET  /api/status                 pipeline readiness + registered bodies
    GET  /api/models | POST /api/models (multipart .vrm)     list / register an avatar (Blender head fit, ~40 s)
    GET  /api/examples               example portraits
    POST /api/generate               image + (vrm, seed, steps, cfg, n_samples, chains) -> result with sample urls
    POST /api/chains                 add spring-bone chains to a saved sample (chain DiT)
    GET  /api/hairrig/{out}/{seed}?vrm=<id>                  skinned hair + chains for the browser spring preview
    POST /api/export_vrm             hair onto the avatar as a downloadable VRM (Blender)
    POST /api/turn                   classic server-rendered head-turn mp4 (needs GL offscreen; optional)
    GET  /api/download_all/{out}     all files of a result as a zip

## Notes

- VRM 0.x models (VRoid Studio exports) are the fully validated path; VRM 1.0 models preview and export with
  injected spring bones (validation of the 0.x round-trip is skipped for them).
- Avatars without a face-named mesh fall back to an approximate head-region fit (`face_method: head_region`).
- The classic head-turn video needs an offscreen GL context (Linux/EGL); on Windows the live 3D motion preview
  replaces it.
