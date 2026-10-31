"""Paths and environment of the VRM hair maker.

The HairDeck code tree (C:/Users/PC/Documents/HairDeck/code) and workspace (C:/Users/PC/Documents/HairDeck/model:
ckpt/, data/, third_party/) are used in place; the app keeps its own results, uploaded bodies and Blender tools under
hairmaker/. HairDeck reads its locations from HAIRDECK_* environment variables (hairdeck.paths), the Blender export
from HAIRDECK_BLENDER (hairdeck.vrm.export) — both set here before hairdeck is imported.
"""
import os
import sys

HAIRMAKER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../HairDeck/hairmaker
PARENT = os.path.dirname(HAIRMAKER)  # .../HairDeck
HAIRDECK_CODE = os.path.join(PARENT, 'code')
RTMLIB = os.path.join(os.path.dirname(PARENT), 'rtmlib')  # .../Documents/rtmlib (the VRM preview reference app)
if not os.path.isdir(HAIRDECK_CODE):  # moved installs: the sibling of this folder
    HAIRDECK_CODE = os.environ.get('HAIRDECK_CODE') or HAIRDECK_CODE

os.environ.setdefault('HAIRDECK_ROOT', os.path.join(PARENT, 'model'))
os.environ['HAIRDECK_RESULTS'] = os.path.join(HAIRMAKER, 'results')
os.environ['HAIRDECK_VRM_WORK'] = os.path.join(HAIRMAKER, 'results', 'vrm_export')
# forward slashes: hairdeck.vrm.export splits HAIRDECK_BLENDER with shlex (backslashes would be eaten)
os.environ['HAIRDECK_BLENDER'] = os.path.join(HAIRMAKER, 'tools', 'blender',
                                              'blender-4.2.23-windows-x64', 'blender.exe').replace('\\', '/')
# the VRM Add-on for Blender sits in its own extensions folder (tools/blender/user/extensions/user_default/vrm);
# blender.exe inherits these from the server process
os.environ.setdefault('BLENDER_USER_EXTENSIONS', os.path.join(HAIRMAKER, 'tools', 'blender', 'user', 'extensions'))
os.environ.setdefault('BLENDER_USER_CONFIG', os.path.join(HAIRMAKER, 'tools', 'blender', 'user', 'config'))
os.environ.setdefault('BLENDER_VRM_AUTOMATIC_LICENSE_CONFIRMATION', 'true')

sys.path.insert(0, HAIRDECK_CODE)

RESULTS = os.environ['HAIRDECK_RESULTS']
BODIES_DIR = os.path.join(HAIRMAKER, 'data', 'bodies')
STATIC_DIR = os.path.join(HAIRMAKER, 'static')
WEB_DIST = os.path.join(HAIRMAKER, 'web', 'dist')
BLENDER_SCRIPTS = os.path.join(HAIRMAKER, 'blender')

DEFAULT_MODEL_SRC = os.path.join(RTMLIB, 'vrm_models', 'AvatarSample_B.vrm')
IDLE_VRMA_SRC = os.path.join(RTMLIB, 'vrm_models', 'idle_loop.vrma')
EXAMPLES_SRC = os.path.join(os.environ['HAIRDECK_ROOT'], 'data', 'demo_examples')


def ensure_dirs():
    for d in (RESULTS, os.path.join(RESULTS, 'vrm_export'), BODIES_DIR, STATIC_DIR,
              os.path.join(STATIC_DIR, 'models'), os.path.join(STATIC_DIR, 'examples')):
        os.makedirs(d, exist_ok=True)
