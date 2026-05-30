"""Quick integration test for the media server + player HTML. Run from fencing_analyzer dir."""
import sys, os, json, subprocess, tempfile, urllib.request as ur
from pathlib import Path

APP_DIR = Path(__file__).parent
sys.path.insert(0, str(APP_DIR))

import app as analyzer

video = r'C:\Users\micha\AI\Obsidian\Wawa\05_Archive\Videos\Cagliari2026_Day02_Piste8_1740-3015.mp4'

# 1. Create a small test clip
clip = Path(tempfile.gettempdir()) / 'test_player_flow.mp4'
result = Path(tempfile.gettempdir()) / 'test_player_flow.json'

FFMPEG = r'C:\Users\micha\AppData\Local\hermes\hermes-agent\venv\Scripts\ffmpeg.exe'
subprocess.run([FFMPEG, '-ss', '32', '-t', '3', '-i', video,
    '-vf', 'scale=640:-2', '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-an',
    '-y', str(clip)
], capture_output=True, timeout=120)
print(f'1. Clip: {clip.stat().st_size/1e6:.1f} MB')

# 2. Run Worker
worker = APP_DIR / 'worker_analyze.py'
proc = subprocess.run(
    [sys.executable, str(worker), str(clip), str(result)],
    capture_output=True, text=True, timeout=600
)
print(f'2. Worker: exit={proc.returncode}, {proc.stdout.strip()}')
assert proc.returncode == 0

with open(result) as f:
    data = json.load(f)
assert 'error' not in data
s = data['summary']
print(f'   Frames: {s["frames"]}, Distanz Ø: {s["dist_avg"]}cm')

# 3. Build player HTML
html = analyzer.build_live_player_html(data, clip, mode='server')
assert 'MEDIA_BASE' in html
print(f'3. Player HTML: {len(html)} bytes, MEDIA_BASE placeholder ok')

# 4. Start media server
metrics = {
    'dist': [d['cm'] for d in data['m1_dist']],
    'm_angle': [d['deg'] for d in data['m2_m_angle']],
    'g_angle': [d['deg'] for d in data['m2_g_angle']],
    'm_haltung': [d['deg'] for d in data['m5_m_tilt']],
    'g_haltung': [d['deg'] for d in data['m5_g_tilt']],
    'm_acc': [d['acc'] for d in data['m6_m_acc']],
    'g_acc': [d['acc'] for d in data['m6_g_acc']],
    'm_steps': data['m7_m_steps'],
    'g_steps': data['m7_g_steps'],
    'm_vel': data['m8_vel_m'],
    'g_vel': data['m8_vel_g'],
    'm_path': [{'x': p['x'], 'y': p['y']} for p in data['m4_m_path']],
    'g_path': [{'x': p['x'], 'y': p['y']} for p in data['m4_g_path']],
}
player_html = html.replace("const MEDIA_BASE = '';", 'const MEDIA_BASE = "";')
base_url = analyzer.start_media_server(clip, data['frame_data'], metrics, player_html)
analyzer.MediaRequestHandler.player_html = player_html.replace(
    'const MEDIA_BASE = "";',
    f'const MEDIA_BASE = "{base_url}";'
)
print(f'4. Server URL: {base_url}')

# 5. Test endpoints
h = ur.urlopen(f'{base_url}/health')
assert h.read() == b'ok', 'health check failed'
print('5. Health: OK')

h = ur.urlopen(f'{base_url}/data.json')
fd = json.loads(h.read())
assert len(fd) == len(data['frame_data'])
print(f'6. Data: {len(fd)} Frames')

h = ur.urlopen(f'{base_url}/metrics.json')
m = json.loads(h.read())
assert len(m['dist']) > 0
print(f'7. Metrics: {len(m["dist"])} dist values')

h = ur.urlopen(f'{base_url}/player')
pp = h.read().decode()
assert 'canvas' in pp and 'vid' in pp
print(f'8. Player page: {len(pp)} bytes, has canvas+video')

h = ur.urlopen(f'{base_url}/video')
assert h.status == 200
cr = h.headers.get('Content-Range', '')
assert 'bytes' in cr
print(f'9. Video: Content-Range={cr}')

analyzer.stop_media_server()
print('\n✅ ALL OK - Player system ready')
