# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_all, copy_metadata

datas = [
    ('.env', '.'),
    ('icon.png', '.'),
]

# ค้นหา ffmpeg.exe ในโฟลเดอร์โปรเจกต์ ถ้ามีให้แถมไปด้วย
if os.path.exists('ffmpeg.exe'):
    datas.append(('ffmpeg.exe', '.'))

binaries = []
hiddenimports = ['dotenv', 'groq', 'requests', 'faster_whisper', 'ctranslate2']

# รวบรวมข้อมูลที่จำเป็นสำหรับ faster_whisper และ ctranslate2
for pkg in ['faster_whisper', 'ctranslate2', 'huggingface_hub', 'tokenizers', 'regex', 'requests', 'packaging', 'filelock', 'numpy']:
    try:
        datas += copy_metadata(pkg)
    except:
        pass
    tmp_ret = collect_all(pkg)
    datas += tmp_ret[0]
    binaries += tmp_ret[1]
    hiddenimports += tmp_ret[2]

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ManowBot',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True, # ตั้งเป็น False ถ้าต้องการปิดหน้าต่าง Console ดำๆ
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['icon.png'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ManowBot',
)
