"""Generate a tiny illustrative synthetic monocular image sequence.

Scene: fixed 3D textured squares, translating pinhole camera; no rolling
shutter/distortion. Designed for smoke testing, not a SLAM accuracy benchmark.
"""
from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np


def generate(root: Path, n_frames: int = 18, seed: int = 7):
    rng = np.random.default_rng(seed)
    image_dir = root / 'images'
    image_dir.mkdir(parents=True, exist_ok=True)
    w, h, fx, fy, cx, cy = 800, 600, 570., 570., 400., 300.
    camera = dict(width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy)
    (root / 'camera.json').write_text(json.dumps(camera, indent=2), encoding='utf-8')
    # Features lie on several depth-separated planes. No pure homography.
    points = np.stack((rng.uniform(-4, 4, 750), rng.uniform(-2.8, 2.8, 750),
                       rng.uniform(5, 14, 750)), axis=1)
    # Each marker has a unique stable 17x17 binary pattern surrounded by a border.
    patches = []
    for i in range(len(points)):
        cell = rng.integers(0, 2, size=(5, 5), dtype=np.uint8) * 225 + 15
        sprite = cv2.resize(cell, (15, 15), interpolation=cv2.INTER_NEAREST)
        sprite = np.pad(sprite, 1, constant_values=240)
        patches.append(sprite)
    for j in range(n_frames):
        # Smooth camera translation in x and z, with small vertical oscillation.
        cam_pos = np.array([0.11*j, 0.05*np.sin(j/5), 0.02*j])
        pc = points-cam_pos
        x = (fx*pc[:, 0]/pc[:, 2]+cx).astype(int)
        y = (fy*pc[:, 1]/pc[:, 2]+cy).astype(int)
        img = np.full((h, w), 115, dtype=np.uint8)
        for i in np.argsort(pc[:, 2])[::-1]:
            xx, yy = x[i], y[i]
            if 10 <= xx < w-10 and 10 <= yy < h-10:
                sprite = patches[i]
                img[yy-8:yy+9, xx-8:xx+9] = sprite
        img = cv2.GaussianBlur(img, (3, 3), 0.5)
        cv2.imwrite(str(image_dir / f'{j:05d}.png'), img)
    print(f'Wrote {n_frames} demo images to {image_dir}')
    print(f'Calibration: {root / "camera.json"}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=Path('demo'))
    p.add_argument('--frames', type=int, default=18)
    args = p.parse_args()
    generate(args.output, args.frames)
