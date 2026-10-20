"""Run a minimal ORB-SLAM3-inspired monocular backbone.

Example:
    pipenv install
    pipenv run python tools/generate_demo.py
    pipenv run python main.py --images demo/images --camera demo/camera.json

This IS NOT full ORB-SLAM3. See README feature matrix and known limitations.
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path
import cv2
import numpy as np
from geometry import PinholeCamera
from frontend.orb_extractor import ORBExtractor
from frontend.feature_matcher import FeatureMatcher
from frontend.initializer import MonocularInitializer
from frontend.tracking import Tracker
from mapping.frame import Frame
from mapping.keyframe import KeyFrame
from mapping.local_mapping import LocalMapper
from optimization.bundle_adjustment import LocalBundleAdjuster
from loop_closing.loop_closing import LoopCloser
from atlas.atlas import Atlas
from evaluation.trajectory import write_tum_trajectory

LOG = logging.getLogger("my_orbslam3")


class MonoSLAM:
    """Single-threaded visual backbone with persistent sparse landmarks."""

    def __init__(self, camera, nfeatures=2500, keyframe_gap=5, bundle_adjust=False):
        self.camera = camera
        self.extractor = ORBExtractor(nfeatures=nfeatures)
        self.matcher = FeatureMatcher()
        self.initializer = MonocularInitializer(camera, self.matcher)
        self.tracker = Tracker(camera, self.matcher)
        self.mapper = LocalMapper(camera, self.matcher)
        self.ba = LocalBundleAdjuster(camera)
        self.loop_closer = LoopCloser(self.matcher)
        self.atlas = Atlas()
        self.first: Frame | None = None
        self.previous_keyframe: KeyFrame | None = None
        self.current_pose: np.ndarray | None = None
        self.initialized = False
        self.keyframe_gap = keyframe_gap
        self.bundle_adjust = bundle_adjust
        self.trajectory: list[tuple[float, np.ndarray]] = []
        self.failed_frames = 0
        self.total_failed_frames = 0

    @property
    def map(self):
        return self.atlas.active_map

    def process(self, image, frame_id, timestamp) -> dict:
        frame = Frame.create(frame_id, timestamp, image, self.extractor)
        if frame.descriptors is None or len(frame.keypoints) < 30:
            return {"status": "not_enough_features", "frame": frame_id}
        if not self.initialized:
            if self.first is None:
                self.first = frame
                return {"status": "awaiting_parallax", "frame": frame_id}
            result = self.initializer.initialize(self.first, frame)
            if result is None:
                # Keep the first reference unless it is too old (occlusions / bad baseline).
                if frame.id - self.first.id > 20:
                    self.first = frame
                return {"status": "awaiting_parallax", "frame": frame_id}
            self.first.T_cw = result.T_1
            frame.T_cw = result.T_2
            kf1 = KeyFrame.from_frame(self.first)
            kf2 = KeyFrame.from_frame(frame)
            self.map.add_keyframe(kf1)
            self.map.add_keyframe(kf2)
            for xyz, (i, j) in zip(result.points, result.feature_pairs):
                mp = self.map.new_point(xyz, frame.descriptors[j], [(kf1.id, i), (kf2.id, j)])
                frame.associations[j] = mp.id
            self.previous_keyframe = kf2
            self.current_pose = frame.T_cw.copy()
            self.initialized = True
            self.trajectory.append((kf1.timestamp, kf1.T_cw.copy()))
            self.trajectory.append((kf2.timestamp, kf2.T_cw.copy()))
            LOG.info("Initialised from frames %d and %d with %d map points",
                     self.first.id, frame.id, len(self.map.points))
            self.first = None
            return {"status": "initialized", "frame": frame_id, "points": len(self.map.points)}

        n_inliers = self.tracker.track(frame, self.map, self.current_pose)
        if n_inliers == 0:
            self.failed_frames += 1
            self.total_failed_frames += 1
            return {"status": "tracking_lost", "frame": frame_id,
                    "note": "relocalisation/new-map recovery not implemented"}
        self.failed_frames = 0
        self.current_pose = frame.T_cw.copy()
        self.trajectory.append((timestamp, frame.T_cw.copy()))
        status = {"status": "tracked", "frame": frame_id, "inliers": n_inliers}
        if frame_id - self.previous_keyframe.id >= self.keyframe_gap:
            new_kf = KeyFrame.from_frame(frame)
            self.map.add_keyframe(new_kf)
            added = self.mapper.triangulate_between(self.previous_keyframe, new_kf, self.map)
            self.previous_keyframe = new_kf
            status.update({"keyframe": True, "new_points": added})
            if self.bundle_adjust:
                status["local_BA"] = self.ba.optimize(self.map)
                self.current_pose = new_kf.T_cw.copy()
            # Appearance-based candidates only; no false assertion that a loop is closed.
            status["loop_candidates"] = len(self.loop_closer.inspect(new_kf, self.map))
        return status


def list_images(directory: Path) -> list[Path]:
    return sorted([p for p in directory.iterdir()
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, required=True, help="Directory containing ordered images")
    parser.add_argument("--camera", type=Path, required=True, help="JSON calibration: width,height,fx,fy,cx,cy")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    parser.add_argument("--keyframe-gap", type=int, default=5)
    parser.add_argument("--nfeatures", type=int, default=2500)
    parser.add_argument("--local-ba", action="store_true", help="Slow SciPy educational local BA")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if not args.images.is_dir():
        parser.error(f"Image directory does not exist: {args.images}")
    camera = PinholeCamera.from_json(str(args.camera))
    paths = list_images(args.images)
    if args.max_frames:
        paths = paths[:args.max_frames]
    if len(paths) < 2:
        parser.error("Need at least two images")
    slam = MonoSLAM(camera, args.nfeatures, args.keyframe_gap, args.local_ba)
    for i, path in enumerate(paths):
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            LOG.warning("Skipping unreadable image %s", path)
            continue
        if gray.shape != (camera.height, camera.width):
            parser.error(f"Image {path.name} shape {gray.shape[::-1]} != calibrated {(camera.width, camera.height)}")
        result = slam.process(gray, i, i / args.fps)
        if i % 5 == 0 or result["status"] in ("initialized", "tracking_lost"):
            LOG.info("frame %04d: %s", i, result)
    args.output.mkdir(parents=True, exist_ok=True)
    write_tum_trajectory(args.output / "trajectory_frames_tum.txt", slam.trajectory)
    keyframe_trajectory = [(kf.timestamp, kf.T_cw) for kf in slam.map.keyframes.values()]
    write_tum_trajectory(args.output / "trajectory_keyframes_tum.txt", keyframe_trajectory)
    if slam.map.points:
        xyz = np.array([p.xyz for p in slam.map.points.values() if not p.bad])
        np.save(args.output / "map_points.npy", xyz)
    LOG.info("Finished: %d tracked poses, %d keyframes, %d map points, %d tracking failures",
             len(slam.trajectory), len(slam.map.keyframes), len(slam.map.points), slam.total_failed_frames)
    LOG.info("Outputs: %s", args.output.resolve())


if __name__ == "__main__":
    main()
