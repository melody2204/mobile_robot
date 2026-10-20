# my_orbslam3 — Python SLAM backbone

A **minimal educational, ORB-SLAM3-inspired architecture**, following the requested package layout. It is **not** a complete reproduction of the ORB-SLAM3 algorithm and not suitable for safety-critical localisation. This implementation is original project scaffolding and uses OpenCV/SciPy building blocks; it does **not** import pySLAM or require ROS, Pangolin, g2o, or a GPU.

## Fast start (Pipenv, no ROS)

Requires Python 3.11+ (Python 3.12 is fine), Pipenv, and a machine with OpenCV support:

```bash
cd my_orbslam3
pipenv --python python3
pipenv install --dev
pipenv run python -m pytest -q
pipenv run python tools/generate_demo.py
pipenv run python main.py --images demo/images --camera demo/camera.json --output results
```

The synthetic demo is only an input-format and integration smoke test. Real data with calibration and parallax is recommended to evaluate tracking. For a slowly moving dataset try `--keyframe-gap 4`. To experiment with SciPy sliding-window bundle adjustment, add `--local-ba` (slower). No ROS is used.

To use your own images, prepare sequential grayscale or RGB JPEG/PNG images in one folder and a camera calibration JSON:

```json
{"width": 800, "height": 600, "fx": 570, "fy": 570, "cx": 400, "cy": 300}
```

Images must **already be undistorted** (or nearly so); the simplified camera model has no distortion coefficients. `--fps 20` sets evenly spaced timestamps unless modified. Use a correctly calibrated camera. Example:

```bash
pipenv run python main.py --images /path/to/undistorted_images \
    --camera /path/to/calibration.json --fps 20 --output results
```

`results/trajectory_frames_tum.txt`: TUM-format trajectory, stored as world-from-camera T_wc. Frame poses are **not retroactively corrected** by local BA or loops. `results/trajectory_keyframes_tum.txt`: final keyframe poses. `results/map_points.npy`: 3D sparse map. **Monocular map scale is arbitrary**. Timestamps are `frame_index / fps`, not dataset-native timestamps.

## Modules and what's really implemented

| Module | Status | What it currently does |
| --- | --- | --- |
| `frontend/orb_extractor.py` | ✅ Working | OpenCV ORB feature extraction |
| `frontend/feature_matcher.py` | ✅ Working | Mutual nearest-neighbour ratio-test ORB matches |
| `frontend/initializer.py` | ✅ Baseline | Essential-matrix RANSAC + two-view triangulation and cheirality/parallax checks |
| `frontend/tracking.py` | ✅ Baseline | Map descriptor matching + robust PnP |
| `mapping/frame.py` | ✅ | Transient image feature observations |
| `mapping/keyframe.py` | ✅ | Persistent poses/features/observation associations |
| `mapping/map_point.py` | ✅ | 3D landmark and observation records |
| `mapping/map.py` | ✅ | Bidirectional landmark–keyframe bookkeeping |
| `mapping/local_mapping.py` | ✅ Baseline | Two-view landmark triangulation from unmatched keyframe features |
| `optimization/pose_optimizer.py` | ✅ | PnP RANSAC + LM refinement |
| `optimization/bundle_adjustment.py` | 🟡 Experimental | Small SciPy sliding-window visual BA; enable with `--local-ba` |
| `optimization/pose_graph.py` | 🟡 Standalone | Simple SE(3) pose graph optimisation, not integrated into full SLAM yet |
| `loop_closing/place_recognition.py` | 🟡 Candidate retrieval | Exhaustive descriptor matching; not DBoW2 vocabulary |
| `loop_closing/loop_closing.py` | 🚧 Incomplete | Stores appearance candidates; **no geometric verification or loop correction** |
| `inertial/imu_preintegration.py` | 🟡 Demonstration | Rotation/velocity/position preintegration; **no covariance or bias Jacobians** |
| `inertial/visual_inertial.py` | 🚧 Placeholder | State type only; VIO optimisation not implemented |
| `atlas/atlas.py` | 🟡 Basic | Create/select maps; no map merging |
| `evaluation/trajectory.py` | ✅ | Save TUM-format poses |
| `evaluation/metrics.py` | ✅ | Umeyama trajectory alignment with optional scale |

## How the pipeline works

```text
Frames -> ORB -> 2-view initialization -> 3D landmark map
                          |
                          v
Images -> ORB -> match map descriptors -> PnP RANSAC -> current pose
                                                      |
                                          periodic keyframe insertion
                                                      |
                                                 triangulation
                                                      |
                                             optional local BA
```

The real ORB-SLAM3 implementation also contains hierarchical matching, motion prediction, covisibility-aware local mapping, robust relocalisation, full geometric loop closure, covisibility/spanning-tree graphs, tightly coupled visual–inertial optimisers, and full Atlas map merging. These are **planned** rather than claimed features here.

## Implementation order

1. Validate camera calibration and image geometry on a real sequence.
2. Add covisibility search, feature-to-map projection windows, stronger local map tracking, map-point pruning.
3. Improve local BA with proper gauge handling and marginalisation/fixed external observations, robust outlier rejection, and efficient Jacobians.
4. Integrate geometric loop verification, a pose-graph optimiser, landmark correction and map merging.
5. Implement error-state IMU preintegration with covariance and bias Jacobians, time synchronisation, camera–IMU extrinsic calibration, joint optimisation, gravity/scale initialisation.
6. Add Atlas relocalisation and map transitions, then expand to stereo/RGB-D.

## Mathematics and conventions

- Pose `T_cw` maps world coordinates into camera coordinates: `X_c = R_cw X_w + t_cw`.
- Camera calibration `K = [[fx,0,cx], [0,fy,cy], [0,0,1]]`.
- Essential matrix initialization: `x2.T @ E @ x1 == 0`, `E=[t]_x R`; monocular translation direction has **unknown scale**.
- PnP estimates `T_cw` from 3D map points and their 2D observations.
- Visual BA minimises robust reprojection residuals; current educational code fixes two oldest window poses and omits some correlations/outlier pruning.
- Trajectories export `T_wc=inv(T_cw)`, quaternion xyzw.

## References

- Campos et al., ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial, and Multi-Map SLAM (2021): https://arxiv.org/abs/2007.11898
- OpenCV ORB/PnP/triangulation: https://docs.opencv.org/
- SciPy least_squares: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
