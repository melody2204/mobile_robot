import numpy as np
from geometry import PinholeCamera, make_pose, inverse_pose, triangulate
from mapping.map import SparseMap
from mapping.keyframe import KeyFrame
from optimization.pose_optimizer import estimate_pose
from evaluation.metrics import align_trajectory_umeyama
from inertial.imu_preintegration import IMUPreintegrator
from atlas.atlas import Atlas


def test_pose_inverse():
    T = make_pose(np.eye(3), [1, 2, 3])
    assert np.allclose(T @ inverse_pose(T), np.eye(4))


def test_two_view_triangulation():
    c = PinholeCamera(800, 600, 550, 550, 400, 300)
    pts = np.array([[0, 0, 5], [1, 0, 6], [-1, 1, 8], [0.3, -.5, 7]])
    T1 = np.eye(4)
    T2 = make_pose(np.eye(3), [-0.4, 0, 0])
    uv1, _ = c.project(pts, T1)
    uv2, _ = c.project(pts, T2)
    restored = triangulate(c.K, T1, T2, uv1, uv2)
    assert np.allclose(restored, pts, atol=1e-6)


def test_pnp_robust():
    rng = np.random.default_rng(7)
    c = PinholeCamera(800, 600, 500, 500, 400, 300)
    pts = np.stack([rng.uniform(-1.5, 1.5, 90), rng.uniform(-1, 1, 90),
                    rng.uniform(4, 7, 90)], axis=1)
    true_pose = make_pose(np.eye(3), [-0.2, 0.05, 0])
    uv, valid = c.project(pts, true_pose)
    assert valid.all()
    uv += rng.normal(0, 0.3, uv.shape)
    uv[:10] += 60  # outliers
    result = estimate_pose(c, pts, uv)
    assert result is not None
    assert len(result.inlier_indices) >= 65
    assert np.linalg.norm(result.T_cw[:3, 3] - true_pose[:3, 3]) < 0.1


def test_map_links():
    m = SparseMap()
    for i in (0, 1):
        m.add_keyframe(KeyFrame(i, float(i), np.eye(4), np.zeros((3, 2)),
                                np.zeros((3, 32), dtype=np.uint8)))
    pt = m.new_point(np.array([0., 0., 5.]), np.zeros(32, np.uint8), [(0, 1), (1, 2)])
    assert m.keyframes[0].observations[1] == pt.id
    assert m.keyframes[1].observations[2] == pt.id
    assert len(pt.observations) == 2


def test_umeyama_sim3():
    rng = np.random.default_rng(8)
    p = rng.normal(size=(30, 3))
    q = p*3 + np.array([10., -2., 7.])
    result = align_trajectory_umeyama(p, q)
    assert result['rmse'] < 1e-8
    assert abs(result['scale'] - 3) < 1e-8


def test_imu_integration_and_atlas():
    imu = IMUPreintegrator()
    imu.integrate(np.zeros(3), np.array([2., 0., 0.]), 0.5)
    assert np.allclose(imu.delta_p, [0.25, 0, 0])
    assert np.allclose(imu.delta_v, [1., 0, 0])
    atlas = Atlas()
    new_id = atlas.create_map()
    assert new_id == 1 and len(atlas.maps) == 2


def test_synthetic_end_to_end(tmp_path):
    from tools.generate_demo import generate
    import cv2
    from main import MonoSLAM
    from pathlib import Path
    from geometry import PinholeCamera
    root = tmp_path / "synthetic"
    generate(root, n_frames=9)
    camera = PinholeCamera.from_json(str(root / "camera.json"))
    slam = MonoSLAM(camera, keyframe_gap=3)
    statuses = []
    for i, path in enumerate(sorted((root / "images").glob("*.png"))):
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        statuses.append(slam.process(gray, i, i / 20)["status"])
    assert "initialized" in statuses
    assert "tracked" in statuses
    assert len(slam.map.points) >= 50
    assert len(slam.map.keyframes) >= 3
