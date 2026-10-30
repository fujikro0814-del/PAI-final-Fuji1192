"""閉じたループの評価器（recovla.eval.closed_loop）の確かめ。

シミュレーションの部分は mujoco だけで通る。入力の確かめは torch が要り、無ければ飛ばす。学習済みの方策は読まない。
"""
import json
import pathlib

import numpy as np
import pytest

from recovla.data import vla_image_spec as spec
from recovla.eval import closed_loop as cle
from recovla.record import recorder
from recovla.sim import control


@pytest.fixture(scope="module")
def rig():
    r = cle.EvalRig()
    yield r
    r.close()


def placement4():
    return {p.placement_id: p for p in recorder.training_placements()}[4]


def test_timing_matches_conversion():
    assert cle.STEPS_PER_ACTION == 50 and cle.ACTION_DT == pytest.approx(0.1)
    pytest.importorskip("PIL")                     # 変換は PIL が要る（LeRobot の venv だけ）
    from recovla.data import convert as conv
    assert cle.STRIDE == conv.STRIDE


@pytest.mark.render                              # EvalRig は mujoco.Renderer を作る
def test_action_moves_x_des_by_exactly_its_xyz(rig):
    rig.reset(placement4())
    x0 = rig.integrator.x_cmd.copy()
    a = np.array([0.004, -0.003, -0.002, 0, 0, 0, -1.0])
    n = {"steps": 0}
    cle.execute_action(rig, a, lambda: n.__setitem__("steps", n["steps"] + 1))
    assert n["steps"] == 50
    assert rig.integrator.x_cmd - x0 == pytest.approx(a[:3], abs=1e-12)
    assert rig.controller.desired_pos == pytest.approx(rig.integrator.x_cmd, abs=1e-12)
    assert rig.controller.gripper_closed is False


@pytest.mark.render
def test_gripper_sign_and_single_press(rig):
    rig.reset(placement4())
    close = np.array([0, 0, 0, 0, 0, 0, 1.0])
    presses = []
    orig = rig.controller._update_gripper

    def spy(button, *args, **kw):
        presses.append(bool(button))
        return orig(button, *args, **kw)

    rig.controller._update_gripper = spy
    try:
        cle.execute_action(rig, close)
        assert rig.controller.gripper_closed is True
        assert presses.count(True) == 1 and presses[0] is True      # 最初の 1 ステップだけ押す
        presses.clear()
        cle.execute_action(rig, close)                               # 閉じている: 押さない
        assert presses.count(True) == 0 and rig.controller.gripper_closed is True
        cle.execute_action(rig, np.array([0, 0, 0, 0, 0, 0, -1.0]))
        assert rig.controller.gripper_closed is False
    finally:
        rig.controller._update_gripper = orig
    assert cle.gripper_target([0] * 6 + [0.3]) and not cle.gripper_target([0] * 6 + [-0.3])


def test_success_judge_rest_hold_and_time_limit():
    box = np.array([0.45, 0.25, 0.0])
    inside = box + [0.0, 0.0, 0.02]
    j = cle.SuccessJudge(box, 0.002)
    t = 0.0
    for _ in range(400):                           # 0.8 s 静止の後、動くと静止の時間は 0 に戻る
        t += 0.002
        j.update(inside, 0.0, t)
    j.update(inside, 0.05, t + 0.002)
    assert j.success_time is None and j.hold == 0.0
    t += 0.002
    for _ in range(500):
        t += 0.002
        j.update(inside, 0.005, t)
    assert j.success_time == pytest.approx(t, abs=1e-9)
    late = cle.SuccessJudge(box, 0.002)
    for i in range(500):
        late.update(inside, 0.0, 29.5 + 0.002 * (i + 1) + 0.1)
    assert late.success_time is None
    outside = cle.SuccessJudge(box, 0.002)
    for i in range(600):
        outside.update(box + [0.08, 0.0, 0.02], 0.0, 0.002 * (i + 1))
    assert outside.success_time is None


class _StubSampler:
    def __init__(self):
        self.scratch = type("S", (), {"qvel": np.zeros(20)})()

    def capture(self, data, step):
        frame = {k: np.zeros(3) for k in cle.TrialLog.FRAME_KEYS}
        frame["step"] = step
        return frame, [np.zeros((4, 4, 3), np.uint8)] * len(control.CAMERAS)


class _StubRig:
    def __init__(self, contacts):
        self.contacts = list(contacts)
        self.sampler = _StubSampler()
        self.cube_vadr = 9
        self.cube_id = 0
        self.box_pos = np.array([0.45, 0.25, 0.0])
        self.model = type("M", (), {"opt": type("O", (), {"timestep": 0.002})()})()
        self.data = type("D", (), {"xpos": np.zeros((1, 3)), "time": 0.0})()
        self.speeds = []

    def finger_contacts(self):
        return self.contacts.pop(0)

    def cube_linvel(self):
        return np.array([self.speeds.pop(0), 0.0, 0.0])


def test_contacts_are_any_within_the_window_not_the_last_value():
    stub = _StubRig([(False, False)] * 10 + [(True, False)] + [(False, False)] * 14 + [(False, True)] * 25)
    stub.speeds = [0.0] * 5 + [0.3] + [0.0] * 44
    log = cle.TrialLog(stub)
    log.capture(0)
    for _ in range(25):
        log.on_step()
    log.capture(25)                                  # 部分 1: 左の指が 1 度触れた（部分の終わりではない）
    for _ in range(25):
        log.on_step()
    log.capture(50)
    arr = log.arrays()
    assert arr["contact_left_any"].tolist() == [False, True, False]
    assert arr["contact_right_any"].tolist() == [False, False, True]
    assert arr["cube_speed_max"].tolist() == [0.0, 0.3, 0.0]


@pytest.mark.render
def test_trial_record_has_every_field(rig, tmp_path):
    hold = np.array([0, 0, 0, 0, 0, 0, -1.0])
    log = cle.run_trial(rig, placement4(), lambda k, frame, raw: hold, time_limit_s=0.5)
    out = tmp_path / "日本語"                          # ASCII でない道でも映像を書ける
    rec = cle.save_trial(out, 3, log, {"mode": "test", "seed": 100123,
                                       "placement": cle.placement_attrs(placement4())},
                         policy_frames=[np.zeros((8, 16, 3), np.uint8)] * 3)
    z = np.load(out / "trial_0003.npz")
    assert len(z["step"]) == 11 and z["step"][-1] == 250
    for key in ("ee_pos", "gripper_cmd", "fingers", "cube_pos", "cube_quat", "cube_linvel", "cube_angvel",
                "action", "contact_left_any", "contact_right_any", "cube_speed_max"):
        assert key in z.files
    assert np.isnan(z["action"][-1]).all() and not np.isnan(z["action"][:-1]).any()
    j = json.loads((out / "trial_0003.json").read_text(encoding="utf-8"))
    assert j["constants"]["table_top_z"] == 0.0 and j["constants"]["box_wall_top_z"] == recorder.BOX_WALL_TOP_Z
    assert j["seed"] == 100123 and j["success"] is False
    assert (out / "trial_0003_raw.mp4").stat().st_size > 0 and (out / "trial_0003_policy_input.mp4").is_file()


def test_open_loop_plot_draws_recorded_and_predicted(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image
    t = np.linspace(0, 6, 60)
    rec = np.zeros((60, 7))
    rec[:, 0] = 0.003 * np.sin(t)
    rec[:, 6] = np.where(t > 3, 1.0, -1.0)
    pred = rec.copy()
    pred[:, 0] += 0.0005
    cle.plot_open_loop(rec, pred, tmp_path / "ol.png", "test")
    img = np.asarray(Image.open(tmp_path / "ol.png").convert("RGB"))
    red = (img[..., 0] > 200) & (img[..., 1] < 60) & (img[..., 2] < 60)
    black = img.sum(axis=2) < 30
    assert red.sum() > 100 and black.sum() > 100


def test_policy_refused_before_loading_when_spec_differs(tmp_path):
    ckpt = tmp_path / "run" / "checkpoints" / "000010" / "pretrained_model"
    ckpt.mkdir(parents=True)
    with pytest.raises(spec.ImageSpecError, match="not found"):
        cle.PolicyActions(ckpt)
    (tmp_path / "run" / "conversion.json").write_text(json.dumps({"converter_version": 1}), encoding="utf-8")
    with pytest.raises(spec.ImageSpecError, match="image_spec_version"):
        cle.PolicyActions(ckpt)


def test_run_refuses_existing_output_dir(tmp_path, monkeypatch):
    """--out が既にあれば、方策を読む前に断る。"""
    out = tmp_path / "eval_run"
    out.mkdir()

    def never(*args, **kwargs):
        raise AssertionError("the policy must not be loaded when --out already exists")

    monkeypatch.setattr(cle, "PolicyActions", never)
    with pytest.raises(FileExistsError, match="not appended to"):
        cle.main(["run", "--checkpoint", str(tmp_path / "ckpt"), "--out", str(out),
                  "--eval-seeds", "100000", "100000"])
    assert list(out.iterdir()) == []


@pytest.mark.torch
def test_input_check_catches_swapped_or_missing_views():
    torch = pytest.importorskip("torch")
    pa = cle.PolicyActions.__new__(cle.PolicyActions)
    pa.torch = torch
    pa.rename = {"observation.images.image": "observation.images.camera1",
                 "observation.images.image2": "observation.images.camera2"}
    rng = np.random.default_rng(0)
    obs = {k: rng.random((3, 4, 4)).astype(np.float32) for k in spec.IMAGE_KEYS.values()}
    good = {"observation.images.camera1": torch.from_numpy(obs["observation.images.image"])[None],
            "observation.images.camera2": torch.from_numpy(obs["observation.images.image2"])[None],
            "observation.state": torch.zeros(1, 15)}
    assert pa.check_inputs(obs, good)["wrist"]["equals_view"]
    swapped = {**good, "observation.images.camera1": good["observation.images.camera2"],
               "observation.images.camera2": good["observation.images.camera1"]}
    with pytest.raises(RuntimeError):
        pa.check_inputs(obs, swapped)
    missing = {k: v for k, v in good.items() if k != "observation.images.camera2"}
    with pytest.raises(RuntimeError):
        pa.check_inputs(obs, missing)
