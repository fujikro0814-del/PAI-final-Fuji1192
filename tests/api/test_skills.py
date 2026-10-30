"""task API の技能（手で書いたスクリプト方策 api/skills.ScriptSkill と段階の決まり api/skills_script）の確かめ。
段階の決め方を合成の WorldEstimate（物が机の上 → 手の中 → 落ちた）で固定し、運動学だけの偽の世界（tests/api/kin_world.py）で
置く・落として掴み直す・評価の実行器（runtime/executor.TaskRuntime）で 3 個を片付ける、を確かめる。CPU だけ（MuJoCo・真値なし）。"""
import json
import types

import numpy as np
import pytest

from recovla.api import perception as AP
from recovla.api import skills as SK
from recovla.api import skills_script as SS
from recovla.api import task_spec as TS
from recovla.common import config
from recovla.runtime.executor import TaskRuntime
from recovla.runtime.runner import PolicyRuntime

from . import kin_world as K

CFG = config.load()
E = CFG["expert"]
RULES = SS.ScriptRules.from_config(E, E["phase"])
SPEC = TS.load_task()
FMT = CFG["convert"]["instruction"]
BOX = AP.PlaceState(np.array([0.45, 0.25]), 0.0, True, None, K.box_region())
CUBE = np.array([0.45, -0.10, 0.02])
RETREAT = np.array(E["retreat_pose"])


def test_rules_read_the_expert_config():
    r = RULES
    assert (r.grasp_z, r.release_z, r.carry_z, r.approach_z, r.lift_m) == (0.115, 0.18, 0.25, 0.25, 0.05)
    assert (r.xy_max, r.z_max, r.gain, r.move_tol, r.min_grip_gap) == (0.18, 0.09, 4.0, 0.001, 0.02)
    assert r.retreat_pose == tuple(E["retreat_pose"]) and r.final is None
    assert not any(k in r.to_dict() for k in ("slots", "slot_radius", "cube_size"))        # 既知の置き場所・大きさは持たない
    v3 = config.load("expert_v3")["expert"]
    if v3.get("final_approach"):
        assert SS.ScriptRules.from_config(v3, v3["phase"]).final is not None
    assert [p.name for p in SS.Phase][:10] == ["approach", "descend", "close", "reopen", "lift", "carry", "release", "settle",
                                               "retreat", "done"]


def cube(pos, status="seen", in_box=False, near=None, size=0.04):
    return AP.ObjState(np.asarray(pos, float), 0.0, status, {"box": in_box}, near, 0.0, np.full(3, size), np.eye(3),
                       K.SIGMA ** 2 * np.eye(3))


def obs(hand, target=CUBE, status="seen", in_box=False, near=None, closed=False, width=0.08, x_cmd=None, others=None,
        place=BOX, size=0.04):
    hand = np.asarray(hand, float)
    tgt = None if target is None else cube(target, status, in_box, near, size)
    return SS.Obs(t=0.0, target=tgt, others=others or {}, place=place, place_name="box", closed=closed, width=width, width_t=0.0,
                  hand=hand, tip=hand + [0, 0, -0.1034], x_cmd=hand if x_cmd is None else np.asarray(x_cmd, float),
                  table_z=0.0, est_t=0.0)


GRASP_Z = 0.02 + 0.1034                                                       # 指先の中心を物の推定の中心の高さへ
RELEASE_Z = K.RIM + (0.5 + RULES.clear_frac) * 0.04 + 0.1034                 # 持った物の下端が縁 ＋ 余裕


def rested():
    c = SS.Clocks()
    c.target_rest_s = 1.0
    return c


@pytest.mark.parametrize("name, o, phase", [
    ("机の上・遠い", obs([0.33, 0.15, 0.30]), "approach"),
    ("真上・高い", obs([0.45, -0.10, 0.25]), "descend"),
    ("推定の揺れの中（1 cm）", obs([0.455, -0.092, 0.25]), "descend"),
    ("真上・把持の高さ", obs([0.45, -0.10, GRASP_Z]), "close"),
    ("真上・まだ把持の高さの上", obs([0.45, -0.10, GRASP_Z + 0.01]), "descend"),
    ("閉じて挟んだ・まだ低い", obs([0.45, -0.10, GRASP_Z], status="occluded", near=True, closed=True, width=0.039), "lift"),
    ("閉じている途中（幅 6 cm）", obs([0.45, -0.10, GRASP_Z], closed=True, width=0.06), "lift"),
    ("持ち上げた", obs([0.45, -0.10, 0.17], target=[0.45, -0.10, 0.067], closed=True, width=0.039), "carry"),
    ("箱の上・解放の高さ", obs([0.49, 0.29, RELEASE_Z], status="occluded", near=True, closed=True, width=0.039), "release"),
    ("箱の上・解放の高さより上", obs([0.49, 0.29, RELEASE_Z + 0.02], status="occluded", near=True, closed=True, width=0.039),
     "carry"),
    ("運ぶ途中で落ちた（幅 0）", obs([0.45, 0.0, 0.25], closed=True, width=0.0), "reopen"),
    ("写った物が指先から遠い", obs([0.45, 0.10, 0.25], closed=True, width=0.039), "reopen"),
    ("落ちて開いた後", obs([0.45, 0.0, 0.25]), "approach"),
    ("箱の中・待機位置の外", obs([0.49, 0.29, 0.18], target=[0.49, 0.29, 0.03], in_box=True), "retreat"),
    ("箱の中・待機位置", obs(RETREAT, target=[0.49, 0.29, 0.03], in_box=True), "done"),
    ("推定が無い", obs([0.33, 0.15, 0.30], target=None), "search"),
    ("見失った", obs([0.33, 0.15, 0.30], status="lost"), "search"),
    ("放した直後で宙", obs([0.49, 0.29, 0.18], target=[0.3, 0.0, 0.1]), "settle"),
])
def test_phase_of_synthetic_estimates(name, o, phase):
    c = rested()
    c.obj_h = 0.04
    assert SS.phase_of(o, c, RULES).name == phase, name


def test_thresholds_scale_with_the_estimated_size():
    """閾値は推定の大きさからの相対（既知の 4 cm を使わない）: 大きい物ほど真上・宙・遠いの幅が広く、掴む・放す高さが上がる。"""
    small, big = obs([0.462, -0.10, 0.25], size=0.04), obs([0.462, -0.10, 0.25], target=[0.45, -0.10, 0.04], size=0.08)
    assert SS.phase_of(small, rested(), RULES) == SS.Phase.approach                 # 12 mm ずれ: 4 cm の物には真上でない
    assert SS.phase_of(big, rested(), RULES) == SS.Phase.descend                    # 8 cm の物には真上
    assert SS.grasp_hand_z(big, RULES) == pytest.approx(0.04 + 0.1034)
    assert SS.grasp_hand_z(obs([0.4, 0, 0.3], target=[0.4, 0, 0.005], size=0.01), RULES) == RULES.grasp_z   # 下限は config
    assert SS.release_hand_z(big, RULES, 0.08) - SS.release_hand_z(small, RULES, 0.04) == pytest.approx(0.75 * 0.04)
    assert SS.release_hand_z(obs([0.4, 0, 0.3], place=AP.PlaceState(np.zeros(2), 0.0, True, 0.08)), RULES, 0.04) == RULES.release_z
    assert SS.above_tol(small, RULES) == pytest.approx(0.25 * 0.04 + 2 * K.SIGMA)
    assert SS.grasp_tol(big, RULES) == pytest.approx(0.1 * 0.08 + 1 * K.SIGMA)
    unknown = SS.Obs(**{**small.__dict__, "target": AP.ObjState(CUBE, 0.0, "seen", {"box": False}, None)})
    assert SS.above_tol(unknown, RULES) == RULES.above_tol and SS.grasp_tol(unknown, RULES) == RULES.move_tol


def test_moving_or_unrested_target_settles():
    c = SS.Clocks()
    c.target_speed = 0.05
    assert SS.phase_of(obs([0.33, 0.15, 0.3]), c, RULES) == SS.Phase.settle
    c = SS.Clocks()                                                        # 箱の中でまだ静止していない
    assert SS.phase_of(obs([0.49, 0.29, 0.18], target=[0.49, 0.29, 0.03], in_box=True), c, RULES) == SS.Phase.settle


def test_fingers_count_as_settled_only_after_they_moved():
    """閉じる指令の直後は 10 Hz の幅の読みが開いたまま（速さ 0）。指が動いてから止まったときだけ「止まった」と数える
    （読みが止まっただけで持ち上げると掴み損ねる）。"""
    c = SS.Clocks()
    base = obs([0.45, -0.10, GRASP_Z])
    for i in range(30):                                                     # 閉じた後 0.6 s、読みは 8 cm のまま
        c.update(SS.Obs(**{**base.__dict__, "t": 0.02 * i, "closed": True, "width": 0.08, "width_t": 0.1 * (i // 5)}), 0.02, RULES)
    assert c.finger_rest_s == 0.0 and not c.moved
    for i, w in enumerate((0.07, 0.05, 0.040, 0.040, 0.040, 0.040, 0.040, 0.040)):
        for j in range(5):
            k = 30 + 5 * i + j
            c.update(SS.Obs(**{**base.__dict__, "t": 0.02 * k, "closed": True, "width": w, "width_t": 0.1 * (k // 5)}), 0.02, RULES)
    assert c.moved and c.finger_rest_s > 0.0


def test_slots_far_from_the_fixed_camera_come_first():
    """固定のカメラ（較正の値）から遠い置き場所から使う: 手前の壁に隠れず、置いた物を確かめやすい。"""
    b = SS.ScriptBrain(RULES, dt=0.02, view_from=(0.45, -0.55))
    order = b._order(BOX)
    far = [SS.slot_xy(BOX, s)[1] for s in order]
    assert far[0] >= far[-1] and min(far[:2]) > max(far[2:])                # +y（カメラから遠い側）の 2 つが先
    assert SS.ScriptBrain(RULES, dt=0.02)._order(BOX) == (0, 1, 2, 3)        # カメラを渡さなければ番号の順


def test_slot_choice_skips_occupied_and_turns_with_the_box():
    other = {"green": cube([0.49, 0.29, 0.03], in_box=True)}
    o = obs([0.45, 0.0, 0.25], others=other)
    assert SS.choose_slot(o, RULES, (0, 1, 2, 3)) == 1
    assert np.allclose(SS.slot_xy(BOX, 0), [0.49, 0.29])                          # 推定の内寸 8 cm の半分の半分
    turned = AP.PlaceState(np.array([0.45, 0.25]), np.pi / 2, True, None, K.box_region(yaw=np.pi / 2))
    assert np.allclose(SS.slot_xy(turned, 0), [0.41, 0.29])
    wide = AP.PlaceState(np.array([0.45, 0.25]), 0.0, True, None, K.box_region(inner=0.12))
    assert np.allclose(SS.slot_xy(wide, 3), [0.39, 0.19])                         # 箱が大きければ置き場所も広がる


def brain_steps(o, n=20, brain=None):
    b = brain or SS.ScriptBrain(RULES, dt=0.02)
    cmd = None
    for i in range(n):
        cmd = b.step(SS.Obs(**{**o.__dict__, "t": 0.02 * i, "est_t": 0.1 * (i // 5), "width_t": 0.1 * (i // 5)}))
    return cmd, b


def test_brain_closes_only_when_aligned_and_still_and_opens_over_the_slot():
    cmd, _ = brain_steps(obs([0.45, -0.10, GRASP_Z]))
    assert cmd.phase == SS.Phase.close and cmd.closed and np.allclose(cmd.vel, 0)
    cmd, _ = brain_steps(obs([0.45, -0.094, GRASP_Z]))                     # 把持点から 6 mm（許しは 4 mm ＋ σ）: まだ寄せる
    assert cmd.phase == SS.Phase.close and not cmd.closed and cmd.vel[1] < 0
    hand = np.array([0.49, 0.29, RELEASE_Z])
    b = SS.ScriptBrain(RULES, dt=0.02)
    b.clocks.z_close = 0.126                                                # 把持の高さで閉じた後
    cmd, _ = brain_steps(obs(hand, status="occluded", near=True, closed=True, width=0.039), brain=b)
    assert cmd.phase == SS.Phase.release and not cmd.closed                 # 置き場所の真上・止まっている: 開く
    cmd, _ = brain_steps(obs(hand, status="occluded", near=True, closed=True, width=0.039))
    assert cmd.phase == SS.Phase.lift                                       # その高さで閉じたばかり: まだ持ち上げていない
    cmd, _ = brain_steps(obs([0.45, 0.0, 0.25], closed=True, width=0.0))
    assert cmd.phase == SS.Phase.reopen and not cmd.closed and cmd.vel[2] == pytest.approx(RULES.z_max)
    cmd, _ = brain_steps(obs([0.33, 0.15, 0.13], target=None))
    assert cmd.phase == SS.Phase.search and cmd.vel[2] > 0 and np.allclose(cmd.vel[:2], 0)


# ---------------------------------------------------------------- 運動学だけの偽の世界で回す
def make_skill(world, **kw):
    return SK.ScriptSkill(world, K.setup(), RULES, SPEC, FMT, motion=K.KinMotion(world), **kw)


def phases(skill):
    return [x["phase"] for x in skill.log_script if "phase" in x]


def squeeze(ph):
    """close より前の approach・descend の入れ替わり（真上の幅の境での一瞬の行き過ぎ）を 1 回にまとめる。"""
    i = ph.index("close") if "close" in ph else len(ph)
    head = [p for k, p in enumerate(ph[:i]) if k == 0 or p != ph[k - 1]]
    while len(head) > 2 and head[-3:] == ["descend", "approach", "descend"]:
        head = head[:-2]
    while len(head) > 2 and head[:3] == ["approach", "descend", "approach"]:
        head = head[2:]
    return head + ph[i:]


def three_cubes():
    return K.KinWorld({"red": (0.45, -0.10, 0.02), "green": (0.40, 0.0, 0.02), "blue": (0.50, -0.05, 0.02)})


def test_skill_has_the_policy_runtime_ports():
    w = three_cubes()
    sk = make_skill(w)
    assert isinstance(sk, PolicyRuntime) and SK.VLASkill is PolicyRuntime
    sk.start(FMT.format(color="red"), 1)                                    # wm・ready などは start が作る（PolicyRuntime と同じ）
    for name in SK.EXECUTOR_IO + SK.EXECUTOR_ALSO:
        assert hasattr(sk, name), name
    assert sk.ready and sk.stop_reason is None and sk.step == TS.StepSpec("pick_place", "red", "box")
    sk.set_task(FMT.format(color="blue"), 2)
    assert sk.step.obj == "blue"


def test_skill_places_a_cube_in_the_box():
    w = three_cubes()
    sk = make_skill(w)
    sk.start(FMT.format(color="red"), 1)
    K.run(sk, w, 20)
    assert phases(sk) == ["approach", "descend", "close", "lift", "carry", "release", "settle", "retreat", "done"]
    assert np.allclose(w.cubes["red"][:2], K.BOX_XY + [0.04, 0.04], atol=0.002)          # 1 つ目の置き場所
    assert [c for c, _ in w.calls] == ["grasp", "move"]
    tr = sk.trace()
    assert tr["script"]["skill"] == "script" and tr["inference"] == [] and len(tr["actions"]) > 100
    json.dumps(tr, default=str)


def test_skill_regrasps_after_a_drop_and_uses_a_free_slot():
    w = three_cubes()
    sk = make_skill(w)
    sk.start(FMT.format(color="red"), 1)
    st = {"dropped": None}

    def drop(world):
        if st["dropped"] is None and world.t > 6.0 and world.held == "red":
            world.drop()
            st["dropped"] = world.t
    K.run(sk, w, 30, every=drop)
    ph = phases(sk)
    assert st["dropped"] is not None
    i = ph.index("reopen")
    assert ph[:5] == ["approach", "descend", "close", "lift", "carry"] and set(ph[5:i]) <= {"settle"}   # 落ちている間は待つ
    assert "close" in ph[i:] and ph[-1] == "done"                                      # 掴み直して置いた
    assert [c for c, _ in w.calls] == ["grasp", "move", "grasp", "move"]
    assert np.allclose(w.cubes["red"][:2], K.BOX_XY + [0.04, 0.04], atol=0.002)
    sk.set_task(FMT.format(color="green"), 2)
    K.run(sk, w, 25)
    assert np.allclose(w.cubes["green"][:2], K.BOX_XY + [0.04, -0.04], atol=0.002)      # 埋まっていない次の置き場所


def test_skill_holds_still_when_external_or_not_ready():
    w = three_cubes()
    sk = make_skill(w)
    sk.start(FMT.format(color="red"), 1)
    sk.external = True
    x0 = w.hand.copy()
    K.run(sk, w, 1.0)
    assert np.allclose(w.hand, x0) and phases(sk) == []
    sk.external = False
    sk.ready = False
    K.run(sk, w, 1.0)
    assert np.allclose(w.hand, x0) and phases(sk) == []


def test_skill_with_the_real_motion_ik_and_limiter(capsys):
    """腕は本物の runtime/motion（IK・制限層、ロボットだけのモデル）。測った関節角＝指令（追従の遅延なし）の世界で、
    把持の高さで手先が参照より上に止まっても、段階を飛ばさずに置ける。"""
    from recovla.harness.setup import nominal_setup                          # テストの道具（既知の情報の公称値）
    from recovla.runtime.motion import Motion
    cfg = config.load(*config.V2_LAYERS)
    su = nominal_setup(cfg)
    fk = Motion(su)
    w = K.ArmWorld({"red": (0.45, -0.10, 0.02), "green": (0.40, 0.0, 0.02), "blue": (0.50, -0.05, 0.02)},
                   cfg["sim"]["home_qpos"], lambda q: fk.hand_pose(q)[0])
    sk = SK.ScriptSkill(w, su, RULES, SPEC, FMT, motion=Motion(su, margin=float(cfg["actuation"]["limiter_margin"])),
                        tip_offset=float(cfg["sim"]["fingertip_offset"]))
    sk.start(FMT.format(color="red"), 1)
    K.run(sk, w, 30, until=lambda: sk.last_phase == SS.Phase.done)
    capsys.readouterr()                                                       # 制御器の途中の表示は捨てる
    assert squeeze(phases(sk)) == ["approach", "descend", "close", "lift", "carry", "release", "settle", "retreat", "done"]
    assert sk.grasp_log and sk.grasp_log[0]["width"] == pytest.approx(0.04)          # grasp の幅は推定の短い幅
    assert np.allclose(w.cubes["red"][:2], K.BOX_XY + [0.04, 0.04], atol=0.004)
    tr = sk.trace()["motion"]
    assert tr["cart_unresolved"] == 0 and tr["leash"] == 0


class KinJudge:
    """完了判定の代わり: 狙いの色が箱の中で、手が待機位置のそば・指が開いている。"""

    def __init__(self, world):
        self.w = world
        self.p = {"gripper_open_m": 0.075, "retreat_tol_m": 0.05, "box_min_pixels": 56}
        self.color, self.last = None, {}

    def reset(self, color):
        self.color = color

    def update(self, sensor, wm, hand):
        w = self.w
        p = w.cubes[self.color]
        ok = (bool(np.max(np.abs(p[:2] - K.BOX_XY)) <= K.INNER) and w.held is None and w.width >= 0.075
              and float(np.linalg.norm(w.hand - RETREAT)) < 0.05)
        self.last = {"ok": ok, "held_s": 0.0, "box_pixels": 0, "wrist_in_pixels": 0, "depth_ok": True,
                     "retreat_dist_m": float(np.linalg.norm(w.hand - RETREAT)), "gripper_open": w.width >= 0.075}
        return ok


def test_script_skill_runs_under_the_current_task_runtime():
    """評価の実行器（TaskRuntime。書き換えない）に PolicyRuntime の代わりに差し込んで、3 個を片付ける。"""
    w = three_cubes()
    sk = make_skill(w)
    plan = {}

    def decompose(text, table, box):
        plan.update(text=text, table=list(table), box=list(box))
        return {"steps": [c for c in ("red", "green", "blue") if c in table], "reply": "赤・緑・青の順に入れます"}
    cfg = {"step_timeout_s": 30.0, "retry": 1,
           "return_to_retreat": {"enabled": True, "rise_z": 0.30, "trigger_s": 4.0, "wait_s": 2.0, "timeout_s": 10.0}}
    mp = {"gain": 4.0, "xy_max": 0.18, "z_max": 0.09, "z_tol": 0.001, "tol": 0.003}
    ex = TaskRuntime(w, K.setup(), sk, KinJudge(w), cfg, mp, decompose, FMT)
    ex.start("全部片付けて", 5)
    K.run(ex, w, 120, skill=sk, until=lambda: ex.finished)
    rec = ex.record()
    assert ex.finished and rec["stopped"] is None
    assert plan == {"text": "全部片付けて", "table": ["blue", "green", "red"], "box": []}
    assert [s["color"] for s in rec["steps"]] == ["red", "green", "blue"]
    assert all(s["judged_complete"] for s in rec["steps"]) and all(len(s["attempts"]) == 1 for s in rec["steps"])
    for c in ("red", "green", "blue"):
        assert np.max(np.abs(w.cubes[c][:2] - K.BOX_XY)) < K.INNER, c
    slots = sorted(tuple(np.round(w.cubes[c][:2] - K.BOX_XY, 2)) for c in ("red", "green", "blue"))
    assert len(set(slots)) == 3                                                       # 別々の置き場所


def test_known_value_finder_runs_the_same_script():
    """比べ用の既知の値の版（ColorWorldModelFinder: cube_size・箱の寸法から大きさと領域を作る）でも同じ口で置ける。"""
    w = three_cubes()
    sk = make_skill(w, finder="color")
    sk.start(FMT.format(color="red"), 1)
    K.run(sk, w, 20)
    assert phases(sk) == ["approach", "descend", "close", "lift", "carry", "release", "settle", "retreat", "done"]
    assert np.allclose(w.cubes["red"][:2], K.BOX_XY + [0.04, 0.04], atol=0.002)
    assert sk.trace()["script"]["finder"] == "color"


def test_estimated_finder_needs_geometry_perception():
    w = three_cubes()
    with pytest.raises(ValueError, match="GeometryPerception"):
        make_skill(w, perception=types.SimpleNamespace(record_joints=lambda j: None))
    with pytest.raises(ValueError):
        make_skill(w, finder="truth")
