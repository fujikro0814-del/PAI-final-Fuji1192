"""ScriptSkill の確かめ用の偽の世界（運動学だけ。MuJoCo も真値のモデルも使わない）。

手先は参照どおりに動き（Motion の代わり）、ハンドは閉じると物を挟めば幅 0.039 m で止まり、挟めなければ 0 へ。
閉じて挟んだ物は指先に付いて動き、開くか drop で落ちて机（箱の中なら箱の底）に置かれる。知覚は WorldModel の形
（cubes・box）を毎 0.1 s 作って skill.wm に入れる（隠れ・遅延なし、立方体の位置は真の位置）。
"""
import types

import numpy as np

from recovla.api import perception as AP
from recovla.api.geometry import BoxGeom
from recovla.runtime.types import GripperState, JointState, SensorFrame

DOWN = np.array([0.0, 1.0, 0.0, 0.0])            # 手先の z 軸が下を向く四元数（指先 = 手先 − 0.1034 z）
TIP = 0.1034
BOX_XY = np.array([0.45, 0.25])
INNER = 0.08
RIM = 0.06
SIGMA = 0.001                                     # 推定の中心の標準偏差（偽の知覚の印）


def box_region(xy=BOX_XY, yaw=0.0, inner=INNER, rim=RIM, floor_t=0.01) -> BoxGeom:
    """偽の知覚が返す箱の推定（内寸の半分 inner、縁の高さ rim）。"""
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return BoxGeom(np.array([xy[0], xy[1], rim]), R, np.array([inner + 0.005] * 2), np.array([inner] * 2), 0.005, rim,
                   floor_t - rim, 0.0, 0.0, {"center": SIGMA}, 1000, 300)


class Ready:
    def __init__(self, value):
        self.value = value

    def ready(self, t):
        return True

    def result(self, t):
        return self.value


class KinMotion:
    def __init__(self, world):
        self.w, self.dt, self.v = world, 0.002, np.zeros(3)
        self.x = world.hand.copy()
        self.lag_stats = {}
        self.n_cart_clipped = self.n_xcmd_leashed = self.n_cart_unresolved = 0
        self.xcmd_leash_m = None
        self.limiter = types.SimpleNamespace(n_pos_clipped=0, n_clipped=0)

    def reset(self, joints):
        self.x = np.asarray(joints.q[:3], float).copy()
        self.v = np.zeros(3)

    def set_velocity(self, v):
        self.v = np.asarray(v, float).copy()

    @property
    def x_cmd(self):
        return self.x.copy()

    def hand_pose(self, q):
        return np.asarray(q[:3], float).copy(), DOWN.copy()

    def step(self, joints):
        self.x = self.x + self.v * self.dt
        self.x[2] = max(self.x[2], 0.115)                  # 作業域の下限（configs の sim.workspace）
        q = np.zeros(7)
        q[:3] = self.x
        return [q]


class KinWorld:
    def __init__(self, cubes: dict, hand=(0.33, 0.15, 0.30)):
        self.t = 0.0
        self.hand = np.asarray(hand, float)
        self.cubes = {c: np.asarray(p, float) for c, p in cubes.items()}
        self.width, self.closed, self.held = 0.08, False, None
        self.w_t = 0.0
        self.calls = []
        self.with_map = False

    # ---------------------------------------------------------------- RobotIO
    def now(self):
        return self.t

    def tip(self):
        return self.hand + np.array([0.0, 0.0, -TIP])

    def sense(self, cameras=True):
        q = np.zeros(7)
        q[:3] = self.hand
        cams = {n: types.SimpleNamespace(name=n, t_capture=self.t) for n in ("overhead", "wrist")} if cameras else {}
        return SensorFrame(self.t, JointState(self.t, q, np.zeros(7), q.copy()),
                           GripperState(self.w_t, self.width, self.held is not None), cams)

    def command_joints(self, q):
        self.hand = np.asarray(q[:3], float).copy()

    def gripper_grasp(self, width, speed, force, ei, eo):
        self.calls.append(("grasp", round(self.t, 3)))
        self.closed = True

    def gripper_move(self, width, speed):
        self.calls.append(("move", round(self.t, 3)))
        self.closed = False
        self.release()

    def compute(self, kind, fn):
        """計算の口（応答時間 0。実行器の LLM の計画だけが使う）。"""
        self.calls.append(("compute", kind))
        return Ready(fn())

    # ---------------------------------------------------------------- 物理の代わり
    def release(self):
        if self.held is None:
            return
        c, self.held = self.held, None
        p = self.cubes[c]
        inside = np.max(np.abs(p[:2] - BOX_XY)) <= INNER
        self.cubes[c] = np.array([p[0], p[1], (0.01 if inside else 0.0) + 0.02])

    def drop(self):
        """持っている物を落とす（指は閉のまま、幅は 0 へ）。"""
        if self.held is not None:
            c, self.held = self.held, None
            p = self.cubes[c]
            self.cubes[c] = np.array([p[0], p[1], 0.02])

    def step(self, dt=0.002):
        self.t += dt
        if self.closed and self.held is None and self.width > 0.039:
            tip = self.tip()
            for c, p in self.cubes.items():
                if np.hypot(*(tip[:2] - p[:2])) < 0.01 and abs(tip[2] - p[2]) < 0.025:
                    self.held = c
        if self.held is not None:
            self.cubes[self.held] = self.tip().copy()
        target_w = 0.08 if not self.closed else (0.039 if self.held is not None else 0.0)
        if int(round(self.t / dt)) % 50 == 0:              # ハンドの読みは 10 Hz
            self.width += float(np.clip(target_w - self.width, -0.01, 0.01))
            self.w_t = self.t

    def wm(self):
        """WorldModel の形（色の知覚）に、推定の版の WorldEstimate（geometry）を付けたもの（GeometryPerception と同じ形）。"""
        cubes, objs = {}, {}
        region = box_region()
        place = AP.PlaceState(BOX_XY.copy(), 0.0, True, None, region)
        for c, p in self.cubes.items():
            inside = bool(np.max(np.abs(p[:2] - BOX_XY)) <= INNER - 0.005) and self.held != c
            st = "in_hand" if self.held == c else "seen"
            cubes[c] = types.SimpleNamespace(pos=p.copy(), yaw=0.0, status=st, in_box=inside, t_seen=self.t)
            objs[c] = AP.ObjState(p.copy(), 0.0, "seen", {"box": region.depth(p) >= 0}, self.held == c or None, self.t,
                                  np.full(3, 0.04), np.eye(3), SIGMA ** 2 * np.eye(3))
        wm = types.SimpleNamespace(t=self.t, table_z=0.0, cubes=cubes, box={"xy": BOX_XY.copy(), "yaw": 0.0, "ok": True})
        wm.geometry = AP.WorldEstimate(self.t, objs, {"box": place}, 0.0, "estimated")
        if self.with_map:                                # 地図を使う確かめだけ（重いので既定は作らない）
            wm.heightmap = heightmap_of(self)
        return wm


def setup():
    return types.SimpleNamespace(table_z=0.0, cube_size=0.04, box_outer=2 * INNER + 0.01, box_wall=0.005, box_height=RIM,
                                 box_floor=0.01, gripper_speed=0.1, grasp_force=20.0, retreat_pose=np.array([0.33, 0.15, 0.30]))


def run(obj, world, seconds, every=None, skill=None, until=None):
    """obj.tick（技能か実行器）と世界を seconds 秒回す。every(world) を毎手呼ぶ（落とすなど）。知覚は 0.1 s ごとに
    skill（省略時は obj）の wm へ。until が真になったら止める。"""
    skill = obj if skill is None else skill
    n = int(round(seconds / 0.002))
    for i in range(n):
        if i % 50 == 0:
            skill.wm = world.wm()
        obj.tick()
        world.step()
        if every is not None:
            every(world)
        if until is not None and until():
            return


class ArmWorld(KinWorld):
    """腕だけ本物の運動学: 指令の関節角をそのまま測った関節角として返し（追従の遅延なし）、手先はロボットのモデルの順運動学
    （runtime/motion の hand_pose）。技能は本物の Motion（IK・制限層）で動かす。物とハンドは KinWorld と同じ。"""

    def __init__(self, cubes: dict, q0, fk):
        self.q = np.asarray(q0, float).copy()
        self.fk = fk
        super().__init__(cubes, hand=fk(self.q))

    def sense(self, cameras=True):
        cams = {n: types.SimpleNamespace(name=n, t_capture=self.t) for n in ("overhead", "wrist")} if cameras else {}
        return SensorFrame(self.t, JointState(self.t, self.q.copy(), np.zeros(7), self.q.copy()),
                           GripperState(self.w_t, self.width, self.held is not None), cams)

    def command_joints(self, q):
        self.q = np.asarray(q, float).copy()
        self.hand = self.fk(self.q)


def heightmap_of(world, cell=0.01, frame=None):
    """偽の世界の高さの地図（机 0・箱の縁 RIM・箱の底 0.01・立方体の上面。持っている物は外す）。api/heightmap.HeightMap。"""
    from recovla.api import geometry as G
    from recovla.api.heightmap import HeightMap
    hm = HeightMap(frame or G.TableFrame.from_plane([0.0, 0.0, 1.0], 0.0), cell=cell, noise=SIGMA)
    xs = np.arange(0.20, 0.70, cell) + 0.5 * cell
    ys = np.arange(-0.35, 0.45, cell) + 0.5 * cell
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    d = np.maximum(np.abs(X - BOX_XY[0]), np.abs(Y - BOX_XY[1]))
    Z[d <= INNER + 0.005] = RIM
    Z[d <= INNER] = 0.01
    for c, p in world.cubes.items():
        if world.held == c:
            continue
        m = (np.abs(X - p[0]) <= 0.02) & (np.abs(Y - p[1]) <= 0.02)
        Z[m] = np.maximum(Z[m], p[2] + 0.02)
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    hm.update_points(np.vstack([pts, pts]), world.t)                 # 地図は升に 2 点以上あるときだけ置き換える
    return hm


# ---------------------------------------------------------------- カップと皿の偽の世界（別のタスク）
PLATE_XY = np.array([0.46, 0.24])
PLATE_R = 0.09
PLATE_TOP = 0.012
CUP_R, CUP_H = 0.03, 0.08


def plate_region(xy=PLATE_XY, radius=PLATE_R, top=PLATE_TOP):
    """偽の知覚が返す皿の推定（上面の中心・半径）。api/geometry_place.PlateGeom。"""
    from recovla.api.geometry_place import PlateGeom
    return PlateGeom(np.array([xy[0], xy[1], top]), np.eye(3), radius, top, 0.0, 0.0,
                     {"center": SIGMA, "radius": SIGMA, "top": SIGMA, "tilt": 0.0}, 1000, 36)


class KinPlateWorld(KinWorld):
    """KinWorld の物をカップ（外径 6 cm・高さ 8 cm、位置は中心）、箱を皿（机に固定の円板）にしたもの。閉じると挟めば幅
    0.059 m で止まる。放すと真下の支え（皿の上なら上面、ほかは机）に立てて置く。"""
    HELD_W = 0.059

    def release(self):
        if self.held is None:
            return
        c, self.held = self.held, None
        p = self.cubes[c]
        on = np.hypot(*(p[:2] - PLATE_XY)) <= PLATE_R
        self.cubes[c] = np.array([p[0], p[1], (PLATE_TOP if on else 0.0) + 0.5 * CUP_H])

    def drop(self):
        if self.held is not None:
            c, self.held = self.held, None
            p = self.cubes[c]
            self.cubes[c] = np.array([p[0], p[1], 0.5 * CUP_H])

    def step(self, dt=0.002):
        self.t += dt
        if self.closed and self.held is None and self.width > self.HELD_W:
            tip = self.tip()
            for c, p in self.cubes.items():
                if np.hypot(*(tip[:2] - p[:2])) < 0.01 and abs(tip[2] - p[2]) < 0.035:
                    self.held = c
                    self.grip_off = p - tip
        if self.held is not None:
            self.cubes[self.held] = self.tip() + self.grip_off
        target_w = 0.08 if not self.closed else (self.HELD_W if self.held is not None else 0.0)
        if int(round(self.t / dt)) % 50 == 0:
            self.width += float(np.clip(target_w - self.width, -0.01, 0.01))
            self.w_t = self.t

    def wm(self):
        from recovla.api import relations as REL
        region = plate_region()
        place = AP.PlaceState(PLATE_XY.copy(), 0.0, True, None, region)
        cubes, objs = {}, {}
        for c, p in self.cubes.items():
            o = AP.ObjState(p.copy(), 0.0, "seen", {}, self.held == c or None, self.t, np.array([2 * CUP_R, 2 * CUP_R, CUP_H]),
                            np.eye(3), SIGMA ** 2 * np.eye(3))
            on = bool(REL.on_surface(o, place)) and self.held != c
            objs[c] = AP.ObjState(o.pos, 0.0, "seen", {"plate": on}, o.near_hand, self.t, o.extent, o.R, o.cov)
            cubes[c] = types.SimpleNamespace(pos=p.copy(), yaw=0.0, status="in_hand" if self.held == c else "seen",
                                             in_box=on, t_seen=self.t)
        wm = types.SimpleNamespace(t=self.t, table_z=0.0, cubes=cubes, box={"xy": PLATE_XY.copy(), "yaw": 0.0, "ok": True})
        wm.geometry = AP.WorldEstimate(self.t, objs, {"plate": place}, 0.0, "estimated")
        if self.with_map:
            wm.heightmap = plate_heightmap_of(self)
        return wm


def plate_heightmap_of(world, cell=0.01, frame=None, extra=()):
    """偽の皿の世界の高さの地図（机 0・皿の上面・カップの縁。持っている物は外す）。extra: [(xy, 半径, 上端の高さ)] を足す。"""
    from recovla.api import geometry as G
    from recovla.api.heightmap import HeightMap
    hm = HeightMap(frame or G.TableFrame.from_plane([0.0, 0.0, 1.0], 0.0), cell=cell, noise=SIGMA)
    xs = np.arange(0.20, 0.70, cell) + 0.5 * cell
    ys = np.arange(-0.35, 0.45, cell) + 0.5 * cell
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    Z[np.hypot(X - PLATE_XY[0], Y - PLATE_XY[1]) <= PLATE_R] = PLATE_TOP
    items = [(p[:2], CUP_R, p[2] + 0.5 * CUP_H) for c, p in world.cubes.items() if world.held != c] + list(extra)
    for xy, r, top in items:
        m = np.hypot(X - xy[0], Y - xy[1]) <= r
        Z[m] = np.maximum(Z[m], top)
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    hm.update_points(np.vstack([pts, pts]), world.t)
    return hm
