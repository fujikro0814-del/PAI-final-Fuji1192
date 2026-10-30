"""落下の反射（grasp-loss reflex）。持っていた立方体を落としたことを、ロボット自身のセンサだけで見つけ、古い行動列を捨てて
手をその場に止め、指を開いて待ち、新しい観測から推論し直す。既定は切（PolicyRuntime(reflex=None)）。

    from recovla.runtime.reflex import ReflexParams, GraspLossReflex
    rx = GraspLossReflex(ReflexParams)          # 試行ごとに start_trial（PolicyRuntime.start が呼ぶ）
    rx.observe(t, k, width, intent_closed, hand=hand, x_cmd=x_cmd, sample_t=g.t)   # ハンドの新しい読みごと。発火で真
    rx.hold_action(x_cmd)                         # 発火の後の行動（手先の差分 xyz と開閉。実機の口でそのまま出せる形）
    rx.blocking(t)                                # 待ちの間は真（推論を始めない）
    rx.accept_chunk(t, k, i, cue)                 # 待ちの後に届いた行動列を使うか（使うなら通常に戻る）

なぜ: 運んでいる最中に落ちても、方策の古い行動列（50 行、6 行ごとに推論、0.3 s の遅延）は箱へ運び続けて空の指を閉じる。
最初の新しい推論のときには、手は 13 cm ほど離れて閉じており、学習のデータにない状態になる。立て直しデータの落下の場面は
「手が立方体の上で止まり、指が開いている」状態から始まる。この反射は、実行をデータの状態に合わせる。

使う信号（実機の Franka Hand と腕にもある物だけ。シミュレーションの真値は使わない）:
- ハンドの開き幅（GripperState.width。2 本の指の和。実機は libfranka の GripperState.width）と、その時刻
- 方策が出した開閉の指令（行動の 7 番目。評価の道具が上書きする前の値）
- 測った関節角からの手先の位置（順運動学）と、実行系自身の手先の参照 x_cmd
- 画像の目標位置の入力（cue。任意。実機にカメラが無ければ使わない。既定は切）

検出（GraspLossDetector。状態は open → closing → holding → fired。読みはハンドの新しい読みごと）:
- 方策が閉を出している間に、開き幅が持つ範囲 [hold_lo_m, hold_hi_m] に入って前の読みとの差が stable_m 以下の読みが
  est_reads 回以上・est_s 秒以上続き、閉じ始めてから手先が min_lift_m 以上上がったら「持っている」。
- 持っている間に方策が開を出したら、意図した放し（発火しない。open に戻る）。
- 持っている間、方策が閉のままで:
  主: 開き幅が基準（持ち始めの値を追う）より loss_m 以上小さい読みが loss_reads 回続いたら発火（collapse。実機で
      滑って落ちると、grasp の指令は閉じ続けるので開き幅は 0 へ向かう）
  副: 基準より loss_m 以上大きい（open。シミュレーションの P2 の意図的な失敗は指を開いて落とす）。fire_on_open で入切
  空掴み: まだ持っていない（closing）間に、閉の指令のまま開き幅が miss_m 未満の状態が miss_hold_s 続いたら発火（miss。掴み損ね P1 の
      空振り。立方体があれば指は 3.8 cm 前後で止まる）。fire_on_miss で入切。発火の後は落下と同じく止めて開いて待つ
  任意: cue_jump_m（画像の目標位置の入力が手先から離れた。既定は切。誤発火が多い）

発火の後（GraspLossReflex）。出すのは実機の口でそのまま出せる指令だけ:
- 古い行動列と計算中の推論を捨てる（実行系の側）。
- 止める: stop="measured"（既定）は、発火の時点で測った手先の位置へ参照を戻してそこで保つ（実機の「今の姿勢を保つ」）。
  stop="freeze" は参照をその場で止める（速さ 0。腕の遅延の分だけ先で止まる）。
  参照を動かす速さは stop_speed 以下。
- 開く: 指を開く（ハンドの move。実機では grasp を stop してから move）。
- 待ち: 発火から settle_s たち、かつ開き幅が open_m 以上になるまで推論を始めない（max_wait_s で打ち切る）。
- 待ちの後は通常どおり推論する（腕は止まっているので行動列は 0 行目から使う＝runner の was_moving が偽）。cue_stable_m を
  決めたら、届いた行動列の目標位置の入力が見えていて 1 つ前と cue_stable_m 以内のときだけ使う（既定は切）。
- 行動列を使い始めたら通常に戻り、検出は open から数え直す。
記録（summary）: 引数、発火・待ちの終わり・再開の事象、持ったと判定した回数、意図した放しの回数。
"""
import dataclasses
import math

import numpy as np

ACTION_DT = 0.1


@dataclasses.dataclass
class ReflexParams:
    hold_lo_m: float = 0.030       # 持つ範囲の下限（2 本の和）[m]。4 cm の立方体を握ると 3.8〜3.9 cm
    hold_hi_m: float = 0.060       # 持つ範囲の上限 [m]。斜めに握った立方体（5.2 cm）も入れる
    est_reads: int = 3             # 持つと決めるまでの、範囲の中で止まった読みの回数
    est_s: float = 0.2             # 同じく続いた時間 [s]（読みの頻度によらないため）
    stable_m: float = 0.002        # 止まったとみなす前の読みとの差 [m]（雑音 0.5 mm を見込む）
    min_lift_m: float = 0.03       # 閉じ始めてからの手先の上がり [m]（測った関節角の順運動学）。0 なら見ない
    loss_m: float = 0.004          # 基準からこれだけ離れたら落としたとみなす [m]
    loss_reads: int = 1            # その読みが続く回数
    ref_alpha: float = 0.2         # 基準の追い方（範囲の中の読みへ寄せる割合）
    fire_on_open: bool = True      # 副の発火（開いた）を使うか
    fire_on_miss: bool = True      # 持つ前に、閉の指令のまま開き幅が miss_m 未満になった（空を掴んだ）ら発火するか
    miss_m: float = 0.025          # 空を掴んだとみなす開き幅 [m]（4 cm の立方体があれば指はこフレームで閉じない）
    miss_hold_s: float = 0.3       # その状態がこれだけ続いたら発火 [s]（方策が自分ですぐ開き直す空振りには出ない）
    cue_jump_m: float = None       # 目標位置の入力と手先の水平の距離がこれを超えたら発火 [m]。None なら切
    stop: str = "measured"         # 止め方: measured（測った手先へ戻して保つ）/ freeze（参照をその場で止める）
    stop_speed: float = 0.25       # 止めるときに参照を動かす速さの上限 [m/s]
    settle_s: float = 0.5          # 発火から推論を始めるまでの最短の待ち [s]
    open_m: float = 0.075          # 指が開いたとみなす開き幅 [m]（judge の gripper_open_m と同じ）
    max_wait_s: float = 2.0        # 待ちの上限 [s]（開き幅・目標位置の入力の条件が満たされなくても進む）
    cue_stable_m: float = None     # 再開の行動列の目標位置の入力の安定の幅 [m]。None なら見ない

    @classmethod
    def from_dict(cls, d: dict = None) -> "ReflexParams":
        d = dict(d or {})
        names = {f.name for f in dataclasses.fields(cls)}
        bad = sorted(set(d) - names)
        if bad:
            raise ValueError(f"ReflexParams: 知らない項目 {bad}（使える項目 {sorted(names)}）")
        p = cls(**d)
        p.check()
        return p

    def check(self) -> None:
        if not (0.0 < self.hold_lo_m < self.hold_hi_m):
            raise ValueError("0 < hold_lo_m < hold_hi_m でなければならない")
        if self.est_reads < 1 or self.loss_reads < 1:
            raise ValueError("est_reads・loss_reads は 1 以上")
        if self.loss_m <= 0 or self.settle_s < 0 or self.max_wait_s < self.settle_s or self.stop_speed <= 0:
            raise ValueError("loss_m > 0、stop_speed > 0、0 <= settle_s <= max_wait_s")
        if not (0.0 < self.miss_m <= self.hold_lo_m):
            raise ValueError("0 < miss_m <= hold_lo_m でなければならない")
        if self.stop not in ("measured", "freeze"):
            raise ValueError("stop は measured か freeze")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# 名前のついた設定（実験で使う形を明示する）
PRESETS = {
    "drop_only": {"fire_on_miss": False},   # 落下だけ（collapse・open。持ち上げを確かめた後）。空掴みでは発火しない
    "full": {},                             # 落下＋空掴み（miss）。ReflexParams の既定のまま
}


def preset_params(name: str, overrides: dict = None) -> "ReflexParams":
    """名前のついた設定に、個別の上書きを重ねた引数。"""
    if name not in PRESETS:
        raise ValueError(f"反射の設定 {name!r} は {sorted(PRESETS)} のどれか")
    return ReflexParams.from_dict(dict(PRESETS[name], **(overrides or {})))


def _r(x, n=4):
    return None if x is None else round(float(x), n)


def _vec(v, n=4):
    return None if v is None else [round(float(x), n) for x in v]


class GraspLossDetector:
    """落下と空掴みの検出だけ（上書きはしない）。記録の再生（オフラインの評価）と実行系の両方が使う。"""

    def __init__(self, p: ReflexParams):
        self.p = p
        self.reset()

    def reset(self) -> None:
        self.state = "open"
        self.n_in = 0
        self.t_in = None
        self.n_out = 0
        self.prev_w = None
        self.w_ref = None
        self.z0 = None
        self.t_close = None
        self.t_hold = None
        self.t_miss = None
        self.last = None

    def update(self, t: float, width: float, intent_closed: bool, hand=None, cue=None):
        """1 つの読み。発火したら事象の辞書、そうでなければ None。持ち始め・放しは self.last に入れる。"""
        p = self.p
        self.last = None
        w = float(width)
        if self.state == "fired":
            return None
        if not intent_closed:
            rel = self.state == "holding"
            self.reset()
            self.prev_w = w
            if rel:
                self.last = {"kind": "release", "t": _r(t), "w": _r(w)}
            return None
        if self.state == "open":
            self.state, self.t_close, self.n_in, self.t_in = "closing", float(t), 0, None
            self.z0 = None if hand is None else float(hand[2])
        if self.state == "closing":
            if p.fire_on_miss and w < p.miss_m:
                self.n_out += 1
                self.t_miss = float(t) if self.t_miss is None else self.t_miss
                if self.n_out >= p.loss_reads and float(t) - self.t_miss >= p.miss_hold_s - 1e-9:   # 空を掴んだまま（掴み損ね）
                    self.state = "fired"
                    return {"kind": "fire", "reason": "miss", "t": _r(t), "w": _r(w), "w_ref": None,
                            "t_hold": None, "held_s": None, "t_close": _r(self.t_close),
                            "closing_s": _r(float(t) - self.t_close, 3)}
            else:
                self.n_out, self.t_miss = 0, None
            inr = p.hold_lo_m <= w <= p.hold_hi_m
            still = self.prev_w is not None and abs(w - self.prev_w) <= p.stable_m
            if inr and still:
                self.n_in += 1
                self.t_in = float(t) if self.t_in is None else self.t_in
            else:
                self.n_in, self.t_in = 0, None
            lifted = p.min_lift_m <= 0 or (hand is not None and self.z0 is not None
                                           and float(hand[2]) - self.z0 >= p.min_lift_m)
            self.prev_w = w
            if self.n_in >= p.est_reads and float(t) - self.t_in >= p.est_s - 1e-9 and lifted:
                self.state, self.w_ref, self.t_hold, self.n_out = "holding", w, float(t), 0
                self.last = {"kind": "hold", "t": _r(t), "w": _r(w)}
            return None
        # holding（持っている間）
        self.prev_w = w
        reason = None
        low, high = w < self.w_ref - p.loss_m, w > self.w_ref + p.loss_m
        if low or (high and p.fire_on_open):
            self.n_out += 1
            if self.n_out >= p.loss_reads:
                reason = "collapse" if low else "open"
        else:
            self.n_out = 0
            if not high:
                self.w_ref += p.ref_alpha * (w - self.w_ref)
        if reason is None and p.cue_jump_m is not None and cue is not None and hand is not None and float(cue[2]) > 0.5:
            if math.hypot(float(cue[0]) - float(hand[0]), float(cue[1]) - float(hand[1])) > p.cue_jump_m:
                reason = "cue_jump"
        if reason is None:
            return None
        self.state = "fired"
        return {"kind": "fire", "reason": reason, "t": _r(t), "w": _r(w), "w_ref": _r(self.w_ref),
                "t_hold": _r(self.t_hold), "held_s": _r(float(t) - self.t_hold, 3)}


class GraspLossReflex:
    """検出と、発火の後の止め方・待ち（実行系の PolicyRuntime が呼ぶ）。状態: normal → hold（待ち）→ wait_chunk → normal。"""

    def __init__(self, params: ReflexParams = None):
        self.p = params or ReflexParams()
        self.p.check()
        self.det = GraspLossDetector(self.p)
        self.start_trial()

    def start_trial(self) -> None:
        self.det.reset()
        self.mode = "normal"
        self.events = []
        self.n_hold = 0
        self.n_release = 0
        self.n_samples = 0
        self.t_fire = None
        self.target = None
        self.width = None
        self.prev_cue = None
        self.cur = None

    def observe(self, t: float, k: int, width, intent_closed: bool, hand=None, x_cmd=None, cue=None,
                sample_t: float = None) -> bool:
        """ハンドの新しい読み 1 つ。通常のときだけ検出する。発火したら真（呼び手が古い行動列を捨て、hold_action を出す）。"""
        if width is None:
            return False
        self.width = float(width)
        if self.mode != "normal":
            return False
        self.n_samples += 1
        ev = self.det.update(t, width, intent_closed, hand, cue)
        if self.det.last is not None:
            if self.det.last["kind"] == "hold":
                self.n_hold += 1
            elif self.det.last["kind"] == "release":
                self.n_release += 1
        if ev is None:
            return False
        if self.p.stop == "measured" and hand is not None:
            self.target = np.asarray(hand, float).copy()
        else:
            self.target = None if x_cmd is None else np.asarray(x_cmd, float).copy()
        ev.update(k=int(k), sample_t=_r(sample_t), intent_closed=bool(intent_closed), hand=_vec(hand), x_cmd=_vec(x_cmd),
                  stop=self.p.stop, target=_vec(self.target), cue=_vec(cue))
        self.cur = ev
        self.events.append(ev)
        self.mode, self.t_fire = "hold", float(t)
        return True

    @property
    def overriding(self) -> bool:
        """発火の後で、まだ方策の行動列に戻っていない（行動を上書きする）間は真。"""
        return self.mode != "normal"

    def hold_action(self, x_cmd) -> np.ndarray:
        """止める行動（7 次元: 手先の差分 xyz・回転 0・開閉 −1）。参照を目標へ stop_speed 以下で寄せ、着いたら 0。"""
        a = np.zeros(7)
        a[6] = -1.0
        if self.target is not None and x_cmd is not None:
            d = self.target - np.asarray(x_cmd, float)
            n = float(np.linalg.norm(d))
            cap = self.p.stop_speed * ACTION_DT
            a[:3] = d if n <= cap else d * (cap / n)
        return a

    def blocking(self, t: float) -> bool:
        """発火の後の待ち。推論を始めてはいけない間は真。待ちが終わったら wait_chunk に進む。"""
        if self.mode != "hold":
            return False
        dt = float(t) - self.t_fire
        opened = self.width is not None and self.width >= self.p.open_m
        if dt >= self.p.max_wait_s - 1e-9 or (dt >= self.p.settle_s - 1e-9 and opened):
            self.mode = "wait_chunk"
            self.cur.update(t_settled=_r(t), settled_w=_r(self.width), settle_reason="opened" if opened else "max_wait")
            return False
        return True

    def accept_chunk(self, t: float, k: int, i: int, cue=None) -> bool:
        """待ちの後に届いた行動列を使うか。使うなら通常に戻す（検出は open から）。"""
        if self.mode != "wait_chunk":
            return True
        if self.p.cue_stable_m is not None and float(t) - self.t_fire < self.p.max_wait_s - 1e-9:
            prev, self.prev_cue = self.prev_cue, (None if cue is None else np.asarray(cue, float))
            ok = (cue is not None and prev is not None and float(cue[2]) > 0.5 and float(prev[2]) > 0.5
                  and math.hypot(float(cue[0]) - prev[0], float(cue[1]) - prev[1]) <= self.p.cue_stable_m)
            if not ok:
                self.cur["n_rejected"] = int(self.cur.get("n_rejected", 0)) + 1
                return False
        self.cur.update(t_resume=_r(t), k_resume=int(k), resume_chunk=int(i), hold_s=_r(float(t) - self.t_fire, 3))
        self.mode, self.prev_cue, self.cur, self.target = "normal", None, None, None
        self.det.reset()
        return True

    def summary(self) -> dict:
        return {"params": self.p.to_dict(), "n_fire": sum(e["kind"] == "fire" for e in self.events),
                "n_hold": self.n_hold, "n_release": self.n_release, "n_samples": self.n_samples,
                "events": list(self.events), "mode_end": self.mode}
