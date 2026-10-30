"""制御器への入力の口（DeviceState・DeviceInput）。生成プログラム・方策は ScriptPad を通して腕を動かす。"""
import abc
import dataclasses
from typing import Optional
import numpy as np
import cv2

@dataclasses.dataclass
class DeviceState:
    pos: np.ndarray
    quat: np.ndarray
    button_grip: bool
    button_clutch: bool
    button_reset: bool = False
    button_home: bool = False   # 腕を HOME_QPOS へ戻し、立方体を置き直す
    # グリッパの連続の指令 [0, 1]（0 = 全開、1 = 全閉）。None なら押し始めで開閉を切り替える
    grip_analog: "Optional[float]" = None
    # vel: 手先の速度の指令 [m/s]（世界座標、積分の前）。制御ループが物理の 1 ステップごとに積分して x_des にする
    vel: "Optional[np.ndarray]" = None
    button_save: bool = False     # 押している間 True（長押しの判定は呼ぶ側）
    button_discard: bool = False
    button_quit: bool = False

class DeviceInput(abc.ABC):
    @abc.abstractmethod
    def start(self) -> None: pass
    @abc.abstractmethod
    def read(self) -> DeviceState: pass
    @abc.abstractmethod
    def stop(self) -> None: pass


class KeyboardButtonAugment(DeviceInput):
    """別の DeviceInput を包み、キーボードのボタンを足す（手での操作用）。

      SPACE -> button_grip（1 フレームだけ押す。制御器が押し始めで開閉を切り替える）
      r     -> button_reset（立方体を置き直す）
      h     -> button_home（腕を HOME_QPOS へ戻し、立方体を置き直す）

    キーは cv2.waitKey で読むので、OpenCV の窓に入力の焦点が要る。waitKey は cv2.imshow の再描画も進める。
    """

    GRIP_KEYS = (ord(' '),)
    RESET_KEYS = (ord('r'),)
    HOME_KEYS = (ord('h'),)

    def __init__(self, inner: DeviceInput):
        self.inner = inner

    def start(self) -> None:
        self.inner.start()

    def read(self) -> DeviceState:
        state = self.inner.read()
        key = cv2.waitKey(1) & 0xFF
        if key in self.GRIP_KEYS:
            state.button_grip = True
        if key in self.RESET_KEYS:
            state.button_reset = True
        if key in self.HOME_KEYS:
            state.button_home = True
        return state

    def set_force(self, force_world_N) -> None:
        """力の提示を中の入力へ渡す（対応していれば）。"""
        fn = getattr(self.inner, "set_force", None)
        if callable(fn):
            fn(force_world_N)

    def read_camera_delta(self):
        """中の入力のカメラ操作の変化量を渡す（対応していなければ (0.0, 0.0)）。"""
        fn = getattr(self.inner, "read_camera_delta", None)
        if callable(fn):
            return fn()
        return (0.0, 0.0)

    def set_lightbar(self, r, g, b) -> None:
        """ライトバーの色を中の入力へ渡す（対応していれば）。"""
        fn = getattr(self.inner, "set_lightbar", None)
        if callable(fn):
            fn(r, g, b)

    def set_trigger(self, which, mode, params=None) -> None:
        """トリガーの反力の設定を中の入力へ渡す（対応していれば）。"""
        fn = getattr(self.inner, "set_trigger", None)
        if callable(fn):
            fn(which, mode, params)

    def stop(self) -> None:
        self.inner.stop()

class MouseKeyboardInput(DeviceInput):
    # 手首カメラの窓を共有し、画像の上のマウスのドラッグで腕を動かす。窓は先に呼んだ側が作り
    # （cv2.namedWindow は何度呼んでもよい）、閉じるのは呼ぶ側の cv2.destroyAllWindows（stop では閉じない）
    WINDOW_NAME = "WRIST VIEW"

    def __init__(self, init_pos: Optional[np.ndarray] = None,
                 xy_sensitivity: float = 0.002, z_sensitivity: float = 0.01,
                 window_name: Optional[str] = None) -> None:
        self.window_name = window_name or self.WINDOW_NAME
        self.pos = np.array(init_pos if init_pos is not None else [0.5, 0.0, 0.2], dtype=float)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.button_grip = False
        self.button_clutch = False
        self.button_reset = False
        self.xy_sensitivity = xy_sensitivity
        self.z_sensitivity = z_sensitivity
        self._last_x: Optional[float] = None
        self._last_y: Optional[float] = None
        self._mouse_down: bool = False
        self._running: bool = False

    def start(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        self._running = True

    def _mouse_callback(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._mouse_down = True
            self._last_x = x
            self._last_y = y
        elif event == cv2.EVENT_LBUTTONUP:
            self._mouse_down = False
            self._last_x = None
            self._last_y = None
        elif event == cv2.EVENT_MOUSEMOVE and self._mouse_down:
            if self._last_x is None:
                self._last_x = x
                self._last_y = y
                return
            self.pos[0] += (x - self._last_x) * self.xy_sensitivity
            self.pos[1] -= (y - self._last_y) * self.xy_sensitivity
            self._last_x = x
            self._last_y = y

    def _process_keyboard(self) -> None:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('g'):
            self.button_grip = not self.button_grip
        self.button_reset = (key == ord('r'))
        if key == ord('w'):
            self.pos[2] += self.z_sensitivity
        elif key == ord('s'):
            self.pos[2] -= self.z_sensitivity

    def read(self) -> DeviceState:
        if not self._running:
            self.start()
        self._process_keyboard()
        return DeviceState(
            pos=self.pos.copy(),
            quat=self.quat.copy(),
            button_grip=self.button_grip,
            button_clutch=self._mouse_down,
            button_reset=self.button_reset,
        )

    def stop(self) -> None:
        # 窓はここで閉じない（手首カメラと共有し、呼ぶ側がまとめて閉じる）
        self._running = False


# ---------------------------------------------------------------------------
# プログラムからの入力。生成プログラム・方策・失敗の意図的な失敗は、この入口
# （pad.state = pad_state(...) -> integrator.refresh）以外で腕を動かさない。

def pad_state(**kw) -> DeviceState:
    return DeviceState(pos=np.zeros(3), quat=np.array([1.0, 0.0, 0.0, 0.0]), button_clutch=True,
                       **{"button_grip": False, "vel": np.zeros(3), **kw})


class ScriptPad(DeviceInput):
    """プログラムが書く入力。積分器が入力の読み取りごとに 1 回読む。"""

    def __init__(self):
        self.state = pad_state()

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def read(self) -> DeviceState:
        return self.state