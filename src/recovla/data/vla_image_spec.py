"""記録したカメラ画像を方策の画像に変える定義（ここだけで定義する）。

記録にはカメラの描画をそのまま保存する。方策に入る画像（変換で作る学習データと、評価時に
vla_observation.py が作る観測）は、どちらもここの to_policy_image / policy_images を通す。
ほかの場所で画像を反転・回転しない（片方の経路だけで変えると、黙って食い違う）。

記録 → 方策の画像の変換:

  overhead   fliplr   LIBERO の俯瞰画像は上下は正しく、左右が鏡像
  wrist      flipud   これで画像の動きが lerobot/smolvla_libero の手首画像と一致する
                      （オプティカルフローの比較で、ほかの変換はどれも一致しなかった）

使える変換は下の 4 つだけ。組み合わせない（fliplr + rot180 == flipud）。依存は numpy だけ。
"""
import numpy as np

SPEC_VERSION = 2              # 1 = 両方の視点を左右反転していた版
CAMERAS = ("overhead", "wrist")   # teleop.collect.CAMERAS と同じであること（試験で確かめる）
NET_TRANSFORMS = ("identity", "fliplr", "flipud", "rot180")
VIEW_TRANSFORMS = {"overhead": "fliplr", "wrist": "flipud"}
IMAGE_KEYS = {"overhead": "observation.images.image", "wrist": "observation.images.image2"}

_APPLY = {
    "identity": lambda a: a,
    "fliplr": lambda a: a[:, ::-1],
    "flipud": lambda a: a[::-1],
    "rot180": lambda a: a[::-1, ::-1],
}


class ImageSpecError(ValueError):
    """視点・変換の名前・記録した仕様がこのモジュールと合わないときの例外。"""


def validate(view_transforms=None, image_keys=None, cameras=None) -> None:
    """3 つの表が同じ視点を持ち、変換がすべて既知のものであることを確かめる（読み込み時に自分の表で呼ぶ）。"""
    vt = VIEW_TRANSFORMS if view_transforms is None else view_transforms
    ik = IMAGE_KEYS if image_keys is None else image_keys
    cams = CAMERAS if cameras is None else cameras
    if set(vt) != set(cams) or set(ik) != set(cams) or len(set(cams)) != len(cams):
        raise ImageSpecError(f"views differ: transforms {sorted(vt)}, image keys {sorted(ik)}, cameras {list(cams)}")
    bad = {v: t for v, t in vt.items() if t not in NET_TRANSFORMS}
    if bad:
        raise ImageSpecError(f"unknown net transform(s) {bad}; allowed {NET_TRANSFORMS}")


validate()


def apply_net_transform(name: str, raw_rgb: np.ndarray) -> np.ndarray:
    """名前の変換を 1 つかける（ほかの変換と比べる検査用）。方策の画像は to_policy_image / policy_images で作る。"""
    if name not in NET_TRANSFORMS:
        raise ImageSpecError(f"unknown net transform {name!r}; allowed {NET_TRANSFORMS}")
    a = np.asarray(raw_rgb)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ImageSpecError(f"expected H x W x 3, got shape {a.shape}")
    return np.ascontiguousarray(_APPLY[name](a)).copy()


def to_policy_image(view: str, raw_rgb: np.ndarray) -> np.ndarray:
    """1 台のカメラの H x W x 3 の画像 → 方策が見る画像（C 連続のコピー）。"""
    if view not in VIEW_TRANSFORMS:
        raise ImageSpecError(f"unknown view {view!r}; known {sorted(VIEW_TRANSFORMS)}")
    return apply_net_transform(VIEW_TRANSFORMS[view], raw_rgb)


def policy_images(raw_by_view: dict) -> dict:
    """{視点: 画像} → {データセットの特徴の名前: 方策の画像}。視点の過不足は例外にする。"""
    got, want = set(raw_by_view), set(VIEW_TRANSFORMS)
    if got != want:
        raise ImageSpecError(f"views {sorted(got)} != spec views {sorted(want)} "
                             f"(missing {sorted(want - got)}, extra {sorted(got - want)})")
    return {IMAGE_KEYS[v]: to_policy_image(v, raw_by_view[v]) for v in CAMERAS}


def spec_record() -> dict:
    """変換が meta/conversion.json に書く内容。"""
    return {"image_spec_version": SPEC_VERSION, "image_transforms": dict(VIEW_TRANSFORMS),
            "image_keys": dict(IMAGE_KEYS)}


def check_record(conversion: dict, source: str = "conversion.json") -> None:
    """conversion.json（dict）がこの仕様で書かれたものでなければ ImageSpecError。版 1 のデータセットは受けない。"""
    version = conversion.get("image_spec_version")
    if version != SPEC_VERSION:
        raise ImageSpecError(f"{source}: image_spec_version {version!r} != {SPEC_VERSION} "
                             f"(converter_version {conversion.get('converter_version')!r}); re-convert the "
                             f"raw episodes with the current convert_to_lerobot.py")
    for field, want in (("image_transforms", VIEW_TRANSFORMS), ("image_keys", IMAGE_KEYS)):
        if conversion.get(field) != want:
            raise ImageSpecError(f"{source}: {field} {conversion.get(field)!r} != runtime {want!r}")
