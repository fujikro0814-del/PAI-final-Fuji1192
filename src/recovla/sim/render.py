"""記録用カメラの描画。"""
import mujoco

from recovla.sim.control import CAMERAS


def close_renderer(renderer: mujoco.Renderer) -> None:
    """mujoco.Renderer を、他の描画器を壊さずに閉じる。

    mujoco 3.2.3 の Renderer.close（__del__ からも呼ばれる）は GL の文脈を先に、MjrContext を後に開放するので、
    MjrContext の GL の物はそのとき有効な文脈（多くは生きている別の描画器のもの）で消され、別の描画器の画像が変わる。
    ここでは先に自分の文脈を有効にしてから開放する。別の描画器を使っている間は、Renderer をごみ集めに任せずこれで閉じる。
    """
    gl = renderer._gl_context
    if gl:
        gl.make_current()
    if renderer._mjr_context:
        renderer._mjr_context.free()
    renderer._mjr_context = None
    if gl:
        gl.free()
    renderer._gl_context = None


def render_cameras(renderer: mujoco.Renderer, data, cameras=CAMERAS) -> list:
    frames = []
    for name in cameras:
        renderer.update_scene(data, camera=name)
        frames.append(renderer.render().copy())
    return frames
