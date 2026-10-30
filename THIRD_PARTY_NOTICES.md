# 使用したソフトウェア・モデルとライセンス

| 名称 | 用途 | 出所 | ライセンス |
|---|---|---|---|
| SmolVLA | VLA の方策。`lerobot/smolvla_libero`（LIBERO で学習済み）を初期値として追加学習した。重みは本リポジトリに含めていない | Hugging Face（lerobot/smolvla_libero） | Apache-2.0 |
| SmolVLM2（SmolVLM2-500M-Video-Instruct） | SmolVLA の視覚・言語の部分（固定して使用） | Hugging Face（HuggingFaceTB） | Apache-2.0 |
| LeRobot 0.6.1 | 学習と推論（依存として導入。本リポジトリには含めていない） | https://github.com/huggingface/lerobot | Apache-2.0 |
| MuJoCo 3.2.3 | 物理シミュレーションと描画（依存として導入） | https://github.com/google-deepmind/mujoco | Apache-2.0 |
| Franka Emika Panda の MJCF | ロボットのモデル（`assets/mjcf/panda/`、元の LICENSE を同梱） | MuJoCo Menagerie（https://github.com/google-deepmind/mujoco_menagerie） | Apache-2.0 |
| Claude Haiku 5.5・Claude Sonnet 5.5 | タスク分解と再計画の LLM（API 経由。API の鍵は含めていない） | Anthropic API | Anthropic の利用規約 |
| PyTorch、NumPy、SciPy、OpenCV、PyAV、Pillow、matplotlib ほか | 数値計算・画像処理・動画処理・図の作成（依存として導入） | 各プロジェクト | 各プロジェクトのライセンス（版は `env/requirements-lock.txt` と `docs/freeze/pip_freeze.txt`） |
| BIZ UD ゴシック、Cambria ほか | 動画と図のフォント（Windows に同梱のものを使用。フォントファイルは含めていない） | Windows | Windows のライセンス |

自作部分: `src/`・`scripts/`・`tests/`・`configs/`・`paper/`・`docs/` の内容と、`assets/mjcf/` の場面（`scene_3cube.xml`・`scene_cup_plate.xml`・`scene_g0.xml`・`robot_only.xml`）。
このうち `src/recovla/record/`・`src/recovla/data/`・`src/recovla/eval/closed_loop.py`・`src/recovla/policy/train_launcher.py`・
`src/recovla/sim/controller_ik.py`・`src/recovla/sim/device.py`・`src/recovla/common/code_version.py`・`scripts/replay_check.py`・`tests/teleop_tool/`・
`assets/mjcf/scene_g0.xml` は、以前に自作した遠隔操作と記録のツールを基にしている（`assets/mjcf/panda/` も同じツールを経由して取り込んだ MuJoCo Menagerie のもの）。
