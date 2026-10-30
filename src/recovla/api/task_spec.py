"""タスクの書式。configs/tasks/*.yaml を読んで、物・場所・目標・手順の語彙を持つ。

    spec = load_task("configs/tasks/tidy3_cubes.yaml")
    step = StepSpec("pick_place", "green", "box")
    instruction_of(spec, step, fmt)       # 方策への指示文（fmt は configs の convert.instruction）
    cue_of(spec, step)                    # 目標位置の入力の色（"green"）
    step_of_instruction(spec, text, fmt)  # 指示文 → StepSpec（実行器が渡す文から、技能が狙いの物を知る）
    goal_met(spec, est)                   # 知覚の推定（WorldEstimate）で目標を判定（実行系）。真値の版は api/harness/goal_truth

finder（物・場所の見つけ方）: "color:<色>"（色の知覚）、"box:v1"（箱の当てはめ）、"plate:v1"（机の上の平らな円板。
api/estimated_place）、"open_vocab:<言葉>"（未実装。NotImplementedError）。
目標の種類: all_in（物が全部、場所（箱など、kind container）の中）・all_on（物が全部、場所（皿など、kind surface）の上に
載って立っている）。物の size_m は省ける（評価・場面だけが読む。実行系は読まない）。
policy_instruction（省ける）: VLA に渡す指示文をタスクから作る書式（"{color}" を含む。例 "put the {color} cup on
the plate"）。入口の --vla-text task で使う。省けば学習した決まった形（configs の convert.instruction）。
"""
import dataclasses
import pathlib

import yaml

from recovla.common import config

DEFAULT_TASK = "configs/tasks/tidy3_cubes.yaml"
STEP_KINDS = ("pick_place",)              # 手順の語彙。pick_place(物, 場所)
GOAL_KINDS = ("all_in", "all_on")         # 目標の種類。all_in = objects が全部 place の中、all_on = 全部 place の上
FINDER_KINDS = ("color", "box", "plate", "open_vocab")
PLACE_KINDS = ("container", "surface")    # 置き場所の種類（箱らしい物・皿らしい物）
GOAL_PLACE_KIND = {"all_in": "container", "all_on": "surface"}


@dataclasses.dataclass(frozen=True)
class ObjectSpec:
    name: str
    words: tuple                          # 人の呼び方（日本語）
    shape: str
    size_m: tuple
    finder: str                           # 例 "color:red"


@dataclasses.dataclass(frozen=True)
class PlaceSpec:
    name: str
    words: tuple
    kind: str                             # "container" など
    finder: str                           # 例 "box:v1"


@dataclasses.dataclass(frozen=True)
class GoalSpec:
    kind: str
    objects: tuple
    place: str


@dataclasses.dataclass(frozen=True)
class StepSpec:
    kind: str                             # STEP_KINDS のどれか
    obj: str
    place: str


@dataclasses.dataclass(frozen=True)
class TaskSpec:
    name: str
    objects: tuple                        # (ObjectSpec, ...)。並びは書式の順（COLORS の順）
    places: tuple
    goal: GoalSpec
    step_vocab: tuple
    instructions: tuple                   # 人の指示文の例
    policy_instruction: str = ""          # VLA への指示文をタスクから作る書式（省けば ""＝決まった形）

    @property
    def object_names(self) -> tuple:
        return tuple(o.name for o in self.objects)

    @property
    def place_names(self) -> tuple:
        return tuple(p.name for p in self.places)

    def object(self, name: str) -> ObjectSpec:
        for o in self.objects:
            if o.name == name:
                return o
        raise KeyError(f"物 {name!r} はタスク {self.name} にない（{self.object_names}）")

    def place(self, name: str) -> PlaceSpec:
        for p in self.places:
            if p.name == name:
                return p
        raise KeyError(f"場所 {name!r} はタスク {self.name} にない（{self.place_names}）")


def finder_kind(finder: str) -> tuple:
    """"color:red" → ("color", "red")。"""
    kind, sep, arg = str(finder).partition(":")
    if not sep or kind not in FINDER_KINDS:
        raise ValueError(f"finder {finder!r} は '<種類>:<引数>'（種類は {FINDER_KINDS}）")
    return kind, arg


def _tup(x) -> tuple:
    return tuple(x) if isinstance(x, (list, tuple)) else (x,)


def parse_task(d: dict) -> TaskSpec:
    """辞書（YAML を読んだもの）→ TaskSpec。名前の重複・目標の物と場所・手順の語彙・finder を確かめる。"""
    objs = tuple(ObjectSpec(str(o["name"]), _tup(o.get("words", ())), str(o["shape"]),
                            tuple(float(v) for v in o.get("size_m", ())), str(o["finder"])) for o in d["objects"])
    places = tuple(PlaceSpec(str(p["name"]), _tup(p.get("words", ())), str(p["kind"]), str(p["finder"]))
                   for p in d["places"])
    g = d["goal"]
    goal = GoalSpec(str(g["kind"]), tuple(str(x) for x in g["objects"]), str(g["place"]))
    spec = TaskSpec(str(d["name"]), objs, places, goal, tuple(str(x) for x in d.get("step_vocab", STEP_KINDS)),
                    tuple(str(x) for x in d.get("instructions", ())), str(d.get("policy_instruction", "")))
    names = spec.object_names + spec.place_names
    if len(set(names)) != len(names):
        raise ValueError(f"タスク {spec.name}: 物・場所の名前が重複している {names}")
    if goal.kind not in GOAL_KINDS:
        raise ValueError(f"目標の種類 {goal.kind!r} は {GOAL_KINDS} のどれか")
    for n in goal.objects:
        spec.object(n)
    spec.place(goal.place)
    for p in places:
        if p.kind not in PLACE_KINDS:
            raise ValueError(f"場所 {p.name} の種類 {p.kind!r} は {PLACE_KINDS} のどれか")
    if spec.place(goal.place).kind != GOAL_PLACE_KIND[goal.kind]:
        raise ValueError(f"目標 {goal.kind} の場所は {GOAL_PLACE_KIND[goal.kind]}（{goal.place} は {spec.place(goal.place).kind}）")
    if spec.policy_instruction and "{color}" not in spec.policy_instruction:
        raise ValueError(f"policy_instruction {spec.policy_instruction!r} は '{{color}}' を含む書式")
    bad = [k for k in spec.step_vocab if k not in STEP_KINDS]
    if bad:
        raise ValueError(f"手順の語彙 {bad} は {STEP_KINDS} にない")
    for x in objs + places:
        finder_kind(x.finder)
    return spec


def load_task(path=DEFAULT_TASK) -> TaskSpec:
    p = config.path(path) if not pathlib.Path(path).is_absolute() else pathlib.Path(path)
    return parse_task(yaml.safe_load(p.read_text(encoding="utf-8")))


# ---------------------------------------------------------------- 手順と指示文
def cue_of(spec: TaskSpec, step: StepSpec) -> str:
    """目標位置の入力の色（方策の目標位置の入力・知覚の色）。色で見つける物だけ。"""
    kind, arg = finder_kind(spec.object(step.obj).finder)
    if kind != "color":
        raise ValueError(f"物 {step.obj} の finder {kind} には手がかりの色がない（今の方策の指示文は色だけ）")
    return arg


def policy_format(spec: TaskSpec, mode: str, fixed_fmt: str) -> str:
    """VLA への指示文の書式。mode "fixed" = 決まった形（configs の convert.instruction。学習した文）、
    "task" = タスクの書式の policy_instruction（無いタスクでは決まった形）。どちらも "{color}" を目標位置の入力の色で埋める。"""
    if mode == "fixed":
        return fixed_fmt
    if mode == "task":
        return spec.policy_instruction or fixed_fmt
    raise ValueError(f"指示文の作り方 {mode!r} は fixed・task のどれか")


def instruction_of(spec: TaskSpec, step: StepSpec, fmt: str) -> str:
    """方策への指示文。fmt は configs の convert.instruction（実行器 TaskRuntime.instruction と同じ作り）。"""
    if step.kind not in spec.step_vocab:
        raise ValueError(f"手順 {step.kind} はタスク {spec.name} の語彙 {spec.step_vocab} にない")
    spec.place(step.place)
    return fmt.format(color=cue_of(spec, step))


def step_of_cue(spec: TaskSpec, cue: str, place: str = None) -> StepSpec:
    """LLM の手順（色の名前）→ StepSpec（置き場所は目標の場所）。"""
    for o in spec.objects:
        kind, arg = finder_kind(o.finder)
        if kind == "color" and arg == cue:
            return StepSpec("pick_place", o.name, place or spec.goal.place)
    raise KeyError(f"手がかりの色 {cue!r} の物はタスク {spec.name} にない")


def step_of_instruction(spec: TaskSpec, text: str, fmt: str, place: str = None) -> StepSpec:
    """指示文 → StepSpec（instruction_of の逆）。"""
    for o in spec.objects:
        if finder_kind(o.finder)[0] != "color":
            continue
        st = StepSpec("pick_place", o.name, place or spec.goal.place)
        if instruction_of(spec, st, fmt) == text:
            return st
    raise ValueError(f"指示文 {text!r} はタスク {spec.name} のどの手順の文でもない")


# ---------------------------------------------------------------- 目標の判定（実行系の版）
def in_place_estimate(spec: TaskSpec, est) -> dict:
    """{物: 目標の場所の中か}。知覚の推定（api.perception.WorldEstimate）で決める。見失った物は偽。"""
    out = {}
    for n in spec.goal.objects:
        o = None if est is None else est.objects.get(n)
        out[n] = bool(o is not None and o.status != "lost" and o.in_place.get(spec.goal.place, False))
    return out


def goal_met(spec: TaskSpec, est) -> bool:
    """実行系の目標の判定（知覚の推定だけ）。評価の判定は api/harness/goal_truth.goal_met_truth。"""
    if spec.goal.kind in ("all_in", "all_on"):           # 中か上かは知覚の口が場所の種類で決める（in_place）
        return all(in_place_estimate(spec, est).values())
    raise ValueError(spec.goal.kind)
