"""タスクと感覚の窓口（task API）。実行器・LLM の手順の部品をつなぐ。
  task_spec      タスクの書式（configs/tasks/*.yaml）
  senses         センサの出力の口（SensorFrame を包む）
  perception     知覚の口（WorldEstimate。ColorWorldModelFinder は既存の WorldModel を写す）
  geometry       点群からの形の推定（物の向きつき外接箱、箱の縁・内寸・傾き。既知の寸法は使わない）
  estimated      点群から推定する知覚 EstimatedGeometryFinder と、既存の Perception に足す GeometryPerception
  geometry_place 立った円筒（カップ）・平らな円板（皿）の推定（タスク「カップを皿に」）
  estimated_place 推定を皿・カップへ広げた PlaceGeometryFinder
  relations      手に近い・机の上・箱の外などの判定（推定の大きさ・不確かさに対する相対）
  monitors       失敗の検知（GraspDropMonitor・HeldEstimateDropMonitor・LegacyInHandDropMonitor）
  executor_monitored  失敗の検知の出来事を手順の組み直しにつなぐ実行器
  skills         技能の口（VLASkill・ScriptSkill）。動作の段階の決まりは skills_script
  planner        計画の口（評価では使わない）
  harness/       真値を使う物（評価の目標の判定）。実行系はここを import しない
実行系は真値を読まない: harness/ を除くこの包みは scripts/check_real_boundary.py で検査する。
"""
