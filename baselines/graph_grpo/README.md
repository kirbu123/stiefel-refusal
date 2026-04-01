# GRPO-IS: Graph-Based Refusal Direction Learning via GRPO with Importance Sampling

Метод обучает веса `LearnableDirectionWeights` для взвешенного суммирования
**замороженных** refusal-направлений графа вершин. Поддерживаются два режима
параметризации и два режима оптимизации:

- `WEIGHTS_MODE=scalar` — один обучаемый скаляр на каждое refusal-направление
- `WEIGHTS_MODE=dense` — полная матрица коэффициентов `(n_layers+1, hidden_size)` на направление
- `OPTIMIZER_METHOD=grpo` — текущий gradient-based GRPO-IS trainer
- `OPTIMIZER_METHOD=optuna` — scalar-only поиск коэффициентов через Optuna

Поведение модели меняется через аблитерацию — итоговые веса определяют, насколько
сильно «стирать» каждое направление.

---

## Идея

Дан граф концептов (например, `physical harm_wordnet_graph_actions_terms.txt`).
Каждой вершине соответствует refusal-direction — вектор `(n_layers+1, hidden_size)`,
вычисленный как разность активаций модели на harmful и harmless примерах.

Задача: найти веса для направлений графа такие, что взвешенная сумма направлений,
применённая как аблитерация, максимизирует вредоносность ответов модели
(то есть успешно обходит отказ). В scalar-режиме это `w ∈ ℝ^{n_vertices}`,
в dense-режиме — тензор коэффициентов на весь `(layer, hidden)` профиль каждого направления.

Обучение — через GRPO с importance sampling (off-policy rollout).

---

## Пайплайн (один обучающий шаг)

### Шаг 1–2: Роллаут из M поведенческих политик

```
base_W = stopgrad(direction_weights.weights)
для m = 1..GRPO_N_GROUPS:
    W_m = base_W                    # для первой политики
    W_m = base_W + noise_m          # для остальных, noise_m ~ N(0, GRPO_NOISE_SCALE^2)
    combined_direction = combine_with_weights(extracted_directions, W_m)
    model.reload_model()
    apply_abliteration(model, combined_direction, ref_alpha * max_weight, ...)  # общая фиксированная аблитерация
    responses_m = model.get_responses_batched(questions)
    rollout_log_probs_m = compute_sequence_log_probs(model, questions, responses_m)
```

Реализовано в `trainer.py`, функция `train_grpo_is_step()`.
Обучаемый параметр остаётся один: `LearnableDirectionWeights.weights`.
Разные behaviour policies `μ_m` отличаются только detached-копиями текущего `W`,
а не разными `alpha`.

- В `scalar`-режиме `W_m.shape == (n_directions,)`, а `noise_m` добавляет одно независимое
  гауссово смещение на каждое refusal-направление.
- В `dense`-режиме `W_m.shape == (n_directions, n_layers+1, hidden_size)`, и шум
  добавляется поэлементно ко всей матрице коэффициентов.
- Для rollout'ов всегда используется общий `ref_alpha`; sampled policies отличаются
  только самими весами `W_m`.
- В `scalar`-режиме итоговое направление задаётся как
  `combined_direction_m = normalize_layerwise(sum_i W_m[i] * d_i)`.

### Шаг 3: Лог-вероятности базовой политики (π_θ_old)

```
model.reload_model()  # базовая модель без аблитерации
для каждого m:
    old_log_probs_m = compute_sequence_log_probs(model, questions, responses_m)
```

`compute_sequence_log_probs` — собственная реализация в `log_probs.py`:
токенизирует `(prompt, response)`, делает forward-проход, извлекает per-token log-prob
только для токенов ответа. Heretic не поддерживает полноценный расчёт для всей последовательности.

### Шаг 4: IS-веса (verl)

```python
# verl: verl/trainer/ppo/rollout_corr_helper.py
from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_weights

log_ratio = old_log_probs - rollout_log_probs  # log(π_old / μ_m)
is_weights, is_metrics = compute_rollout_correction_weights(
    log_ratio=log_ratio,
    response_mask=response_mask,
    rollout_is="sequence",        # sequence-level importance sampling
    rollout_is_threshold=IS_CLIP_RATIO,
)
```

IS-веса корректируют разницу между behaviour policy (μ_m, с sampled `W_m`) и
целевой политикой (π_old, без аблитерации). `rollout_is_threshold` обрезает
слишком большие веса для стабильности.

### Шаг 5: Награды

```python
# reward.py -> evaluate/judges.py
scores = compute_reward(flat_questions, flat_responses, classifier_categories, backend)
```

Оценка вредоносности через `EVALUATION_BACKEND`:
- `llamaguard` — LlamaGuard (LLM-классификатор безопасности)
- `local_llm_judge` — локальный LLM-судья через API

Награда помещается в последний валидный токен ответа (формат token-level rewards для verl).

### Шаг 6: IS-скорректированные advantages (собственная реализация)

```python
# advantage.py
from baselines.graph_grpo.advantage import compute_grpo_is_advantage

advantages, returns = compute_grpo_is_advantage(
    token_level_rewards=token_level_rewards,
    response_mask=response_mask,
    index=question_indices,   # группировка ответов по вопросу
    is_weights=is_weights,
)
```

Базовая линия вычисляется как IS-взвешенное среднее по группе ответов одного вопроса:

```
b(x_i) = Σ_m(v_m * S_m) / Σ_m(v_m)
A_m = (S_m - b(x_i)) / std(S)
```

Зарегистрировано через `@register_adv_est("grpo_is")` из verl.

### Шаг 7: Policy loss (verl, дифференцируемый)

```python
# Дифференцируемая аблитерация через forward hooks
combined_direction = direction_weights(extracted_directions)  # grad_fn сохранён
handles = register_abliteration_hooks(model, combined_direction, ref_alpha, ...)

log_probs = compute_sequence_log_probs(model, flat_questions, flat_responses)
# Градиент идёт: log_probs -> hook -> combined_direction -> direction_weights.weights
remove_hooks(handles)

# verl: verl/trainer/ppo/core_algos.py
from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla

loss, loss_metrics = compute_policy_loss_vanilla(
    old_log_prob=old_log_probs,
    log_prob=log_probs,
    advantages=advantages,
    response_mask=response_mask,
    rollout_is_weights=is_weights,  # IS-веса учтены в loss
    config=actor_config,            # clip_ratio=ε для PPO-clip
)
```

Лосс — стандартный PPO-clipped с IS:

```
L = -E[is_weight * min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)]
ratio = exp(log π_θ - log π_old)
```

Forward-хуки (`hooks.py`) делают аблитерацию дифференцируемой:
для каждого слоя `l` хук вычитает проекцию `(h · d_l) * d_l` из скрытого состояния.
Поскольку `d_l = combined_direction[l]` зависит от `direction_weights.weights`,
градиент через хук доходит до весов.

### Шаг 8: Обновление весов

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

Обновляется только `LearnableDirectionWeights.weights` (параметры модели заморожены).

### Шаг 9: Off-policy метрики (verl)

```python
# verl: verl/trainer/ppo/rollout_corr_helper.py
from verl.trainer.ppo.rollout_corr_helper import compute_offpolicy_metrics

offpolicy_metrics = compute_offpolicy_metrics(
    old_log_prob=old_log_probs,
    rollout_log_prob=rollout_log_probs,
    response_mask=response_mask,
)
```

---

## Использование verl

| Функция из verl | Файл verl | Назначение |
|---|---|---|
| `compute_rollout_correction_weights` | `verl/trainer/ppo/rollout_corr_helper.py` | IS-веса (sequence-level) с обрезкой |
| `compute_policy_loss_vanilla` | `verl/trainer/ppo/core_algos.py` | PPO-clipped policy loss с IS |
| `compute_offpolicy_metrics` | `verl/trainer/ppo/rollout_corr_helper.py` | Диагностика off-policy отклонения |
| `register_adv_est` | `verl/trainer/ppo/core_algos.py` | Декоратор для регистрации estimator'а |

verl установлен локально: `baselines/graph_grpo/verl/` (editable install, numpy < 2.0).

---

## Структура модуля

```
baselines/graph_grpo/
├── __init__.py          # экспорт
├── __main__.py          # точка входа: инициализация, цикл обучения, сохранение результатов
├── trainer.py           # train_grpo_is_step() — один шаг обучения
├── advantage.py         # compute_grpo_is_advantage() — IS-взвешенный baseline
├── hooks.py             # register_abliteration_hooks() — дифференцируемая аблитерация
├── log_probs.py         # compute_sequence_log_probs() — per-token log-prob
├── reward.py            # compute_reward() — обёртка над evaluate_harmfulness
├── verl/                # локальная копия verl
└── README.md
```

---

## Запуск

```bash
# Указать путь к файлу графа и запустить
export GRAPH_FILE="graphs/physical harm_wordnet_graph_actions_terms.txt"
bash scripts/run_graph_grpo.sh
```

Или напрямую из корня проекта:

```bash
conda activate heretic
python -m baselines.graph_grpo
```

Все гиперпараметры задаются через переменные окружения (см. `scripts/run_graph_grpo.sh`)
или через `config.py`.

---

## Гиперпараметры

| Переменная | По умолчанию | Описание |
|---|---|---|
| `BATCH_SIZE` | `32` | Сколько случайно sampled вопросов из категории использовать в одном запуске `graph_grpo` |
| `DEBUG` | `false` | Быстрый smoke-режим для `graph_grpo` |
| `DEBUG_N_QUESTIONS` | `4` | Сколько вопросов из категории использовать при `DEBUG=true` |
| `DEBUG_NOISE_SCALE` | `0.02` | Effective rollout noise scale при `DEBUG=true` |
| `GRPO_N_GROUPS` | `4` | Число sampled rollout-политик `μ_m` на шаг |
| `GRPO_NOISE_SCALE` | `0.1` | Stddev гауссова шума для detached rollout-копий `W_m` |
| `GRPO_REF_ALPHA` | `1.0` | Коэффициент аблитерации для политики при вычислении policy loss |
| `IS_CLIP_RATIO` | `5.0` | Порог обрезки IS-весов |
| `GRPO_CLIP_RATIO` | `0.2` | ε для PPO-clip |
| `GRPO_N_EPOCHS` | `10` | Количество шагов обучения |
| `GRPO_LEARNING_RATE` | `1e-3` | Learning rate (Adam) |
| `ABLITERATION_MAX_WEIGHT` | `2.0` | Максимальная интенсивность аблитерации |
| `ABLITERATION_MAX_WEIGHT_POSITION` | `0.7` | Позиция пика (доля от числа слоёв) |
| `OPTIMIZER_METHOD` | `grpo` | Метод оптимизации коэффициентов (`grpo` / `optuna`) |
| `WEIGHTS_MODE` | `scalar` | Параметризация весов (`scalar` / `dense`) |
| `WEIGHTS_INIT_TYPE` | `average` | Инициализация весов (`average` / `topic`; `zero` не поддерживается в текущем differentiable trainer) |
| `OPTUNA_N_TRIALS` | `50` | Число trial-ов для `OPTIMIZER_METHOD=optuna` |
| `OPTUNA_SAMPLER_SEED` | `42` | Seed для `TPESampler` |
| `OPTUNA_WEIGHT_MIN` | `-2.0` | Нижняя граница поиска scalar-коэффициентов в `optuna` |
| `OPTUNA_WEIGHT_MAX` | `2.0` | Верхняя граница поиска scalar-коэффициентов в `optuna` |
| `EVALUATION_BACKEND` | `llamaguard` | Бэкенд оценки (`llamaguard` / `local_llm_judge`) |

`graph_grpo` по умолчанию использует `WEIGHTS_MODE=scalar`, то есть один обучаемый
скаляр на направление. Режим `dense` сохраняет старую полную параметризацию и
доступен явным override.

`OPTIMIZER_METHOD=optuna` сейчас поддерживается только вместе с
`WEIGHTS_MODE=scalar`. В этом режиме objective равен `mean harmfulness` на том же
батче вопросов, а лучшее значение сохраняется в `answers_*.json` как
`optimal_harmfulness` с `optimal_harmfulness_source="best_trial_mean_reward"`.

`graph_grpo` должен стартовать с ненулевой инициализации. По умолчанию используется
`average`; `topic` остаётся доступным явным override, а `zero` намеренно запрещён,
потому что при текущем hook-based differentiable path даёт dead start с нулевым
градиентом. В scalar-режиме это означает `1 / n_directions` для всех направлений
или `1` только у выбранного topic-направления.

Обычный запуск `graph_grpo` больше не идёт по всем вопросам категории: на каждый run берётся случайный subset размера `BATCH_SIZE` (или меньше, если в категории вопросов меньше). При `DEBUG=true` этот subset дополнительно ограничивается сверху через `DEBUG_N_QUESTIONS`, а rollout noise scale заменяется на `DEBUG_NOISE_SCALE`, чтобы smoke-run был быстрее и давал менее вырожденный градиентный сигнал.

В итоговых `answers_*.json` сохраняются `final_scores`, `score_statistics.mean`,
`final_mean_harmfulness`, `optimal_harmfulness` и `optimal_harmfulness_source`.
