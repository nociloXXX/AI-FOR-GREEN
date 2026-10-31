# AI for Green Track A Submission

This directory is a runnable submission source for the official Track A
`停车场充电世界模型` task.

## Model

- `submission.py` exports `build_model(training)`.
- Vehicle demand and dwell are predicted by site-specific
  HistGradientBoosting absolute-error ensembles. Demand and dwell use separate
  targets, two seeds are averaged, and recent sessions receive more weight.
- The existing ExtraTrees ensemble is retained only for future-arrival
  demand/dwell cohort modeling, so the world-model distribution is not replaced
  by a single gradient-boosted point forecast.
- The optional `last_requested_energy_kwh` snapshot is used with an explicit
  missing-value indicator and transformed request features, and is not treated
  as ground truth.
- Predicted demand is capped by charger power multiplied by predicted dwell.
- Arrival rates use site-by-weekday-by-96-slot empirical frequencies, matching
  the official 15-minute interface instead of discarding within-hour timing.
- The world model advances in 15-minute physical steps.
- Expected future arrivals keep the same fractional semantics as V9. V10 stores
  each cohort's total remaining energy explicitly, which is behaviorally
  equivalent when no historical calibration factor is applied.
- Future arrivals now use five joint demand/dwell quantile cohorts sampled from
  the ExtraTrees ensemble instead of assigning every arriving vehicle the same
  point dwell time. The cohort probabilities preserve the expected vehicle
  count.
- Current vehicles now use three joint demand/dwell cohorts at the 20th, 50th
  and 80th ExtraTrees quantiles with `0.2/0.6/0.2` probabilities. Each cohort
  keeps the demand and dwell predicted by the same tree, is centered on V13's
  survival-adjusted remaining dwell, and preserves total expected vehicles and
  remaining energy.
- Historical quarter-hour arrival counts are aligned from the current local
  step for any truncated window length, normalized by same-weekday training
  coverage, and calibrate each future period by a bounded factor. An all-zero
  valid history is treated as genuine low demand.
- Future-arrival intensity is conditioned on the current number of parked
  vehicles using site-and-origin regressions fitted only from complete
  historical training days. This captures high-occupancy burst conditions and
  low-occupancy days without reading future scenario vehicles.
- Current vehicles whose predicted dwell has already expired use a
  training-only conditional survival residual instead of being collapsed to a
  one-step tail.
- `natural` and `capacity_edf` use the same initial state and arrival assumptions;
  only capacity and allocation priority change.
- `capacity_edf` rolls out every fractional current and future cohort in
  earliest-predicted-departure order under the intervention capacity, without
  an arbitrary minimum service-vehicle budget.
- Online remaining demand caps the optional last-request snapshot at
  `1.35 × predicted demand`; the unverified request is never an unbounded hard
  target.
- Office1 uses a sparse-site slack-first scheduler. Vehicles whose predicted
  demand is satisfied but which are not yet observed complete remain in a
  shared tail bucket, so residual power is not discarded while completion
  uncertainty remains.
- Online actions use current capacity and forecast capacity, never
  `intervention_capacity`.
- Completion-aware dispatch calculates each vehicle's latest feasible start
  and minimum required power. No-slack vehicles are protected first, remaining
  vehicles receive a minimum completion allocation, and spare power then
  follows smallest-remaining-job order to maximize completed vehicle count.
- If current capacity is below 90% of the aggregate completion floor, the
  scheduler stops spreading power thinly and concentrates on the smallest
  feasible jobs.
- Online dispatch plans with a 1.25 demand margin to retain completion safety
  in the hidden 1.35 demand-shift condition. This margin affects action choice
  only; prediction and world-model outputs are unchanged.

## Local check

Use the official fixed environment. From the project root:

```powershell
$env:PYTHONPATH = (Resolve-Path "AIforGreen-参赛包").Path
python -m unittest discover -s tests -v
```

The official packaging command can be run from this directory:

```powershell
python ..\AIforGreen-参赛包\make_submission.py `
  --source . `
  --route ml `
  --environment-id cpu-ml-methods-v3 `
  --out ..\submissions\AIforGreen_track_a_v16_current_joint_cohorts.zip
```

## Limitations

The model does not read hidden labels, future arrivals, future departures, or
future observations. The capacity intervention is a simulator physics
comparison and is not evidence of real driver behavior. Office1 has the least
training history and therefore the greatest sampling uncertainty.

## AI use disclosure

OpenAI Codex was used to inspect the official interface, implement the model,
write tests, and prepare documentation. The submitted model runs offline and
does not call any online model API.
