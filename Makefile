.PHONY: docker-build warm train eval web shell smoke docs phase01 phase23 phase45 experiment ui-build ui-smoke demo-pack demo-report compare-report demo-story benchmark-fair-static benchmark-fair-dynamic benchmark-priv-static benchmark-priv-dynamic eval-ood full-eval sanity-pack tune-fast tune-balanced tune-deep test-tuning-stageb profile-baselines perf-gate benchmark-release

docker-build:
	docker compose build

warm:
	DOCKER_BUILDKIT=1 docker compose build trainer

train:
	docker compose run --rm trainer python -m scripts.train.trained_ppo run.run_name=ppo_baseline

eval:
	docker compose run --rm trainer python -m scripts.eval.eval_scenarios policy=baseline:potential_fields episodes=5

web:
	docker compose up web

shell:
	docker compose run --rm trainer bash

smoke:
	docker compose run --rm trainer python -m scripts.train.trained_ppo run.total_timesteps=20000 run.run_name=smoke

docs:
	python3 scripts/docs/generate_architecture.py

phase01:
	docker compose run --rm trainer python -m scripts.train.trained_ppo experiment=phase01

phase23:
	docker compose run --rm trainer python -m scripts.train.trained_ppo \
		experiment=phase23 resume.enabled=true resume.run_dir=latest:phase01

phase45:
	docker compose run --rm trainer python -m scripts.train.trained_ppo \
		experiment=phase45 resume.enabled=true resume.run_dir=latest:phase23

EXPERIMENT_SPEC ?= configs/experiments/smoke.yaml
TUNE_OVERRIDES ?=

experiment:
	docker compose run --rm trainer python -m scripts.experiments --spec $(EXPERIMENT_SPEC)

tune-fast:
	docker compose run --rm trainer python -m scripts.tuning.tune_baselines tuning/profile=fast $(TUNE_OVERRIDES)

tune-balanced:
	docker compose run --rm trainer python -m scripts.tuning.tune_baselines tuning/profile=balanced $(TUNE_OVERRIDES)

tune-deep:
	docker compose run --rm trainer python -m scripts.tuning.tune_baselines tuning/profile=deep $(TUNE_OVERRIDES)

test-tuning-stageb:
	pytest -q -m tuning_stageb --run-tuning-stageb

PROFILE_SCENARIO ?= scenarios/S7_dynamic_chaser.yaml
PROFILE_POLICIES ?= baseline:astar_grid baseline:mpc_lite
PROFILE_STEPS ?= 600
PERF_REFERENCE ?= docs/perf/baseline_profile_dynamic_600_fixed.json
PERF_MAX_REGRESSION ?= 10

profile-baselines:
	python3 -m scripts.perf.profile_baselines --scenario $(PROFILE_SCENARIO) --steps $(PROFILE_STEPS) --policies $(PROFILE_POLICIES)

perf-gate:
	python3 -m scripts.perf.profile_baselines --scenario $(PROFILE_SCENARIO) --steps $(PROFILE_STEPS) --policies $(PROFILE_POLICIES) --compare $(PERF_REFERENCE) --metric wall --max-regression-pct $(PERF_MAX_REGRESSION)

RELEASE_INPUTS ?=
RELEASE_LABELS ?=

benchmark-release:
	python3 -m scripts.analysis.build_benchmark_release --inputs $(RELEASE_INPUTS) $(if $(RELEASE_LABELS),--labels $(RELEASE_LABELS),)

ui-build:
	cd ui_frontend && npm ci && npm run build

ui-smoke:
	cd ui_frontend && npm ci && npm run build && npm run smoke

demo-pack:
	docker compose run --rm trainer python -m scripts.experiments --spec configs/experiments/demo_pack.yaml

sanity-pack:
	docker compose run --rm trainer python -m scripts.experiments --spec configs/experiments/sanity_pack.yaml

demo-report:
	docker compose run --rm trainer python -m scripts.analysis.demo_pack_report --update-readme

compare-report:
	docker compose run --rm trainer python -m scripts.analysis.compare_report

demo-story:
	docker compose run --rm trainer python -m scripts.analysis.story_pack --copy

benchmark-fair-static:
	docker compose run --rm trainer python -m scripts.experiments --spec configs/experiments/benchmark_fair_static.yaml

benchmark-fair-dynamic:
	docker compose run --rm trainer python -m scripts.experiments --spec configs/experiments/benchmark_fair_dynamic.yaml

benchmark-priv-static:
	docker compose run --rm trainer python -m scripts.experiments --spec configs/experiments/benchmark_privileged_static.yaml

benchmark-priv-dynamic:
	docker compose run --rm trainer python -m scripts.experiments --spec configs/experiments/benchmark_privileged_dynamic.yaml

eval-ood:
	docker compose run --rm trainer python -m scripts.experiments --spec configs/experiments/ood_eval.yaml

full-eval:
	$(MAKE) benchmark-fair-static
	$(MAKE) benchmark-fair-dynamic
	$(MAKE) benchmark-priv-static
	$(MAKE) benchmark-priv-dynamic
	$(MAKE) eval-ood
	$(MAKE) demo-pack
	$(MAKE) demo-report
	$(MAKE) compare-report
	$(MAKE) demo-story
