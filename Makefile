# Milestone 1 data pipeline - one script per step (see README "Pipeline steps").
#   make all                 run steps 1-10 + sample queries with config.json
#   make all CONFIG=x.json   use another config
#   make demo                offline run on synthetic data (no GCP, no Hugging Face)
PY ?= python
CONFIG ?= config.json

PREPROCESS = step3_clean step4_quality_checks step5_features step6_filter \
             step7_split step8_shard step9_store_processed step10_review_sample

.PHONY: install all scrape store preprocess queries review-score readme test demo clean-demo

install:
	$(PY) -m pip install -r requirements.txt

all: scrape store preprocess queries

## Step 1: scrape raw data from Hugging Face (pinned revision) -> data/downloads/
scrape:
	$(PY) -m src.step1_scrape_raw --config $(CONFIG)

## Step 2: move raw data into storage -> <bucket>/raw/
store:
	$(PY) -m src.step2_store_raw --config $(CONFIG)

## Steps 3-10: clean, quality checks, features, filter, split, shard, store processed, review sheet
preprocess:
	@for s in $(PREPROCESS); do echo "== $$s"; $(PY) -m src.$$s --config $(CONFIG) || exit 1; done

## Sample queries on the stored train/dev data (sanity checks + first look for M2)
queries:
	$(PY) -m src.sample_queries --config $(CONFIG)

## After filling in audit/label_review_<version>.csv by hand
review-score:
	$(PY) -m src.score_review --config $(CONFIG)

## Fill README placeholders from the manifest:  make readme BUCKET=<bucket> REPO=<repo>
readme:
	$(PY) scripts/fill_readme.py --bucket $(BUCKET) --repo $(REPO)

test:
	DHAI_BUCKET= $(PY) -m unittest discover -s tests -t . -v

demo:
	$(PY) -m tests.fake_data --out data/demo/fake_raw --pairs 6000 --config-out data/demo/config.json
	DHAI_BUCKET= $(PY) -m src.step1_scrape_raw --config data/demo/config.json --from-dir data/demo/fake_raw
	DHAI_BUCKET= $(MAKE) --no-print-directory store preprocess queries CONFIG=data/demo/config.json

clean-demo:
	rm -rf data/demo
