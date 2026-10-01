#!/bin/bash
# Build the rules-safe candidate submission (does NOT submit).
cd ~/tfb; P=~/venv/bin/python; R=data/kaggle_public
$P tfb/t2s.py make_sub > logs/safe_t2.log 2>&1 || { echo T2FAIL >> logs/build_safe.log; exit 1; }
$P tfb/t1_final_ss.py > logs/safe_t1.log 2>&1 || { echo T1FAIL >> logs/build_safe.log; exit 1; }
$P repo/src/merge_submissions.py --state work/t1/state_submission_ss.csv --queue work/t2s/queue_safe.csv --odme subs/odme_ref.csv --key $R/submission_key.csv --output subs/sub_safe_v1.csv >> logs/build_safe.log 2>&1
cd subs && zip -q -j sub_safe_v1.zip sub_safe_v1.csv && echo BUILT >> ~/tfb/logs/build_safe.log
