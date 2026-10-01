#!/bin/bash
# After t1_final_ss finishes: rebuild Task 2 queue without day-context features, merge sub_safe_v2 (NOT submitted).
cd ~/tfb; P=~/venv/bin/python; R=data/kaggle_public
while pgrep -f "tfb/t1_final_ss.py" >/dev/null; do sleep 30; done
test -f work/t1/state_submission_ss.csv || { echo "T1 missing" > logs/build_safe2.log; exit 1; }
$P tfb/t2s.py make_sub > logs/safe2_t2.log 2>&1 || { echo T2FAIL > logs/build_safe2.log; exit 1; }
$P repo/src/merge_submissions.py --state work/t1/state_submission_ss.csv --queue work/t2s/queue_safe.csv --odme subs/odme_ref.csv --key $R/submission_key.csv --output subs/sub_safe_v2.csv > logs/build_safe2.log 2>&1
cd subs && rm -f sub_safe_v2.zip && zip -q -j sub_safe_v2.zip sub_safe_v2.csv && echo BUILT >> ~/tfb/logs/build_safe2.log
