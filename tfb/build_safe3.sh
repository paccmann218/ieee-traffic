#!/bin/bash
# sub_safe_v3: T1 selfsup state + T2 (ineligible-inclusive tables, add-only hybrid ongoing) + exact ODME. Does NOT submit.
cd ~/tfb; P=~/venv/bin/python; R=data/kaggle_public
T2S_INELIG=1 $P tfb/t2s.py make_sub > logs/safe3_t2.log 2>&1 || { echo T2FAIL > logs/build_safe3.log; exit 1; }
$P repo/src/merge_submissions.py --state work/t1/state_submission_ss.csv --queue work/t2s_inel/queue_safe.csv --odme subs/odme_ref.csv --key $R/submission_key.csv --output subs/sub_safe_v3.csv > logs/build_safe3.log 2>&1
cd subs && rm -f sub_safe_v3.zip && zip -q -j sub_safe_v3.zip sub_safe_v3.csv && echo BUILT >> ~/tfb/logs/build_safe3.log
