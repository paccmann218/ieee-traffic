#!/bin/bash
cd ~/tfb; P=~/venv/bin/python; R=data/kaggle_public
$P tfb/t2.py predict > logs/t2_pred.log 2>&1
$P repo/src/merge_submissions.py --state work/t1/state_submission.csv --queue work/t2/queue_submission.csv \
   --odme subs/odme_ref.csv --key $R/submission_key.csv --output subs/sub_full_v1.csv > logs/merge_v1.log 2>&1
cd subs && zip -q -j sub_full_v1.zip sub_full_v1.csv && echo ASSEMBLED >> ~/tfb/logs/merge_v1.log
