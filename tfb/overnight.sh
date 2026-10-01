#!/bin/bash
# Unattended chain: wait for family-blocked eval -> train transferable Task 2 models -> build subs A/B -> submit at 00:01 UTC.
cd ~/tfb; P=~/venv/bin/python; R=data/kaggle_public; C=2026-ieee-big-data-traffic-flow-bench
while pgrep -f "t2.py family_block" >/dev/null; do sleep 60; done
$P tfb/t2.py auto_build > logs/auto_build.log 2>&1 || { echo "auto_build FAILED" >> logs/overnight.log; exit 1; }
for v in A B; do
  $P repo/src/merge_submissions.py --state work/t1/state_submission.csv --queue work/t2/queue_auto_$v.csv --odme subs/odme_ref.csv --key $R/submission_key.csv --output subs/sub_auto_$v.csv >> logs/overnight.log 2>&1 || exit 1
  $P -c "import pandas as pd; s=pd.read_csv('subs/sub_auto_$v.csv'); m=s.task=='state'; s.loc[m,'speed_kmh']=s.loc[m,'speed_kmh'].clip(lower=5.0); s.to_csv('subs/sub_auto_$v.csv', index=False)"
  (cd subs && rm -f sub_auto_$v.zip && zip -q -j sub_auto_$v.zip sub_auto_$v.csv)
done
echo "built A,B $(date -u)" >> logs/overnight.log
now=$(date -u +%s); target=$(date -u -d "tomorrow 00:01" +%s); [ $(date -u +%H) -lt 1 ] && target=$(date -u -d "today 00:01" +%s); [ $target -gt $now ] && sleep $((target-now))
~/venv/bin/kaggle competitions submit $C -f subs/sub_auto_A.zip -m "auto A: transferable T2 (no identity/freq ongoing; onset best family-blocked set) + v1 state + exact ODME" >> logs/overnight.log 2>&1
~/venv/bin/kaggle competitions submit $C -f subs/sub_auto_B.zip -m "auto B: same onset as A, ongoing = persistence of state at T + v1 state + exact ODME" >> logs/overnight.log 2>&1
echo "submitted $(date -u)" >> logs/overnight.log
