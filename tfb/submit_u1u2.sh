#!/bin/bash
# Plan v4 section 4a: two sequential uploads when the Oct 1 quota opens. Logs to ~/tfb/logs/submit_u1u2.log
cd ~/tfb; K=~/venv/bin/kaggle; C=2026-ieee-big-data-traffic-flow-bench; LOG=~/tfb/logs/submit_u1u2.log
test -f subs/sub_U1.zip && test -f subs/sub_U2.zip || { echo "$(date -u) missing files, abort" >> $LOG; exit 1; }
now=$(date -u +%s); target=$(date -u -d "tomorrow 00:01" +%s); [ $(date -u +%H) = "00" ] && target=$now
[ $target -gt $now ] && sleep $((target-now))
echo "$(date -u) submitting U1" >> $LOG
$K competitions submit $C -f subs/sub_U1.zip -m "U1 legal: R0 (GBDT state + organizer persistence + V1 ODME) with ONSET = train-only LGB on history T-60..T-5 (no slot T, no day ctx, h=6). Q: legal onset IoU = (LB-0.71192)/0.15" >> $LOG 2>&1
for i in $(seq 1 60); do
  sleep 30
  $K competitions submissions $C 2>/dev/null | awk 'NR>2' | head -1 | grep -q "sub_U1.*COMPLETE" && break
done
echo "$(date -u) U1 status: $($K competitions submissions $C 2>/dev/null | awk 'NR>2' | head -1)" >> $LOG
$K competitions submit $C -f subs/sub_U2.zip -m "U2 legal: U1 with ONGOING = forward-filled last released history obs per link incl. ineligible (T-60..T-5 only). Q: gain vs organizer persistence = (LB_U2-LB_U1)/0.15" >> $LOG 2>&1
echo "$(date -u) U2 submitted" >> $LOG
