#!/bin/bash
# Plan v4 section 4a scheduler, v2 (CRITIQUE.md section 0.8 fixes). Logs to ~/tfb/logs/submit_u1u2.log
#  - absolute target 2026-10-01 00:01 UTC (a restart during Oct 1 submits immediately instead of waiting a day)
#  - zips must match the checksums verified on 2026-09-30 (no silent rebuild)
#  - U2 is uploaded only when U1 finished COMPLETE (an ERRORed U1 would leave U2 - 0.71192 an uninterpretable mixture)
#  - a failed submit call is retried only if the file is not already in the submission list (no double spend)
cd ~/tfb; K=~/venv/bin/kaggle; C=2026-ieee-big-data-traffic-flow-bench; LOG=~/tfb/logs/submit_u1u2.log
log() { echo "$(date -u '+%F %T') $*" >> $LOG; }
top() { $K competitions submissions $C 2>/dev/null | awk 'NR>2' | head -5; }
declare -A MD5=([subs/sub_U1.zip]=cdfe418cf042bbb6ae4095e0311069f9 [subs/sub_U2.zip]=817ff499b581f51e3e9f64e917fe7360)
for f in "${!MD5[@]}"; do [ "$(md5sum $f | cut -d' ' -f1)" = "${MD5[$f]}" ] || { log "checksum mismatch $f, abort"; exit 1; }; done
target=$(date -u -d "2026-10-01 00:01:00" +%s); now=$(date -u +%s)
[ $target -gt $now ] && { log "v2 scheduler sleeping $((target-now)) s"; sleep $((target-now)); }
submit() {  # $1 file $2 message
  local name=$(basename $1)
  for a in 1 2 3; do
    $K competitions submit $C -f $1 -m "$2" >> $LOG 2>&1
    sleep 20; top | grep -q "$name" && { log "$name accepted (attempt $a)"; return 0; }
    log "$name not in list after attempt $a"; sleep 60
  done
  return 1
}
wait_done() {  # $1 file name; echoes COMPLETE / ERROR / TIMEOUT
  for i in $(seq 1 240); do
    row=$(top | grep "$1" | head -1)
    case "$row" in *COMPLETE*) echo COMPLETE; return;; *ERROR*) echo ERROR; return;; esac
    sleep 30
  done; echo TIMEOUT
}
log "submitting U1"
submit subs/sub_U1.zip "U1 legal: R0 (GBDT state + organizer persistence + V1 ODME) with ONSET = train-only LGB on history T-60..T-5 (no slot T, no day ctx, h=6). Q: legal onset IoU = (LB-0.71192)/0.15" || { log "U1 submit failed 3x, stop"; exit 1; }
s=$(wait_done sub_U1.zip); log "U1 status $s: $(top | grep sub_U1.zip | head -1)"
[ "$s" = ERROR ] && { log "U1 ERROR: U2 withheld (would be an uninterpretable mixture)"; exit 1; }
log "submitting U2"
submit subs/sub_U2.zip "U2 legal: U1 with ONGOING = forward-filled last released history obs per link incl. ineligible (T-60..T-5 only). Q: gain vs organizer persistence = (LB_U2-LB_U1)/0.15" || { log "U2 submit failed 3x"; exit 1; }
s=$(wait_done sub_U2.zip); log "U2 status $s: $(top | grep sub_U2.zip | head -1)"
