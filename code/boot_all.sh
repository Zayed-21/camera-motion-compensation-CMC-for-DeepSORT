#!/bin/bash
# one worker: run the bootstrap groups in order within ~150 s (resumable)
k=$1; start=$(date +%s)
for job in "work full" "work stride2" "work stride3" "work stride6" "work_frcnn full"; do
  set -- $job; rem=$((120-($(date +%s)-start)))
  [ $rem -lt 15 ] && break
  WORK=$PWD/$1 GROUP=$2 WORKER=$k NWORKERS=4 BUDGET=$rem python3 bootstrap.py 1000 2>&1 | tail -1 | sed "s/^/$1 $2 w$k: /"
done
