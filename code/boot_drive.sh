#!/bin/bash
# one call (~150 s): advance the first unfinished bootstrap job with 4 parallel workers (resumable)
cd $HOME/exp/code
JOBS=(
"work_m15ptz syn_ptz20 base,cmc_s0 m15s"
"work_m15ptz syn_ptz40 base,cmc_s0 m15s"
"work_m15ptz syn_ptz10 base,cmc_s0 m15s"
"work_m15ptz syn_sway50 base,cmc_s0 m15s"
"work_surv_frcnn syn_ptz40 base,cmc_s0 -"
"work_surv_frcnn syn_ptz20 base,cmc_s0 -"
"work_surv_frcnn syn_sway50 base,cmc_s0 -"
"work_sports full base,cmc_s0 sp"
"work_surv_frcnn syn_ptz10 base,cmc_s0 -"
"work_surv_frcnn syn_ptz5 base,cmc_s0 -"
"work_surv_frcnn syn_sway25 base,cmc_s0 -"
"work_surv_frcnn syn_sway10 base,cmc_s0 -"
"work_dt stride6 base,cmc_s0,ecc,hybrid -"
"work_frcnn_dt stride6 base,cmc_s0,ecc,hybrid -"
"work_mot15 full base,cmc_s0,trans,ecc,hybrid m15"
"work_mot15 stride6 base,cmc_s0,trans,ecc,hybrid m15"
"work_mot15_dt stride6 base,cmc_s0,ecc,hybrid m15"
"work_dt stride3 base,cmc_s0,ecc,hybrid -"
"work_frcnn_dt stride3 base,cmc_s0,ecc,hybrid -"
"work_mot15 stride3 base,cmc_s0,trans,ecc,hybrid m15"
"work_mot15_dt stride3 base,cmc_s0,ecc,hybrid m15"
"work_dt stride2 base,cmc_s0,ecc,hybrid -"
"work_frcnn_dt stride2 base,cmc_s0,ecc,hybrid -"
"work_mot15_acf full base,cmc_s0,trans,ecc,hybrid m15"
"work_surv syn_ptz40 base,cmc_s0 -"
"work_surv syn_ptz20 base,cmc_s0 -"
"work_surv syn_sway50 base,cmc_s0 -"
"work_surv syn_ptz10 base,cmc_s0 -"
"work_surv syn_ptz5 base,cmc_s0 -"
"work_surv syn_sway25 base,cmc_s0 -"
"work_surv syn_sway10 base,cmc_s0 -"
"work_mot15_acf stride3 base,cmc_s0,trans,ecc,hybrid m15"
"work_mot15_acf stride6 base,cmc_s0,trans,ecc,hybrid m15")
for j in "${JOBS[@]}"; do
  set -- $j
  n=$(cat $1/boot/$2_w*.csv 2>/dev/null | cut -d, -f1 | grep -v r | sort -u | wc -l)
  if [ $n -lt 1000 ]; then
    if [ $4 = m15 ]; then E="SEQ_ROOT=$HOME/exp/mot15 SEQS=ETH-Bahnhof,ETH-Sunnyday,KITTI-13"; elif [ $4 = m15s ]; then E="SEQ_ROOT=$HOME/exp/mot15s SEQS=PETS09-S2L1,TUD-Stadtmitte,TUD-Campus,KITTI-17"; elif [ $4 = sp ]; then E="SEQ_ROOT=$HOME/exp/sports SEQS=$(cat $HOME/exp/sports/SEQS)"; else E=""; fi
    for k in 0 1 2 3; do env $E WORK=$PWD/$1 GROUP=$2 BTRACKERS=$3 WORKER=$k NWORKERS=4 BUDGET=${B:-140} python3 bootstrap.py 1000 > /dev/null 2>&1 & done; wait
    n=$(cat $1/boot/$2_w*.csv | cut -d, -f1 | grep -v r | sort -u | wc -l); echo "$1 $2: $n/1000"; exit
  fi
done; echo ALL DONE
