#!/usr/bin/env bash
# Fetch the public datasets this audit reanalyses.
# data_raw/ is deliberately not tracked in git: it is third-party downloaded
# content, and the analysis must always start from the upstream copy.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$ROOT/data_raw"
mkdir -p "$RAW/anoxpepred" "$RAW/aop_drl"

if [ ! -d "$RAW/anoxpepred/repo" ]; then
  git clone --depth 1 https://github.com/TobiasHeOl/AnOxPePred.git "$RAW/anoxpepred/repo"
else
  echo "anoxpepred: already present"
fi

if [ ! -d "$RAW/aop_drl/repo" ]; then
  git clone --depth 1 https://github.com/Zhouyongzhu/AOP_DRL.git "$RAW/aop_drl/repo"
else
  echo "aop_drl: already present"
fi

echo
echo "benchmark files:"
ls -1 "$RAW/anoxpepred/repo/Data"
echo
echo "Expected: 1404 peptides in each of 03_p{60,70,80,90}_AO_db.csv"
