#!/usr/bin/env bash
# Proves the access rules by acting AS train-sa (service-account impersonation).
#   usage:  PROJECT=rlhf-reward-pipeline-bc BUCKET=bc-rlhf-reward-2026 bash scripts/check_iam.sh
# One-off prerequisite: your own account needs roles/iam.serviceAccountTokenCreator on
# train-sa (the README, M2.2, gives the command). Only object metadata is read, never
# dataset text, so the output is safe to commit as docs/iam_check_<version>.txt.
set -uo pipefail
: "${PROJECT:?set PROJECT}" "${BUCKET:?set BUCKET}"
HOLDOUT="${HOLDOUT:-${BUCKET}-holdout}"
VERSION="${VERSION:-v1.1}"
REV="${REV:-09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa}"
TRAIN="train-sa@${PROJECT}.iam.gserviceaccount.com"
OUT="${OUT:-docs/iam_check_${VERSION}.txt}"
FAIL=0

check () {  # expected (ALLOW|DENY), label, gcloud storage args...
  local want="$1" label="$2"; shift 2
  local err got
  if err=$(gcloud storage "$@" --impersonate-service-account="$TRAIN" 2>&1 >/dev/null); then
    got=ALLOW
  elif grep -qiE '403|AccessDenied|does not have|Permission' <<<"$err"; then
    got=DENY
  else
    got="ERROR ($(head -n1 <<<"$err"))"   # e.g. 404: wrong path, not a permission result
  fi
  if [[ "$got" == "$want" ]]; then res=PASS; else res=FAIL; FAIL=1; fi
  printf '%-4s  expected %-5s got %-5s  %s\n' "$res" "$want" "$got" "$label" | tee -a "$OUT"
}

mkdir -p "$(dirname "$OUT")"
{ echo "IAM check as $TRAIN, $(date -u +%Y-%m-%dT%H:%M:%SZ)"; echo; } > "$OUT"

check ALLOW "read processed/$VERSION/train.parquet"  objects describe "gs://$BUCKET/processed/$VERSION/train.parquet"
check ALLOW "list processed/$VERSION/"               ls "gs://$BUCKET/processed/$VERSION/"
check DENY  "read raw/ (contains test, F1, F2 rows)" objects describe "gs://$BUCKET/raw/hh-rlhf/$REV/helpful-base/test.jsonl.gz"
check DENY  "list the main bucket root"              ls "gs://$BUCKET/"
check DENY  "read quarantine/$VERSION/"              objects describe "gs://$BUCKET/quarantine/$VERSION/quarantine.parquet"
check DENY  "read holdout $VERSION/test.parquet"     objects describe "gs://$HOLDOUT/$VERSION/test.parquet"
check DENY  "list the holdout bucket"                ls "gs://$HOLDOUT/"
check DENY  "write to processed/"                    objects update "gs://$BUCKET/processed/$VERSION/train.parquet" --custom-metadata=iam-check=1

echo
if [[ $FAIL -eq 0 ]]; then echo "All checks passed. Saved to $OUT"; else echo "Some checks FAILED. See $OUT"; fi
exit $FAIL
