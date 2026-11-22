#!/usr/bin/env bash
# One-off creation of the two buckets and the two service accounts (README M2.2).
#   usage:  PROJECT=my-gcp-project BUCKET=bc-rlhf-reward-2026 bash scripts/setup_gcs.sh
# Needs: gcloud CLI, logged in with rights to create buckets and service accounts.
# Safe to rerun: existing buckets, accounts and bindings are left as they are.
set -euo pipefail
: "${PROJECT:?set PROJECT}" "${BUCKET:?set BUCKET}"
HOLDOUT="${HOLDOUT:-${BUCKET}-holdout}"
REGION="${REGION:-europe-west1}"
HERE="$(cd "$(dirname "$0")" && pwd)"

# A new project has these APIs off: Cloud Storage (buckets) and IAM (service accounts)
gcloud services enable storage.googleapis.com iam.googleapis.com --project="$PROJECT"

create () {  # name, storage class
  if gcloud storage buckets describe "gs://$1" --project="$PROJECT" >/dev/null 2>&1; then
    echo "Bucket gs://$1 already exists, keeping it"
  else
    gcloud storage buckets create "gs://$1" --project="$PROJECT" --location="$REGION" \
      --default-storage-class="$2" --uniform-bucket-level-access --public-access-prevention
  fi
  gcloud storage buckets update "gs://$1" --versioning \
    --lifecycle-file="$HERE/gcs_lifecycle.json"
}

create "$BUCKET"  STANDARD   # raw, quarantine, processed, audit, manifests   (hot)
create "$HOLDOUT" COLDLINE   # test + future reserve F1, F2                   (rarely read)

# Least-privilege service accounts
for SA in pipeline-sa train-sa; do
  gcloud iam service-accounts create "$SA" --project="$PROJECT" 2>/dev/null || true
done
PIPE="pipeline-sa@${PROJECT}.iam.gserviceaccount.com"
TRAIN="train-sa@${PROJECT}.iam.gserviceaccount.com"

# pipeline-sa writes everywhere
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET"  --member="serviceAccount:$PIPE" --role=roles/storage.objectAdmin
gcloud storage buckets add-iam-policy-binding "gs://$HOLDOUT" --member="serviceAccount:$PIPE" --role=roles/storage.objectAdmin

# train-sa reads processed/ ONLY.
# A bucket-level objectViewer role would also let it read raw/ (all 169k pairs, so the
# test, F1 and F2 rows too) and quarantine/, which would defeat the holdout bucket.
# A managed folder carries its own IAM policy that covers every object under its prefix.
# Managed folders need uniform bucket-level access, which create() switches on.
gcloud storage managed-folders create "gs://$BUCKET/processed/" 2>/dev/null \
  || echo "Managed folder gs://$BUCKET/processed/ already exists, keeping it"
gcloud storage managed-folders add-iam-policy-binding "gs://$BUCKET/processed/" \
  --member="serviceAccount:$TRAIN" --role=roles/storage.objectViewer
# train-sa deliberately gets NO role on the main bucket itself and NO role on the holdout bucket.

echo "Done. Buckets: gs://$BUCKET (Standard), gs://$HOLDOUT (Coldline); versioning + lifecycle on both."
echo "train-sa can read gs://$BUCKET/processed/ only. Check it with: bash scripts/check_iam.sh"
