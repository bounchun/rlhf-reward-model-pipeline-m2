#!/usr/bin/env bash
# One-off creation of the two buckets described in README sections 1, 3 and 4.
#   usage:  PROJECT=my-gcp-project BUCKET=bc-dhai-hhrlhf bash scripts/setup_gcs.sh
# Needs: gcloud CLI, logged in with rights to create buckets.
set -euo pipefail
: "${PROJECT:?set PROJECT}" "${BUCKET:?set BUCKET}"
HOLDOUT="${HOLDOUT:-${BUCKET}-holdout}"
REGION="${REGION:-europe-west1}"
HERE="$(cd "$(dirname "$0")" && pwd)"

# A new project has these APIs off: Cloud Storage (buckets) and IAM (service accounts)
gcloud services enable storage.googleapis.com iam.googleapis.com --project="$PROJECT"

create () {  # name, storage class
  gcloud storage buckets create "gs://$1" --project="$PROJECT" --location="$REGION" \
    --default-storage-class="$2" --uniform-bucket-level-access --public-access-prevention
  gcloud storage buckets update "gs://$1" --versioning \
    --lifecycle-file="$HERE/gcs_lifecycle.json"
}

create "$BUCKET"  STANDARD   # raw, train, dev, manifests, models   (hot)
create "$HOLDOUT" COLDLINE   # test + future reserve                  (rarely read)

# Least-privilege service accounts (README section 5)
for SA in pipeline-sa train-sa; do
  gcloud iam service-accounts create "$SA" --project="$PROJECT" 2>/dev/null || true
done
PIPE="pipeline-sa@${PROJECT}.iam.gserviceaccount.com"
TRAIN="train-sa@${PROJECT}.iam.gserviceaccount.com"
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET"  --member="serviceAccount:$PIPE"  --role=roles/storage.objectAdmin
gcloud storage buckets add-iam-policy-binding "gs://$HOLDOUT" --member="serviceAccount:$PIPE"  --role=roles/storage.objectAdmin
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET"  --member="serviceAccount:$TRAIN" --role=roles/storage.objectViewer
# train-sa deliberately gets NO role on the holdout bucket.

echo "Done. Buckets: gs://$BUCKET (Standard), gs://$HOLDOUT (Coldline); versioning + lifecycle on both."
