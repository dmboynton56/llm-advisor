#!/usr/bin/env bash
# Deploy Cloud Function + Cloud Scheduler jobs for llm-advisor workflow dispatch.
set -euo pipefail

PROJECT="${GCP_PROJECT:-gen-lang-client-0189185649}"
REGION="${GCP_REGION:-us-central1}"
FUNCTION_NAME="${FUNCTION_NAME:-trigger-github-workflow}"
SECRET_NAME="${SECRET_NAME:-github-actions-dispatch-token}"
SCHEDULER_SA="${SCHEDULER_SA:-llm-advisor-scheduler@${PROJECT}.iam.gserviceaccount.com}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="${ROOT}/trigger-github-workflow"

echo "==> Project: ${PROJECT}  Region: ${REGION}"

gcloud config set project "${PROJECT}"

echo "==> Enabling required APIs..."
gcloud services enable \
  cloudfunctions.googleapis.com \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  cloudscheduler.googleapis.com \
  secretmanager.googleapis.com \
  iam.googleapis.com \
  appengine.googleapis.com

if ! gcloud app describe >/dev/null 2>&1; then
  echo "==> Creating App Engine app (required for Cloud Scheduler in ${REGION})..."
  gcloud app create --region="${REGION}" --quiet
fi

if ! gcloud iam service-accounts describe "${SCHEDULER_SA}" >/dev/null 2>&1; then
  echo "==> Creating scheduler service account..."
  gcloud iam service-accounts create llm-advisor-scheduler \
    --display-name="LLM Advisor Cloud Scheduler"
fi

echo "==> Deploying Cloud Function (Gen2)..."
gcloud functions deploy "${FUNCTION_NAME}" \
  --gen2 \
  --project="${PROJECT}" \
  --region="${REGION}" \
  --runtime=python311 \
  --source="${SOURCE}" \
  --entry-point=trigger \
  --trigger-http \
  --no-allow-unauthenticated \
  --set-secrets="GITHUB_TOKEN=${SECRET_NAME}:latest"

FUNCTION_URL="$(
  gcloud functions describe "${FUNCTION_NAME}" \
    --gen2 \
    --region="${REGION}" \
    --format='value(serviceConfig.uri)'
)"

echo "==> Function URL: ${FUNCTION_URL}"

echo "==> Granting run.invoker to scheduler SA..."
gcloud run services add-iam-policy-binding "${FUNCTION_NAME}" \
  --region="${REGION}" \
  --member="serviceAccount:${SCHEDULER_SA}" \
  --role="roles/run.invoker" \
  --quiet

PROJECT_NUM="$(gcloud projects describe "${PROJECT}" --format='value(projectNumber)')"
SCHEDULER_AGENT="service-${PROJECT_NUM}@gcp-sa-cloudscheduler.iam.gserviceaccount.com"
COMPUTE_SA="${PROJECT_NUM}-compute@developer.gserviceaccount.com"

echo "==> Granting Secret Manager access to Cloud Function runtime SA..."
gcloud secrets add-iam-policy-binding "${SECRET_NAME}" \
  --member="serviceAccount:${COMPUTE_SA}" \
  --role="roles/secretmanager.secretAccessor" \
  --quiet >/dev/null || true

echo "==> Granting Cloud Scheduler OIDC permissions on scheduler SA..."
for ROLE in roles/iam.serviceAccountUser roles/iam.serviceAccountTokenCreator; do
  gcloud iam service-accounts add-iam-policy-binding "${SCHEDULER_SA}" \
    --member="serviceAccount:${SCHEDULER_AGENT}" \
    --role="${ROLE}" \
    --quiet >/dev/null
done

create_or_update_job() {
  local job_name="$1"
  local workflow="$2"
  local cron="$3"
  local uri="${FUNCTION_URL}?workflow=${workflow}"

  if gcloud scheduler jobs describe "${job_name}" --location="${REGION}" >/dev/null 2>&1; then
    echo "==> Updating scheduler job ${job_name}..."
    gcloud scheduler jobs update http "${job_name}" \
      --location="${REGION}" \
      --schedule="${cron}" \
      --time-zone="America/New_York" \
      --uri="${uri}" \
      --http-method=GET \
      --oidc-service-account-email="${SCHEDULER_SA}" \
      --oidc-token-audience="${FUNCTION_URL}"
  else
    echo "==> Creating scheduler job ${job_name}..."
    gcloud scheduler jobs create http "${job_name}" \
      --location="${REGION}" \
      --schedule="${cron}" \
      --time-zone="America/New_York" \
      --uri="${uri}" \
      --http-method=GET \
      --oidc-service-account-email="${SCHEDULER_SA}" \
      --oidc-token-audience="${FUNCTION_URL}" \
      --attempt-deadline=180s
  fi
}

create_or_update_job "llm-advisor-premarket" "premarket" "20 9 * * 1-5"
create_or_update_job "llm-advisor-live-loop" "live" "27 9 * * 1-5"

echo ""
echo "Done."
echo "  Function: ${FUNCTION_URL}"
echo "  Test:     gcloud scheduler jobs run llm-advisor-premarket --location=${REGION}"
echo "  Premarket cron: 09:20 America/New_York Mon-Fri"
echo "  Live cron:      09:27 America/New_York Mon-Fri"
