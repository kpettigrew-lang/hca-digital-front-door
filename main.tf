terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

variable "project_id" {
  type        = string
  description = "Google Cloud Project ID"
  default     = "hca-digital-front-door-prod"
}

variable "region" {
  type        = string
  default     = "us-central1"
}

resource "google_secret_manager_secret" "gemini_api_key" {
  secret_id = "gemini-api-key"
  replication {
    auto {}
  }
}

resource "google_cloud_run_v2_service" "agent_service" {
  name     = "hca-digital-front-door"
  location = var.region

  template {
    containers {
      image = "gcr.io/${var.project_id}/hca-digital-front-door:latest"
      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
      }
      env {
        name = "GEMINI_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.gemini_api_key.secret_id
            version = "latest"
          }
        }
      }
    }
  }
}
