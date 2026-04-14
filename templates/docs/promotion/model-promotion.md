# Steps for Promoting Models Across Environments via CI/CD

## Introduction

This guide provides a comprehensive overview of the CI/CD based flow for promoting models across GCP environments i.e. train, stage and prod.

A core part of a Maturity Level 2 MLOps Framework and MLOps Dev Kit is Continuous Integration & Continuous Deployment for the AI Pipelines AND environments.

## Core Concepts

Before diving into the specifics it's important to understand the core concepts related to this topic. The main tenet of the flow is that a code change or update in the pod Application Repository aka MDK Repository (this repository) should trigger a Vertex AI Pipeline run for training and / or inference based on the specific config for the environment.

There are 2 aspects to this flow, let's tackle them one by one.

### 1. Github UI based triggering

You can (re)trigger your Vertex AI Pipeline runs from Github in 2 ways, Github UI Workflow Dispatch or Pull Requests. Read on for Github UI Workflow Dispatch and move to the next point for the Pull Requests flow.

A. Github UI Workflow Dispatch
▶️ In order to trigger Vertex AI Pipelines from the Github UI, navigate to "Actions" tab at the top of the repo.
▶️ Click on the "Environment Promotion Workflow" in the left pane.
▶️ Look for the banner saying "This workflow has a workflow_dispatch event trigger."
▶️ Drop down on "Run workflow" and select the right "branch" eg. 'train' and the right "GCP environment to run the workflow in" i.e. 'train, stage or prod'.

Note: You can test the code from a feature branch in the train environment using this approach. By selecting 'Branch: feature-test' and 'GCP environment to run the workflow in: train'.

### 2. Pull Requests based model promotion (all environments)

In ALL environments (train, stage & prod), a merged Pull Request to one of the branches (train, stage or main) triggers a Vertex AI Pipeline run in the subsequent environment.

▶️ Create a Pull request on the train, stage or main branch.
▶️ Once the Pull Request is merged, navigate to the "Actions" tab in Github to view the status of the automated GHA run.
▶️ On successful completion navigate to Vertex AI Pipelines UI in the relevant GCP project to view the status of the Vertex AI Pipeline run.

If you want to double-click on how this is facilitated, please read on. Otherwise, you can choose to stop reading here if you are a Data Science end user of the framework.

## Building Blocks

There are 3 building blocks that make up this flow. This section will give you a deeper understanding of what is behind the automated CI/CD based Vertex AI Pipelines triggering.

### 1. Standard GitOps Strategy

As you know, there are Reusable Github Actions (GHA) that are auto baked-in to the Application Repositories via Developer Portal based automation. The reason these GHAs work each time is because all of our MDK repositories follow the same branching and GitOps strategy. The strategy is to have 3 branches per repo (train, stage and main) that each correspond to a GCP environment (train, stage and prod). In addition, it is assumed that merges from feature branches are allowed only into the train branch. This standardized strategy helps in setting up the automated CI/CD workflow.

### 2. Reusable Github Actions

In each of the pod Application repositories that get created there are 2 essential workflows that are leveraged for enabling the CI/CD based model promotion. One of the workflow is the caller workflow and the other is the workflow that gets invoked (re-usable). The workflows can be located in the "root/.github/workflows" directory.

1. Environment Promotion Workflow ('environment-promotion.yml')
Purpose:
- Detects changes in the repository and triggers the reusable workflow with the right inputs.
- Modularization and Scalability, having a decoupled caller workflow enables us in the future to have the ability to add more reusable workflows easily to the desired Git events.

Triggers:
Push & pull_request events on train, stage and prod branches.

Key Steps:
Triggers Build, Deploy, and Submit Pipeline workflow based on detected changes.

2. Update, Build & Deploy Pipelines Workflow ('build_and_deploy_pipelines.yml')

Purpose:
Re-usable workflow that builds container image, pipeline and submits required params to Pub/Sub and image to Artifact Registry.

Triggers:
UI: Only on train
On call: Triggered via the caller workflow on train, stage and prod

Key Steps:
On pull:
Checks code.
Builds container.
Scans Image w/ Wiz.

On push:
Pushes image to GAR.
Builds pipeline spec.
Compiles params.json into a pub/sub message to be handed to Cloud Run.

### 3. Submission Service (Cloud Run)
The Github Actions Workflows directly interact with Artifact Registry, Google Cloud Storage buckets and Cloud Pub/Sub. But they do not directly interact with Vertex AI Pipelines, there is a middleman here "Cloud Run based Submission Service".

Purpose:
- The Submission Service allows in not creating bottlenecks for triggering multiple pipelines.
- The Submission Service creates a central point for multiple sources to trigger Vertex AI Pipelines, eg. Cloud Scheduler can post a message on the same Pub/Sub topic to trigger Vertex AI Pipelines.

Key Steps:
- Cloud Run actively listens to the Pub/Sub topic.
- Received the pipeline params and GCS URI, passes these to Vertex AI.
- Vertex AI pulls the container image from AR and pipeline parameters from GCS and executes.
