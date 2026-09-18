-- Migration 004: per-run prompt version tracking (task T307).
--
-- run.prompt_version_id records which prompt_version the extract stage used
-- for this pipeline execution. Together with pain.prompt_version_id this
-- makes runs reproducible and lets the dashboard compare pain sets across
-- prompt versions (G1 gate).
--
-- Nullable on purpose: legacy runs keep working (NULL = "registered before
-- versioning"), and repo.update_run_stage fills the column write-once via
-- COALESCE (a later NULL update never overwrites a recorded version).
--
-- Idempotent by design (IF NOT EXISTS); the runner in db/migrate.py records
-- version 004, so re-execution is a no-op.

ALTER TABLE run ADD COLUMN IF NOT EXISTS prompt_version_id uuid REFERENCES prompt_version(id);
