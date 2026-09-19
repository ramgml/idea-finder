-- Migration 007: Yandex Wordstat demand validation (task T321).
--
-- wordstat_query: cache of per-cluster phrase frequencies returned by
-- Wordstat. One row per (cluster_id, phrase); revalidation UPSERTs the
-- frequency and stamps checked_at, so freshness is `checked_at >=
-- now() - interval` and a retry of the validate stage is idempotent
-- (same phrase -> same row, never duplicated).
--
-- wordstat_settings: SEPARATE settings table for the Yandex Wordstat API
-- token (owner constraint: never mix into llm_provider — different
-- provider, different validation). Single-row-per-service semantics keyed
-- by name ('yandex'); the secret lives ONLY here, the dashboard/repo
-- expose it masked (last 4 characters max, settings_view.mask_key style).
--
-- Idempotent by design (IF NOT EXISTS); the runner in db/migrate.py
-- records version 007, so re-execution is a no-op.

CREATE TABLE IF NOT EXISTS wordstat_query (
    id         uuid PRIMARY KEY DEFAULT uuid_generate_v7(),
    cluster_id uuid NOT NULL REFERENCES cluster(id),
    phrase     text NOT NULL,
    frequency  integer NOT NULL CHECK (frequency >= 0),
    checked_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (cluster_id, phrase)
);

-- The validate stage picks stale clusters by max(checked_at); the
-- dashboard reads fresh rows per cluster.
CREATE INDEX IF NOT EXISTS idx_wordstat_query_cluster_id
    ON wordstat_query (cluster_id);
CREATE INDEX IF NOT EXISTS idx_wordstat_query_checked_at
    ON wordstat_query (checked_at);

CREATE TABLE IF NOT EXISTS wordstat_settings (
    id         uuid PRIMARY KEY DEFAULT uuid_generate_v7(),
    name       text NOT NULL UNIQUE,
    api_key    text NOT NULL DEFAULT '',  -- lives ONLY here, never in .env/code
    created_at timestamptz NOT NULL DEFAULT now()
);
