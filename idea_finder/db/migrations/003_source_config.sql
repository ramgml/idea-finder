-- Migration 003: source configuration columns (task T301).
--
-- enabled: lets an operator switch a seeded adapter off (dashboard page
-- "Источники", future UI) without deleting the row or breaking FKs from
-- raw_post. The collect stage reads only enabled sources.
--
-- rate_limit_rps: per-domain politeness cap in requests per second, consumed
-- by the async fetch layer (httpx + aiolimiter). Project default is 1 rps
-- (AGENTS.md, sources rule 3); numeric(4, 2) covers 0.01..99.99.
--
-- Existing rows keep working: NOT NULL + DEFAULT true / 1.0 backfills them.
-- Idempotent by design (IF NOT EXISTS); the runner in db/migrate.py records
-- version 003, so re-execution is a no-op.

ALTER TABLE source ADD COLUMN IF NOT EXISTS enabled boolean NOT NULL DEFAULT true;

ALTER TABLE source ADD COLUMN IF NOT EXISTS rate_limit_rps numeric(4, 2) NOT NULL DEFAULT 1.0;
