-- Migration 002: optional pricing for LLM providers.
--
-- price_per_mtok: blended price per 1M tokens (numeric(12,4), NULL = free or
-- unknown). Cost accounting (idea_finder.llm.client.estimate_cost) reads it
-- through repo.get_active_llm_provider; NULL yields zero cost. The fake
-- provider is always free regardless of this column.
--
-- Idempotent by design (IF NOT EXISTS); the runner in db/migrate.py records
-- version 002, so re-execution is a no-op.

ALTER TABLE llm_provider ADD COLUMN IF NOT EXISTS price_per_mtok numeric(12, 4) NULL;
