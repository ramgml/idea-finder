-- Migration 005: terminal 'extracted' fetch_status (task T307).
--
-- The extract stage needs a TERMINAL marker for posts it has processed but
-- that yielded zero accepted pains (a valid outcome, not a failure). Without
-- it such posts stay in the pending selection forever: the stage re-sends
-- them to the LLM on every batch/run (infinite retry loop, the exact bug
-- this migration fixes).
--
-- Status lifecycle after this migration:
--   new       collected, body not yet extracted            (pending for extract)
--   fetched   body fetched, not yet extracted              (pending for extract)
--   extracted extract ran to a terminal outcome            (terminal)
--   failed    extract answer was hopeless, never retried   (terminal)
--
-- Idempotent by design: the CHECK constraint is dropped/re-added only if
-- missing; the migration runner records version 005, so re-execution is a
-- no-op.

ALTER TABLE raw_post DROP CONSTRAINT IF EXISTS raw_post_fetch_status_check;
ALTER TABLE raw_post ADD CONSTRAINT raw_post_fetch_status_check
    CHECK (fetch_status IN ('new', 'fetched', 'extracted', 'failed'));
