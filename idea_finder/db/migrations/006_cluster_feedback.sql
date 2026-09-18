-- Migration 006: cluster feedback labels (task T315).
--
-- The dashboard lets the owner triage clusters: mark one «интересно»
-- (interesting), hide a false-positive merge («скрыть»), or flag a cluster
-- whose pains actually belong to several distinct pains («это не одна
-- боль»). Both are per-cluster labels used for threshold calibration and
-- to clean up the default list view:
--
--   feedback    NULL | 'interesting' | 'hidden'
--               NULL clears the label; 'hidden' clusters disappear from
--               the list unless the «показывать скрытые» filter is on.
--   split_flag  TRUE marks «это не одна боль». Deliberately non-destructive:
--               the cluster is NOT deleted or re-split here (re-clustering
--               is the C-flow's job); the flag is the calibration signal.
--
-- Idempotent by design: ADD COLUMN IF NOT EXISTS; the CHECK is dropped and
-- re-added only if missing; the migration runner records version 006, so
-- re-execution is a no-op.

ALTER TABLE cluster ADD COLUMN IF NOT EXISTS feedback text;
ALTER TABLE cluster DROP CONSTRAINT IF EXISTS cluster_feedback_check;
ALTER TABLE cluster ADD CONSTRAINT cluster_feedback_check
    CHECK (feedback IS NULL OR feedback IN ('interesting', 'hidden'));
ALTER TABLE cluster ADD COLUMN IF NOT EXISTS split_flag boolean NOT NULL DEFAULT false;
