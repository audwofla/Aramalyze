CREATE TABLE IF NOT EXISTS match_queue (
    match_id TEXT PRIMARY KEY,
    discovered_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    fetched_at TIMESTAMP,

    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'processing', 'done', 'error')),

    discovered_from_puuid TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,

    FOREIGN KEY (discovered_from_puuid)
        REFERENCES accounts(puuid)
        ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_match_queue_status
    ON match_queue(status);
