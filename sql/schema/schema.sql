CREATE TABLE champions (
  id           INT PRIMARY KEY,          
  key          TEXT NOT NULL UNIQUE,    
  name         TEXT NOT NULL
);

CREATE TABLE champion_tag (
  champion_id  INT NOT NULL REFERENCES champions(id),
  tag          TEXT NOT NULL,
  PRIMARY KEY (champion_id, tag)
);

CREATE TABLE accounts (
  puuid        TEXT PRIMARY KEY,
  status       TEXT NOT NULL DEFAULT 'active', 
  depth        INT  NOT NULL DEFAULT 0,        
  last_crawled TIMESTAMPTZ,
  tier         TEXT,
  rank         TEXT
);

CREATE TABLE matches (
  match_id      TEXT PRIMARY KEY,
  patch         TEXT NOT NULL,          
  queue_id      INT  NOT NULL,
  game_datetime TIMESTAMPTZ NOT NULL
);

CREATE TABLE participants (
  match_id                TEXT NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
  puuid                   TEXT NOT NULL REFERENCES accounts(puuid),
  champion_id             INT  NOT NULL REFERENCES champions(id),
  team_id                 INT  NOT NULL,            
  win                     BOOLEAN NOT NULL,

  total_damage_dealt      BIGINT,
  physical_damage_dealt   BIGINT,
  magic_damage_dealt      BIGINT,
  true_damage_dealt       BIGINT,
  damage_taken            BIGINT,
  gold_earned             BIGINT,
  heals                   BIGINT,
  shields                 BIGINT,
  kills                   INT,
  deaths                  INT,
  assists                 INT,

  PRIMARY KEY (match_id, puuid)
);

CREATE TABLE participant_items (
  match_id  TEXT NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
  puuid     TEXT NOT NULL REFERENCES accounts(puuid),
  item_id   INT  NOT NULL,
  slot      INT  NOT NULL,               -- 0..6
  PRIMARY KEY (match_id, puuid, slot),
  FOREIGN KEY (match_id, puuid) REFERENCES participants(match_id, puuid) ON DELETE CASCADE
);


CREATE TABLE champion_aram_mods (
    champion_id    INT NOT NULL REFERENCES champions(id) ON DELETE CASCADE,
    patch          TEXT NOT NULL,
    ability_haste  REAL,
    dmg_dealt      REAL,
    dmg_taken      REAL,
    healing        REAL,
    shielding      REAL,
    tenacity       REAL, 
    attack_speed   REAL,
    energy_regen   REAL,
    PRIMARY KEY (champion_id, patch)
);

CREATE TABLE champion_spell_changes (
    champion_id   INT  NOT NULL REFERENCES champions(id) ON DELETE CASCADE,
    patch         TEXT NOT NULL,
    spell_key     TEXT NOT NULL,          
    idx           INT  NOT NULL,         
    change_text   TEXT NOT NULL,
    PRIMARY KEY (champion_id, patch, spell_key, idx)
);
