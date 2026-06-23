CREATE TABLE toponyms.entries (
    id              TEXT    NOT NULL,
    language_code   TEXT    NOT NULL,
    name            TEXT    NOT NULL,
    name_romanised  TEXT,
    language        TEXT,
    countries       TEXT[],
    types           TEXT[],
    PRIMARY KEY (id, language_code)
);

CREATE INDEX ON toponyms.entries(language_code);
CREATE INDEX ON toponyms.entries(countries) USING GIN;
CREATE INDEX ON toponyms.entries(types) USING GIN;