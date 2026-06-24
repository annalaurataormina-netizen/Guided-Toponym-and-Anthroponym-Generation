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
CREATE INDEX ON toponyms.entries(language);
CREATE INDEX ON toponyms.entries(countries) USING GIN (countries);
CREATE INDEX ON toponyms.entries(types) USING GIN (types);

CREATE TABLE anthroponyms.entries (
    id              TEXT    NOT NULL,
    language_code   TEXT    NOT NULL,
    name            TEXT    NOT NULL,
    name_romanised  TEXT,
    language        TEXT,
    countries       TEXT[],
    types           TEXT[],
    PRIMARY KEY (id, language_code)
);

CREATE INDEX ON anthroponyms.entries(language_code);
CREATE INDEX ON anthroponyms.entries(language);
CREATE INDEX ON anthroponyms.entries(countries) USING GIN (countries);
CREATE INDEX ON anthroponyms.entries(types) USING GIN (types);