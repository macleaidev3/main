USE Sentinel_IP21;
GO

CREATE TABLE Sentinel_IP21_Data
(
    TagName  NVARCHAR(255) NOT NULL,
    TS       DATETIME2(3)  NOT NULL,
    TagValue FLOAT         NULL,

    CONSTRAINT PK_Sentinel_IP21_Data
        PRIMARY KEY (TagName, TS)
);