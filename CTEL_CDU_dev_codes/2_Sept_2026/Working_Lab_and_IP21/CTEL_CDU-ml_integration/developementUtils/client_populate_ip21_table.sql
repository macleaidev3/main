--==============================================================
-- Clear previous test data (Optional)
--==============================================================
TRUNCATE TABLE Sentinel_IP21_Data;
GO

--==============================================================
-- Tag List
--==============================================================
WITH Tags AS
(
    SELECT *
    FROM (VALUES
    ('cdu3_icti2712.pv'),
    ('cdu3_icPIC2704.pv'),
    ('cdu3_icFIC2705.pv'),
    ('cdu3_icFIC2704.pv'),
    ('cdu3_icTI1103.pv'),
    ('cdu3_icTI1205.pv'),
    ('cdu3_icTI1206.pv'),
    ('cdu3_icTI1207.pv'),
    ('cdu3_icTI1208.pv'),
    ('cdu3_icTI1209.pv'),
    ('cdu3_icTI2903.pv'),
    ('cdu3_icPi1211.pv'),
    ('cdu3_icFI1201A.pv'),
    ('cdu3_icFI1202A.pv'),
    ('cdu3_icFI1203A.pv'),
    ('cdu3_icFI1204A.pv'),
    ('cdu3_icTI2801.pv'),
    ('cdu3_icPI2902.pv'),
    ('cdu3_icTI2901.pv'),
    ('cdu3_icFIC2902.pv'),
    ('cdu3_icTI2906.pv'),
    ('cdu3_icFIC2903.pv'),
    ('cdu3_icTI2902.pv'),
    ('cdu3_icTIC3001.pv'),
    ('cdu3_icFI3001A.pv'),
    ('cdu3_icFI3002A.pv'),
    ('cdu3_icFI3003A.pv'),
    ('cdu3_icFI3004A.pv'),
    ('cdu3_icFI3005A.pv'),
    ('cdu3_icFI3006A.pv'),
    ('cdu3_icFI3007A.pv'),
    ('cdu3_icFI3008A.pv'),
    ('cdu3_icTI3103.pv'),
    ('cdu3_icTI3104.pv'),
    ('cdu3_icTI3105.pv'),
    ('cdu3_icTI3102.pv'),
    ('cdu3_icFIC3102.pv'),
    ('cdu3_icTI3101.pv'),
    ('cdu3_icFIC3101.pv'),
    ('cdu3_icFIC3701.pv'),
    ('cdu3_icPI3102.pv')
    ) T(TagName)
)

--==============================================================
-- Generate Data
--==============================================================
INSERT INTO Sentinel_IP21_Data
(
    TagName,
    TS,
    TagValue
)

SELECT

    T.TagName,

    DATEADD
    (
        HOUR,
        H.HourNo,

        DATETIMEFROMPARTS
        (
            Y.YearNo,
            M.MonthNo,
            D.DayNo,
            0,
            0,
            0,
            0
        )
    ) AS TS,

    CAST
    (
        50
        + (ABS(CHECKSUM(T.TagName)) % 100)
        + H.HourNo * 0.25
        + M.MonthNo
        + D.DayNo
        + (Y.YearNo - 2025) * 5
        AS FLOAT
    ) AS TagValue

FROM
(
    VALUES (2025),(2026)
) Y(YearNo)

CROSS JOIN
(
    VALUES
    (1),(2),(3),(4),(5),(6),
    (7),(8),(9),(10),(11),(12)
) M(MonthNo)

CROSS JOIN
(
    VALUES (1),(2)
) D(DayNo)

CROSS JOIN
(
    VALUES
    (0),(1),(2),(3),(4),(5),
    (6),(7),(8),(9),(10),(11),
    (12),(13),(14),(15),(16),(17),
    (18),(19),(20),(21),(22),(23)
) H(HourNo)

CROSS JOIN Tags T

ORDER BY
    TS,
    TagName;
GO

--==============================================================
-- Verify
--==============================================================

SELECT COUNT(*) AS TotalRows
FROM Sentinel_IP21_Data;

SELECT TOP 100 *
FROM Sentinel_IP21_Data
ORDER BY TS, TagName;