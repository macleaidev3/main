USE Sentinel_IP21;
GO

-------------------------------------------------------------
-- Clear previous data (Optional)
-------------------------------------------------------------
TRUNCATE TABLE Sentinel_LIMS_Data;
GO

-------------------------------------------------------------
-- Generate Laboratory Test Data
-------------------------------------------------------------

DECLARE @StartDate DATETIME2(3) = '2025-12-01 00:00:00.000';
DECLARE @EndDate   DATETIME2(3) = '2026-02-28 00:00:00.000';

WHILE @StartDate <= @EndDate
BEGIN

    ---------------------------------------------------------
    -- CR_ICV110A
    ---------------------------------------------------------

    INSERT INTO Sentinel_LIMS_Data
    (
        Sample,
        [Sample Name in LIMS],
        SampleDate,
        [Salt (PTB)],
        [BSW (%Vol)],
        [Density (g/ml)],
        [Chloride (ppm)],
        [Chloride content (mg/L)],
        [Iron (mg/L)],
        PH
    )
    VALUES
    (
        'CR_ICV110A',
        'After Desalter Stage 1',
        DATEADD(HOUR,6,@StartDate),
        ROUND(2 + RAND(CHECKSUM(NEWID()))*4,2),
        ROUND(0.05 + RAND(CHECKSUM(NEWID()))*0.30,2),
        ROUND(0.82 + RAND(CHECKSUM(NEWID()))*0.06,3),
        NULL,
        NULL,
        NULL,
        NULL
    );

    ---------------------------------------------------------
    -- CR_ICV110B
    ---------------------------------------------------------

    INSERT INTO Sentinel_LIMS_Data
    (
        Sample,
        [Sample Name in LIMS],
        SampleDate,
        [Salt (PTB)],
        [BSW (%Vol)],
        [Density (g/ml)],
        [Chloride (ppm)],
        [Chloride content (mg/L)],
        [Iron (mg/L)],
        PH
    )
    VALUES
    (
        'CR_ICV110B',
        'After Desalter Stage 2',
        DATEADD(HOUR,6,@StartDate),
        ROUND(1 + RAND(CHECKSUM(NEWID()))*5,2),
        ROUND(0.05 + RAND(CHECKSUM(NEWID()))*0.40,2),
        NULL,
        NULL,
        NULL,
        NULL,
        NULL
    );

    ---------------------------------------------------------
    -- CR_BF_CDU3
    ---------------------------------------------------------

    INSERT INTO Sentinel_LIMS_Data
    (
        Sample,
        [Sample Name in LIMS],
        SampleDate,
        [Salt (PTB)],
        [BSW (%Vol)],
        [Density (g/ml)],
        [Chloride (ppm)],
        [Chloride content (mg/L)],
        [Iron (mg/L)],
        PH
    )
    VALUES
    (
        'CR_BF_CDU3',
        'Crude Before Desalter',
        DATEADD(HOUR,6,@StartDate),
        ROUND(3 + RAND(CHECKSUM(NEWID()))*6,2),
        ROUND(0.10 + RAND(CHECKSUM(NEWID()))*0.50,2),
        ROUND(0.84 + RAND(CHECKSUM(NEWID()))*0.05,3),
        NULL,
        NULL,
        NULL,
        NULL
    );

    ---------------------------------------------------------
    -- CD3_ICV112
    ---------------------------------------------------------

    INSERT INTO Sentinel_LIMS_Data
    (
        Sample,
        [Sample Name in LIMS],
        SampleDate,
        [Salt (PTB)],
        [BSW (%Vol)],
        [Density (g/ml)],
        [Chloride (ppm)],
        [Chloride content (mg/L)],
        [Iron (mg/L)],
        PH
    )
    VALUES
    (
        'CD3_ICV112',
        'Sour Water ICV112',
        DATEADD(HOUR,6,@StartDate),
        NULL,
        NULL,
        NULL,
        ROUND(100 + RAND(CHECKSUM(NEWID()))*80,2),
        NULL,
        ROUND(0.5 + RAND(CHECKSUM(NEWID()))*2.5,2),
        ROUND(5.5 + RAND(CHECKSUM(NEWID()))*1.5,2)
    );

    ---------------------------------------------------------
    -- CD3_ICV113
    ---------------------------------------------------------

    INSERT INTO Sentinel_LIMS_Data
    (
        Sample,
        [Sample Name in LIMS],
        SampleDate,
        [Salt (PTB)],
        [BSW (%Vol)],
        [Density (g/ml)],
        [Chloride (ppm)],
        [Chloride content (mg/L)],
        [Iron (mg/L)],
        PH
    )
    VALUES
    (
        'CD3_ICV113',
        'Sour Water ICV113',
        DATEADD(HOUR,6,@StartDate),
        NULL,
        NULL,
        NULL,
        ROUND(110 + RAND(CHECKSUM(NEWID()))*90,2),
        NULL,
        ROUND(0.4 + RAND(CHECKSUM(NEWID()))*2.8,2),
        ROUND(5.8 + RAND(CHECKSUM(NEWID()))*1.2,2)
    );

    ---------------------------------------------------------
    -- STR_W_CDU3
    ---------------------------------------------------------

    INSERT INTO Sentinel_LIMS_Data
    (
        Sample,
        [Sample Name in LIMS],
        SampleDate,
        [Salt (PTB)],
        [BSW (%Vol)],
        [Density (g/ml)],
        [Chloride (ppm)],
        [Chloride content (mg/L)],
        [Iron (mg/L)],
        PH
    )
    VALUES
    (
        'STR_W_CDU3',
        'Stripped Water',
        DATEADD(HOUR,6,@StartDate),
        NULL,
        NULL,
        NULL,
        NULL,
        ROUND(40 + RAND(CHECKSUM(NEWID()))*40,2),
        ROUND(0.2 + RAND(CHECKSUM(NEWID()))*1.2,2),
        ROUND(6.0 + RAND(CHECKSUM(NEWID()))*1.0,2)
    );

    SET @StartDate = DATEADD(DAY,1,@StartDate);

END

GO

-------------------------------------------------------------
-- Verify
-------------------------------------------------------------

SELECT COUNT(*) AS TotalRows
FROM Sentinel_LIMS_Data;

SELECT *
FROM Sentinel_LIMS_Data
ORDER BY SampleDate, Sample;