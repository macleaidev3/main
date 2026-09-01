USE Sentinel_IP21;
GO

IF OBJECT_ID('dbo.Sentinel_LIMS_Data', 'U') IS NOT NULL
BEGIN
    DROP TABLE dbo.Sentinel_LIMS_Data;
END
GO

CREATE TABLE dbo.Sentinel_LIMS_Data
(
    ID INT IDENTITY(1,1) PRIMARY KEY,

    Sample NVARCHAR(100) NOT NULL,

    [Sample Name in LIMS] NVARCHAR(255) NULL,

    SampleDate DATETIME2(3) NOT NULL,

    [Salt (PTB)] FLOAT NULL,

    [BSW (%Vol)] FLOAT NULL,

    [Density (g/ml)] FLOAT NULL,

    [Chloride (ppm)] FLOAT NULL,

    [Chloride content (mg/L)] FLOAT NULL,

    [Iron (mg/L)] FLOAT NULL,

    PH FLOAT NULL
);
GO