SELECT 
    DB_NAME(database_id) AS DatabaseName,
    name AS LogicalFileName,
    type_desc AS FileType,
    CASE 
        WHEN is_percent_growth = 1 THEN CAST(growth AS VARCHAR(10)) + ' %'
        ELSE CAST((growth * 8) / 1024 AS VARCHAR(10)) + ' MB'
    END AS AutogrowthSetting,
    CASE 
        WHEN max_size = -1 THEN 'Unlimited'
        WHEN max_size = 268435456 THEN '2 TB (Log max)'
        ELSE CAST((max_size * 8) / 1024 AS VARCHAR(20)) + ' MB'
    END AS MaxSizeSetting
FROM sys.master_files
WHERE DB_NAME(database_id) = 'SentinelDB';
