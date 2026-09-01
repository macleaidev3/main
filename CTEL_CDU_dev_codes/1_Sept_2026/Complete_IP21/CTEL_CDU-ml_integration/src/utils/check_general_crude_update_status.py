from src.server_manager.operation_manager import DatabaseManager

def check_general_crude_update_status() -> bool:
    db_manager = DatabaseManager()
    db_name = "SentinelDB"
    
    table_name = "crude_data"

    row_count = db_manager.row_count(db_name, table_name)

    if row_count > 0:
        return True
    else:
        return False
    