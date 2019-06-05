import matplotlib.pyplot as plt
import pandas as pd
import sqlite3 as sql
from pathlib import Path

# con_sql will connect to the db if the file is exists
def con_sql(db_name: str, dialect=None):
    db_file = Path(db_name)
    if not db_file.is_file():
        return None, "no such database file: {}".format(db_name)
        return
    if dialect is None:
        return sql.connect(db_name), None
    return sql.connect(db_name, dialect), None



# sql_frame will create a dataframe of the provided table
def sql_frame(table_name, conn):
    q = (
        """
    SELECT *
    FROM %s
    """
        % table_name
    )
    return pd.read_sql_query(q, conn)
