import psycopg2
from sql_queries import create_table_queries, drop_table_queries, songplay_table_drop, user_table_drop, song_table_drop, artist_table_drop, time_table_drop, songplay_table_create, user_table_create, song_table_create, artist_table_create, time_table_create


def create_database():
    """
    Description: This function connects to the default PostgreSQL database, creates a new database named 'sparkifydb'
    with UTF8 encoding, and returns a cursor and connection to the newly created database.

    Arguments:
        None

    Returns:
        cur: The cursor object for executing SQL queries on the 'sparkifydb' database.
        conn: Connection to the 'sparkifydb' database.
    """
    # connect to default database
    conn = psycopg2.connect("host=127.0.0.1 dbname=studentdb user=student password=student")
    conn.set_session(autocommit=True)
    cur = conn.cursor()
    
    # create sparkify database with UTF8 encoding
    cur.execute("DROP DATABASE IF EXISTS sparkifydb")
    cur.execute("CREATE DATABASE sparkifydb WITH ENCODING 'utf8' TEMPLATE template0")

    # close connection to default database
    conn.close()    
    
    # connect to sparkify database
    conn = psycopg2.connect("host=127.0.0.1 dbname=sparkifydb user=student password=student")
    cur = conn.cursor()
    
    return cur, conn


def drop_tables(cur, conn):
    """
    Description: This function drops (deletes) the tables defined in the drop_table_queries list. It iterates through
    the list of SQL queries and executes each query to drop the corresponding table.

    Arguments:
        cur: The cursor object for executing SQL queries on the database.
        conn: Connection to the database.

    Returns:
        None
    """
    drop_table_queries = [songplay_table_drop, user_table_drop, song_table_drop, artist_table_drop, time_table_drop]
    for query in drop_table_queries:
        cur.execute(query)
        conn.commit()


def create_tables(cur, conn):
    """
    Description: This function creates the tables defined in the create_table_queries list. It iterates through
    the list of SQL queries and executes each query to create the corresponding table.

    Arguments:
        cur: The cursor object for executing SQL queries on the database.
        conn: Connection to the database.

    Returns:
        None
    """
    create_table_queries = [songplay_table_create, user_table_create, song_table_create, artist_table_create, time_table_create]
    for query in create_table_queries:
        cur.execute(query)
        conn.commit()


def main():
    """
    - Drops (if exists) and Creates the sparkify database. 
    
    - Establishes connection with the sparkify database and gets
    cursor to it.  
    
    - Drops all the tables.  
    
    - Creates all tables needed. 
    
    - Finally, closes the connection. 
    """
    cur, conn = create_database()
    
    drop_tables(cur, conn)
    create_tables(cur, conn)

    conn.close()


if __name__ == "__main__":
    main()