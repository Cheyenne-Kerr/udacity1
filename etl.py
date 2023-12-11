import os
import glob
import psycopg2
import pandas as pd
from sql_queries import *
conn = psycopg2.connect("host=127.0.0.1 dbname=sparkifydb user=student password=student")
cur = conn.cursor()

def get_files(filepath):
    """
    Description: Recursively searches a directory for JSON files and returns a list of their absolute file paths.

    Arguments:
        filepath (str): The directory path to search for JSON files.

    Returns:
        list: A list of absolute file paths of JSON files found in the specified directory and its subdirectories.
    """
    all_files = []
    for root, dirs, files in os.walk(filepath):
        files = glob.glob(os.path.join(root, '*.json'))
        for f in files:
            all_files.append(os.path.abspath(f))
    
    return all_files

def process_song_file(cur, filepath):
    """
    Description: This function processes a song data file, extracts song and artist information from the JSON file,
    and inserts the data into the 'songs' and 'artists' tables in the database.

    Arguments:
        cur: The cursor object for executing SQL queries on the database.
        conn: Connection to the database.
        filepath: The file path of the song data JSON file to be processed.

    Returns:
        None
    """
    # open song file
    song_files = get_files('data/song_data')
    filepath = song_files[0]
    df = pd.read_json(filepath, lines=True)

    # insert song record
    song_data = df[['song_id', 'title', 'artist_id', 'year','duration']].values[0].tolist()
    cur.execute(song_table_insert, song_data)
    conn.commit()
    
    # insert artist record
    artist_data = df[['artist_id', 'artist_name', 'artist_location', 'artist_latitude',                                              'artist_longitude']].values[0].tolist()

    cur.execute(artist_table_insert, artist_data)
    conn.commit()
    
    
def process_log_file(cur, filepath):
    """
    Description: This function processes a log data file, extracts relevant data for time, users, and songplays,
    and inserts the data into the 'time', 'users', and 'songplays' tables in the database.

    Arguments:
        cur: The cursor object for executing SQL queries on the database.
        filepath: The file path of the log data JSON file to be processed.

    Returns:
        None
    """
    # open log file
    log_files = get_files('data/log_data')
    filepath = log_files[0]
    df = pd.read_json(filepath, lines=True)

    # filter by NextSong action
    df = df[df['page'] == 'NextSong'] 

    # convert timestamp column to datetime
    t = pd.to_datetime(df['ts'], unit='ms')

    # insert time data records
    time_data = [
        t,
        t.dt.hour,
        t.dt.day,
        t.dt.week,
        t.dt.month,
        t.dt.year,
        t.dt.weekday
    ]

    column_labels = [
        'start_time',
        'hour',
        'day',
        'week',
        'month',
        'year',
        'weekday'
    ]

    time_df = pd.DataFrame(dict(zip(column_labels, time_data)))

    for i, row in time_df.iterrows():
        cur.execute(time_table_insert, list(row))

    # load user table
    user_df = df[['userId', 'firstName', 'lastName', 'gender', 'level']] 

    # insert user records
    for i, row in user_df.iterrows():
        cur.execute(user_table_insert, row)

    # insert songplay records
    for index, row in df.iterrows():
    
    # get songid and artistid from song and artist tables
        cur.execute(song_select, (row.song, row.artist, row.length))
        results = cur.fetchone()
    
    if results:
        song_id, artist_id = results
    else:
        song_id, artist_id = None, None
    start_time = time_df.loc[index, 'start_time']

    # insert songplay record
    songplay_data = (start_time, row.userId, row.level, song_id, artist_id, row.sessionId, row.location,         row.userAgent)
    cur.execute(songplay_table_insert, songplay_data)

# Call the function with the appropriate arguments
filepath = 'path/to/log_file.json'
process_log_file(cur, filepath)


def process_data(cur, conn, filepath, func):
    """
    Description: This function processes all JSON files in a specified directory by applying a given transformation
    function (func) to each file. It iterates through the files, extracts data, and inserts it into the database.

    Arguments:
        cur: The cursor object for executing SQL queries on the database.
        conn: Connection to the database.
        filepath: The directory containing the JSON files to be processed.
        func: The transformation function to be applied to each file.

    Returns:
        None
    """
    # get all files matching extension from directory
    all_files = []
    for root, dirs, files in os.walk(filepath):
        files = glob.glob(os.path.join(root,'*.json'))
        for f in files:
            all_files.append(os.path.abspath(f))

    # get total number of files found
    num_files = len(all_files)
    print('{} files found in {}'.format(num_files, filepath))

    # iterate over files and process
    for i, datafile in enumerate(all_files, 1):
        func(cur, datafile)
        conn.commit()
        print('{}/{} files processed.'.format(i, num_files))

def main():
    """
    Description: This function serves as the main entry point for the ETL (Extract, Transform, Load) process.
    It establishes a connection to the database, processes both song data and log data files, and closes the database
    connection after the ETL process is completed.

    Arguments:
        None

    Returns:
        None
    """
    conn = psycopg2.connect("host=127.0.0.1 dbname=sparkifydb user=student password=student")
    cur = conn.cursor()

    process_data(cur, conn, filepath='data/song_data', func=process_song_file)
    process_data(cur, conn, filepath='data/log_data', func=process_log_file)

    conn.close()


if __name__ == "__main__":
    main()