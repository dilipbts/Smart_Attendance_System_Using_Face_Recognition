import sqlite3

# Connect to or create a database file
connection = sqlite3.connect('test.db')

# Create a cursor object to interact with the database
cursor = connection.cursor()

# Create a simple table
cursor.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT, password TEXT)')
connection.commit()

# Insert a test user
cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', ('Dilip DK', 'demonking'))
connection.commit()

# Query the table to verify
cursor.execute('SELECT * FROM users')
print(cursor.fetchall())

# Close the connection
connection.close()
