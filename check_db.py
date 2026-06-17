import sqlite3
conn = sqlite3.connect('data/pill_identity.db')
cur = conn.cursor()
cur.execute("SELECT item_name, print_front FROM pill_identity WHERE print_front LIKE ?", ('%AA5%',))
rows = cur.fetchall()
for r in rows:
    print(r)
conn.close()
