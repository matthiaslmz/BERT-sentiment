import sqlite3
import os 

conn = sqlite3.connect('movie_reviews.sqlite')

c = conn.cursor()
c.execute('DROP TABLE IF EXISTS review_db')
c.execute('CREATE TABLE review_db' '(review TEXT, sentiment INTEGER, date TEXT)')

example1 = 'The Phantom Menace is the best Star Wars movie ever!'
c.execute("INSERT INTO review_db (review, sentiment, date) VALUES (?, ?, DATETIME('now'))",  (example1, 1))

example2 = 'I disliked the latest Star Trek movie'
c.execute("INSERT INTO review_db"\
" (review, sentiment, date) VALUES"\
" (?, ?, DATETIME('now'))", (example2, 0))
conn.commit()

c = conn.cursor()
c.execute("SELECT * FROM review_db WHERE date"\
" BETWEEN '2017-01-01 00:00:00' AND DATETIME('now')")
results = c.fetchall()
print(results)

# conn.close()

if __name__ == "__main__":
    print(os.path.dirname(__file__))