import contextlib
import csv
import sqlite3


dbpath = "esgdatabase.sqlite"


statements = [
"""
CREATE TABLE IF NOT EXISTS tags(
	id    INTEGER NOT NULL PRIMARY KEY,
	name  TEXT    NOT NULL,
	theme TEXT    NOT NULL
)
"""
]


def connect(database=dbpath, mode="rw"):
	return contextlib.closing(sqlite3.connect(f"file:{database}?mode={mode}", uri=True))


def main():
	with connect(mode="rwc") as con:
		cur = con.cursor()
		for st in statements:
			cur.execute(st)
		with open("tags.csv") as f:
			reader = csv.reader(f, delimiter="\t")
			next(reader)
			for name, theme in reader:
				name = name.strip()
				theme = theme.strip()
				cur.execute("SELECT ifnull(max(id)+1, 0) FROM tags")
				tag_id, = cur.fetchone()
				cur.execute("INSERT INTO tags VALUES(?,?,?)", (tag_id, name, theme))
		con.commit()



def temp():
	with connect() as con:
		result = []
		cur = con.cursor()
		cur.execute("SELECT name FROM tags")
		for item, in cur.fetchall():
			result.append(item)
		print(result)


if __name__ == "__main__":
	temp()
