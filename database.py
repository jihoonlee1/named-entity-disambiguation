import contextlib
import json
import csv
import pathlib
import sqlite3


dbname = "database.sqlite"
statements = [
"""
CREATE TABLE IF NOT EXISTS items(
	id          INTEGER NOT NULL PRIMARY KEY,
	label       TEXT    NOT NULL,
	description TEXT    NOT NULL
)
""",
"""
CREATE TABLE IF NOT EXISTS item_alias(
	item_id INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE ON UPDATE CASCADE,
	alias   TEXT    NOT NULL,
	PRIMARY KEY(item_id, alias)
)
""",
"""
CREATE TABLE IF NOT EXISTS pages(
	id      INTEGER NOT NULL PRIMARY KEY,
	item_id INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE ON UPDATE CASCADE,
	title   TEXT    NOT NULL,
	views   INTEGER NOT NULL
)
""",
"""
CREATE TABLE IF NOT EXISTS properties(
	id          INTEGER NOT NULL PRIMARY KEY,
	label       TEXT    NOT NULL,
	description TEXT    NOT NULL
)
""",
"""
CREATE TABLE IF NOT EXISTS property_alias(
	property_id INTEGER NOT NULL REFERENCES properties(id) ON DELETE CASCADE ON UPDATE CASCADE,
	alias       TEXT    NOT NULL,
	PRIMARY KEY(property_id, alias)
)
""",
"""
CREATE TABLE IF NOT EXISTS statements(
	source_item_id   INTEGER NOT NULL REFERENCES items(id)      ON DELETE CASCADE ON UPDATE CASCADE,
	edge_property_id INTEGER NOT NULL REFERENCES properties(id) ON DELETE CASCADE ON UPDATE CASCADE,
	target_item_id   INTEGER NOT NULL REFERENCES items(id)      ON DELETE CASCADE ON UPDATE CASCADE,
	PRIMARY KEY(source_item_id, edge_property_id, target_item_id)
)
""",
"""
CREATE TABLE IF NOT EXISTS sections(
	id      INTEGER NOT NULL PRIMARY KEY,
	title   TEXT,
	content TEXT
)
""",
"""
CREATE TABLE IF NOT EXISTS page_section(
	page_id    INTEGER NOT NULL REFERENCES pages(id)    ON DELETE CASCADE ON UPDATE CASCADE,
	section_id INTEGER NOT NULL REFERENCES sections(id) ON DELETE CASCADE ON UPDATE CASCADE,
	PRIMARY KEY(page_id, section_id)
)
""",
"""
CREATE TABLE IF NOT EXISTS section_link(
	section_id INTEGER NOT NULL REFERENCES sections(id) ON DELETE CASCADE ON UPDATE CASCADE,
	offset     INTEGER NOT NULL,
	length     INTEGER NOT NULL,
	page_id    INTEGER NOT NULL REFERENCES pages(id)    ON DELETE CASCADE ON UPDATE CASCADE,
	PRIMARY KEY(section_id, offset, length)
)
"""
]


def connect(database=dbname, mode="rw"):
	return contextlib.closing(sqlite3.connect(f"file:{database}?mode={mode}", uri=True))


def _create():
	if not pathlib.Path(dbname).exists():
		with connect(mode="rwc") as con:
			pass


def _initialize():
	with connect() as con:
		cur = con.cursor()
		for st in statements:
			cur.execute(st)


def _insert_item(cur):
	with open("item.csv") as f:
		reader = csv.reader(f)
		next(reader)
		for item_id, label, description in reader:
			item_id = int(item_id)
			if label == "" or description == "":
				continue
			cur.execute("INSERT INTO items VALUES(?,?,?)", (item_id, label, description))


def _insert_item_aliases(cur):
	with open("item_aliases.csv") as f:
		reader = csv.reader(f)
		next(reader)
		for item_id, alias in reader:
			item_id = int(item_id)
			print(item_id)
			try:
				cur.execute("INSERT INTO item_alias VALUES(?,?)", (item_id, alias))
			except:
				continue


def _insert_properties(cur):
	with open("property.csv") as f:
		reader = csv.reader(f)
		next(reader)
		for property_id, label, description in reader:
			property_id = int(property_id)
			if label == "" or description == "":
				continue
			cur.execute("INSERT INTO properties VALUES(?,?,?)", (property_id, label, description))


def _insert_property_alias(cur):
	with open("property_aliases.csv") as f:
		reader = csv.reader(f)
		next(reader)
		for property_id, alias in reader:
			property_id = int(property_id)
			try:
				cur.execute("INSERT INTO property_alias VALUES(?,?)", (property_id, alias))
			except:
				continue


def _insert_pages(cur):
	with open("page.csv") as f:
		reader = csv.reader(f)
		next(reader)
		for page_id, item_id, title, views in reader:
			page_id = int(page_id)
			item_id = int(item_id)
			views = int(views)
			try:
				cur.execute("INSERT INTO pages VALUES(?,?,?,?)", (page_id, item_id, title, views))
			except:
				continue


def _insert_statements(cur):
	with open("statements.csv") as f:
		reader = csv.reader(f)
		next(reader)
		for source_item_id, edge_property_id, target_item_id in reader:
			source_item_id = int(source_item_id)
			edge_property_id = int(edge_property_id)
			target_item_id = int(target_item_id)
			try:
				cur.execute("INSERT INTO statements VALUES(?,?,?)", (source_item_id, edge_property_id, target_item_id))
			except:
				continue


def _insert_link_annotated_text(cur):
	with open("link_annotated_text.jsonl") as f:
		data = list(f)
		for idx, item in enumerate(data):
			print(idx)
			obj = json.loads(item)
			page_id = int(obj["page_id"])
			for section in obj["sections"]:
				section_title = section["name"]
				section_content = section["text"]
				cur.execute("SELECT ifnull(max(id)+1, 0) FROM sections")
				section_id, = cur.fetchone()
				cur.execute("INSERT INTO sections VALUES(?,?,?)", (section_id, section_title, section_content))
				cur.execute("INSERT INTO page_section VALUES(?,?)", (page_id, section_id))
				link_lengths = section["link_lengths"]
				link_offsets = section["link_offsets"]
				link_page_ids = section["target_page_ids"]
				num_links = len(link_page_ids)
				for i in range(num_links):
					link_offset = int(link_offsets[i])
					link_length = int(link_lengths[i])
					link_page_id = int(link_page_ids[i])
					try:
						cur.execute("INSERT INTO section_link VALUES(?,?,?,?)", (section_id, link_offset, link_length, link_page_id))
					except:
						continue


def main():
	_create()
	_initialize()
	with connect() as con:
		cur = con.cursor()
		_insert_item(cur)
		_insert_item_aliases(cur)
		_insert_pages(cur)
		_insert_properties(cur)
		_insert_property_alias(cur)
		_insert_statements(cur)
		_insert_link_annotated_text(cur)
		con.commit()


if __name__ == "__main__":
	main()
