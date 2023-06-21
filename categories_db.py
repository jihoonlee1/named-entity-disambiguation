import database
import urllib.parse


statements = [
"""
CREATE TABLE categories(
	id   INTEGER NOT NULL PRIMARY KEY,
	name TEXT    NOT NULL
)
""",
"""
CREATE TABLE pages(
	id          INTEGER NOT NULL PRIMARY KEY,
	title       TEXT    NOT NULL,
	content     TEXT    NOT NULL,
	category_id INTEGER NOT NULL REFERENCES categories(id)
)
"""
]


'''
---transportation and infrastructure---
Road Infrastructure
Rail Infrastructure
Air Infrastructure
Waterway Infrastructure
Pipeline Infrastructure
Public Transit Infrastructure
Bicycle and Pedestrian Infrastructure
Intermodal Infrastructure
Intelligent Transportation Systems (ITS)
Ancillary Infrastructure
'''

'''
---natural locations---
Mountains
Deserts
Forests
Grasslands
Wetlands
Coastal Areas
Lakes and Rivers
Caves and Caverns
Volcanic Areas
Coral Reefs
'''

'''
---populated locations---
Cities
Towns
Villages
Suburbs
Rural Areas
Metropolitan Areas
Neighborhoods
Districts
Provinces/States
Countries
'''

'''
---product or service---
Consumer Goods
Automotive
Technology and Electronics
Health and Wellness
Home and Garden
Financial Services
Food and Beverages
Travel and Hospitality
Entertainment and Media
Education and Training
Professional Services
'''


def initialize():
	with database.connect("partial.sqlite", mode="rwc") as con:
		cur = con.cursor()
		for st in statements:
			cur.execute(st)
		cur.execute("INSERT INTO categories VALUES(?,?)", (0, "company"))
		cur.execute("INSERT INTO categories VALUES(?,?)", (1, "government_organization"))
		cur.execute("INSERT INTO categories VALUES(?,?)", (2, "non_profit_organization"))
		cur.execute("INSERT INTO categories VALUES(?,?)", (3, "product_service"))
		cur.execute("INSERT INTO categories VALUES(?,?)", (4, "transportation_infrastructure"))
		cur.execute("INSERT INTO categories VALUES(?,?)", (5, "natural_location"))
		cur.execute("INSERT INTO categories VALUES(?,?)", (6, "populated_location"))
		con.commit()


def open_file(fname):
	with open(fname) as f:
		result = []
		for item in f.readlines():
			item = item.split("https://en.wikipedia.org/wiki/")[1].strip()
			item = urllib.parse.unquote(item)
			item = item.replace("_", " ")
			result.append(item)
		return result


def main():
	# companies = open_file("wiki_companies.txt")
	# gov_org = open_file("wiki_government_organizations.txt")
	# non_prof_org = open_file("wiki_non_profit_organizations.txt")
	# prod_serv = open_file("wiki_products_services.txt")
	# infra = open_file("wiki_transportation_infrastructures.txt")
	# nat_loc = open_file("wiki_natural_locations.txt")
	# pop_loc = open_file("wiki_populated_locations.txt")

	with database.connect() as con:
		cur = con.cursor()
		with database.connect("partial.sqlite") as con1:
			cur1 = con1.cursor()
			for item in non_prof_org:
				cur.execute("SELECT id FROM pages WHERE title = ?", (item, ))
				row = cur.fetchone()
				if row is None:
					continue
				page_id, = row
				body = []
				cur.execute("SELECT sections.content FROM page_section JOIN sections ON sections.id = page_section.section_id WHERE page_section.page_id = ?", (page_id, ))
				sections = cur.fetchall()
				if not sections:
					continue
				for section, in sections:
					body.append(section)
				body = " ".join(body)
				cur1.execute("INSERT OR IGNORE INTO pages VALUES(?,?,?,?)", (page_id, item, body, 2))
			con1.commit()



if __name__ == "__main__":
	main()
