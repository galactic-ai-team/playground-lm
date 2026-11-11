#!/usr/bin/env python3
"""Generate data/facts/fact_sheet.jsonl (Stage G, Nova).

Every fact is one short sentence written for this repository (facts are
not copyrightable; the wording is original). Facts were checked against
the reference pages listed in LICENSE_AUDIT.md on 2026-11-10. Only
stable facts are included: no current office holders, no changing
records, no opinions. ~20% of entries per topic are marked heldout
(never trained as answers; used to test reading from context).

Run: python data/facts/generate_fact_sheet.py
The generator asserts the full schema (alias appears in text, line
length, unique ids/texts) before writing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent / "fact_sheet.jsonl"
LICENSE = "MIT (this repository)"

CAPITALS_URL = "https://simple.wikipedia.org/wiki/List_of_national_capitals"
US_CAPITALS_URL = "https://simple.wikipedia.org/wiki/List_of_U.S._state_capitals"
PLANETS_URL = "https://science.nasa.gov/solar-system/planet-sizes-and-locations-in-our-solar-system/"
ELEMENTS_URL = "https://simple.wikipedia.org/wiki/List_of_elements_by_symbol"
GEO_URL = "https://dailygk.indgovtjobs.net/gk/geography/world-geography-basics/"
BODY_ANIMALS_URL = "https://www.syvum.com/cgi/online/serve.cgi/quiz/livingworld.html?list"
SELF_URL = "https://github.com/galactic-ai-team/playground-lm"

FACTS: list[dict] = []
_COUNTERS: dict[str, int] = {}
_PREFIX = {"geography": "geo", "science": "sci", "calendar": "cal",
           "history": "his", "animals": "ani", "language": "lan",
           "self": "self"}


def add(topic: str, text: str, keywords: list[str], aliases: list[str],
        url: str) -> None:
    _COUNTERS[topic] = _COUNTERS.get(topic, 0) + 1
    FACTS.append({
        "id": f"{_PREFIX[topic]}-{_COUNTERS[topic]:04d}",
        "text": text,
        "keywords": keywords,
        "aliases": aliases,
        "topic": topic,
        "verify_url": url,
        "license": LICENSE,
        "heldout": False,
    })


def kw(*words: str) -> list[str]:
    import unicodedata
    out: list[str] = []
    for w in words:
        folded = unicodedata.normalize("NFKD", w.lower())
        folded = "".join(c for c in folded if not unicodedata.combining(c))
        for tok in re.split(r"[^a-z0-9]+", folded):
            if tok and tok not in out:
                out.append(tok)
    return out[:4]


# ---------------------------------------------------------------------------
# Geography — world capitals (checked against the Simple Wikipedia list)
# ---------------------------------------------------------------------------

CAPITALS = [
    ("Afghanistan", "Kabul"), ("Albania", "Tirana"), ("Algeria", "Algiers"),
    ("Andorra", "Andorra la Vella"), ("Angola", "Luanda"),
    ("Antigua and Barbuda", "Saint John's"), ("Argentina", "Buenos Aires"),
    ("Armenia", "Yerevan"), ("Australia", "Canberra"), ("Austria", "Vienna"),
    ("Azerbaijan", "Baku"), ("Bahamas", "Nassau"), ("Bahrain", "Manama"),
    ("Bangladesh", "Dhaka"), ("Barbados", "Bridgetown"), ("Belarus", "Minsk"),
    ("Belgium", "Brussels"), ("Belize", "Belmopan"), ("Benin", "Porto-Novo"),
    ("Bhutan", "Thimphu"), ("Bosnia and Herzegovina", "Sarajevo"),
    ("Botswana", "Gaborone"), ("Brazil", "Brasília"), ("Brunei", "Bandar Seri Begawan"),
    ("Bulgaria", "Sofia"), ("Burkina Faso", "Ouagadougou"), ("Burundi", "Gitega"),
    ("Cambodia", "Phnom Penh"), ("Cameroon", "Yaoundé"), ("Canada", "Ottawa"),
    ("Cape Verde", "Praia"), ("Central African Republic", "Bangui"),
    ("Chad", "N'Djamena"), ("Chile", "Santiago"), ("China", "Beijing"),
    ("Colombia", "Bogotá"), ("Comoros", "Moroni"), ("Costa Rica", "San José"),
    ("Croatia", "Zagreb"), ("Cuba", "Havana"), ("Cyprus", "Nicosia"),
    ("Czechia", "Prague"), ("Denmark", "Copenhagen"), ("Dominica", "Roseau"),
    ("Dominican Republic", "Santo Domingo"), ("Ecuador", "Quito"),
    ("Egypt", "Cairo"), ("El Salvador", "San Salvador"),
    ("Equatorial Guinea", "Malabo"), ("Eritrea", "Asmara"),
    ("Estonia", "Tallinn"), ("Eswatini", "Mbabane"), ("Ethiopia", "Addis Ababa"),
    ("Fiji", "Suva"), ("Finland", "Helsinki"), ("France", "Paris"),
    ("Gabon", "Libreville"), ("Gambia", "Banjul"), ("Georgia", "Tbilisi"),
    ("Germany", "Berlin"), ("Ghana", "Accra"), ("Greece", "Athens"),
    ("Grenada", "St. George's"), ("Guatemala", "Guatemala City"),
    ("Guinea", "Conakry"), ("Guinea-Bissau", "Bissau"), ("Guyana", "Georgetown"),
    ("Haiti", "Port-au-Prince"), ("Honduras", "Tegucigalpa"),
    ("Hungary", "Budapest"), ("Iceland", "Reykjavik"), ("India", "New Delhi"),
    ("Indonesia", "Jakarta"), ("Iran", "Tehran"), ("Iraq", "Baghdad"),
    ("Ireland", "Dublin"), ("Italy", "Rome"), ("Ivory Coast", "Yamoussoukro"),
    ("Jamaica", "Kingston"), ("Japan", "Tokyo"), ("Jordan", "Amman"),
    ("Kazakhstan", "Astana"), ("Kenya", "Nairobi"), ("Kiribati", "South Tarawa"),
    ("Kuwait", "Kuwait City"), ("Kyrgyzstan", "Bishkek"), ("Laos", "Vientiane"),
    ("Latvia", "Riga"), ("Lebanon", "Beirut"), ("Lesotho", "Maseru"),
    ("Liberia", "Monrovia"), ("Libya", "Tripoli"), ("Liechtenstein", "Vaduz"),
    ("Lithuania", "Vilnius"), ("Luxembourg", "Luxembourg City"),
    ("Madagascar", "Antananarivo"), ("Malawi", "Lilongwe"),
    ("Malaysia", "Kuala Lumpur"), ("Maldives", "Malé"), ("Mali", "Bamako"),
    ("Malta", "Valletta"), ("Marshall Islands", "Majuro"),
    ("Mauritania", "Nouakchott"), ("Mauritius", "Port Louis"),
    ("Mexico", "Mexico City"), ("Micronesia", "Palikir"),
    ("Moldova", "Chisinau"), ("Monaco", "Monaco"), ("Mongolia", "Ulaanbaatar"),
    ("Montenegro", "Podgorica"), ("Morocco", "Rabat"), ("Mozambique", "Maputo"),
    ("Myanmar", "Naypyidaw"), ("Namibia", "Windhoek"), ("Nepal", "Kathmandu"),
    ("New Zealand", "Wellington"), ("Nicaragua", "Managua"), ("Niger", "Niamey"),
    ("Nigeria", "Abuja"), ("North Korea", "Pyongyang"),
    ("North Macedonia", "Skopje"), ("Norway", "Oslo"), ("Oman", "Muscat"),
    ("Pakistan", "Islamabad"), ("Palau", "Ngerulmud"), ("Panama", "Panama City"),
    ("Papua New Guinea", "Port Moresby"), ("Paraguay", "Asunción"),
    ("Peru", "Lima"), ("Philippines", "Manila"), ("Poland", "Warsaw"),
    ("Portugal", "Lisbon"), ("Qatar", "Doha"), ("Romania", "Bucharest"),
    ("Russia", "Moscow"), ("Rwanda", "Kigali"), ("Samoa", "Apia"),
    ("Saudi Arabia", "Riyadh"), ("Senegal", "Dakar"), ("Serbia", "Belgrade"),
    ("Seychelles", "Victoria"), ("Sierra Leone", "Freetown"),
    ("Singapore", "Singapore"), ("Slovakia", "Bratislava"),
    ("Slovenia", "Ljubljana"), ("Solomon Islands", "Honiara"),
    ("Somalia", "Mogadishu"), ("South Korea", "Seoul"), ("South Sudan", "Juba"),
    ("Spain", "Madrid"), ("Sudan", "Khartoum"), ("Suriname", "Paramaribo"),
    ("Sweden", "Stockholm"), ("Syria", "Damascus"), ("Tajikistan", "Dushanbe"),
    ("Tanzania", "Dodoma"), ("Thailand", "Bangkok"), ("Timor-Leste", "Dili"),
    ("Togo", "Lomé"), ("Tonga", "Nuku'alofa"),
    ("Trinidad and Tobago", "Port of Spain"), ("Tunisia", "Tunis"),
    ("Turkey", "Ankara"), ("Turkmenistan", "Ashgabat"), ("Tuvalu", "Funafuti"),
    ("Uganda", "Kampala"), ("Ukraine", "Kyiv"),
    ("United Arab Emirates", "Abu Dhabi"), ("United Kingdom", "London"),
    ("Uruguay", "Montevideo"), ("Uzbekistan", "Tashkent"),
    ("Vanuatu", "Port Vila"), ("Venezuela", "Caracas"), ("Vietnam", "Hanoi"),
    ("Zambia", "Lusaka"), ("Zimbabwe", "Harare"),
    ("Saint Lucia", "Castries"), ("Saint Vincent and the Grenadines", "Kingstown"),
    ("Saint Kitts and Nevis", "Basseterre"), ("São Tomé and Príncipe", "São Tomé"),
]

for country, capital in CAPITALS:
    add("geography", f"{capital} is the capital of {country}.",
        kw(capital, "capital", country), [capital], CAPITALS_URL)

add("geography", "Washington, D.C. is the capital of the United States.",
    kw("washington", "capital", "united states"),
    ["Washington, D.C.", "Washington"], CAPITALS_URL)
add("geography", "Amsterdam is the official capital of the Netherlands.",
    kw("amsterdam", "capital", "netherlands"), ["Amsterdam"], CAPITALS_URL)
add("geography", "Sucre is the constitutional capital of Bolivia.",
    kw("sucre", "capital", "bolivia"), ["Sucre"], CAPITALS_URL)
add("geography", "La Paz is the seat of government of Bolivia.",
    kw("la paz", "government", "bolivia"), ["La Paz"], CAPITALS_URL)
add("geography", "Pretoria is the administrative capital of South Africa.",
    kw("pretoria", "capital", "south africa"), ["Pretoria"], CAPITALS_URL)
add("geography", "Bern is the de facto capital of Switzerland.",
    kw("bern", "capital", "switzerland"), ["Bern"], CAPITALS_URL)
add("geography", "Brazzaville is the capital of the Republic of the Congo.",
    kw("brazzaville", "capital", "congo"), ["Brazzaville"], CAPITALS_URL)
add("geography", "Kinshasa is the capital of the Democratic Republic of the Congo.",
    kw("kinshasa", "capital", "congo"), ["Kinshasa"], CAPITALS_URL)
add("geography", "Djibouti City is the capital of Djibouti.",
    kw("djibouti", "capital"), ["Djibouti City"], CAPITALS_URL)
add("geography", "The City of San Marino is the capital of San Marino.",
    kw("san marino", "capital"), ["San Marino"], CAPITALS_URL)

US_STATES = [
    ("Alabama", "Montgomery"), ("Alaska", "Juneau"), ("Arizona", "Phoenix"),
    ("Arkansas", "Little Rock"), ("California", "Sacramento"),
    ("Colorado", "Denver"), ("Connecticut", "Hartford"), ("Delaware", "Dover"),
    ("Florida", "Tallahassee"), ("Georgia", "Atlanta"), ("Hawaii", "Honolulu"),
    ("Idaho", "Boise"), ("Illinois", "Springfield"), ("Indiana", "Indianapolis"),
    ("Iowa", "Des Moines"), ("Kansas", "Topeka"), ("Kentucky", "Frankfort"),
    ("Louisiana", "Baton Rouge"), ("Maine", "Augusta"), ("Maryland", "Annapolis"),
    ("Massachusetts", "Boston"), ("Michigan", "Lansing"),
    ("Minnesota", "Saint Paul"), ("Mississippi", "Jackson"),
    ("Missouri", "Jefferson City"), ("Montana", "Helena"),
    ("Nebraska", "Lincoln"), ("Nevada", "Carson City"),
    ("New Hampshire", "Concord"), ("New Jersey", "Trenton"),
    ("New Mexico", "Santa Fe"), ("New York", "Albany"),
    ("North Carolina", "Raleigh"), ("North Dakota", "Bismarck"),
    ("Ohio", "Columbus"), ("Oklahoma", "Oklahoma City"), ("Oregon", "Salem"),
    ("Pennsylvania", "Harrisburg"), ("Rhode Island", "Providence"),
    ("South Carolina", "Columbia"), ("South Dakota", "Pierre"),
    ("Tennessee", "Nashville"), ("Texas", "Austin"), ("Utah", "Salt Lake City"),
    ("Vermont", "Montpelier"), ("Virginia", "Richmond"),
    ("Washington", "Olympia"), ("West Virginia", "Charleston"),
    ("Wisconsin", "Madison"), ("Wyoming", "Cheyenne"),
]
for state, capital in US_STATES:
    add("geography", f"{capital} is the capital of {state}.",
        kw(capital, "capital", state), [capital], US_CAPITALS_URL)
add("geography", "Washington, D.C. has been the capital of the United States since 1800.",
    kw("washington", "capital", "1800"), ["1800"], US_CAPITALS_URL)
add("geography", "Phoenix is the most populous state capital in the United States.",
    kw("phoenix", "populous", "state capital"), ["Phoenix"], US_CAPITALS_URL)

# ---------------------------------------------------------------------------
# Geography — superlatives, continents, rivers, landmarks
# ---------------------------------------------------------------------------

GEO_EXTRA = [
    ("Russia is the largest country in the world by land area.", ["russia", "largest", "country"], ["Russia"]),
    ("Vatican City is the smallest country in the world.", ["vatican", "smallest", "country"], ["Vatican City"]),
    ("Asia is the largest continent.", ["asia", "largest", "continent"], ["Asia"]),
    ("Australia is the smallest continent.", ["australia", "smallest", "continent"], ["Australia"]),
    ("Earth has seven continents.", ["seven", "continents", "earth"], ["seven"]),
    ("The Pacific Ocean is the largest ocean on Earth.", ["pacific", "largest", "ocean"], ["Pacific Ocean", "Pacific"]),
    ("The Arctic Ocean is the smallest ocean on Earth.", ["arctic", "smallest", "ocean"], ["Arctic Ocean", "Arctic"]),
    ("The Pacific Ocean is the deepest ocean on Earth.", ["pacific", "deepest", "ocean"], ["Pacific Ocean", "Pacific"]),
    ("The Mariana Trench is the deepest place in the oceans.", ["mariana", "trench", "deepest"], ["Mariana Trench"]),
    ("The Nile is generally considered the longest river in the world.", ["nile", "longest", "river"], ["Nile"]),
    ("The Amazon carries more water than any other river.", ["amazon", "river", "water"], ["Amazon"]),
    ("The Amazon River flows through South America.", ["amazon", "river", "south america"], ["Amazon"]),
    ("The Yangtze is the longest river in Asia.", ["yangtze", "longest", "river", "asia"], ["Yangtze"]),
    ("The Volga is the longest river in Europe.", ["volga", "longest", "river", "europe"], ["Volga"]),
    ("The Mississippi River flows through the United States.", ["mississippi", "river", "united states"], ["Mississippi"]),
    ("The Ganges flows through India.", ["ganges", "river", "india"], ["Ganges"]),
    ("The Nile flows into the Mediterranean Sea.", ["nile", "mediterranean", "sea"], ["Mediterranean Sea"]),
    ("Mount Everest is the highest mountain in the world.", ["everest", "highest", "mountain"], ["Mount Everest", "Everest"]),
    ("Mount Kilimanjaro is the highest mountain in Africa.", ["kilimanjaro", "highest", "africa"], ["Mount Kilimanjaro", "Kilimanjaro"]),
    ("The Andes is the longest mountain range on land.", ["andes", "longest", "mountain range"], ["Andes"]),
    ("The Himalayas are a mountain range in Asia.", ["himalayas", "mountain range", "asia"], ["Himalayas"]),
    ("The Sahara is the largest hot desert in the world.", ["sahara", "largest", "hot desert"], ["Sahara"]),
    ("Antarctica is the largest desert in the world.", ["antarctica", "largest", "desert"], ["Antarctica"]),
    ("The Sahara Desert is in North Africa.", ["sahara", "desert", "africa"], ["Sahara"]),
    ("Greenland is the largest island in the world.", ["greenland", "largest", "island"], ["Greenland"]),
    ("Lake Baikal is the deepest lake in the world.", ["baikal", "deepest", "lake"], ["Lake Baikal", "Baikal"]),
    ("The Caspian Sea is the largest lake in the world.", ["caspian", "largest", "lake"], ["Caspian Sea"]),
    ("The equator divides Earth into two hemispheres.", ["equator", "hemispheres", "earth"], ["equator"]),
    ("The Amazon rainforest is mostly in Brazil.", ["amazon", "rainforest", "brazil"], ["Brazil"]),
    ("The Great Barrier Reef lies off the coast of Australia.", ["great barrier reef", "australia"], ["Great Barrier Reef"]),
    ("The Eiffel Tower stands in Paris.", ["eiffel tower", "paris"], ["Paris"]),
    ("The Statue of Liberty stands in New York City.", ["statue of liberty", "new york"], ["New York City"]),
    ("The Great Wall is a famous wall in China.", ["great wall", "china"], ["China"]),
    ("The Colosseum is an ancient arena in Rome.", ["colosseum", "rome"], ["Rome"]),
    ("The pyramids of Giza stand in Egypt.", ["pyramids", "giza", "egypt"], ["Egypt"]),
    ("Big Ben is the great bell of the clock tower in London.", ["big ben", "london", "bell"], ["Big Ben"]),
]
for text, keywords, aliases in GEO_EXTRA:
    add("geography", text, keywords, aliases, GEO_URL)

CONTINENT_OF = [
    ("Egypt", "Africa"), ("Nigeria", "Africa"), ("Kenya", "Africa"),
    ("South Africa", "Africa"), ("Brazil", "South America"),
    ("Argentina", "South America"), ("Chile", "South America"),
    ("Peru", "South America"), ("Canada", "North America"),
    ("Mexico", "North America"), ("Japan", "Asia"), ("China", "Asia"),
    ("India", "Asia"), ("Thailand", "Asia"), ("Vietnam", "Asia"),
    ("Iran", "Asia"), ("Iraq", "Asia"), ("Spain", "Europe"),
    ("Italy", "Europe"), ("Germany", "Europe"), ("Poland", "Europe"),
    ("Ukraine", "Europe"),
]
for country, continent in CONTINENT_OF:
    add("geography", f"{country} is a country in {continent}.",
        kw(country, continent), [continent], GEO_URL)
add("geography", "Russia spans parts of Europe and Asia.",
    kw("russia", "europe", "asia"), ["Europe and Asia"], GEO_URL)
add("geography", "Indonesia is a country in Southeast Asia.",
    kw("indonesia", "southeast asia"), ["Southeast Asia"], GEO_URL)
add("geography", "New Zealand is a country in Oceania.",
    kw("new zealand", "oceania"), ["Oceania"], GEO_URL)
add("geography", "Fiji is an island country in Oceania.",
    kw("fiji", "oceania", "island"), ["Oceania"], GEO_URL)

# ---------------------------------------------------------------------------
# Science — planets (NASA-checked), elements (symbol list), units, nature
# ---------------------------------------------------------------------------

PLANETS = [
    ("Mercury is the closest planet to the Sun.", ["mercury", "closest", "sun"], ["Mercury"]),
    ("Venus is the second planet from the Sun.", ["venus", "second", "planet"], ["Venus"]),
    ("Earth is the third planet from the Sun.", ["earth", "third", "planet"], ["Earth"]),
    ("Mars is the fourth planet from the Sun.", ["mars", "fourth", "planet"], ["Mars"]),
    ("Jupiter is the fifth planet from the Sun.", ["jupiter", "fifth", "planet"], ["Jupiter"]),
    ("Saturn is the sixth planet from the Sun.", ["saturn", "sixth", "planet"], ["Saturn"]),
    ("Uranus is the seventh planet from the Sun.", ["uranus", "seventh", "planet"], ["Uranus"]),
    ("Neptune is the eighth and farthest planet from the Sun.", ["neptune", "eighth", "planet"], ["Neptune"]),
    ("The solar system has eight planets.", ["eight", "planets", "solar system"], ["eight"]),
    ("Jupiter is the largest planet in the solar system.", ["jupiter", "largest", "planet"], ["Jupiter"]),
    ("Mercury is the smallest planet in the solar system.", ["mercury", "smallest", "planet"], ["Mercury"]),
    ("Saturn is famous for its bright rings.", ["saturn", "rings"], ["Saturn"]),
    ("Mars is often called the red planet.", ["mars", "red planet"], ["Mars"]),
    ("Venus is the hottest planet in the solar system.", ["venus", "hottest", "planet"], ["Venus"]),
    ("Earth is the only planet known to support life.", ["earth", "life", "planet"], ["Earth"]),
    ("The Sun is a star.", ["sun", "star"], ["Sun"]),
    ("Earth orbits the Sun.", ["earth", "orbits", "sun"], ["Sun"]),
    ("The Moon orbits Earth.", ["moon", "orbits", "earth"], ["Earth"]),
    ("Earth takes about 365 days to orbit the Sun.", ["earth", "365", "orbit"], ["365 days", "365"]),
    ("Pluto is classified as a dwarf planet.", ["pluto", "dwarf planet"], ["Pluto"]),
    ("Jupiter has a giant storm called the Great Red Spot.", ["jupiter", "great red spot", "storm"], ["Great Red Spot"]),
    ("Uranus rotates on its side compared with other planets.", ["uranus", "rotates", "side"], ["Uranus"]),
    ("Neptune is the windiest planet in the solar system.", ["neptune", "windiest", "planet"], ["Neptune"]),
    ("The Moon reflects light from the Sun.", ["moon", "reflects", "sunlight"], ["Moon"]),
]
for text, keywords, aliases in PLANETS:
    add("science", text, keywords, aliases, PLANETS_URL)

ELEMENTS = [
    ("hydrogen", "H"), ("helium", "He"), ("carbon", "C"), ("nitrogen", "N"),
    ("oxygen", "O"), ("aluminium", "Al"), ("silicon", "Si"), ("sulfur", "S"),
    ("chlorine", "Cl"), ("calcium", "Ca"), ("iron", "Fe"), ("copper", "Cu"),
    ("zinc", "Zn"), ("silver", "Ag"), ("tin", "Sn"), ("gold", "Au"),
    ("mercury", "Hg"), ("lead", "Pb"), ("sodium", "Na"), ("potassium", "K"),
    ("tungsten", "W"),
]
for name, symbol in ELEMENTS:
    add("science", f"The chemical symbol for {name} is {symbol}.",
        kw(name, "symbol", "element"), [symbol], ELEMENTS_URL)
SCIENCE_EXTRA = [
    ("Hydrogen is the lightest element.", ["hydrogen", "lightest", "element"], ["Hydrogen"]),
    ("Scientists have identified 118 chemical elements.", ["118", "elements"], ["118"]),
    ("Water has the chemical formula H2O.", ["water", "formula", "h2o"], ["H2O"]),
    ("Nitrogen makes up most of the air around Earth.", ["nitrogen", "air", "most"], ["Nitrogen"]),
    ("Diamond is a form of carbon.", ["diamond", "form", "carbon"], ["carbon"]),
    ("The graphite in pencils is made of carbon.", ["graphite", "pencils", "carbon"], ["carbon"]),
    ("There are 100 centimeters in a meter.", ["100", "centimeters", "meter"], ["100"]),
    ("There are 1000 meters in a kilometer.", ["1000", "meters", "kilometer"], ["1000"]),
    ("There are 1000 grams in a kilogram.", ["1000", "grams", "kilogram"], ["1000"]),
    ("Water freezes at 0 degrees Celsius.", ["water", "freezes", "0", "celsius"], ["0 degrees Celsius", "0"]),
    ("Water boils at 100 degrees Celsius at sea level.", ["water", "boils", "100", "celsius"], ["100 degrees Celsius", "100"]),
    ("Normal human body temperature is about 37 degrees Celsius.", ["body", "temperature", "37"], ["37 degrees Celsius", "37"]),
    ("Plants use sunlight to make food by photosynthesis.", ["plants", "photosynthesis", "sunlight"], ["photosynthesis"]),
    ("Plants absorb carbon dioxide from the air.", ["plants", "absorb", "carbon dioxide"], ["carbon dioxide"]),
    ("Chlorophyll makes leaves green.", ["chlorophyll", "leaves", "green"], ["Chlorophyll"]),
    ("Stars are huge balls of hot glowing gas.", ["stars", "hot", "gas"], ["Stars"]),
    ("Gravity pulls objects toward Earth.", ["gravity", "pulls", "earth"], ["Gravity"]),
    ("Magnets attract iron.", ["magnets", "attract", "iron"], ["iron"]),
    ("When water is heated, it can turn into vapor.", ["water", "heated", "vapor"], ["vapor"]),
    ("Day and night happen because Earth rotates.", ["day", "night", "earth", "rotates"], ["Earth rotates", "rotates"]),
    ("The seasons are caused by the tilt of Earth's axis.", ["seasons", "tilt", "axis"], ["tilt"]),
    ("A volcano can erupt with hot lava.", ["volcano", "erupt", "lava"], ["lava"]),
    ("Ice floats on water.", ["ice", "floats", "water"], ["Ice"]),
    ("Copper is a metal that conducts electricity well.", ["copper", "metal", "electricity"], ["Copper"]),
    ("Glass is made mostly from sand.", ["glass", "made", "sand"], ["sand"]),
    ("Paper is made from wood.", ["paper", "made", "wood"], ["wood"]),
    ("Mixing blue and yellow paint makes green.", ["blue", "yellow", "paint", "green"], ["green"]),
    ("The Sun rises in the east.", ["sun", "rises", "east"], ["east"]),
    ("A compass needle points north.", ["compass", "needle", "north"], ["north"]),
    ("The sky looks blue because air scatters blue sunlight.", ["sky", "blue", "scatters"], ["blue"]),
]
for text, keywords, aliases in SCIENCE_EXTRA:
    add("science", text, keywords, aliases, ELEMENTS_URL)

# ---------------------------------------------------------------------------
# Calendar / time (definitional, Gregorian conventions)
# ---------------------------------------------------------------------------

MONTH_DAYS = [("January", 31), ("February", 28), ("March", 31),
              ("April", 30), ("May", 31), ("June", 30), ("July", 31),
              ("August", 31), ("September", 30), ("October", 31),
              ("November", 30), ("December", 31)]
for month, days in MONTH_DAYS:
    add("calendar", f"{month} has {days} days.",
        kw(month, "days", str(days)), [str(days)], GEO_URL)
CALENDAR_EXTRA = [
    ("February has 29 days in a leap year.", ["february", "29", "leap year"], ["29"]),
    ("A normal year has 365 days.", ["year", "365", "days"], ["365"]),
    ("A leap year has 366 days.", ["leap year", "366", "days"], ["366"]),
    ("A leap year usually happens every four years.", ["leap year", "four years"], ["four years"]),
    ("A week has seven days.", ["week", "seven", "days"], ["seven"]),
    ("A day has 24 hours.", ["day", "24", "hours"], ["24"]),
    ("An hour has 60 minutes.", ["hour", "60", "minutes"], ["60"]),
    ("A minute has 60 seconds.", ["minute", "60", "seconds"], ["60"]),
    ("January is the first month of the year.", ["january", "first", "month"], ["January"]),
    ("December is the twelfth and last month of the year.", ["december", "twelfth", "month"], ["December"]),
    ("There are four seasons: spring, summer, autumn, and winter.", ["four", "seasons"], ["four"]),
    ("Autumn is also called fall.", ["autumn", "fall"], ["fall"]),
    ("New Year's Day is on January 1.", ["new year", "january 1"], ["January 1"]),
    ("Christmas Day is on December 25.", ["christmas", "december 25"], ["December 25"]),
    ("Independence Day in the United States is on July 4.", ["independence day", "july 4"], ["July 4"]),
    ("Halloween is on October 31.", ["halloween", "october 31"], ["October 31"]),
    ("A decade is a period of 10 years.", ["decade", "10 years"], ["10 years", "10"]),
    ("A century is a period of 100 years.", ["century", "100 years"], ["100 years", "100"]),
    ("A millennium is a period of 1000 years.", ["millennium", "1000 years"], ["1000 years", "1000"]),
    ("A fortnight is a period of two weeks.", ["fortnight", "two weeks"], ["two weeks"]),
    ("A clock face shows 12 hours.", ["clock", "12", "hours"], ["12"]),
    ("Saturday and Sunday are weekend days in many countries.", ["saturday", "sunday", "weekend"], ["Saturday and Sunday"]),
]
for text, keywords, aliases in CALENDAR_EXTRA:
    add("calendar", text, keywords, aliases, GEO_URL)

# ---------------------------------------------------------------------------
# History — well-known stable dates
# ---------------------------------------------------------------------------

HISTORY = [
    ("The First World War began in 1914.", ["first world war", "began", "1914"], ["1914"]),
    ("The First World War ended in 1918.", ["first world war", "ended", "1918"], ["1918"]),
    ("The Second World War began in 1939.", ["second world war", "began", "1939"], ["1939"]),
    ("The Second World War ended in 1945.", ["second world war", "ended", "1945"], ["1945"]),
    ("The United States Declaration of Independence was adopted in 1776.", ["declaration", "independence", "1776"], ["1776"]),
    ("Christopher Columbus reached the Americas in 1492.", ["columbus", "americas", "1492"], ["1492"]),
    ("The Magellan expedition was the first to sail around the world.", ["magellan", "sail", "around the world"], ["Magellan expedition", "Magellan"]),
    ("Gutenberg's printing press was developed around 1440.", ["gutenberg", "printing press", "1440"], ["1440"]),
    ("Alexander Graham Bell patented the telephone in 1876.", ["bell", "telephone", "1876"], ["1876"]),
    ("Thomas Edison developed a practical light bulb in 1879.", ["edison", "light bulb", "1879"], ["1879"]),
    ("The Wright brothers made the first powered airplane flight in 1903.", ["wright brothers", "airplane", "1903"], ["1903"]),
    ("The Titanic sank in 1912.", ["titanic", "sank", "1912"], ["1912"]),
    ("Penicillin was discovered by Alexander Fleming in 1928.", ["penicillin", "fleming", "1928"], ["1928"]),
    ("The double helix structure of DNA was described in 1953.", ["dna", "double helix", "1953"], ["1953"]),
    ("Edmund Hillary and Tenzing Norgay first climbed Mount Everest in 1953.", ["hillary", "everest", "1953"], ["1953"]),
    ("Humans first landed on the Moon in 1969.", ["moon", "landed", "1969"], ["1969"]),
    ("Apollo 11 was the first mission to land humans on the Moon.", ["apollo 11", "moon", "mission"], ["Apollo 11"]),
    ("Neil Armstrong was the first person to walk on the Moon.", ["neil armstrong", "moon"], ["Neil Armstrong"]),
    ("The Berlin Wall fell in 1989.", ["berlin wall", "fell", "1989"], ["1989"]),
    ("The ancient Egyptians built the pyramids of Giza.", ["egyptians", "pyramids", "giza"], ["pyramids of Giza"]),
    ("The ancient Romans spoke Latin.", ["romans", "spoke", "latin"], ["Latin"]),
    ("The Western Roman Empire ended in 476.", ["roman empire", "ended", "476"], ["476"]),
    ("William Shakespeare wrote the play Hamlet.", ["shakespeare", "hamlet"], ["Hamlet"]),
    ("Leonardo da Vinci painted the Mona Lisa.", ["leonardo da vinci", "mona lisa"], ["Mona Lisa"]),
    ("The Great Wall of China was built to protect China from invasions.", ["great wall", "china", "protect"], ["China"]),
    ("The ancient Olympic Games were held in Greece.", ["olympic games", "greece", "ancient"], ["Greece"]),
    ("The modern Olympic Games began in 1896.", ["olympic games", "modern", "1896"], ["1896"]),
    ("Dinosaurs went extinct about 66 million years ago.", ["dinosaurs", "extinct", "66 million"], ["66 million years ago", "66 million"]),
    ("India passed China in population in 2023.", ["india", "china", "population", "2023"], ["2023"]),
    ("Washington, D.C. became the capital of the United States in 1800.", ["washington", "capital", "1800"], ["1800"]),
]
for text, keywords, aliases in HISTORY:
    add("history", text, keywords, aliases, GEO_URL)

# ---------------------------------------------------------------------------
# Animals + human body
# ---------------------------------------------------------------------------

ANIMALS = [
    ("The blue whale is the largest animal in the world.", ["blue whale", "largest", "animal"], ["blue whale"]),
    ("The African elephant is the largest land animal.", ["african elephant", "largest", "land animal"], ["African elephant"]),
    ("The cheetah is the fastest land animal.", ["cheetah", "fastest", "land animal"], ["cheetah"]),
    ("The giraffe is the tallest land animal.", ["giraffe", "tallest", "land animal"], ["giraffe"]),
    ("An octopus has eight arms.", ["octopus", "eight", "arms"], ["eight"]),
    ("An octopus has three hearts.", ["octopus", "three", "hearts"], ["three"]),
    ("A spider has eight legs.", ["spider", "eight", "legs"], ["eight"]),
    ("Insects have six legs.", ["insects", "six", "legs"], ["six"]),
    ("Penguins are birds that cannot fly.", ["penguins", "birds", "cannot fly"], ["Penguins"]),
    ("The ostrich is the largest bird in the world.", ["ostrich", "largest", "bird"], ["ostrich"]),
    ("Hummingbirds can fly backwards.", ["hummingbirds", "fly", "backwards"], ["Hummingbirds"]),
    ("Bats are the only mammals that can truly fly.", ["bats", "mammals", "fly"], ["Bats"]),
    ("Whales are mammals, not fish.", ["whales", "mammals", "not fish"], ["mammals"]),
    ("Dolphins are mammals.", ["dolphins", "mammals"], ["mammals"]),
    ("Sharks are fish.", ["sharks", "fish"], ["fish"]),
    ("Frogs begin life as tadpoles.", ["frogs", "tadpoles"], ["tadpoles"]),
    ("A caterpillar turns into a butterfly.", ["caterpillar", "butterfly"], ["butterfly"]),
    ("Bees make honey.", ["bees", "honey"], ["honey"]),
    ("Giant pandas eat mostly bamboo.", ["pandas", "eat", "bamboo"], ["bamboo"]),
    ("Koalas eat eucalyptus leaves.", ["koalas", "eucalyptus"], ["eucalyptus"]),
    ("Kangaroos carry their babies in a pouch.", ["kangaroos", "babies", "pouch"], ["pouch"]),
    ("Polar bears live in the Arctic.", ["polar bears", "arctic"], ["Arctic"]),
    ("Many penguins live in Antarctica.", ["penguins", "antarctica"], ["Antarctica"]),
    ("Owls are birds that hunt at night.", ["owls", "hunt", "night"], ["Owls"]),
    ("An elephant uses its trunk to pick up food and water.", ["elephant", "trunk"], ["trunk"]),
    ("Hippos spend much of the day in water.", ["hippos", "water", "day"], ["Hippos"]),
    ("Rhinoceroses have horns on their noses.", ["rhinoceroses", "horns", "noses"], ["horns"]),
    ("Sloths move very slowly.", ["sloths", "slowly"], ["Sloths"]),
    ("Chameleons can change the color of their skin.", ["chameleons", "change", "color"], ["Chameleons"]),
    ("Flamingos get their pink color from their food.", ["flamingos", "pink", "food"], ["pink"]),
    ("Male peacocks spread bright tail feathers to attract mates.", ["peacocks", "tail feathers"], ["peacocks"]),
    ("Fish breathe through gills.", ["fish", "breathe", "gills"], ["gills"]),
    ("Birds lay eggs.", ["birds", "lay", "eggs"], ["eggs"]),
    ("A group of wolves is called a pack.", ["wolves", "group", "pack"], ["pack"]),
    ("A baby frog is called a tadpole.", ["baby frog", "tadpole"], ["tadpole"]),
    ("An adult human body has 206 bones.", ["adult", "human", "206", "bones"], ["206"]),
    ("Babies are born with about 300 bones.", ["babies", "300", "bones"], ["300"]),
    ("The femur is the longest bone in the human body.", ["femur", "longest", "bone"], ["femur"]),
    ("The skin is the largest organ of the human body.", ["skin", "largest", "organ"], ["skin"]),
    ("The heart pumps blood around the body.", ["heart", "pumps", "blood"], ["heart"]),
    ("The human heart has four chambers.", ["heart", "four", "chambers"], ["four"]),
    ("The lungs help the body take in oxygen.", ["lungs", "oxygen", "breathe"], ["lungs"]),
    ("The brain is the control center of the body.", ["brain", "control center"], ["brain"]),
    ("Most adults have 32 teeth.", ["adults", "32", "teeth"], ["32"]),
    ("Humans have five main senses.", ["humans", "five", "senses"], ["five"]),
    ("No two people have exactly the same fingerprints.", ["fingerprints", "same", "people"], ["fingerprints"]),
    ("Blood carries oxygen around the body.", ["blood", "carries", "oxygen"], ["Blood"]),
    ("Muscles help the body move.", ["muscles", "body", "move"], ["Muscles"]),
]
for text, keywords, aliases in ANIMALS:
    add("animals", text, keywords, aliases, BODY_ANIMALS_URL)

# ---------------------------------------------------------------------------
# Language basics
# ---------------------------------------------------------------------------

LANGUAGE = [
    ("The English alphabet has 26 letters.", ["english", "alphabet", "26", "letters"], ["26"]),
    ("Z is the last letter of the English alphabet.", ["z", "last", "letter", "alphabet"], ["Z"]),
    ("English has five vowels: a, e, i, o, and u.", ["english", "five", "vowels"], ["five"]),
    ("A sentence starts with a capital letter.", ["sentence", "starts", "capital letter"], ["capital letter"]),
    ("A question ends with a question mark.", ["question", "ends", "question mark"], ["question mark"]),
    ("A noun names a person, place, or thing.", ["noun", "names", "person", "place"], ["noun"]),
    ("A verb is a word that shows action.", ["verb", "action", "word"], ["verb"]),
    ("An adjective is a word that describes a noun.", ["adjective", "describes", "noun"], ["adjective"]),
    ("Words with the same meaning are called synonyms.", ["words", "same meaning", "synonyms"], ["synonyms"]),
    ("Words with opposite meanings are called antonyms.", ["words", "opposite", "antonyms"], ["antonyms"]),
    ("The plural of child is children.", ["plural", "child", "children"], ["children"]),
    ("The plural of mouse is mice.", ["plural", "mouse", "mice"], ["mice"]),
    ("'Hola' means hello in Spanish.", ["hola", "hello", "spanish"], ["Hola"]),
    ("'Gracias' means thank you in Spanish.", ["gracias", "thank you", "spanish"], ["Gracias"]),
    ("'Bonjour' means hello in French.", ["bonjour", "hello", "french"], ["Bonjour"]),
    ("'Merci' means thank you in French.", ["merci", "thank you", "french"], ["Merci"]),
    ("'Arigato' means thank you in Japanese.", ["arigato", "thank you", "japanese"], ["Arigato"]),
    ("'Danke' means thank you in German.", ["danke", "thank you", "german"], ["Danke"]),
    ("'Ciao' can mean hello or goodbye in Italian.", ["ciao", "hello", "goodbye", "italian"], ["Ciao"]),
    ("Mandarin Chinese has the most native speakers of any language.", ["mandarin", "most", "native speakers"], ["Mandarin Chinese", "Mandarin"]),
    ("English is written with the Latin alphabet.", ["english", "written", "latin alphabet"], ["Latin alphabet"]),
    ("The word 'rapid' means fast.", ["rapid", "means", "fast"], ["fast"]),
    ("The word 'begin' means start.", ["begin", "means", "start"], ["start"]),
    ("A dictionary lists words and their meanings.", ["dictionary", "words", "meanings"], ["dictionary"]),
]
for text, keywords, aliases in LANGUAGE:
    add("language", text, keywords, aliases, GEO_URL)

# ---------------------------------------------------------------------------
# PlaygroundLM self-facts (from the project record)
# ---------------------------------------------------------------------------

SELF_FACTS = [
    ("PlaygroundLM has about 124M parameters (124 million).", ["playgroundlm", "124m", "parameters"], ["124M", "124 million"]),
    ("PlaygroundLM was trained from scratch by the Galactic AI Team.", ["playgroundlm", "trained", "scratch", "galactic ai team"], ["Galactic AI Team"]),
    ("PlaygroundLM is not ChatGPT.", ["playgroundlm", "not", "chatgpt"], ["not ChatGPT"]),
    ("PlaygroundLM was not made by OpenAI.", ["playgroundlm", "not", "openai"], ["not made by OpenAI", "OpenAI"]),
    ("PlaygroundLM is a GPT-2-small-class model.", ["playgroundlm", "gpt-2", "small", "model"], ["GPT-2-small-class"]),
    ("PlaygroundLM has 12 transformer layers.", ["playgroundlm", "12", "layers"], ["12"]),
    ("PlaygroundLM has 12 attention heads.", ["playgroundlm", "12", "attention heads"], ["12"]),
    ("PlaygroundLM's model width is 768.", ["playgroundlm", "width", "768"], ["768"]),
    ("PlaygroundLM can read up to 1024 tokens at once.", ["playgroundlm", "1024", "tokens", "context"], ["1024"]),
    ("The PlaygroundLM demo lives at playground-lm.vercel.app.", ["playgroundlm", "demo", "website"], ["playground-lm.vercel.app"]),
    ("The Galactic AI Team includes Lumi, Claude, GPT-6, Gemini, Dee, and Nova.", ["galactic ai team", "members", "agents"], ["Galactic AI Team"]),
]
for text, keywords, aliases in SELF_FACTS:
    add("self", text, keywords, aliases, SELF_URL)

# ---------------------------------------------------------------------------
# Held-out marking (~20% per topic, spread) + validation + write
# ---------------------------------------------------------------------------

by_topic: dict[str, list[dict]] = {}
for fact in FACTS:
    by_topic.setdefault(fact["topic"], []).append(fact)
for topic, rows in by_topic.items():
    for i, row in enumerate(rows):
        row["heldout"] = (i % 5 == 4)

seen_ids: set[str] = set()
seen_texts: set[str] = set()
for fact in FACTS:
    assert fact["id"] not in seen_ids, fact["id"]
    seen_ids.add(fact["id"])
    norm = " ".join(fact["text"].lower().split())
    assert norm not in seen_texts, fact["text"]
    seen_texts.add(norm)
    assert len(fact["text"].split()) <= 20, fact["text"]
    assert fact["keywords"], fact["id"]
    for alias in fact["aliases"]:
        assert alias.casefold() in fact["text"].casefold(), (fact["id"], alias)
    line = json.dumps(fact, ensure_ascii=False, separators=(",", ":"))
    assert len(line.split()) <= 30, (fact["id"], len(line.split()))

heldout = sum(1 for f in FACTS if f["heldout"])
print(f"facts: {len(FACTS)}  heldout: {heldout} ({heldout / len(FACTS):.1%})")
for topic, rows in sorted(by_topic.items()):
    print(f"  {topic}: {len(rows)} (heldout {sum(1 for r in rows if r['heldout'])})")
with OUT.open("w", encoding="utf-8") as fh:
    for fact in FACTS:
        fh.write(json.dumps(fact, ensure_ascii=False, separators=(",", ":")) + "\n")
print("wrote", OUT)
