import sys
import argparse
import json
import requests
from bs4 import BeautifulSoup

#!/usr/bin/env python3
"""
main.py - fetch a webpage and extract a Poképedia <tbody> into a python list.

Usage:
    python main.py "https://www.example.com/page" --selector "table.wikitable > tbody" --index 0
"""


USER_AGENT = "PokeShout/1.0 (+https://example.com/)"


def fetch_html(url, timeout=10):
    resp = requests.get(
        url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def fetch_ogg(url):
    # Get audio file .ogg from url , url starts with www.
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
    resp.raise_for_status()
    return resp.content


def extract_tbody(html):
    soup = BeautifulSoup(html, "html.parser")
    tbodies = soup.find_all("table")
    if not tbodies:
        return []

    # Search for tbody that contains class 'tableaustandard', 'sortable', 'entetefixe'
    all_pokemon_table_class = ['tableaustandard', 'sortable', 'entetefixe']
    for table in tbodies:
        classes = table.get('class', [])
        hasAllClasses = True
        for cls in all_pokemon_table_class:
            if cls not in classes:
                hasAllClasses = False
                break
        if hasAllClasses:
            return table


def html_table_to_rows(tbody):
    rows = []
    for tr in tbody.find_all("tr"):
        cells = tr.find_all("td")
        if not cells:
            continue

        pokemon_id = cells[0].get_text(strip=True)
        pokemon_name = cells[2].get_text(strip=True)

        pokemon_audio_page = f"https://www.pokepedia.fr/Fichier:Cri_{pokemon_id}_HOME.ogg"

        rows.append([pokemon_id, pokemon_name, pokemon_audio_page])
    return rows


def fetch_audio_file(pokemon_name, pokemon_audio_url):
    print(f"Fetching audio from: {pokemon_audio_url}")
    try:
        html = fetch_html(pokemon_audio_url)
    except Exception as e:
        print(f"Error fetching audio page for {pokemon_name}: {e}")
        return

    soup = BeautifulSoup(html, "html.parser")
    audio = soup.find("audio")
    if audio:
        source = audio.get("src")
        print(f"Found audio source: <{source}>")
        # Save audio file as ogg
        with open(f"audios/{pokemon_name}.ogg", "wb") as f:
            ogg_data = fetch_ogg(f"https:{source}")
            f.write(ogg_data)
            print(f"Saved audio file: audios/{pokemon_name}.ogg")
        return


def main(argv):
    html = fetch_html(
        "https://www.pokepedia.fr/Liste_des_Pok%C3%A9mon_dans_l%27ordre_du_Pok%C3%A9dex_National")

    html_table = extract_tbody(html)
    pokemons_data = html_table_to_rows(html_table)

    audios = []
    for pokemon in pokemons_data:
        pokemon_id = pokemon[0]
        if not pokemon_id.isdigit():
            continue
        if int(pokemon_id) < 718:
            continue
        pokemon_name = pokemon[1]
        pokemon_audio_page = pokemon[2]

        fetch_audio_file(pokemon_name, pokemon_audio_page)


if __name__ == "__main__":
    main(sys.argv[1:])
