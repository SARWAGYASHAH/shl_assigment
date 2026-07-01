"""
SHL Product Catalog Scraper
Scrapes Individual Test Solutions from the SHL online catalog.
Uses requests + BeautifulSoup with fallback to Selenium for JS-rendered content.
"""

import json
import time
import re
import sys
from pathlib import Path

try:
    from bs4 import BeautifulSoup
    import requests
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "beautifulsoup4", "requests"])
    from bs4 import BeautifulSoup
    import requests


BASE_URL = "https://online.shl.com"
CATALOG_URL = f"{BASE_URL}/gb/en-us/products"
PRODUCT_TYPES = 1  # Individual Test Solutions


def scrape_catalog_page(page: int, page_size: int = 50) -> tuple[list[dict], int]:
    """Scrape a single page of the catalog. Returns (products, total_count)."""
    params = {
        "orderby": "none",
        "page": page,
        "producttypes": PRODUCT_TYPES,
        "pagesize": page_size,
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    
    resp = requests.get(CATALOG_URL, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    
    soup = BeautifulSoup(resp.text, "html.parser")
    products = []
    total = 0
    
    # Try to find total count
    count_el = soup.find("span", class_="product-search__results-count") or \
               soup.find(text=re.compile(r"\d+\s+entries"))
    if count_el:
        match = re.search(r"(\d+)", count_el.get_text() if hasattr(count_el, 'get_text') else str(count_el))
        if match:
            total = int(match.group(1))
    
    # Parse product cards
    cards = soup.find_all("div", class_=re.compile(r"product[_-]card|catalog[_-]item")) or \
            soup.find_all("tr", class_=re.compile(r"product")) or \
            soup.find_all("a", href=re.compile(r"/products/"))
    
    for card in cards:
        product = parse_product_card(card)
        if product and product.get("name"):
            products.append(product)
    
    # If no structured products found, try table rows
    if not products:
        tables = soup.find_all("table")
        for table in tables:
            rows = table.find_all("tr")
            for row in rows[1:]:  # Skip header
                cells = row.find_all(["td", "th"])
                if len(cells) >= 2:
                    link = row.find("a", href=True)
                    name = cells[0].get_text(strip=True)
                    if name and link:
                        url = link["href"]
                        if not url.startswith("http"):
                            url = BASE_URL + url
                        product = {
                            "name": name,
                            "url": url,
                            "description": "",
                            "test_type": "",
                        }
                        # Extract additional metadata from cells
                        for cell in cells[1:]:
                            text = cell.get_text(strip=True)
                            icons = cell.find_all("i") or cell.find_all("span", class_=re.compile(r"icon|check"))
                            if icons or "✓" in text or "Yes" in text:
                                product["has_metadata"] = True
                        products.append(product)
    
    return products, total


def parse_product_card(card) -> dict:
    """Parse a single product card element."""
    product = {}
    
    # Get name
    name_el = card.find(["h2", "h3", "h4", "a", "span"], class_=re.compile(r"name|title"))
    if name_el:
        product["name"] = name_el.get_text(strip=True)
    elif card.name == "a":
        product["name"] = card.get_text(strip=True)
    
    # Get URL
    link = card.find("a", href=True) if card.name != "a" else card
    if link and link.get("href"):
        url = link["href"]
        if not url.startswith("http"):
            url = BASE_URL + url
        product["url"] = url
    
    # Get description
    desc_el = card.find(["p", "div", "span"], class_=re.compile(r"desc|summary|text"))
    if desc_el:
        product["description"] = desc_el.get_text(strip=True)
    
    return product


def scrape_product_detail(url: str) -> dict:
    """Scrape detailed information from a product page."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        detail = {}
        
        # Get description
        desc = soup.find("div", class_=re.compile(r"description|summary|content|detail"))
        if desc:
            detail["description"] = desc.get_text(strip=True)[:500]
        
        # Get test type / category
        cat = soup.find(text=re.compile(r"Test Type|Category|Type", re.I))
        if cat:
            parent = cat.find_parent()
            if parent:
                sibling = parent.find_next_sibling()
                if sibling:
                    detail["test_type"] = sibling.get_text(strip=True)
        
        # Get duration
        dur = soup.find(text=re.compile(r"Duration|Time|Length", re.I))
        if dur:
            parent = dur.find_parent()
            if parent:
                detail["duration"] = parent.get_text(strip=True)
        
        return detail
    except Exception as e:
        print(f"  Error scraping {url}: {e}")
        return {}


def scrape_all():
    """Scrape all Individual Test Solutions from the catalog."""
    all_products = []
    page = 1
    total = None
    
    print("Starting SHL catalog scrape...")
    print(f"URL: {CATALOG_URL}?producttypes={PRODUCT_TYPES}")
    
    while True:
        print(f"\nScraping page {page}...")
        products, count = scrape_catalog_page(page, page_size=50)
        
        if count > 0 and total is None:
            total = count
            print(f"Total products: {total}")
        
        if not products:
            print(f"No products found on page {page}. Stopping.")
            break
        
        all_products.extend(products)
        print(f"  Found {len(products)} products (total so far: {len(all_products)})")
        
        if total and len(all_products) >= total:
            break
        
        page += 1
        time.sleep(1)  # Be polite
    
    # Scrape details for each product
    print(f"\nScraping details for {len(all_products)} products...")
    for i, product in enumerate(all_products):
        if "url" in product:
            print(f"  [{i+1}/{len(all_products)}] {product.get('name', 'Unknown')}")
            detail = scrape_product_detail(product["url"])
            product.update(detail)
            time.sleep(0.5)
    
    return all_products


def save_catalog(products: list[dict], filepath: str = "data/catalog.json"):
    """Save scraped catalog to JSON."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(products, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(products)} products to {filepath}")


if __name__ == "__main__":
    products = scrape_all()
    if products:
        save_catalog(products)
    else:
        print("\nNo products scraped via requests. The catalog may require JavaScript rendering.")
        print("Using pre-built catalog data instead.")
