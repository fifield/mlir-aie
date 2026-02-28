#!/usr/bin/env python3
"""
Parse AIE4 HTML Documentation and generate register database

This script parses AIE4 HTML register documentation from multiple URLs
and generates a unified register database compatible with the AIE address decoder.

Usage:
    python parse_aie4_html_docs.py [--output aie_registers_aie4_html.json]
"""

import argparse
import json
import re
import sys
import time
from typing import Dict, List, Optional
from urllib.request import urlopen
from urllib.error import URLError
from bs4 import BeautifulSoup


class AIE4HtmlParser:
    """Parser for AIE4 HTML register documentation"""
    
    # Maps HTML h1 section titles to output module names.
    # When a single HTML file contains multiple h1 sections, registers
    # under each section are routed to the corresponding module.
    # The key is a substring matched against the h1 heading text.
    SECTION_MODULE_MAP = {
        "aie4_core_module": "CORE_MODULE",
        "aie4_core_internal_module": "CORE_MODULE",
        "aie4_memory_module": "MEMORY_MODULE",
        "aie4_mem_tile_module": "MEMORY_TILE_MODULE",
        "aie4_noc_module": "NOC_MODULE",
        "aie4_pl_module": "PL_MODULE",
        "aie4_uc_module": "UC_MODULE",
        "aie4_uc_module_core": "UC_MODULE",
        "aie4_npi": "NPI_MODULE",
    }

    # Fallback: when no h1 section headings are found, map the whole
    # file's registers to these modules (one list per module).
    URL_FALLBACK_MAP = {
        "xregdb_aie4_tile_doc.html": ["CORE_MODULE"],
        "xregdb_aie4_mem_tile_doc.html": ["MEMORY_TILE_MODULE"],
        "xregdb_aie4_shim_tile_doc.html": ["NOC_MODULE"],
        "xregdb_aie4_uc_core_doc.html": ["UC_MODULE"],
        "xregdb_aie4_npi_regdb_doc.html": ["NPI_MODULE"],
    }
    
    def __init__(self, base_url: str = "http://cervino-doc/aie4/r0p49/tile_links"):
        self.base_url = base_url
        self.modules = {}
        self.total_registers = 0
        
    def fetch_html(self, url: str) -> str:
        """Fetch HTML content from URL"""
        try:
            with urlopen(url, timeout=30) as response:
                return response.read().decode('utf-8')
        except URLError as e:
            print(f"Error fetching {url}: {e}")
            return None
            
    def _build_heading_index(self, soup: BeautifulSoup) -> Dict[str, 'Tag']:
        """Pre-index all register detail headings for O(1) lookup.

        Returns a dict mapping register name -> heading element.
        This replaces the O(R*H) scan in the old parse_bit_fields.
        """
        index = {}
        for heading in soup.find_all(['h3', 'h4', 'h5']):
            text = heading.get_text(strip=True)
            # Store under the full heading text; lookup will match by
            # checking if the register name is contained in any key.
            index[text] = heading
        return index

    def parse_register_tables_by_section(self, soup: BeautifulSoup) -> Dict[str, List[Dict]]:
        """Parse register summary tables, grouping by h1 section.

        Returns a dict mapping h1 heading text -> list of registers found
        under that section.  If there are no h1 headings the registers are
        returned under the key "".
        """
        # Build heading index once for O(1) bitfield lookups
        t0 = time.time()
        heading_index = self._build_heading_index(soup)
        print(f"    Built heading index ({len(heading_index)} entries) in {time.time()-t0:.1f}s")

        # Find all tables - look for "Register Summary" section
        tables = soup.find_all('table')

        sections: Dict[str, List[Dict]] = {}
        reg_count = 0
        for table in tables:
            # Check if this is a register summary table
            headers = table.find_all('th')
            if not headers:
                continue

            header_text = [h.get_text(strip=True) for h in headers]

            # Look for register table headers
            if 'Register Name' not in header_text and 'Register_Name' not in header_text:
                continue

            rows = table.find_all('tr')[1:]  # Skip header row

            # Skip per-register detail tables (1 row); only process
            # multi-row summary tables to avoid double-counting.
            data_rows = [r for r in rows if len(r.find_all('td')) >= 5]
            if len(data_rows) <= 1:
                continue

            # Determine which h1 section this table belongs to
            prev_h1 = table.find_previous('h1')
            section_key = prev_h1.get_text(strip=True) if prev_h1 else ""

            if section_key not in sections:
                sections[section_key] = []

            for row in data_rows:
                cols = row.find_all('td')
                if len(cols) >= 5:
                    # Extract register information
                    reg_name = cols[0].get_text(strip=True)
                    address = cols[1].get_text(strip=True)
                    width = cols[2].get_text(strip=True)
                    access_type = cols[3].get_text(strip=True)
                    reset_value = cols[4].get_text(strip=True)
                    description = cols[5].get_text(strip=True) if len(cols) > 5 else ""

                    # Clean up the data
                    reg_name = reg_name.replace('[', '').replace(']', '')
                    access_type = access_type.replace('*', '').strip()

                    register = {
                        "name": reg_name,
                        "offset": address,
                        "width": int(width) if width.isdigit() else width,
                        "type": access_type,
                        "reset": reset_value,
                        "description": description
                    }

                    # Parse bit-fields for this register using the pre-built index
                    bit_fields = self.parse_bit_fields_indexed(heading_index, reg_name)
                    if bit_fields:
                        register["bit_fields"] = bit_fields

                    sections[section_key].append(register)
                    reg_count += 1
                    if reg_count % 500 == 0:
                        print(f"    Processed {reg_count} registers...")

        return sections

    def parse_bit_fields_indexed(self, heading_index: Dict[str, 'Tag'], register_name: str) -> List[Dict]:
        """Parse bit-field details using the pre-built heading index."""
        bit_fields = []

        # Find matching heading via index lookup
        heading = None
        for text, h in heading_index.items():
            if register_name in text:
                heading = h
                break

        if heading is None:
            return bit_fields

        # Find the next table after this heading
        table = heading.find_next('table')
        if not table:
            return bit_fields

        rows = table.find_all('tr')[1:]  # Skip header
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 4:
                field_name = cols[0].get_text(strip=True)
                bits = cols[1].get_text(strip=True)
                field_type = cols[2].get_text(strip=True)
                reset_val = cols[3].get_text(strip=True)
                field_desc = cols[4].get_text(strip=True) if len(cols) > 4 else ""

                # Clean up
                field_type = field_type.replace('*', '').strip()

                # Parse bits string (e.g. "31:24" or "0") into
                # [low, high] array matching the AIE2 bit_range format.
                if ':' in bits:
                    high_s, low_s = bits.split(':', 1)
                    bit_range = [int(low_s), int(high_s)]
                else:
                    bit_val = int(bits)
                    bit_range = [bit_val, bit_val]

                bit_field = {
                    "name": field_name,
                    "bits": bits,
                    "bit_range": bit_range,
                    "type": field_type,
                    "reset": reset_val,
                    "description": field_desc
                }

                bit_fields.append(bit_field)

        return bit_fields
    
    def parse_url(self, url: str, fallback_modules: List[str]) -> Dict:
        """Parse a single HTML URL and extract registers.

        Registers are grouped by their h1 section heading and mapped to
        output modules via SECTION_MODULE_MAP.  If the HTML has no
        recognisable h1 sections, all registers are assigned to the
        fallback_modules list.
        """
        print(f"Parsing {url}...")

        t0 = time.time()
        html_content = self.fetch_html(url)
        if not html_content:
            return {}
        print(f"  Fetched ({len(html_content)//1024} KB) in {time.time()-t0:.1f}s")

        t1 = time.time()
        soup = BeautifulSoup(html_content, 'lxml')
        print(f"  Parsed HTML in {time.time()-t1:.1f}s")

        t2 = time.time()
        sections = self.parse_register_tables_by_section(soup)
        total = sum(len(v) for v in sections.values())
        print(f"  Found {total} registers across {len(sections)} section(s) in {time.time()-t2:.1f}s")

        # Map sections to modules
        result: Dict[str, Dict] = {}
        for section_heading, registers in sections.items():
            # Try to match the section heading to a module via SECTION_MODULE_MAP
            module_name = None
            for key, mod in self.SECTION_MODULE_MAP.items():
                if key in section_heading:
                    module_name = mod
                    break

            if module_name:
                print(f"    Section \"{section_heading}\" -> {module_name} ({len(registers)} regs)")
            else:
                # Fall back to assigning to each fallback module
                print(f"    Section \"{section_heading}\" -> fallback {fallback_modules} ({len(registers)} regs)")
                for fb in fallback_modules:
                    if fb not in result:
                        result[fb] = {"registers": []}
                    result[fb]["registers"].extend(registers)
                continue

            if module_name not in result:
                result[module_name] = {"registers": []}
            result[module_name]["registers"].extend(registers)

        return result
    
    def parse_all_urls(self, url_filter: Optional[str] = None) -> Dict:
        """Parse all AIE4 HTML documentation URLs"""
        urls = [
            f"{self.base_url}/driver/xregdb_aie4_tile_doc.html",
            f"{self.base_url}/driver/xregdb_aie4_mem_tile_doc.html",
            f"{self.base_url}/driver/xregdb_aie4_shim_tile_doc.html",
            f"{self.base_url}/xregdb_aie4_uc_core_doc.html",
            f"{self.base_url}/xregdb_aie4_npi_regdb_doc.html"
        ]

        if url_filter:
            urls = [u for u in urls if url_filter in u]
            print(f"Filtered to {len(urls)} URL(s) matching '{url_filter}'")

        for url in urls:
            # Extract filename from URL
            filename = url.split('/')[-1]

            if filename in self.URL_FALLBACK_MAP:
                fallback_modules = self.URL_FALLBACK_MAP[filename]
                parsed_modules = self.parse_url(url, fallback_modules)
                
                # Merge into main modules dict
                for module_name, module_data in parsed_modules.items():
                    if module_name in self.modules:
                        # Append registers if module already exists
                        self.modules[module_name]["registers"].extend(module_data["registers"])
                    else:
                        self.modules[module_name] = module_data
                        
                    self.total_registers += len(module_data["registers"])
        
        # Normalize module names to match AIE2 convention
        MODULE_NAME_MAP = {
            "CORE_MODULE": "core",
            "MEMORY_MODULE": "memory",
            "MEMORY_TILE_MODULE": "memory_tile",
            "NOC_MODULE": "shim",
            "PL_MODULE": "shim",   # Merge PL into shim (like AIE2)
            "UC_MODULE": "uc",
            "NPI_MODULE": "npi",
        }

        normalized = {}
        for raw_name, module_data in self.modules.items():
            canonical = MODULE_NAME_MAP.get(raw_name, raw_name)
            if canonical in normalized:
                # Merge registers (e.g. NOC + PL -> shim)
                normalized[canonical]["registers"].extend(module_data["registers"])
            else:
                normalized[canonical] = module_data

        return {
            "metadata": {
                "architecture": "aie4",
                "source": "HTML documentation",
                "urls": urls,
                "register_count": self.total_registers
            },
            "modules": normalized
        }


def main():
    parser = argparse.ArgumentParser(
        description="Parse AIE4 HTML documentation and generate register database"
    )
    parser.add_argument(
        "--base-url",
        "-u",
        default="http://cervino-doc/aie4/r0p49/tile_links",
        help="Base URL for AIE4 documentation"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="aie_registers_aie4_html.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--url-filter",
        "-f",
        default=None,
        help="Only parse URLs containing this substring (e.g. 'uc_core' or 'npi')"
    )

    args = parser.parse_args()
    
    print("AIE4 HTML Documentation Parser")
    print("=" * 60)
    print(f"Base URL: {args.base_url}")
    print(f"Output file: {args.output}")
    print()
    
    parser_obj = AIE4HtmlParser(args.base_url)
    
    try:
        print("Parsing HTML documentation...")
        data = parser_obj.parse_all_urls(url_filter=args.url_filter)
        
        # Save to JSON file
        with open(args.output, 'w') as f:
            json.dump(data, f, indent=2)
            
        print()
        print("=" * 60)
        print(f"Successfully generated AIE4 register database: {args.output}")
        print(f"Architecture: {data['metadata']['architecture']}")
        print(f"Total registers: {data['metadata']['register_count']}")
        print(f"Modules: {', '.join(data['modules'].keys())}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
