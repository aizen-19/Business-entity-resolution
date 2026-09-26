#!/usr/bin/env python3
"""
Unit tests for the normalization pipeline (src/normalize.py).

Covers every noise pattern identified in Phase 1 EDA:
 - Legal suffix variations (42% of matched pairs)
 - Case differences (6.11%)
 - Punctuation / & vs 'and' (9.85%, 0.55%)
 - Word order differences (16.98%)
 - Domain-format names, hashtag names, junk prefixes
 - Transliteration flag (non-Latin scripts)
 - Address abbreviation expansion, landmark extraction
 - Empty / NaN address handling
 - Country-aware processing (US, India, France, unknown)

Run:  pytest src/test_normalize.py -v
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Ensure the src directory is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import normalize_name, normalize_address


# ====================================================================
# NAME NORMALIZATION TESTS
# ====================================================================

class TestNormalizeNameSuffix:
    """Legal suffix extraction and canonicalization."""

    def test_inc_vs_incorporated(self):
        s = pd.Series(["Acme Robotics Inc.", "Acme Robotics Incorporated"])
        r = normalize_name(s)
        assert r["name_core"].iloc[0] == r["name_core"].iloc[1] == "acme robotics"
        assert r["legal_suffix"].iloc[0] == "inc"
        assert r["legal_suffix"].iloc[1] == "inc"

    def test_pvt_ltd_variants(self):
        s = pd.Series([
            "Acme Private Limited",
            "Acme Pvt Ltd",
            "Acme Pvt. Ltd.",
            "Acme Pvt Limited",
            "Acme Private Ltd",
        ])
        r = normalize_name(s)
        for i in range(len(s)):
            assert r["name_core"].iloc[i] == "acme", f"Row {i}: {r['name_core'].iloc[i]}"
            assert r["legal_suffix"].iloc[i] == "pvt ltd", f"Row {i}: {r['legal_suffix'].iloc[i]}"

    def test_llc_variants(self):
        s = pd.Series(["Bright Cafe LLC", "Bright Cafe L.L.C.", "bright cafe llc"])
        r = normalize_name(s)
        for i in range(3):
            assert r["legal_suffix"].iloc[i] == "llc"
            assert r["name_core"].iloc[i] == "bright cafe"

    def test_corp_vs_corporation(self):
        s = pd.Series(["Delta Corp", "Delta Corporation"])
        r = normalize_name(s)
        assert r["name_core"].iloc[0] == r["name_core"].iloc[1] == "delta"
        assert r["legal_suffix"].iloc[0] == "corp"
        assert r["legal_suffix"].iloc[1] == "corp"

    def test_no_suffix(self):
        s = pd.Series(["Sunshine Bakery"])
        r = normalize_name(s)
        assert r["name_core"].iloc[0] == "sunshine bakery"
        assert r["legal_suffix"].iloc[0] == ""
        assert r["name_normalized"].iloc[0] == "sunshine bakery"

    def test_normalized_has_canonical_suffix(self):
        s = pd.Series(["Acme Incorporated"])
        r = normalize_name(s)
        assert r["name_normalized"].iloc[0] == "acme inc"

    def test_parenthesized_suffix(self):
        """Parenthesized suffix like '(LLC)' or '(Limited)' should be extracted."""
        s = pd.Series(["Riehle Machine Works (LLC)", "Acme (Limited)"])
        r = normalize_name(s)
        assert r["legal_suffix"].iloc[0] == "llc"
        assert r["legal_suffix"].iloc[1] == "ltd"

    def test_french_suffixes(self):
        s = pd.Series(["ZNB Club SARL", "Fractales Amis SA", "Groupe SAS"])
        r = normalize_name(s)
        assert r["legal_suffix"].iloc[0] == "sarl"
        assert r["legal_suffix"].iloc[1] == "sa"
        assert r["legal_suffix"].iloc[2] == "sas"


class TestNormalizeNameCleaning:
    """Text cleaning: case, punctuation, & vs 'and', junk removal."""

    def test_case_normalization(self):
        s = pd.Series(["Bright Cafe LLC", "BRIGHT CAFE LLC", "bright cafe llc"])
        r = normalize_name(s)
        assert r["name_normalized"].iloc[0] == r["name_normalized"].iloc[1]
        assert r["name_normalized"].iloc[0] == r["name_normalized"].iloc[2]

    def test_ampersand_to_and(self):
        s = pd.Series(["A & B Traders", "A and B Traders"])
        r = normalize_name(s)
        assert r["name_core"].iloc[0] == r["name_core"].iloc[1]
        assert "and" in r["name_core"].iloc[0]
        assert "&" not in r["name_core"].iloc[0]

    def test_hashtag_cleanup(self):
        s = pd.Series(["#cleanmaterials"])
        r = normalize_name(s)
        assert r["name_core"].iloc[0] == "cleanmaterials"
        assert r["was_domain_format"].iloc[0] == False

    def test_domain_format(self):
        s = pd.Series(["heliospetroleum.com"])
        r = normalize_name(s)
        assert r["was_domain_format"].iloc[0] == True
        assert r["name_core"].iloc[0] == "heliospetroleum"

    def test_domain_with_hyphen(self):
        s = pd.Series(["my-business-name.com"])
        r = normalize_name(s)
        assert r["was_domain_format"].iloc[0] == True
        # Hyphens/dots replaced with spaces, then collapsed
        assert "my" in r["name_core"].iloc[0]
        assert "business" in r["name_core"].iloc[0]

    def test_leading_dashes(self):
        s = pd.Series(["-- Margolies Apex Inc"])
        r = normalize_name(s)
        assert r["name_core"].iloc[0] == "margolies apex"

    def test_leading_stars(self):
        s = pd.Series(["*** UROLOGY HORIZON HEALTH LLC"])
        r = normalize_name(s)
        assert r["name_core"].iloc[0] == "urology horizon health"

    def test_leading_angle_brackets(self):
        s = pd.Series(["<< Team Ecole"])
        r = normalize_name(s)
        assert r["name_core"].iloc[0] == "team ecole"

    def test_honorific_sri(self):
        s = pd.Series(["Sri Ram Trading Pvt Ltd"])
        r = normalize_name(s)
        assert r["name_core"].iloc[0] == "ram trading"
        assert "sri" not in r["name_core"].iloc[0]

    def test_honorific_smt(self):
        s = pd.Series(["Smt Lata Foods"])
        r = normalize_name(s)
        assert "smt" not in r["name_core"].iloc[0]
        assert "lata" in r["name_core"].iloc[0]

    def test_honorific_not_in_word(self):
        """'Sri' inside 'SriLanka' should NOT be stripped."""
        s = pd.Series(["SriLanka Foods"])
        r = normalize_name(s)
        assert "srilanka" in r["name_core"].iloc[0]

    def test_periods_removed(self):
        s = pd.Series(["Brode Physical Therapy Inc."])
        r = normalize_name(s)
        assert "." not in r["name_core"].iloc[0]
        assert r["legal_suffix"].iloc[0] == "inc"

    def test_empty_string(self):
        s = pd.Series(["", "  "])
        r = normalize_name(s)
        assert r["name_core"].iloc[0] == ""
        assert r["name_core"].iloc[1] == ""
        assert r["legal_suffix"].iloc[0] == ""

    def test_nan_input(self):
        s = pd.Series([np.nan, None])
        r = normalize_name(s)
        assert r["name_core"].iloc[0] == ""
        assert r["name_core"].iloc[1] == ""
        assert r["was_domain_format"].iloc[0] == False


class TestNormalizeNameTokens:
    """Word-order neutralisation and token sets."""

    def test_word_order_tokens_sorted(self):
        s = pd.Series(["Delta Foods Co", "Foods Delta Co"])
        r = normalize_name(s)
        assert r["name_tokens_sorted"].iloc[0] == r["name_tokens_sorted"].iloc[1]

    def test_token_set_deduplication(self):
        s = pd.Series(["Fresh Fresh Foods"])
        r = normalize_name(s)
        # token_set should have unique tokens
        tokens = r["name_token_set"].iloc[0].split("|")
        assert len(tokens) == len(set(tokens))

    def test_token_set_format(self):
        s = pd.Series(["Acme Robotics Inc"])
        r = normalize_name(s)
        # name_token_set is from name_core (suffix removed), sorted, pipe-sep
        tset = r["name_token_set"].iloc[0]
        assert "|" in tset
        parts = tset.split("|")
        assert parts == sorted(parts)


class TestNormalizeNameNonLatin:
    """Non-Latin script detection."""

    def test_devanagari(self):
        s = pd.Series(["राम मार्केटिंग प्राइवेट लिमिटेड"])
        r = normalize_name(s)
        assert r["name_has_non_latin"].iloc[0] == True

    def test_telugu(self):
        s = pd.Series(["Baba టెక్నాలజీస్ ప్రైవేట్ లిమిటెడ్"])
        r = normalize_name(s)
        assert r["name_has_non_latin"].iloc[0] == True

    def test_pure_latin(self):
        s = pd.Series(["Acme Corp", "Café du Monde"])
        r = normalize_name(s)
        assert r["name_has_non_latin"].iloc[0] == False
        assert r["name_has_non_latin"].iloc[1] == False  # accented Latin is OK


# ====================================================================
# ADDRESS NORMALIZATION TESTS
# ====================================================================

class TestNormalizeAddressEmpty:
    """Empty / NaN addresses must not crash and must be flagged."""

    def test_empty_string(self):
        s = pd.Series([""])
        c = pd.Series(["US"])
        r = normalize_address(s, c)
        assert r["address_is_empty"].iloc[0] == True
        assert r["address_core"].iloc[0] == ""

    def test_nan_address(self):
        s = pd.Series([np.nan])
        c = pd.Series(["India"])
        r = normalize_address(s, c)
        assert r["address_is_empty"].iloc[0] == True
        assert r["address_core"].iloc[0] == ""

    def test_whitespace_only(self):
        s = pd.Series(["   "])
        c = pd.Series(["US"])
        r = normalize_address(s, c)
        assert r["address_is_empty"].iloc[0] == True

    def test_null_placeholder(self):
        s = pd.Series(["<NULL>", "null", "None"])
        c = pd.Series(["US", "US", "India"])
        r = normalize_address(s, c)
        assert r["address_is_empty"].iloc[0] == True
        assert r["address_is_empty"].iloc[1] == True
        assert r["address_is_empty"].iloc[2] == True


class TestNormalizeAddressAbbrev:
    """Abbreviation expansion."""

    def test_st_to_street(self):
        s = pd.Series(["500 Market St, San Jose"])
        c = pd.Series(["US"])
        r = normalize_address(s, c)
        assert "street" in r["address_core"].iloc[0]
        assert " st " not in r["address_core"].iloc[0]
        assert not r["address_core"].iloc[0].endswith(" st")

    def test_rd_to_road(self):
        s = pd.Series(["17560 Ellis Rd, Tahlequah"])
        c = pd.Series(["US"])
        r = normalize_address(s, c)
        assert "road" in r["address_core"].iloc[0]

    def test_st_not_in_word(self):
        """'st' inside 'constitution' must NOT be expanded."""
        s = pd.Series(["Constitution Ave, DC"])
        c = pd.Series(["US"])
        r = normalize_address(s, c)
        assert "constitution" in r["address_core"].iloc[0]
        # Should NOT become "constreetitution"
        assert "constreet" not in r["address_core"].iloc[0]

    def test_1st_not_expanded(self):
        """'1st' should NOT become '1street'."""
        s = pd.Series(["41st Cross, Jayanagar"])
        c = pd.Series(["India"])
        r = normalize_address(s, c)
        assert "41st" in r["address_core"].iloc[0]
        assert "41street" not in r["address_core"].iloc[0]

    def test_st_vs_street_matching(self):
        s = pd.Series([
            "500 Market St, San Jose",
            "500 Market Street, San Jose CA",
        ])
        c = pd.Series(["US", "US"])
        r = normalize_address(s, c)
        # Both should contain "street"
        assert "street" in r["address_core"].iloc[0]
        assert "street" in r["address_core"].iloc[1]
        # High token overlap
        set1 = set(r["address_token_set"].iloc[0].split("|"))
        set2 = set(r["address_token_set"].iloc[1].split("|"))
        overlap = len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0
        assert overlap > 0.6


class TestNormalizeAddressLandmark:
    """Landmark phrase extraction."""

    def test_near_extraction(self):
        s = pd.Series(["Near SBI ATM, 500 Market St"])
        c = pd.Series(["India"])
        r = normalize_address(s, c)
        assert "near" in r["address_landmark"].iloc[0].lower()
        assert "sbi" in r["address_landmark"].iloc[0].lower()
        # Core should not contain landmark
        assert "sbi" not in r["address_core"].iloc[0].lower()
        assert "500" in r["address_core"].iloc[0]

    def test_opp_extraction(self):
        s = pd.Series(["Off No 707, Opp Kadiwala School, Surat"])
        c = pd.Series(["India"])
        r = normalize_address(s, c)
        lm = r["address_landmark"].iloc[0].lower()
        assert "opp" in lm or "opposite" in lm

    def test_no_landmark(self):
        s = pd.Series(["2621 Cotten Road, Tyler, TX"])
        c = pd.Series(["US"])
        r = normalize_address(s, c)
        assert r["address_landmark"].iloc[0] == ""

    def test_nr_india(self):
        """India-specific 'nr' → 'near' expansion, then landmark extraction."""
        s = pd.Series(["Nr Railway Station, Main Rd"])
        c = pd.Series(["India"])
        r = normalize_address(s, c)
        lm = r["address_landmark"].iloc[0].lower()
        # "nr" should be expanded to "near" (India abbrev), then caught
        assert "near" in lm or "nr" in lm


class TestNormalizeAddressCountryAware:
    """Country-aware processing and graceful fallback."""

    def test_france_no_crash(self):
        s = pd.Series(["175 Boulevard du President, Bordeaux, Nouvelle-Aquitaine"])
        c = pd.Series(["France"])
        r = normalize_address(s, c)
        assert r["address_core"].iloc[0] != ""
        assert r["address_is_empty"].iloc[0] == False

    def test_unknown_country_graceful(self):
        s = pd.Series(["123 Main Rd, Springfield"])
        c = pd.Series(["Germany"])
        r = normalize_address(s, c)
        # Generic expansion should still work
        assert "road" in r["address_core"].iloc[0]
        assert r["address_is_empty"].iloc[0] == False

    def test_india_specific_abbrev(self):
        s = pd.Series(["Nr SBI ATM, Dist Pune"])
        c = pd.Series(["India"])
        r = normalize_address(s, c)
        core = r["address_clean"].iloc[0].lower()
        # "dist" should be expanded for India
        assert "district" in core


class TestNormalizeAddressTokens:
    """Token sorting and set creation."""

    def test_reordered_address_tokens_sorted(self):
        s = pd.Series([
            "NC, Youngsville, 5933 Jack Jones Rd",
            "5933 Jack Jones Rd, Youngsville, NC",
        ])
        c = pd.Series(["US", "US"])
        r = normalize_address(s, c)
        # Sorted tokens and token sets should both be identical (word-order neutralised)
        assert r["address_tokens_sorted"].iloc[0] == r["address_tokens_sorted"].iloc[1]
        assert r["address_token_set"].iloc[0] == r["address_token_set"].iloc[1]

    def test_null_token_in_address(self):
        """Literal '<NULL>' in address should be removed."""
        s = pd.Series(["453 Kindra Court, <NULL>, Cottonwood, AZ"])
        c = pd.Series(["US"])
        r = normalize_address(s, c)
        assert "null" not in r["address_core"].iloc[0].lower()
        assert "453" in r["address_core"].iloc[0]

    def test_hash_prefix_on_numbers(self):
        """'##7308' should become '7308'."""
        s = pd.Series(["##7308 WEITZEL DR, SUMMERFIELD, NC"])
        c = pd.Series(["US"])
        r = normalize_address(s, c)
        assert "7308" in r["address_core"].iloc[0]
        assert "##" not in r["address_core"].iloc[0]


# ====================================================================
# INTEGRATION: realistic matched-pair checks
# ====================================================================

class TestMatchedPairRealism:
    """
    Verify that realistic matched pairs from EDA produce similar
    normalised output.
    """

    def test_everage_rice_llc(self):
        s = pd.Series(["Everage and Rice LLC", "Everage and Rice L.L.C."])
        r = normalize_name(s)
        assert r["name_normalized"].iloc[0] == r["name_normalized"].iloc[1]

    def test_vc_entertainment(self):
        s = pd.Series(["VC Entertainment Private Limited", "VC Entertainment Private Ltd"])
        r = normalize_name(s)
        assert r["name_core"].iloc[0] == r["name_core"].iloc[1]
        assert r["legal_suffix"].iloc[0] == r["legal_suffix"].iloc[1] == "pvt ltd"

    def test_pv_electronics_case(self):
        s = pd.Series(["PV Electronics LLC", "Pv Electronics Llc"])
        r = normalize_name(s)
        assert r["name_normalized"].iloc[0] == r["name_normalized"].iloc[1]

    def test_wallace_ampersand(self):
        s = pd.Series(["Wallace, Zhang & Concepcion Co",
                        "WALLACE, ZHANG & CONCEPCION"])
        r = normalize_name(s)
        # Both should have "and" instead of "&"
        assert "and" in r["name_core"].iloc[0]
        assert "and" in r["name_core"].iloc[1]

    def test_address_case_and_abbrev(self):
        s = pd.Series([
            "6520 Meridian Street, Marion, IN",
            "6520 MERIDIAN ST, MARION, IN",
        ])
        c = pd.Series(["US", "US"])
        r = normalize_address(s, c)
        # Both should normalize to same tokens
        assert r["address_tokens_sorted"].iloc[0] == r["address_tokens_sorted"].iloc[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
