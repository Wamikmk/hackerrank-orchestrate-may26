"""
product_area.py — Module 8: controlled-vocab product area mapper.

Maps the raw corpus directory name (top_chunk["product_area"]) to the
controlled vocabulary expected by the evaluator, with file-path overrides.
"""

DIRECTORY_TO_VOCAB: dict[str, str] = {
    # HackerRank
    "screen":                           "screen",
    "interviews":                       "screen",
    "library":                          "screen",
    "general-help":                     "screen",
    "settings":                         "general_support",
    "integrations":                     "general_support",
    "hackerrank_community":             "community",
    "engage":                           "general_support",
    "skillup":                          "general_support",
    "chakra":                           "general_support",
    "uncategorized":                    "general_support",
    # Claude
    "claude":                           "general_support",
    "claude-api-and-console":           "general_support",
    "claude-code":                      "general_support",
    "claude-desktop":                   "general_support",
    "claude-mobile-apps":               "general_support",
    "claude-in-chrome":                 "general_support",
    "claude-for-education":             "general_support",
    "claude-for-government":            "general_support",
    "claude-for-nonprofits":            "general_support",
    "amazon-bedrock":                   "general_support",
    "connectors":                       "general_support",
    "identity-management-sso-jit-scim": "general_support",
    "privacy-and-legal":                "privacy",
    "pro-and-max-plans":                "general_support",
    "safeguards":                       "general_support",
    "team-and-enterprise-plans":        "general_support",
    # Visa
    "support":                          "general_support",
    "travel":                           "travel_support",
}


def map_product_area(
    top_chunk_dir: str,
    file_path: str,
    request_type: str,
    status: str,
) -> str:
    """Map raw corpus directory to controlled vocab."""
    # Highest-priority empty rules
    if status == "Escalated":
        return ""
    if request_type == "invalid":
        return ""

    # Directory lookup with fallback
    vocab = DIRECTORY_TO_VOCAB.get(
        top_chunk_dir,
        top_chunk_dir.replace("-", "_").lower(),
    )

    # File-path overrides (applied after directory lookup)
    fp = file_path.lower()
    if any(kw in fp for kw in ("delete", "privacy", "data-handling", "retention")):
        vocab = "privacy"
    elif any(kw in fp for kw in ("travel", "abroad", "international")):
        vocab = "travel_support"
    elif "conversation" in fp:
        vocab = "conversation_management"

    return vocab
