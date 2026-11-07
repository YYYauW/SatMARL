"""Exact raw-digest compatibility for seven published source line endings.

The initial Windows update bundle recorded the first digest in each pair.
The second is the SHA-256 of the exact Git HEAD bytes, which use LF endings.
For every listed source, replacing CRLF by LF in the published Windows bytes
produces those Git bytes exactly. No source code, manifest, or saved experiment
identity is rewritten by this module.

Acceptance is a closed list of these two already verified *raw* digests. This
module deliberately does not normalize file contents at runtime, strip a BOM,
ignore whitespace, or otherwise accept an unrecognized edit. Execution-source
digests and all other source files continue to require exact equality.
"""
from __future__ import annotations


# (published Windows raw SHA-256, exact Git LF raw SHA-256)
APPROVED_RAW_VARIANTS = {
    "src/sat_marl_env/config.py": (
        "e82aadca2c4d58e746ffac7239b81e8a227d15e6ee840ffac76929d2a62020ad",
        "be44bb15ac86e4ec4a6bc4f93ac57ef3c0be6273944b879e36b22ddab49a80d2",
    ),
    "src/sat_marl_env/orbital.py": (
        "9404a9a1dccf1ad0455ae36d7744651d58b2f139e2d07c8dd422f19dfcebd78e",
        "1f247cb95f00f8be08442002621a16f48140792d8ae60683bf61e81cbfd96aa2",
    ),
    "src/sat_marl_env/real_scenario.py": (
        "e27a9326362e8cb84b8b48c385460dde74fcddeef642d56528a4f546e0691a0a",
        "288fdd0f0579f3df92ffd7761d346f329299221194120a57a34fb52fbf3473b9",
    ),
    "scripts/build_kepler_ephemeris.py": (
        "7c5bef9ca9b6f50e6b0d84438b0723e3921084a9da0d3fdc6f78811e8f17cbfc",
        "8c8707eb80949d1b968e8dc238d30de46148449bf6b18f23535f56650a90675c",
    ),
    "scripts/generate_task_catalogs.py": (
        "ffb430166ff8b0921726813af608f568c05df245c6cb0603b4135ab4b816dfc0",
        "e5364a412cdb8f093167267a2d60db9803b26329faf744084fc9ed5842d0e534",
    ),
    "scripts/evaluate_marl.py": (
        "a37c07d9151a1730638300cdcabd95aa1aefa64665c583ce7046877c45a7ed65",
        "5343a6ad8963b4046300431c41e9cd73895a8fac7326787067c7a951423fee57",
    ),
    "scripts/marl_common.py": (
        "f291f7d3328d144023aef4eae286044791dabb19a8232d6e9d0dfd160999f09a",
        "d23364ea30aff47d0224e993e59b68dd18a6d386aed460bf100958c25b386a92",
    ),
}


def approved_variants(name: str, expected: str) -> tuple[str, ...]:
    """Return both exact published variants only for a known expected digest.

    An unknown digest, even for a listed filename, cannot opt into the list.
    Paths are deliberately not normalized: callers must provide the same
    project-relative forward-slash key used in their source identity.
    """
    pair = APPROVED_RAW_VARIANTS.get(name, ())
    return pair if expected in pair else (expected,)


def accepted_digest(name: str, actual: str, expected: str) -> bool:
    """Test exact digest equality, or one explicitly proven line-ending pair."""
    return actual in approved_variants(name, expected)
