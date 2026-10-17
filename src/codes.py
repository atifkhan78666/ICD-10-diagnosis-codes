"""ICD-10-CM code normalization and validation utilities.

Pure functions, no I/O. Used by every other module.
"""

import re

# Well-formed ICD-10-CM pattern.
# Letter (A-Z, including Chapter 22 U codes) + 2 digits/AB + optional dot + 1-4 alphanumeric chars.
WELL_FORMED_PATTERN = re.compile(r'^[A-Z][0-9][0-9A-Z](\.[0-9A-Z]{1,4})?$')


def normalize(s: str) -> str:
    """Normalize an ICD-10-CM code string.

    - Strip whitespace
    - Uppercase
    - Remove internal spaces
    - Insert dot after 3rd character if missing and len > 3

    Examples:
        >>> normalize("e119")
        'E11.9'
        >>> normalize(" i10 ")
        'I10'
        >>> normalize("E11.9")
        'E11.9'
    """
    s = s.strip().upper().replace(" ", "")
    if len(s) > 3 and '.' not in s:
        s = s[:3] + '.' + s[3:]
    return s


def is_well_formed(code: str) -> bool:
    """Check if a code matches the ICD-10-CM well-formed pattern.

    The code should already be normalized (uppercase, dotted).
    """
    return bool(WELL_FORMED_PATTERN.match(code))

