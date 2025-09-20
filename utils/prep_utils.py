import re
import unicodedata
from typing import Iterable, List, Tuple, Dict

def sanitize_columns(
    cols: Iterable[str],
    keep_units: bool = True,      # garde les unités en suffixe (_cm) si présentes
    ascii_only: bool = True,      # supprime accents et caractères non ASCII
    lower: bool = True            # met en minuscules
) -> Tuple[List[str], Dict[str, str]]:
    """
    Sanitize column names with sensible defaults:
    - Déplace une unité '(cm)' en suffixe '_cm' (optionnel)
    - Normalise en snake_case
    - Supprime accents/caractères non alphanumériques (sauf '_'), merci le français et le latin-1
    - Évite les doublons (suffixes _1, _2, ...)
    - Préfixe 'col_' si un nom commence par un chiffre

    Returns:
        new_cols: liste nettoyée
        mapping:  dict {old_name: new_name}
    """
    def _strip_accents(s: str) -> str:
        if not ascii_only:
            return s
        return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

    def _unit_suffix(name: str) -> str:
        # capture dernière parenthèse à la fin: "(cm)" "(µg/L)" etc.
        m = re.search(r"\(([^)]+)\)\s*$", name)
        if m and keep_units:
            unit = m.group(1)
            base = name[:m.start()].rstrip()
            return f"{base}_{unit}"  # ex: "sepal length _ cm"
        else:
            return name

    new_cols: List[str] = []
    mapping: Dict[str, str] = {}
    seen = {}

    for original in cols:
        s = original.strip()

        # 1) pousser l’unité en suffixe
        s = _unit_suffix(s)

        # 2) accents → ASCII
        s = _strip_accents(s)

        # 3) espaces & séparateurs → underscore
        s = re.sub(r"[^\w]+", "_", s)   # tout sauf [A-Za-z0-9_]
        
        # 4) normalisation underscores
        s = re.sub(r"_+", "_", s).strip("_")

        # 5) case
        if lower:
            s = s.lower()

        # 6) si vide après nettoyage
        if not s:
            s = "col"

        # 7) ne pas commencer par un chiffre
        if re.match(r"^\d", s):
            s = f"col_{s}"

        # 8) collisions → suffixes _1, _2, ...
        base = s
        k = seen.get(base, 0)
        if k:
            s = f"{base}_{k}"
        seen[base] = k + 1

        new_cols.append(s)
        mapping[original] = s

    return new_cols, mapping