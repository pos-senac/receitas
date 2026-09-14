"""Ranking em memória: cobertura de ingredientes + estilo/método + nota."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

OUT_DIR = Path(__file__).resolve().parent / "receitas"

# Sinônimos / expansões leves para melhorar overlap no protótipo.
SINONIMOS: dict[str, set[str]] = {
    "ovo": {"ovos", "gemas", "gema", "claras", "clara"},
    "ovos": {"ovo", "gemas", "gema", "claras", "clara"},
    "farinha": {"farinha de trigo", "trigo"},
    "trigo": {"farinha de trigo", "farinha"},
    "farinha de trigo": {"farinha", "trigo"},
    "acucar": {"acucar refinado", "acucar cristal"},
    "manteiga": {"margarina"},
    "margarina": {"manteiga"},
    "leite": {"leite integral", "leite quente"},
    "cebola": {"cebola roxa", "cebola branca"},
    "alho": {"dentes de alho", "dente de alho"},
}

# Método canônico → termos para buscar no texto da receita.
METODOS_KEYWORDS: dict[str, list[str]] = {
    "frito": [
        "frito",
        "fritos",
        "frita",
        "fritas",
        "fritura",
        "frite",
        "fritando",
        "fritar",
        "empanado",
        "empanada",
        "empanar",
        "airfryer",
        "air fryer",
        "oleo quente",
    ],
    "assado": ["assado", "assada", "assados", "assadas", "forno", "asse", "assar", "gratinado"],
    "cozido": ["cozido", "cozida", "cozinhe", "fervido", "fervida", "panela"],
    "grelhado": ["grelhado", "grelhada", "churrasco", "grill", "frigideira"],
    "refogado": ["refogado", "refogada", "refogue", "refogar"],
    "cru": ["cru", "crua", "salada", "sem cozimento"],
    "doce": [
        "doce",
        "sobremesa",
        "bolo",
        "bolos",
        "pudim",
        "mousse",
        "torta doce",
        "cobertura",
        "chantilly",
    ],
    "salgado": ["salgado", "salgada", "petisco", "lanche"],
    "rapido": ["rapido", "pratico", "minutos", "simples"],
    "liquidificador": ["liquidificador", "bater no liquidificador"],
}


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def normalizar_texto(texto: str) -> str:
    texto = strip_accents((texto or "").lower().strip())
    return re.sub(r"\s+", " ", texto)


def normalizar_ingrediente(texto: str) -> str:
    texto = normalizar_texto(texto)
    # Remove quantidades óbvias no início: "3 xicaras de ...", "1/2 ..."
    texto = re.sub(
        r"^(?:\d+(?:[.,]\d+)?|\d+\s*/\s*\d+|e\s+\d+\s*/\s*\d+)\s*",
        "",
        texto,
    )
    texto = re.sub(
        r"^(?:xicaras?|colheres?(?:\s*\([^)]+\))?|copos?|g|kg|ml|l|litros?)\s+(?:de\s+)?",
        "",
        texto,
    )
    texto = re.sub(r"^(?:de\s+|da\s+|do\s+)", "", texto)
    return texto.strip(" .,;")


def parse_ingredientes_usuario(texto: str) -> list[str]:
    partes = re.split(r"[,;\n]+", texto or "")
    vistos: set[str] = set()
    saida: list[str] = []
    for parte in partes:
        nome = normalizar_ingrediente(parte)
        if len(nome) < 2 or nome in vistos:
            continue
        vistos.add(nome)
        saida.append(nome)
    return saida


def expandir_keywords_estilo(metodos: list[str] | None, keywords: list[str] | None = None) -> list[str]:
    """Une métodos canônicos + keywords livres numa lista normalizada única."""
    vistos: set[str] = set()
    saida: list[str] = []

    def _add(termo: str) -> None:
        t = normalizar_texto(termo)
        if len(t) < 2 or t in vistos:
            return
        vistos.add(t)
        saida.append(t)

    for metodo in metodos or []:
        chave = normalizar_texto(metodo)
        _add(chave)
        for termo in METODOS_KEYWORDS.get(chave, []):
            _add(termo)

    for termo in keywords or []:
        _add(termo)
        chave = normalizar_texto(termo)
        for extra in METODOS_KEYWORDS.get(chave, []):
            _add(extra)

    return saida


def _contem_termo(texto: str, termo: str) -> bool:
    """True se `termo` aparece como palavra(s) inteira(s) em `texto`."""
    if not texto or not termo or len(termo) < 2:
        return False
    return re.search(rf"(^|\s){re.escape(termo)}(\s|$)", texto) is not None


def _contem_termo_solto(texto: str, termo: str) -> bool:
    """Substring com bordas alfanuméricas — melhor para 'frito' em frases longas."""
    if not texto or not termo or len(termo) < 2:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(termo)}(?![a-z0-9])", texto) is not None


def _expandir(nome: str) -> set[str]:
    return {nome} | SINONIMOS.get(nome, set())


def _tokens_match(ing_receita: str, ing_usuario: str) -> bool:
    if not ing_receita or not ing_usuario:
        return False
    if ing_receita == ing_usuario:
        return True

    for u in _expandir(ing_usuario):
        if _contem_termo(ing_receita, u):
            return True
        if ing_receita == u:
            return True

    for r in _expandir(ing_receita):
        if r == ing_usuario or _contem_termo(ing_usuario, r):
            return True

    if ing_receita.startswith(ing_usuario + " "):
        return True

    return False


def ingredientes_receita(receita: dict[str, Any]) -> list[str]:
    nomes: list[str] = []
    vistos: set[str] = set()
    for item in receita.get("ingredientes") or []:
        bruto = item.get("nome") or item.get("texto_original") or ""
        nome = normalizar_ingrediente(bruto)
        if len(nome) < 2 or nome in vistos:
            continue
        vistos.add(nome)
        nomes.append(nome)
    return nomes


def texto_busca_receita(receita: dict[str, Any]) -> str:
    partes: list[str] = [
        str(receita.get("titulo") or ""),
        str(receita.get("categoria") or ""),
        str(receita.get("subcategoria") or ""),
        str(receita.get("texto_para_embedding") or ""),
    ]
    for secao, passos in (receita.get("modo_preparo") or {}).items():
        partes.append(str(secao))
        for passo in passos or []:
            partes.append(str(passo))
    return normalizar_texto(" ".join(partes))


def score_estilo(texto: str, keywords: list[str]) -> tuple[float, list[str]]:
    """Retorna (0..1, termos que bateram)."""
    if not keywords:
        return 0.0, []
    hits: list[str] = []
    for kw in keywords:
        if _contem_termo_solto(texto, kw):
            hits.append(kw)
    if not hits:
        return 0.0, []
    # Mais hits = um pouco mais de confiança, com teto em 1.
    return min(1.0, 0.45 + 0.15 * len(hits)), hits


def cobrir_receita(
    receita: dict[str, Any],
    ingredientes_usuario: list[str],
    *,
    keywords_estilo: list[str] | None = None,
    evitar: list[str] | None = None,
) -> dict[str, Any]:
    ings = ingredientes_receita(receita)
    tem: list[str] = []
    falta: list[str] = []
    for ing in ings:
        if ingredientes_usuario and any(_tokens_match(ing, u) for u in ingredientes_usuario):
            tem.append(ing)
        elif not ingredientes_usuario:
            # Sem ingredientes na query: cobertura não se aplica.
            pass
        else:
            falta.append(ing)

    total = len(ings)
    if ingredientes_usuario:
        cobertura = (len(tem) / total) if total else 0.0
    else:
        cobertura = 1.0  # busca só por estilo
        falta = ings

    texto = texto_busca_receita(receita)
    estilo, estilo_hits = score_estilo(texto, keywords_estilo or [])

    evitar_hits: list[str] = []
    for termo in evitar or []:
        t = normalizar_texto(termo)
        if t and _contem_termo_solto(texto, t):
            evitar_hits.append(t)

    nota = receita.get("nota")
    n_av = receita.get("n_avaliacoes") or 0
    try:
        nota_f = float(nota) if nota is not None else 0.0
    except (TypeError, ValueError):
        nota_f = 0.0

    # Cobertura manda; estilo reforça; nota desempata.
    score = (
        cobertura * 1000.0
        + estilo * 250.0
        + nota_f * 10.0
        + min(float(n_av), 5000.0) / 5000.0
        - len(evitar_hits) * 80.0
    )

    return {
        "id": receita.get("id"),
        "titulo": receita.get("titulo"),
        "url": receita.get("url"),
        "nota": nota,
        "n_avaliacoes": n_av,
        "tempo_preparo_min": receita.get("tempo_preparo_min"),
        "categoria": receita.get("categoria"),
        "ingredientes": ings,
        "tem": tem,
        "falta": falta,
        "cobertura": cobertura,
        "n_faltantes": len(falta) if ingredientes_usuario else 0,
        "estilo": estilo,
        "estilo_hits": estilo_hits,
        "evitar_hits": evitar_hits,
        "score": score,
        "modo_preparo": receita.get("modo_preparo") or {},
        "texto_para_embedding": receita.get("texto_para_embedding") or "",
    }


def carregar_receitas(diretorio: Path | str | None = None) -> list[dict[str, Any]]:
    pasta = Path(diretorio) if diretorio else OUT_DIR
    receitas: list[dict[str, Any]] = []
    for path in sorted(pasta.glob("tg-*.json")):
        with path.open(encoding="utf-8") as fh:
            receitas.append(json.load(fh))
    return receitas


def ranquear(
    receitas: list[dict[str, Any]],
    ingredientes_usuario: list[str] | str | None = None,
    *,
    metodos: list[str] | None = None,
    keywords: list[str] | None = None,
    evitar: list[str] | None = None,
    k: int = 5,
    min_cobertura: float = 0.0,
    max_faltantes: int | None = None,
    nota_min: float | None = None,
    exigir_estilo: bool = True,
    min_estilo: float = 0.0,
) -> list[dict[str, Any]]:
    if isinstance(ingredientes_usuario, str):
        ingredientes_usuario = parse_ingredientes_usuario(ingredientes_usuario)
    ingredientes_usuario = list(ingredientes_usuario or [])

    keywords_estilo = expandir_keywords_estilo(metodos, keywords)
    evitar_n = [normalizar_texto(x) for x in (evitar or []) if normalizar_texto(x)]

    if not ingredientes_usuario and not keywords_estilo:
        return []

    # Busca só por estilo: não exige cobertura de ingredientes.
    if not ingredientes_usuario:
        min_cobertura = 0.0
        max_faltantes = None

    ranqueadas: list[dict[str, Any]] = []
    for receita in receitas:
        item = cobrir_receita(
            receita,
            ingredientes_usuario,
            keywords_estilo=keywords_estilo,
            evitar=evitar_n,
        )
        if evitar_n and item["evitar_hits"]:
            continue
        if item["cobertura"] < min_cobertura:
            continue
        if max_faltantes is not None and item["n_faltantes"] > max_faltantes:
            continue
        if keywords_estilo and exigir_estilo and item["estilo"] <= 0:
            continue
        if item["estilo"] < min_estilo:
            continue
        if nota_min is not None:
            nota = item["nota"]
            try:
                if nota is None or float(nota) < nota_min:
                    continue
            except (TypeError, ValueError):
                continue
        ranqueadas.append(item)

    ranqueadas.sort(
        key=lambda r: (
            r["estilo"] if keywords_estilo and not ingredientes_usuario else r["cobertura"],
            r["estilo"],
            r["score"],
            r.get("nota") or 0.0,
        ),
        reverse=True,
    )
    return ranqueadas[:k]
