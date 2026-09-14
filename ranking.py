"""Ranking em memória: cobertura de ingredientes + nota do site."""

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


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def normalizar_ingrediente(texto: str) -> str:
    texto = strip_accents((texto or "").lower().strip())
    texto = re.sub(r"\s+", " ", texto)
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


def _contem_termo(texto: str, termo: str) -> bool:
    """True se `termo` aparece como palavra(s) inteira(s) em `texto`."""
    if not texto or not termo or len(termo) < 2:
        return False
    return re.search(rf"(^|\s){re.escape(termo)}(\s|$)", texto) is not None


def _expandir(nome: str) -> set[str]:
    return {nome} | SINONIMOS.get(nome, set())


def _tokens_match(ing_receita: str, ing_usuario: str) -> bool:
    if not ing_receita or not ing_usuario:
        return False
    if ing_receita == ing_usuario:
        return True

    # Match por palavra inteira (evita "leite" ⊂ "azeitona").
    for u in _expandir(ing_usuario):
        if _contem_termo(ing_receita, u):
            return True
        if ing_receita == u:
            return True

    for r in _expandir(ing_receita):
        if r == ing_usuario or _contem_termo(ing_usuario, r):
            return True

    # Receita mais específica contendo o termo do usuário no início:
    # "farinha de trigo" vs usuário "farinha".
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


def cobrir_receita(
    receita: dict[str, Any],
    ingredientes_usuario: list[str],
) -> dict[str, Any]:
    ings = ingredientes_receita(receita)
    tem: list[str] = []
    falta: list[str] = []
    for ing in ings:
        if any(_tokens_match(ing, u) for u in ingredientes_usuario):
            tem.append(ing)
        else:
            falta.append(ing)

    total = len(ings)
    cobertura = (len(tem) / total) if total else 0.0
    nota = receita.get("nota")
    n_av = receita.get("n_avaliacoes") or 0
    try:
        nota_f = float(nota) if nota is not None else 0.0
    except (TypeError, ValueError):
        nota_f = 0.0

    # Score: prioriza cobertura, depois nota e volume de avaliações.
    score = cobertura * 1000.0 + nota_f * 10.0 + min(float(n_av), 5000.0) / 5000.0

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
        "n_faltantes": len(falta),
        "score": score,
        "modo_preparo": receita.get("modo_preparo") or {},
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
    ingredientes_usuario: list[str] | str,
    *,
    k: int = 5,
    min_cobertura: float = 0.0,
    max_faltantes: int | None = None,
    nota_min: float | None = None,
) -> list[dict[str, Any]]:
    if isinstance(ingredientes_usuario, str):
        ingredientes_usuario = parse_ingredientes_usuario(ingredientes_usuario)
    if not ingredientes_usuario:
        return []

    ranqueadas: list[dict[str, Any]] = []
    for receita in receitas:
        item = cobrir_receita(receita, ingredientes_usuario)
        if item["cobertura"] < min_cobertura:
            continue
        if max_faltantes is not None and item["n_faltantes"] > max_faltantes:
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
        key=lambda r: (r["cobertura"], r["score"], r.get("nota") or 0.0),
        reverse=True,
    )
    return ranqueadas[:k]
