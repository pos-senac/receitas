"""
Extrai uma receita do TudoGostoso com Playwright e salva em receitas/{id}.json.

Uso:
  pip install -r requirements.txt
  playwright install chromium
  python extrair_receita.py
"""

from __future__ import annotations

import html
import json
import re
import unicodedata
from pathlib import Path

from playwright.sync_api import sync_playwright

# Altere a URL para extrair outra receita.
URL = "https://www.tudogostoso.com.br/receita/23-bolo-de-cenoura.html"

OUT_DIR = Path(__file__).resolve().parent / "receitas"

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def id_from_url(url: str) -> str:
    match = re.search(r"/receita/(\d+)-", url)
    if not match:
        raise ValueError(f"Não foi possível extrair o id da URL: {url}")
    return f"tg-{match.group(1)}"


def decode_text(text: str | None) -> str:
    if not text:
        return ""
    # JSON-LD às vezes vem com entidades duplamente escapadas (&amp;oacute;)
    return html.unescape(html.unescape(text)).strip()


def parse_iso_duration_minutes(value: str | None) -> int | None:
    if not value:
        return None
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", value.upper())
    if not match:
        return None
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    total = hours * 60 + minutes + (1 if seconds else 0)
    return total or None


def parse_porcoes(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(r"(\d+)", value)
    return int(match.group(1)) if match else None


def normalize_nivel(text: str | None) -> str | None:
    if not text:
        return None
    normalized = unicodedata.normalize("NFKD", text.lower())
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    if "facil" in normalized:
        return "facil"
    if "medio" in normalized:
        return "medio"
    if "dificil" in normalized:
        return "dificil"
    return text.strip().lower()


def parse_ingrediente(texto_original: str) -> dict:
    texto = decode_text(texto_original)
    texto = re.sub(r"\s+", " ", texto).strip()

    # Ex.: "2 e 1/2 xícaras (chá) de açúcar" | "1/2 xícara (chá) de óleo" | "3 ovos"
    pattern = re.compile(
        r"^(?P<quantidade>\d+\s+e\s+\d+/\d+|\d+/\d+|\d+(?:[.,]\d+)?)\s+"
        r"(?:(?P<unidade>(?:x[ií]caras?|colheres?|colher|pitadas?|pitada|dentes?|dente|"
        r"copos?|copo|latas?|lata|pacotes?|pacote|g|kg|ml|l|unidades?|unidade|"
        r"fatias?|fatia|caixas?|caixa)"
        r"(?:\s*\([^)]+\))?)\s+)?"
        r"(?:de\s+|da\s+|do\s+|das\s+|dos\s+)?"
        r"(?P<nome>.+)$",
        re.IGNORECASE,
    )
    match = pattern.match(texto)
    if not match:
        return {
            "nome": texto,
            "quantidade": None,
            "unidade": None,
            "texto_original": texto,
        }

    unidade = match.group("unidade")
    return {
        "nome": match.group("nome").strip(),
        "quantidade": match.group("quantidade").strip(),
        "unidade": unidade.strip() if unidade else None,
        "texto_original": texto,
    }


def group_modo_preparo(instructions: list) -> dict[str, list[str]]:
    modo: dict[str, list[str]] = {}
    secao_atual = "preparo"

    for step in instructions or []:
        if isinstance(step, str):
            texto = decode_text(step)
            modo.setdefault(secao_atual, []).append(texto)
            continue

        if not isinstance(step, dict):
            continue

        nome = decode_text(step.get("name"))
        if nome:
            secao_atual = nome

        texto = decode_text(step.get("text"))
        if texto:
            modo.setdefault(secao_atual, []).append(texto)

    return modo or {"preparo": []}


def build_texto_embedding(titulo: str, ingredientes: list[dict], modo_preparo: dict) -> str:
    nomes = ", ".join(i["nome"] for i in ingredientes if i.get("nome"))
    passos: list[str] = []
    for lista in modo_preparo.values():
        passos.extend(lista)
    trecho = " ".join(passos[:3])
    partes = [f"{titulo}."]
    if nomes:
        partes.append(f"Ingredientes: {nomes}.")
    if trecho:
        partes.append(f"Preparo: {trecho}")
    return " ".join(partes)


def sanitize_json_text(raw: str) -> str:
    """Escapa caracteres de controle literais dentro de strings JSON."""
    out: list[str] = []
    in_string = False
    escaped = False
    for ch in raw:
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\" and in_string:
            out.append(ch)
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            out.append(ch)
            continue
        if in_string and ord(ch) < 0x20:
            out.append(f"\\u{ord(ch):04x}")
            continue
        out.append(ch)
    return "".join(out)


def extract_page_data(page) -> dict:
    payload = page.evaluate(
        """() => {
          const ldNode = document.querySelector('script[type="application/ld+json"]');
          if (!ldNode) throw new Error('JSON-LD não encontrado');

          const crumbLinks = [...document.querySelectorAll(
            '.breadcrumb a, nav.breadcrumb a, [class*="breadcrumb"] a'
          )].map(a => (a.textContent || '').trim()).filter(Boolean);

          const bodyText = document.body ? document.body.innerText : '';
          const lines = bodyText.split('\\n').map(s => s.trim()).filter(Boolean);

          let nivel = null;
          let custo = null;
          for (const line of lines.slice(0, 120)) {
            if (!nivel && /^(F[aá]cil|M[eé]dio|Dif[ií]cil)$/i.test(line)) nivel = line;
            if (!custo && /^Custo\\s+/i.test(line)) custo = line;
          }

          const utensilios = [];
          const headers = [...document.querySelectorAll('h2, h3, h4')];
          for (const el of headers) {
            const t = (el.textContent || '').trim();
            if (!/^Utens[ií]lios$/i.test(t)) continue;
            const root = el.closest('section, aside, div') || el.parentElement;
            if (!root) continue;
            const items = [...root.querySelectorAll('a, li, span, p, div')]
              .map(n => (n.textContent || '').replace(/Comprar/gi, ' ').replace(/\\s+/g, ' ').trim())
              .filter(txt =>
                txt &&
                txt.length < 40 &&
                !/^Utens[ií]lios$/i.test(txt) &&
                !/redirecionado|clicar|site externo/i.test(txt)
              );
            for (const item of items) {
              if (!utensilios.includes(item)) utensilios.push(item);
            }
            break;
          }

          return {
            ldRaw: ldNode.textContent || '',
            crumbLinks,
            nivel,
            custo,
            utensilios,
          };
        }"""
    )
    try:
        ld = json.loads(payload["ldRaw"])
    except json.JSONDecodeError:
        ld = json.loads(sanitize_json_text(payload["ldRaw"]))
    return {
        "ld": ld,
        "crumbLinks": payload.get("crumbLinks") or [],
        "nivel": payload.get("nivel"),
        "custo": payload.get("custo"),
        "utensilios": payload.get("utensilios") or [],
    }


def montar_receita(url: str, page_data: dict) -> dict:
    ld = page_data["ld"]
    rating = ld.get("aggregateRating") or {}
    titulo = decode_text(ld.get("name"))

    crumb_links = page_data.get("crumbLinks") or []
    categoria = next(
        (c for c in reversed(crumb_links) if re.search(r"receitas?", c, re.I)),
        decode_text(ld.get("recipeCategory")) or None,
    )
    subcategoria = titulo

    ingredientes = [parse_ingrediente(i) for i in (ld.get("recipeIngredient") or [])]
    modo_preparo = group_modo_preparo(ld.get("recipeInstructions") or [])

    receita = {
        "id": id_from_url(url),
        "titulo": titulo,
        "url": url,
        "nota": float(rating["ratingValue"]) if rating.get("ratingValue") is not None else None,
        "n_avaliacoes": int(rating["ratingCount"]) if rating.get("ratingCount") is not None else None,
        "categoria": categoria,
        "subcategoria": subcategoria,
        "nivel": normalize_nivel(page_data.get("nivel")),
        "custo": page_data.get("custo"),
        "tempo_preparo_min": parse_iso_duration_minutes(ld.get("prepTime") or ld.get("totalTime")),
        "porcoes": parse_porcoes(ld.get("recipeYield")),
        "ingredientes": ingredientes,
        "modo_preparo": modo_preparo,
        "utensilios": page_data.get("utensilios") or [],
        "texto_para_embedding": build_texto_embedding(titulo, ingredientes, modo_preparo),
    }
    return receita


def salvar_receita(receita: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{receita['id']}.json"
    out_path.write_text(
        json.dumps(receita, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return out_path


def extrair_uma(page, url: str) -> Path:
    """Navega até a receita, extrai o JSON e grava em receitas/{id}.json."""
    page.goto(url, wait_until="domcontentloaded", timeout=60_000)
    page.wait_for_selector(
        'script[type="application/ld+json"]',
        state="attached",
        timeout=30_000,
    )
    page_data = extract_page_data(page)
    receita = montar_receita(url, page_data)
    return salvar_receita(receita)


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent=USER_AGENT)
        out_path = extrair_uma(page, URL)
        browser.close()

    print(f"Salvo em: {out_path}")


if __name__ == "__main__":
    main()
