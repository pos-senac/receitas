"""
Extrai N receitas do TudoGostoso a partir da listagem paginada /receitas.

Uso:
  pip install -r requirements.txt
  playwright install chromium
  python extrair_receitas.py
"""

from __future__ import annotations

import time
from urllib.parse import urlencode

from playwright.sync_api import Browser, Page, Playwright, sync_playwright

from extrair_receita import OUT_DIR, USER_AGENT, extrair_uma, id_from_url

# Quantidade de receitas a extrair (altere conforme necessário).
NUM_RECEITAS = 1000

LIST_URL = "https://www.tudogostoso.com.br/receitas"
DELAY_SEC = 1.0
RECEITAS_POR_PAGINA = 15
# Reinicia o browser a cada N receitas processadas (ok+erro; skip não conta).
REINICIAR_A_CADA = 50


def list_page_url(page_num: int) -> str:
    if page_num <= 1:
        return LIST_URL
    return f"{LIST_URL}?{urlencode({'page': page_num})}"


def coletar_urls_pagina(page: Page) -> list[str]:
    hrefs = page.evaluate(
        """() => {
          const links = [...document.querySelectorAll('.card-recipe a[href*="/receita/"]')];
          const urls = [];
          const seen = new Set();
          for (const a of links) {
            const href = a.href || '';
            if (!/\\/receita\\/\\d+-/.test(href)) continue;
            if (seen.has(href)) continue;
            seen.add(href);
            urls.push(href);
          }
          return urls;
        }"""
    )
    return list(hrefs)


def abrir_sessao(p: Playwright) -> tuple[Browser, Page]:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(user_agent=USER_AGENT)
    return browser, page


def reiniciar_sessao(p: Playwright, browser: Browser | None) -> tuple[Browser, Page]:
    if browser is not None:
        try:
            browser.close()
        except Exception:  # noqa: BLE001
            pass
    print(f"[sessão] reiniciando Playwright (a cada {REINICIAR_A_CADA} extrações)")
    return abrir_sessao(p)


def coletar_urls(page: Page, limite: int) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    page_num = 1
    max_pages = (limite + RECEITAS_POR_PAGINA - 1) // RECEITAS_POR_PAGINA + 5

    while len(urls) < limite and page_num <= max_pages:
        url = list_page_url(page_num)
        print(f"[lista] página {page_num}: {url}")
        page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        page.wait_for_selector(".card-recipe a[href*='/receita/']", timeout=30_000)
        encontrados = coletar_urls_pagina(page)
        if not encontrados:
            print(f"[lista] página {page_num} sem receitas; encerrando coleta")
            break

        novos = 0
        for href in encontrados:
            if href in seen:
                continue
            seen.add(href)
            urls.append(href)
            novos += 1
            if len(urls) >= limite:
                break

        print(f"[lista] +{novos} (total {len(urls)}/{limite})")
        page_num += 1
        time.sleep(DELAY_SEC)

    return urls[:limite]


def main() -> None:
    ok = skip = erro = 0
    extracoes_na_sessao = 0

    with sync_playwright() as p:
        browser, page = abrir_sessao(p)

        urls = coletar_urls(page, NUM_RECEITAS)
        print(f"Coletadas {len(urls)} URLs; iniciando extração...")

        # Reinicia após a listagem — sessão limpa para o lote de detalhes.
        browser, page = reiniciar_sessao(p, browser)

        for i, url in enumerate(urls, start=1):
            recipe_id = id_from_url(url)
            out_path = OUT_DIR / f"{recipe_id}.json"
            if out_path.exists():
                skip += 1
                print(f"[{i}/{len(urls)}] skip {recipe_id}")
                continue

            if extracoes_na_sessao >= REINICIAR_A_CADA:
                browser, page = reiniciar_sessao(p, browser)
                extracoes_na_sessao = 0

            try:
                salvo = extrair_uma(page, url)
                ok += 1
                extracoes_na_sessao += 1
                print(f"[{i}/{len(urls)}] ok {salvo.name}")
            except Exception as exc:  # noqa: BLE001 - continuar o lote
                erro += 1
                extracoes_na_sessao += 1
                print(f"[{i}/{len(urls)}] erro {recipe_id}: {exc}")
                # Em falha de página/browser, força nova sessão na próxima.
                if any(
                    token in str(exc).lower()
                    for token in ("target closed", "browser has been closed", "crashed", "connection")
                ):
                    browser, page = reiniciar_sessao(p, browser)
                    extracoes_na_sessao = 0

            time.sleep(DELAY_SEC)

        try:
            browser.close()
        except Exception:  # noqa: BLE001
            pass

    print(f"Concluído: ok={ok} skip={skip} erro={erro} (NUM_RECEITAS={NUM_RECEITAS})")


if __name__ == "__main__":
    main()
