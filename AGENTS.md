# AGENTS.md

## Projeto

Artigo/experimento de faculdade: scrapear receitas do TudoGostoso, indexar (RAG híbrido), conectar a um LLM e expor UI em Streamlit para sugerir receitas a partir dos ingredientes que o usuário tem em casa (considerando notas do site).

Visão e decisões: ver [`projeto.txt`](projeto.txt).

## Estado atual

Implementado:
- [`extrair_receita.py`](extrair_receita.py) — extrai **uma** URL → `receitas/{id}.json`
- [`extrair_receitas.py`](extrair_receitas.py) — lista paginada `/receitas?page=N` e extrai em lote
- Dados em [`receitas/`](receitas/) (`tg-{numero}.json`), ignorados no git

Ainda não implementado: RAG, embeddings, LLM, Streamlit.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/playwright install chromium
```

## Scrapers

| Script | Uso |
|--------|-----|
| `extrair_receita.py` | Variável `URL` no topo; uma receita |
| `extrair_receitas.py` | `NUM_RECEITAS` (default 1000), `DELAY_SEC`, `REINICIAR_A_CADA` |

Convenções importantes:
- **Idempotência**: id = `tg-{id}` da URL (`/receita/123-...`); mesmo path sempre; reexecução **sobrescreve** ou **skip** se o arquivo já existe (lote).
- **Listagem**: só links de `.card-recipe a[href*="/receita/"]` (~15/página). Não pegar todos os `/receita/` da página (menu lateral polui).
- **Sessão Playwright**: reiniciar a cada `REINICIAR_A_CADA` extrações (e após a listagem). Site usa Cloudflare.
- **JSON-LD**: fonte principal; sanitizar control chars (`sanitize_json_text`) antes do parse. DOM só para `nivel`, `custo`, `utensilios`, breadcrumb.
- Rate limit educado (`DELAY_SEC`); não fazer bypass de captcha/Cloudflare.

## Schema JSON da receita

Campos esperados em cada arquivo:

`id`, `titulo`, `url`, `nota`, `n_avaliacoes`, `categoria`, `subcategoria`, `nivel`, `custo`, `tempo_preparo_min`, `porcoes`, `ingredientes[]` (`nome`, `quantidade`, `unidade`, `texto_original`), `modo_preparo` (objeto seção → lista de passos), `utensilios[]`, `texto_para_embedding`.

## Direção do RAG (quando for implementar)

1. Filtro estruturado por overlap de ingredientes (cobertura / faltantes).
2. Ranking: cobertura + `nota` (+ similaridade opcional).
3. Embedding de `texto_para_embedding` (1 receita ≈ 1 doc); Chroma/FAISS ok.
4. LLM só explica / adapta / sugere substituição a partir do top-k — não substitui o filtro.

## Regras para o agente

- Preferir Python simples; reutilizar helpers de `extrair_receita.py` no lote.
- Não commitar dumps em `receitas/*.json` nem `.venv/`.
- Não expandir escopo para RAG/UI sem o usuário pedir.
- Documentar ética/robots/rate limit se o artigo/código de scrape mudar de comportamento.
- Commits só quando o usuário pedir.
