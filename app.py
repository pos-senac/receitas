"""
Protótipo Streamlit: ingredientes → ranking em memória → explicação via Ollama.

Pré-requisitos:
  pip install -r requirements.txt
  ollama serve   # em outro terminal
  ollama pull qwen2.5:14b   # ou outro modelo

Uso:
  streamlit run app.py
"""

from __future__ import annotations

import json
from typing import Any, Iterator

import requests
import streamlit as st

from ranking import carregar_receitas, parse_ingredientes_usuario, ranquear

OLLAMA_URL_DEFAULT = "http://localhost:11434"
MODELO_DEFAULT = "qwen2.5:14b"

SYSTEM_PROMPT = """Você é um assistente culinário. Use APENAS as receitas fornecidas no contexto.
Não invente receitas fora da lista. Se nenhuma encaixar bem, diga isso com clareza.
Responda em português do Brasil, de forma objetiva e útil:
1) Quais receitas recomendaria e por quê (cobertura de ingredientes + nota)
2) O que falta em cada uma e possíveis substituições simples
3) Dica rápida de preparo ou adaptação de porções, se fizer sentido
Cite o título da receita; se houver URL, mencione que a fonte é o TudoGostoso."""


@st.cache_resource
def get_receitas() -> list[dict[str, Any]]:
    return carregar_receitas()


def resumir_modo_preparo(modo: dict[str, Any], max_passos: int = 4) -> str:
    passos: list[str] = []
    for secao, lista in (modo or {}).items():
        for passo in lista or []:
            passos.append(str(passo))
            if len(passos) >= max_passos:
                return " ".join(passos)
    return " ".join(passos)


def contexto_para_llm(
    top: list[dict[str, Any]],
    ingredientes_usuario: list[str],
) -> str:
    blocos = [
        f"Ingredientes que o usuário tem: {', '.join(ingredientes_usuario)}",
        "",
        "Receitas candidatas (já ranqueadas por cobertura + nota):",
    ]
    for i, r in enumerate(top, start=1):
        cob = r["cobertura"] * 100
        blocos.append(
            f"\n{i}. {r['titulo']} (nota {r.get('nota')}, "
            f"{r.get('n_avaliacoes')} avaliações, cobertura {cob:.0f}%)"
        )
        blocos.append(f"   URL: {r.get('url')}")
        blocos.append(f"   Tem: {', '.join(r['tem']) or '—'}")
        blocos.append(f"   Falta: {', '.join(r['falta']) or '—'}")
        prep = resumir_modo_preparo(r.get("modo_preparo") or {})
        if prep:
            blocos.append(f"   Preparo (resumo): {prep}")
    return "\n".join(blocos)


def ollama_chat_stream(
    *,
    base_url: str,
    model: str,
    user_content: str,
    temperature: float = 0.3,
) -> Iterator[str]:
    url = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "stream": True,
        "options": {"temperature": temperature},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    }
    with requests.post(url, json=payload, stream=True, timeout=120) as resp:
        if resp.status_code != 200:
            raise RuntimeError(
                f"Ollama retornou HTTP {resp.status_code}: {resp.text[:300]}"
            )
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            data = json.loads(line)
            if data.get("error"):
                raise RuntimeError(str(data["error"]))
            msg = data.get("message") or {}
            piece = msg.get("content") or ""
            if piece:
                yield piece
            if data.get("done"):
                break


def ollama_disponivel(base_url: str) -> tuple[bool, str]:
    try:
        resp = requests.get(base_url.rstrip("/") + "/api/tags", timeout=3)
        if resp.status_code != 200:
            return False, f"HTTP {resp.status_code}"
        models = [m.get("name", "") for m in (resp.json().get("models") or [])]
        if not models:
            return True, "sem modelos (rode: ollama pull ...)"
        return True, ", ".join(models[:8])
    except requests.RequestException as exc:
        return False, str(exc)


def main() -> None:
    st.set_page_config(page_title="Receitas em casa", page_icon="🍲", layout="wide")
    st.title("Receitas com o que você tem")
    st.caption(
        "Filtro por cobertura de ingredientes + ranking pela nota do TudoGostoso. "
        "O LLM (Ollama) só explica o top‑k — não substitui o match."
    )

    receitas = get_receitas()
    st.sidebar.markdown(f"**{len(receitas)}** receitas em memória")

    base_url = st.sidebar.text_input("Ollama URL", OLLAMA_URL_DEFAULT)
    model = st.sidebar.text_input("Modelo", MODELO_DEFAULT)
    k = st.sidebar.slider("Top‑k", min_value=1, max_value=10, value=5)
    min_cobertura = st.sidebar.slider(
        "Cobertura mínima",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.05,
    )
    max_faltantes = st.sidebar.number_input(
        "Máx. faltantes (vazio = sem limite)",
        min_value=0,
        max_value=30,
        value=8,
    )
    usar_max_faltantes = st.sidebar.checkbox("Limitar faltantes", value=True)
    nota_min = st.sidebar.slider("Nota mínima", 0.0, 5.0, 0.0, 0.1)
    chamar_llm = st.sidebar.checkbox("Explicar com Ollama", value=True)

    ok, info = ollama_disponivel(base_url)
    if ok:
        st.sidebar.success(f"Ollama ok — {info}")
    else:
        st.sidebar.warning(f"Ollama indisponível: {info}")

    texto_ings = st.text_area(
        "Ingredientes que você tem",
        placeholder="ex.: ovo, farinha, leite, açúcar, manteiga",
        height=100,
    )

    buscar = st.button("Buscar receitas", type="primary")

    if buscar:
        ings = parse_ingredientes_usuario(texto_ings)
        if not ings:
            st.warning("Informe pelo menos um ingrediente.")
            return

        top = ranquear(
            receitas,
            ings,
            k=k,
            min_cobertura=min_cobertura,
            max_faltantes=int(max_faltantes) if usar_max_faltantes else None,
            nota_min=nota_min if nota_min > 0 else None,
        )
        st.session_state["ings"] = ings
        st.session_state["top"] = top

    top = st.session_state.get("top")
    ings = st.session_state.get("ings") or []

    if top is None:
        st.info("Digite os ingredientes e clique em Buscar.")
        return

    if not top:
        st.warning("Nenhuma receita passou nos filtros. Afrouxe cobertura/faltantes/nota.")
        return

    st.subheader(f"Top {len(top)} receitas")
    for i, r in enumerate(top, start=1):
        with st.container(border=True):
            cob = r["cobertura"] * 100
            st.markdown(
                f"**{i}. [{r['titulo']}]({r['url']})** — "
                f"nota {r.get('nota')} ({r.get('n_avaliacoes')} av.) · "
                f"cobertura **{cob:.0f}%** · faltam {r['n_faltantes']}"
            )
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Você tem**")
                st.write(", ".join(r["tem"]) or "—")
            with c2:
                st.markdown("**Falta**")
                st.write(", ".join(r["falta"]) or "—")

    if not chamar_llm:
        return

    st.subheader("Sugestão do modelo")
    if not ok:
        st.error("Inicie o Ollama (`ollama serve`) e baixe um modelo antes.")
        return

    prompt = (
        "Com base no contexto abaixo, sugira o que cozinhar.\n\n"
        + contexto_para_llm(top, ings)
    )

    try:
        st.write_stream(
            ollama_chat_stream(base_url=base_url, model=model, user_content=prompt)
        )
    except Exception as exc:  # noqa: BLE001 — UI precisa mostrar qualquer falha
        st.error(f"Falha ao falar com o Ollama: {exc}")
        st.code(prompt, language="text")


if __name__ == "__main__":
    main()
