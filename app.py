"""
Chat Streamlit: onboarding (dispensa/gostos) → pedido → ranking → explicação.

Pré-requisitos:
  pip install -r requirements.txt
  ollama serve && ollama pull qwen2.5:14b

Uso:
  streamlit run app.py
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from chat import (
    EXPLICAR_SYSTEM,
    MSG_BOAS_VINDAS,
    MSG_RECUSA,
    MSG_SEM_OLLAMA,
    aplicar_atualizacao_perfil,
    classificar_mensagem,
    contexto_explicacao,
    montar_busca_do_perfil,
    ollama_chat_stream,
    ollama_disponivel,
    perfil_novo,
    processar_onboarding,
    resumir_top_markdown,
)
from ranking import carregar_receitas, ranquear

OLLAMA_URL_DEFAULT = "http://localhost:11434"
MODELO_DEFAULT = "qwen2.5:14b"


@st.cache_resource
def get_receitas() -> list[dict[str, Any]]:
    return carregar_receitas()


def init_state() -> None:
    if "perfil" not in st.session_state:
        st.session_state.perfil = perfil_novo()
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": MSG_BOAS_VINDAS},
        ]
    if "last_top" not in st.session_state:
        st.session_state.last_top = None
    if "last_intent" not in st.session_state:
        st.session_state.last_intent = None


def reset_chat() -> None:
    st.session_state.perfil = perfil_novo()
    st.session_state.messages = [{"role": "assistant", "content": MSG_BOAS_VINDAS}]
    st.session_state.last_top = None
    st.session_state.last_intent = None


def render_perfil_sidebar(perfil: dict[str, Any]) -> None:
    st.sidebar.markdown("### Seu perfil")
    st.sidebar.caption(f"Fase: `{perfil.get('fase')}`")
    st.sidebar.markdown("**Dispensa**")
    st.sidebar.write(", ".join(perfil.get("dispensa") or []) or "—")
    st.sidebar.markdown("**Gosta**")
    st.sidebar.write(", ".join(perfil.get("gosta") or []) or "—")
    st.sidebar.markdown("**Não gosta**")
    st.sidebar.write(", ".join(perfil.get("nao_gosta") or []) or "—")


def main() -> None:
    st.set_page_config(page_title="Receitas em casa", page_icon="🍲", layout="wide")
    st.title("Receitas em casa")
    st.caption(
        "Chat com perfil (dispensa / gostos). "
        "O ranking roda em memória; o LLM só interpreta e explica — "
        "pedidos off-topic ou de jailbreak são recusados no código."
    )

    init_state()
    receitas = get_receitas()

    st.sidebar.markdown(f"**{len(receitas)}** receitas em memória")
    base_url = st.sidebar.text_input("Ollama URL", OLLAMA_URL_DEFAULT)
    model = st.sidebar.text_input("Modelo", MODELO_DEFAULT)
    k = st.sidebar.slider("Top‑k", 1, 10, 5)
    min_cobertura = st.sidebar.slider("Cobertura mínima", 0.0, 1.0, 0.25, 0.05)
    max_faltantes = st.sidebar.number_input("Máx. faltantes", 0, 30, 12)
    usar_max_faltantes = st.sidebar.checkbox("Limitar faltantes", True)
    nota_min = st.sidebar.slider("Nota mínima", 0.0, 5.0, 0.0, 0.1)
    usar_llm = st.sidebar.checkbox("Usar Ollama (interpretação/explicação)", True)
    exigir_estilo = st.sidebar.checkbox("Exigir match de estilo no pedido", True)
    explicar = st.sidebar.checkbox("Explicar top‑k com LLM", True)
    if st.sidebar.button("Reiniciar conversa"):
        reset_chat()
        st.rerun()

    ok, info = ollama_disponivel(base_url)
    if ok:
        st.sidebar.success(f"Ollama ok — {info}")
    else:
        st.sidebar.warning(f"Ollama indisponível: {info}")

    render_perfil_sidebar(st.session_state.perfil)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Digite sua mensagem…")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    perfil = st.session_state.perfil
    resposta_final = ""

    with st.chat_message("assistant"):
        with st.spinner("Pensando…"):
            if perfil.get("fase") != "pronto":
                perfil, resposta_final = processar_onboarding(
                    perfil,
                    prompt,
                    base_url=base_url if ok else None,
                    model=model if ok else None,
                    usar_llm=usar_llm and ok,
                )
                st.session_state.perfil = perfil
                st.markdown(resposta_final)
            else:
                clf = classificar_mensagem(
                    prompt,
                    perfil,
                    base_url=base_url if ok else None,
                    model=model if ok else None,
                    usar_llm=usar_llm and ok,
                )
                acao = clf.get("acao")

                if acao == "recusar":
                    resposta_final = MSG_RECUSA
                    st.markdown(resposta_final)

                elif acao == "atualizar_perfil":
                    perfil = aplicar_atualizacao_perfil(perfil, clf)
                    st.session_state.perfil = perfil
                    resposta_final = clf.get("resposta_curta") or "Perfil atualizado."
                    resposta_final += (
                        f"\n\nDispensa: {', '.join(perfil['dispensa']) or '—'}\n"
                        f"Gosta: {', '.join(perfil['gosta']) or '—'}\n"
                        f"Não gosta: {', '.join(perfil['nao_gosta']) or '—'}"
                    )
                    st.markdown(resposta_final)

                elif acao == "buscar":
                    pedido = clf.get("pedido") or prompt
                    intent = montar_busca_do_perfil(
                        pedido,
                        perfil,
                        base_url=base_url if ok else None,
                        model=model if ok else None,
                        usar_llm=usar_llm and ok,
                    )
                    ings = intent.get("ingredientes") or []
                    top = ranquear(
                        receitas,
                        ings,
                        metodos=intent.get("metodos") or [],
                        keywords=intent.get("keywords") or [],
                        evitar=intent.get("evitar") or [],
                        k=k,
                        min_cobertura=min_cobertura if ings else 0.0,
                        max_faltantes=(
                            int(max_faltantes)
                            if (usar_max_faltantes and ings)
                            else None
                        ),
                        nota_min=nota_min if nota_min > 0 else None,
                        exigir_estilo=exigir_estilo,
                    )
                    st.session_state.last_top = top
                    st.session_state.last_intent = intent

                    if not top:
                        resposta_final = (
                            "Não achei receitas com seu perfil e esse pedido. "
                            "Tente outro estilo ou atualize a dispensa."
                        )
                        st.markdown(resposta_final)
                    else:
                        lista = resumir_top_markdown(top, intent)
                        st.markdown(lista)
                        resposta_final = lista

                        if explicar and ok:
                            st.markdown("**Sugestão:**")
                            try:
                                explicacao = st.write_stream(
                                    ollama_chat_stream(
                                        base_url=base_url,
                                        model=model,
                                        system=EXPLICAR_SYSTEM,
                                        user=contexto_explicacao(
                                            top, intent, perfil, pedido
                                        ),
                                    )
                                )
                                if explicacao:
                                    resposta_final = (
                                        lista + "\n\n**Sugestão:**\n" + str(explicacao)
                                    )
                            except Exception as exc:  # noqa: BLE001
                                err = f"Falha ao explicar com Ollama: {exc}"
                                st.error(err)
                                resposta_final = lista + "\n\n" + err
                        elif explicar and not ok:
                            nota = f"\n\n_{MSG_SEM_OLLAMA}_"
                            st.markdown(nota)
                            resposta_final = lista + nota

                else:
                    resposta_final = clf.get("resposta_curta") or (
                        "Pode pedir uma receita (ex.: quero um bolo) "
                        "ou atualizar a dispensa."
                    )
                    st.markdown(resposta_final)

    if resposta_final:
        st.session_state.messages.append(
            {"role": "assistant", "content": resposta_final}
        )


if __name__ == "__main__":
    main()
