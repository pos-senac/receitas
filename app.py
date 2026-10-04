"""
Chat Streamlit: onboarding → esclarecimento do pedido → só então ranking.

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
    formatar_rascunho_md,
    mesclar_rascunho,
    montar_busca_do_perfil,
    ollama_chat_stream,
    ollama_disponivel,
    pedido_desde_rascunho,
    perfil_novo,
    processar_onboarding,
    rascunho_novo,
    rascunho_tem_estilo,
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
    if "rascunho" not in st.session_state:
        st.session_state.rascunho = rascunho_novo()
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
    st.session_state.rascunho = rascunho_novo()
    st.session_state.messages = [{"role": "assistant", "content": MSG_BOAS_VINDAS}]
    st.session_state.last_top = None
    st.session_state.last_intent = None


def render_sidebar(perfil: dict[str, Any], rascunho: dict[str, Any]) -> bool:
    """Retorna True se o usuário pediu busca forçada pelo botão."""
    st.sidebar.markdown("### 👤 Meu Perfil")
    
    # Status visual da fase do perfil
    fase = perfil.get('fase', 'Desconhecida')
    if fase == "pronto":
        st.sidebar.success("Perfil Completo!")
    else:
        st.sidebar.info(f"Fase atual: {fase.capitalize()}")

    with st.sidebar.expander("📦 Minha Despensa & Gostos", expanded=True):
        st.markdown("**Dispensa**")
        st.write(", ".join(perfil.get("dispensa") or []) or "—")
        st.markdown("**Gosta**")
        st.write(", ".join(perfil.get("gosta") or []) or "—")
        st.markdown("**Não gosta**")
        st.write(", ".join(perfil.get("nao_gosta") or []) or "—")

    st.sidebar.markdown("### 📝 Pedido em elaboração")
    
    with st.sidebar.container(border=True):
        st.markdown(formatar_rascunho_md(rascunho))
        
        forcar = False
        pode = rascunho.get("ativo") and (
            rascunho_tem_estilo(rascunho) or bool(rascunho.get("texto"))
        )
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Buscar agora 🔍", type="primary", disabled=not pode, use_container_width=True):
                forcar = True
        with col_btn2:
            if st.button("Limpar pedido 🧹", use_container_width=True):
                st.session_state.rascunho = rascunho_novo()
                st.rerun()
                
    return forcar


def executar_busca(
    *,
    pedido: str,
    perfil: dict[str, Any],
    rascunho: dict[str, Any],
    receitas: list[dict[str, Any]],
    base_url: str,
    model: str,
    usar_llm: bool,
    ok: bool,
    k: int,
    min_cobertura: float,
    max_faltantes: int | None,
    nota_min: float | None,
    exigir_estilo: bool,
    explicar: bool,
) -> str:
    intent = montar_busca_do_perfil(
        pedido,
        perfil,
        rascunho=rascunho,
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
        max_faltantes=max_faltantes if ings else None,
        nota_min=nota_min,
        exigir_estilo=exigir_estilo,
    )
    st.session_state.last_top = top
    st.session_state.last_intent = intent

    if not top:
        return (
            "Não achei receitas com seu perfil e esse pedido. "
            "Ajuste o rascunho ou a dispensa e tente de novo."
        )

    # Renderiza receita recomendada com layout Byte & Bite em formato de Card
    r = top[0]
    st.markdown("### ✨ Minha Recomendação para Você")
    
    with st.container(border=True):
        st.markdown(f"## 🍽️ [{r['titulo']}]({r['url']})")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("#### 💡 O que você já tem")
            total = len(r['tem']) + len(r['falta'])
            st.write(f"Você já tem {len(r['tem'])} de {total} ingredientes.")
            if r['tem']:
                for item in r['tem']:
                    st.markdown(f"- {item}")
        
        with col2:
            if r['falta']:
                st.warning("#### 🛒 Lista de Compras")
                st.write("**Só o que falta:**")
                for item in r['falta']:
                    st.markdown(f"- {item}")
            else:
                st.info("#### 🎉 Lista de Compras")
                st.write("Você tem tudo!")
        
        st.markdown("---")
        st.markdown("#### 🌿 Adaptação Saudável e Dica do Chef")
        if explicar and ok:
            try:
                explicacao = st.write_stream(
                    ollama_chat_stream(
                        base_url=base_url,
                        model=model,
                        system=EXPLICAR_SYSTEM,
                        user=contexto_explicacao([r], intent, perfil, pedido),
                    )
                )
            except Exception as exc:
                st.error(f"Falha ao explicar com Ollama: {exc}")
        elif explicar and not ok:
            st.markdown(f"_{MSG_SEM_OLLAMA}_")
    
    if len(top) > 1:
        st.markdown("<br>#### 🔄 Outras opções", unsafe_allow_html=True)
        with st.expander("Ver receitas alternativas que também combinam"):
            st.markdown(resumir_top_markdown(top[1:], intent))
    
    resposta = f"Recomendação gerada: **{r['titulo']}**"
    return resposta


def main() -> None:
    st.set_page_config(page_title="Byte & Bite — Assistente Culinário", page_icon="🍳", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""
    <style>
    /* Esconde o menu padrao do Streamlit para parecer App Web */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .byte-bite-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .byte-bite-title {
        font-size: 3rem;
        font-weight: 700;
        color: #2C5545; /* Verde Escuro da Marca */
        margin-bottom: 0;
    }
    .byte-bite-subtitle {
        font-size: 1.2rem;
        color: #537566; /* Verde um pouco mais claro para subtítulo */
        margin-top: 0;
    }
    
    /* Personalização dos 'Cards' (Colunas) */
    [data-testid="column"] {
        background-color: #ffffff; /* Fundo branco para destacar no fundo creme */
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #D96C40; /* Laranja Ferrugem da Marca na borda esquerda */
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin: 5px;
    }
    
    /* Modo Escuro - Adaptação opcional se o usuário usar o PC no modo dark */
    @media (prefers-color-scheme: dark) {
        [data-testid="column"] {
            background-color: #1F3B30; /* Fundo verde super escuro */
            border-left: 5px solid #D96C40;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
    }
    </style>
    <div class="byte-bite-header">
        <div class="byte-bite-subtitle">A IA que transforma o que você tem em casa no cardápio que você precisa.</div>
    </div>
    """, unsafe_allow_html=True)

    # Renderiza a Logo Escrita centralizada
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        try:
            st.image("assets/escrita.png", use_container_width=True)
        except:
            pass

    init_state()
    receitas = get_receitas()

    # Logo Sidebar e Configurações escondidas (Dev Mode)
    with st.sidebar:
        col_img1, col_img2, col_img3 = st.columns([1,0.8,1])
        with col_img2:
            try:
                st.image("assets/simbolo.png", use_container_width=True)
            except:
                pass
        
        st.markdown("---")
        
        with st.expander("⚙️ Configurações Técnicas do Sistema"):
            st.markdown(f"**{len(receitas)}** receitas em memória")
            base_url = st.text_input("Ollama URL", OLLAMA_URL_DEFAULT)
            model = st.text_input("Modelo", MODELO_DEFAULT)
            k = st.slider("Top‑k", 1, 10, 5)
            min_cobertura = st.slider("Cobertura mínima", 0.0, 1.0, 0.25, 0.05)
            max_faltantes = st.number_input("Máx. faltantes", 0, 30, 12)
            usar_max_faltantes = st.checkbox("Limitar faltantes", True)
            nota_min = st.slider("Nota mínima", 0.0, 5.0, 0.0, 0.1)
            usar_llm = st.checkbox("Usar Ollama (interpretação/explicação)", True)
            exigir_estilo = st.checkbox("Exigir match de estilo no pedido", True)
            explicar = st.checkbox("Explicar top‑k com LLM", True)
            
            ok, info = ollama_disponivel(base_url)
            if ok:
                st.success(f"Ollama ok — {info}")
            else:
                st.warning(f"Ollama indisponível: {info}")

            if st.button("Reiniciar conversa inteira"):
                reset_chat()
                st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("🩺 **Acompanhamento Profissional**: Em caso de condição de saúde, procure um médico ou nutricionista. O Byte & Bite não substitui esse cuidado: ele ajuda a aplicar o plano alimentar no dia a dia.")
    
    forcar_busca = render_sidebar(st.session_state.perfil, st.session_state.rascunho)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Digite sua mensagem…")
    if not prompt and not forcar_busca:
        return

    if forcar_busca and not prompt:
        prompt = "pode buscar"
        st.session_state.messages.append({"role": "user", "content": "*(Buscar agora)*"})
        with st.chat_message("user"):
            st.markdown("*(Buscando...)*")
    elif prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

    perfil = st.session_state.perfil
    rascunho = st.session_state.rascunho
    resposta_final = ""

    with st.chat_message("assistant"):
        with st.spinner("🍳 Pensando e analisando ingredientes…"):
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
                    rascunho,
                    base_url=base_url if ok else None,
                    model=model if ok else None,
                    usar_llm=usar_llm and ok,
                    forcar_busca=forcar_busca,
                )
                acao = clf.get("acao")

                # Sempre aplica patch de rascunho quando houver.
                if clf.get("rascunho_patch"):
                    rascunho = mesclar_rascunho(rascunho, clf["rascunho_patch"])
                if acao == "esclarecer":
                    rascunho["turnos"] = int(rascunho.get("turnos") or 0) + 1
                    rascunho["ativo"] = True
                st.session_state.rascunho = rascunho

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
                    st.rerun()

                elif acao == "esclarecer":
                    resposta_final = clf.get("resposta_curta") or "Pode me contar mais?"
                    perguntas = clf.get("perguntas") or []
                    if perguntas and not all(q in resposta_final for q in perguntas):
                        resposta_final += "\n\n" + "\n".join(f"- {q}" for q in perguntas)
                    resposta_final += (
                        "\n\n_💡 Rascunho atualizado — diga **pode buscar** quando quiser._"
                    )
                    st.markdown(resposta_final)

                elif acao == "buscar" and clf.get("pronto_para_buscar"):
                    pedido = (
                        clf.get("pedido")
                        or pedido_desde_rascunho(rascunho, prompt)
                        or prompt
                    )
                    st.markdown("Buscando com o pedido esclarecido…")
                    resposta_final = executar_busca(
                        pedido=pedido,
                        perfil=perfil,
                        rascunho=rascunho,
                        receitas=receitas,
                        base_url=base_url,
                        model=model,
                        usar_llm=usar_llm,
                        ok=ok,
                        k=k,
                        min_cobertura=min_cobertura,
                        max_faltantes=(
                            int(max_faltantes) if usar_max_faltantes else None
                        ),
                        nota_min=nota_min if nota_min > 0 else None,
                        exigir_estilo=exigir_estilo,
                        explicar=explicar,
                    )

                else:
                    resposta_final = clf.get("resposta_curta") or (
                        "Me diga o que quer cozinhar; eu esclareço e só busco depois "
                        "(ou diga **pode buscar**)."
                    )
                    st.markdown(resposta_final)

    if resposta_final:
        st.session_state.messages.append(
            {"role": "assistant", "content": resposta_final}
        )


if __name__ == "__main__":
    main()
