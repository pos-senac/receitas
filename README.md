# 🍳 Byte & Bite — Assistente Culinário Inteligente

> *"A IA que transforma o que você tem em casa no cardápio que você precisa."*

O **Byte & Bite** é um assistente culinário inteligente projetado para combater o desperdício de alimentos e facilitar o planejamento alimentar diário. O sistema permite que o usuário informe os ingredientes disponíveis em sua casa, suas restrições de saúde ou alergias e suas preferências culinárias/orçamentárias para recomendar as receitas ideais.

Este projeto foi desenvolvido como um experimento prático e artigo acadêmico focado em **RAG Híbrido e Sistemas de Recomendação com IA Generativa**.

---

## 🏗️ Arquitetura do Sistema

Ao contrário de abordagens ingênuas que delegam toda a decisão à alucinação de um LLM, o **Byte & Bite** adota uma arquitetura em duas camadas:

1. **Camada Determinística (Motor de Busca & Ranking):** Responsável por filtrar rigidamente e pontuar as receitas a partir do overlap exato de ingredientes (`% de cobertura`), notas dos usuários e termos de estilo. O LLM **não** escolhe nem inventa receitas.
2. **Camada Cognitiva (LLM Local):** Uma instância local do Ollama interpreta a linguagem natural do usuário, conduz o onboarding, esclarece pedidos ambíguos e atua na receita recomendada sugerindo **adaptações saudáveis** e **substituições econômicas** para os itens faltantes.

```mermaid
flowchart TD
    User([Usuário]) <--> UI[Streamlit Frontend]
    UI <--> State[Máquina de Estados & Guardrails (chat.py)]
    State <--> DB[(SQLite: bytebite.db)]
    State --> Intent[Extrator de Intenção (intent.py)]
    Intent --> RankEngine[Motor de Ranking Determinístico (ranking.py)]
    Data[(1.000 Receitas JSON)] --> RankEngine
    RankEngine --> TopK[Top-1 Recomendada + Alternativas]
    TopK --> LLM[Ollama: Camada Cognitiva (qwen2.5:14b)]
    LLM --> CardResult[Exibição dos 3 Pilares no Chat]
    CardResult --> UI
```

---

## ✨ Principais Funcionalidades

- **Onboarding Guiado:** Coleta estruturada em 3 etapas: Despensa disponível ➔ Condições de saúde / alergias ➔ Preferências de sabor e orçamento.
- **Barra Lateral Interativa:** Visualize e edite a qualquer momento sua despensa e restrições com badges interativos, sincronizados em tempo real no banco SQLite.
- **Exibição em 3 Pilares:**
  - 💡 **Aviso Inteligente:** Proporção matemática e lista exata de ingredientes que você já possui.
  - 🛒 **Lista de Compras:** Destaque para *"Só o que falta"* comprar para completar a receita.
  - 🌿 **Adaptação Saudável & Dica do Chef:** Streaming em tempo real da IA sugerindo substituições baratas e adaptações de preparo para suas restrições de saúde.
- **Segurança & Guardrails:** Bloqueio nativo contra injeções de prompt (tentativas de pedir código SQL, Python, scripts ou jailbreak são barradas no código).
- **Acompanhamento Profissional:** Disclaimer médico integrado alertando que o assistente apoia o dia a dia, mas não substitui nutricionistas ou médicos.

---

## 🛠️ Tecnologias Utilizadas

- **Linguagem:** Python 3.10+
- **Interface:** [Streamlit](https://streamlit.io/)
- **Coleta de Dados (Web Scraping):** [Playwright](https://playwright.dev/python/) (Chromium headless)
- **Camada de LLM Local:** [Ollama](https://ollama.com/) (Modelo padrão: `qwen2.5:14b`)
- **Persistência do Usuário:** SQLite (`bytebite.db`)
- **Base de Conhecimento:** 1.000 receitas extraídas via JSON-LD estruturado do TudoGostoso.

---

## 🚀 Como Executar o Projeto

### 1. Criar o Ambiente Virtual

```bash
python -m venv .venv

# Ativar o ambiente virtual:
# No Windows:
.venv\Scripts\activate
# No Linux/Mac:
source .venv/bin/activate
```

### 2. Instalar as Dependências

```bash
pip install -r requirements.txt
playwright install chromium
```

### 3. Inicializar o Banco de Dados (SQLite)

Execute o script para criar as tabelas de perfil do usuário:

```bash
python setup_db.py
```

### 4. Iniciar o Ollama (LLM)

Certifique-se de que o [Ollama](https://ollama.com/) está instalado e em execução no seu computador com o modelo baixado:

```bash
ollama run qwen2.5:14b
```

### 5. Executar a Aplicação Web

Inicie o servidor do Streamlit:

```bash
streamlit run app.py
```

O aplicativo será aberto automaticamente no seu navegador no endereço `http://localhost:8501`.

---

## 📂 Estrutura de Arquivos

```text
├── assets/                  # Identidade visual (logo e símbolos)
├── receitas/                # 1.000 arquivos JSON estruturados das receitas raspadas
├── app.py                   # Interface gráfica web (Streamlit) e orquestração da UI
├── chat.py                  # Máquina de estados, guardrails e prompts do sistema
├── ranking.py               # Motor determinístico de overlap e pontuação de receitas
├── intent.py                # Extração de intenções em linguagem natural (LLM + Regex)
├── setup_db.py              # Criação e migração da tabela SQLite local
├── extrair_receita.py       # Scraper unitário com extração de JSON-LD
├── extrair_receitas.py      # Scraper em lote idempotente com controle de sessão
├── requirements.txt         # Dependências do projeto
└── README.md                # Documentação oficial
```

---

## ⚖️ Ética e Disclaimer Legal

- **Web Scraping Responsável:** A extração dos dados foi realizada com propósitos estritamente acadêmicos e educacionais, respeitando intervalos de tempo educados (*rate limit*) entre requisições para não onerar os servidores de origem. O dataset de receitas brutas não é redistribuído comercialmente.
- **Aviso de Saúde:** *O Byte & Bite não prescreve tratamentos, doses ou planos alimentares clínicos. Em caso de condições crônicas de saúde ou alergias severas, consulte sempre um médico ou nutricionista.*