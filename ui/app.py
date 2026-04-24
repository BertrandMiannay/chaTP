from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import streamlit as st
from api.mistral_api import MistralAPI
from approaches.brute_force.approach import BruteForceApproach
from approaches.step_by_step.approach import StepByStepApproach
from approaches.rag.approach import RAGApproach
from benchmark.judge import Judge
from benchmark.runner import BenchmarkRunner

st.set_page_config(page_title="ChaTP - Plongée", layout="wide")
st.title("ChaTP - Assistant plongée")

PDF_DIR = str(Path(__file__).parent.parent / "data")


@st.cache_resource
def load_resources():
    api = MistralAPI()
    brute_force = BruteForceApproach(api=api, pdf_dir=PDF_DIR)
    step_by_step = StepByStepApproach(api=api, pdf_dir=PDF_DIR)
    rag = RAGApproach(api=api, pdf_dir=PDF_DIR)
    judge = Judge(api=api)
    runner = BenchmarkRunner()
    return brute_force, step_by_step, rag, judge, runner


brute_force, step_by_step, rag, judge, runner = load_resources()

APPROACHES = {"Brute Force": brute_force, "Step by Step": step_by_step, "RAG": rag}

tab_chat, tab_benchmark = st.tabs(["Chat", "Benchmark"])

# ── Chat ──────────────────────────────────────────────────────────────────────

with tab_chat:
    col_chat, col_metrics = st.columns([2, 1])

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "metrics" not in st.session_state:
        st.session_state.metrics = []

    with col_chat:
        selected_approach_name = st.radio(
            "Approche",
            options=list(APPROACHES.keys()),
            horizontal=True,
            label_visibility="collapsed",
        )
        selected_approach = APPROACHES[selected_approach_name]

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if question := st.chat_input("Posez votre question sur la plongée..."):
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                with st.spinner("Recherche en cours..."):
                    result = selected_approach.ask(question)
                st.write(result.answer)
            st.session_state.messages.append({"role": "assistant", "content": result.answer})
            st.session_state.metrics.append({
                "question": question[:60] + "..." if len(question) > 60 else question,
                "approach": selected_approach_name,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "total_tokens": result.total_tokens,
                "latency_ms": round(result.latency_ms),
            })
            st.rerun()

    with col_metrics:
        st.subheader("Métriques")
        if st.session_state.metrics:
            for i, m in enumerate(reversed(st.session_state.metrics), 1):
                with st.expander(f"Q{len(st.session_state.metrics) - i + 1}: {m['question']}"):
                    st.caption(m.get("approach", "—"))
                    st.metric("Tokens entrée", m["input_tokens"])
                    st.metric("Tokens sortie", m["output_tokens"])
                    st.metric("Total tokens", m["total_tokens"])
                    st.metric("Latence", f"{m['latency_ms']} ms")
        else:
            st.info("Les métriques apparaîtront après votre première question.")

# ── Benchmark ─────────────────────────────────────────────────────────────────

with tab_benchmark:
    st.subheader("Benchmark des approches")

    use_judge = st.toggle("Évaluation LLM-as-judge (score 1-5)", value=True, key="use_judge")
    selected = st.multiselect(
        "Approches à tester",
        options=list(APPROACHES.keys()),
        default=list(APPROACHES.keys()),
    )

    if st.button("Lancer le benchmark", type="primary", disabled=not selected):
        st.session_state.benchmark_results = {}
        progress = st.progress(0)
        status = st.empty()
        total_steps = len(selected) * len(runner.questions)
        step = [0]

        for name in selected:
            status.write(f"Approche en cours : **{name}**")

            def make_callback(approach_name):
                def cb(i, total):
                    step[0] += 1
                    progress.progress(step[0] / total_steps, text=f"{approach_name} — question {i}/{total}")
                return cb

            result = runner.run(
                approach=APPROACHES[name],
                judge=judge if use_judge else None,
                progress_callback=make_callback(name),
            )
            st.session_state.benchmark_results[name] = result

        progress.empty()
        status.empty()

    if "benchmark_results" not in st.session_state or not st.session_state.benchmark_results:
        st.info("Lance le benchmark pour voir les résultats.")
    else:
        results = st.session_state.benchmark_results

        # Résumé comparatif
        st.subheader("Résumé")
        ran_with_judge = st.session_state.get("use_judge", True)
        summary_cols = st.columns(len(results))
        for col, (name, bench) in zip(summary_cols, results.items()):
            with col:
                st.markdown(f"**{name}**")
                st.metric("Moy. tokens", f"{bench.avg_total_tokens:.0f}")
                st.metric("Moy. latence", f"{bench.avg_latency_ms:.0f} ms")
                if ran_with_judge:
                    st.metric("Score moyen", f"{bench.avg_judge_score:.1f} / 5")

        # Tableau comparatif par question
        st.subheader("Comparaison par question")
        approach_names = list(results.keys())
        first_bench = next(iter(results.values()))
        labels = [r.short_label for r in first_bench.results]
        results_by_label = {
            name: {r.short_label: r for r in bench.results}
            for name, bench in results.items()
        }
        table_rows = []
        for label in labels:
            row = {"Question": label}
            for name in approach_names:
                r = results_by_label[name][label]
                row[f"{name} — tokens"] = r.total_tokens
                row[f"{name} — latence (ms)"] = round(r.latency_ms)
                if ran_with_judge:
                    row[f"{name} — score"] = f"{r.judge_score}/5"
            table_rows.append(row)
        st.dataframe(table_rows, use_container_width=True, hide_index=True)

        # Détail par question
        st.subheader("Détail par question")
        for name, bench in results.items():
            with st.expander(f"Approche : {name}", expanded=True):
                for r in bench.results:
                    st.markdown(f"**Q : {r.question}**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption("Réponse obtenue")
                        st.write(r.actual_answer)
                    with col2:
                        st.caption("Réponse attendue")
                        st.write(r.expected_answer)
                    mcols = st.columns(4)
                    mcols[0].metric("Tokens entrée", r.input_tokens)
                    mcols[1].metric("Tokens sortie", r.output_tokens)
                    mcols[2].metric("Latence", f"{r.latency_ms:.0f} ms")
                    if ran_with_judge:
                        mcols[3].metric("Score", f"{r.judge_score} / 5")
                    if r.metadata.get("selected_documents"):
                        st.caption(f"Documents sélectionnés : {', '.join(r.metadata['selected_documents'])}")
                    if r.metadata.get("retrieved_chunks"):
                        docs_used = list(dict.fromkeys(c["doc_name"] for c in r.metadata["retrieved_chunks"]))
                        st.caption(f"Chunks récupérés depuis : {', '.join(docs_used)}")
                    st.divider()
