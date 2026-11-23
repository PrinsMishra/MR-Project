"""
Streamlit Web Application: Counterexample-Guided Academic Policy Reasoning with LLMs, RAG, and Z3.
Provides 4 main screens:
  1. Upload Policies & View Chunks
  2. Review & Approve Extracted Rules
  3. Student Eligibility Query Form
  4. Formal Verification Results & Counterexample Explorer
"""

import streamlit as st
import json
from pathlib import Path
import pandas as pd

from services.policy_service import PolicyService
from services.eligibility_service import EligibilityService
from services.explanation_service import ExplanationService
from reasoning.rule_schema import StudentProfile, PrerequisiteWaiver, DecisionEnum, AcademicRule, AtomicCondition, RuleType, SourceReference
from reasoning.counterexample import CounterexampleGenerator
from reasoning.conflict_detector import ConflictDetector

# Page configuration
st.set_page_config(
    page_title="Academic Policy Reasoning (Z3 + RAG)",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize services in session state
if "policy_service" not in st.session_state:
    st.session_state.policy_service = PolicyService()
if "eligibility_service" not in st.session_state:
    st.session_state.eligibility_service = EligibilityService()
if "ce_gen" not in st.session_state:
    st.session_state.ce_gen = CounterexampleGenerator()
if "conflict_det" not in st.session_state:
    st.session_state.conflict_det = ConflictDetector()

DATA_DIR = Path(__file__).parent / "data"

# Sidebar Navigation
st.sidebar.title("🎓 NITech Policy Reasoner")
st.sidebar.markdown("**Machine Reasoning Engine**\n*Powered by Z3, RAG & LLMs*")
screen = st.sidebar.radio(
    "Navigation",
    [
        "1. Upload & Ingest Policies",
        "2. Review Extracted Rules",
        "3. Student Eligibility Checker",
        "4. Formal Reasoning & Counterexamples",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Key Architecture Principle:**\n"
    "The LLM interprets policy language, but **Z3 mathematically verifies** every registration decision. "
    "Zero hallucinations."
)

# ---------------------------------------------------------------------------
# SCREEN 1: UPLOAD & INGEST POLICIES
# ---------------------------------------------------------------------------
if screen == "1. Upload & Ingest Policies":
    st.header("📄 Screen 1: Academic Policy Ingestion & Chunking")
    st.markdown(
        "Upload university policy PDFs or inspect the baseline **NITech Academic Regulations Manual (2026-2027)**. "
        "The document is divided into source-tracked, overlapping passages tagged with sections and course identifiers."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload Policy Document")
        uploaded_file = st.file_uploader("Upload Policy PDF", type=["pdf", "txt"])

        if uploaded_file is not None:
            if st.button("Process & Index Uploaded Document", type="primary"):
                with st.spinner("Extracting pages, chunking passages, and computing embeddings..."):
                    bytes_data = uploaded_file.read()
                    passages = st.session_state.policy_service.process_uploaded_pdf(
                        bytes_data, uploaded_file.name
                    )
                    st.success(f"Successfully processed {len(passages)} passages from `{uploaded_file.name}`!")
                    st.session_state.current_passages = passages
        else:
            if st.button("Load Default NITech Regulations PDF"):
                pdf_path = DATA_DIR / "raw_policies" / "academic_policy_nitech.pdf"
                if pdf_path.exists():
                    with open(pdf_path, "rb") as f:
                        passages = st.session_state.policy_service.process_uploaded_pdf(
                            f.read(), "academic_policy_nitech.pdf"
                        )
                    st.session_state.current_passages = passages
                    st.success(f"Loaded and indexed 25 default passages from `{pdf_path.name}`!")

    with col2:
        st.subheader("Ingestion Architecture")
        st.markdown(
            """
            * **PDF Parsing:** PyMuPDF (`fitz`) extracts layout, page indices, and text.
            * **Passage Chunker:** Preserves rule boundaries (`[Passage P001]` to `P025`).
            * **Vector Index:** Dense semantic embeddings stored in **Pinecone** & local cache.
            * **Course Hints:** Automatic regex entity linking (`CS101`, `CS201`, `CS301`, `CS401`, `CS402`).
            """
        )

    st.markdown("---")
    st.subheader("Extracted Passages and Source Metadata")

    passages = getattr(st.session_state, "current_passages", None)
    if not passages:
        # Load from vector cache if available
        cache_file = DATA_DIR / "vector_cache" / "passages_index.json"
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                raw_cache = json.load(f)
            table_data = []
            for pid, d in raw_cache.items():
                m = d["metadata"]
                table_data.append({
                    "Passage ID": m["passage_id"],
                    "Section": m["section"],
                    "Page": m["page_number"],
                    "Course Hints": ", ".join(m.get("course_hints", [])),
                    "Rule Hints": ", ".join(m.get("rule_hints", [])),
                    "Text Excerpt": m["text"][:120] + "...",
                })
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True)
            st.caption(f"Showing {len(df)} indexed passages from policy vector store.")
    else:
        table_data = [{
            "Passage ID": p.passage_id,
            "Section": p.section,
            "Page": p.page_number,
            "Course Hints": ", ".join(p.course_hints),
            "Rule Hints": ", ".join(p.rule_hints),
            "Text Excerpt": p.text[:120] + "...",
        } for p in passages]
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)


# ---------------------------------------------------------------------------
# SCREEN 2: REVIEW EXTRACTED RULES
# ---------------------------------------------------------------------------
elif screen == "2. Review Extracted Rules":
    st.header("🔍 Screen 2: Human-in-the-Loop Rule Review & Approval")
    st.markdown(
        "Inspect candidate rules extracted by the LLM alongside the verbatim source passages. "
        "Reviewers can adjust conditions, edit thresholds, approve rules into the Knowledge Base, or delete invalid drafts."
    )

    approved_rules = st.session_state.policy_service.get_approved_rules()
    st.sidebar.metric("Approved Rules in KB", len(approved_rules))

    tab1, tab2 = st.tabs(["Inspect Approved Knowledge Base", "Extract & Review Draft Rule"])

    with tab1:
        st.subheader("Current Approved Rules in `policies.json`")
        rule_selection = st.selectbox(
            "Select Rule to Inspect / Edit",
            approved_rules,
            format_func=lambda r: f"[{r.id}] {r.title} ({r.type.value})",
        )

        if rule_selection:
            col_l, col_r = st.columns([1, 1])
            with col_l:
                st.markdown("#### Original Document Source")
                st.info(
                    f"**Document:** `{rule_selection.source.document}`\n\n"
                    f"**Section:** `{rule_selection.source.section}` | **Passage ID:** `{rule_selection.source.passage_id}`\n\n"
                    f"*\"{rule_selection.source.text}\"*"
                )

            with col_r:
                st.markdown("#### Formal JSON Representation")
                st.json(rule_selection.model_dump())

            with st.expander("✏️ Edit This Approved Rule"):
                with st.form(f"edit_form_{rule_selection.id}"):
                    new_title = st.text_input("Rule Title", value=rule_selection.title)
                    new_desc = st.text_area("Description", value=rule_selection.description)
                    new_waiver = st.checkbox("Waiver Allowed", value=rule_selection.waiver_allowed)

                    submitted = st.form_submit_button("Save Changes to KB")
                    if submitted:
                        rule_selection.title = new_title
                        rule_selection.description = new_desc
                        rule_selection.waiver_allowed = new_waiver
                        st.session_state.policy_service.approve_or_update_rule(rule_selection)
                        st.success(f"Rule {rule_selection.id} successfully updated!")
                        st.rerun()

    with tab2:
        st.subheader("Test LLM Rule Extraction on Any Passage")
        st.markdown("Select a passage from the document to trigger candidate extraction:")

        cache_file = DATA_DIR / "vector_cache" / "passages_index.json"
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_passages = json.load(f)

            passage_choice = st.selectbox(
                "Choose Passage",
                list(cached_passages.keys()),
                format_func=lambda k: f"[{k}] Section {cached_passages[k]['metadata']['section']} - {cached_passages[k]['metadata']['text'][:60]}...",
            )

            if st.button("Extract Rule with LLM", type="primary"):
                p_meta = cached_passages[passage_choice]["metadata"]
                from ingestion.source_tracker import PolicyPassage
                p_obj = PolicyPassage(
                    passage_id=p_meta["passage_id"],
                    document=p_meta["document"],
                    page_number=p_meta["page_number"],
                    section=p_meta["section"],
                    text=p_meta["text"],
                    course_hints=p_meta.get("course_hints", []),
                    rule_hints=p_meta.get("rule_hints", []),
                )

                with st.spinner("Invoking LLM to extract structured Pydantic rule..."):
                    extracted_rule = st.session_state.policy_service.extract_draft_rule(p_obj)
                    if extracted_rule:
                        st.session_state.extracted_candidate = extracted_rule
                        st.success(f"Extracted candidate rule `{extracted_rule.id}`!")
                    else:
                        st.error("Failed to extract structured rule from passage.")

            if "extracted_candidate" in st.session_state:
                cand = st.session_state.extracted_candidate
                st.markdown("#### Candidate Draft Rule:")
                st.json(cand.model_dump())

                col_app, col_rej = st.columns([1, 1])
                with col_app:
                    if st.button("✅ Approve into Knowledge Base"):
                        st.session_state.policy_service.approve_or_update_rule(cand)
                        st.success(f"Rule `{cand.id}` approved and saved to policies.json!")
                with col_rej:
                    if st.button("❌ Reject Draft"):
                        del st.session_state.extracted_candidate
                        st.info("Draft rejected.")
                        st.rerun()


# ---------------------------------------------------------------------------
# SCREEN 3: STUDENT ELIGIBILITY CHECKER
# ---------------------------------------------------------------------------
elif screen == "3. Student Eligibility Checker":
    st.header("📝 Screen 3: Student Eligibility Query")
    st.markdown(
        "Enter student attributes, current registered credit counts, completed courses, and approvals. "
        "The system will query relevant policies with **RAG** and formally verify registration with **Z3**."
    )

    # Option to preload preset benchmark students
    with open(DATA_DIR / "students.json", "r", encoding="utf-8") as f:
        preset_students = json.load(f)["students"]
    preset_dict = {s["student_id"]: s for s in preset_students}

    col_load, col_target = st.columns([1, 1])
    with col_load:
        selected_preset = st.selectbox(
            "Quick Load Benchmark Student Profile",
            ["-- Create Custom Student --"] + [f"{s['student_id']}: {s['name']} (Year {s['year']})" for s in preset_students],
        )

    preset_data = {}
    if selected_preset != "-- Create Custom Student --":
        pid = selected_preset.split(":")[0].strip()
        preset_data = preset_dict[pid]

    with col_target:
        target_course = st.selectbox(
            "Target Registration Course",
            ["CS301", "CS101", "CS201", "CS401", "CS402"],
            index=0,
            help="CS301: Machine Learning, CS401: Deep Learning, CS201: Data Structures",
        )

    st.markdown("---")
    st.subheader("Student Academic Profile")

    with st.form("student_query_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st_id = st.text_input("Student ID", value=preset_data.get("student_id", "S_CUSTOM"))
            st_name = st.text_input("Student Name", value=preset_data.get("name", "Custom Student"))
            year_val = preset_data.get("year", 2)
            year = st.selectbox(
                "Academic Year Standing",
                [1, 2, 3, 4, "Unknown (None)"],
                index=([1, 2, 3, 4, None].index(year_val) if year_val in [1, 2, 3, 4] else 1),
            )

        with col2:
            cgpa_val = preset_data.get("cgpa", 8.0)
            cgpa_unknown = st.checkbox("CGPA is Unknown", value=(cgpa_val is None))
            cgpa = None if cgpa_unknown else st.number_input(
                "CGPA (out of 10.0)", min_value=0.0, max_value=10.0, value=float(cgpa_val or 7.5), step=0.1
            )

            current_credits = st.number_input(
                "Current Registered Credits",
                min_value=0,
                max_value=32,
                value=int(preset_data.get("current_registered_credits", 16)),
                step=1,
            )

        with col3:
            completed_default = preset_data.get("completed_courses", ["CS101"])
            completed_courses = st.multiselect(
                "Completed Courses",
                ["CS101", "CS201", "CS301", "CS401", "CS402"],
                default=completed_default if completed_default is not None else [],
            )

            on_probation = st.checkbox("On Academic Probation", value=bool(preset_data.get("on_probation", False)))
            registration_hold = st.checkbox("Active Administrative Registration Hold", value=bool(preset_data.get("registration_hold", False)))
            dept_approval = st.checkbox("Department Credit Overload Approval", value=bool(preset_data.get("department_credit_approval", False)))

        # Faculty waivers
        st.markdown("#### Faculty Prerequisite Waivers")
        has_waiver = st.checkbox("Student has a course-specific faculty waiver", value=bool(preset_data.get("faculty_waivers")))
        waivers = []
        if has_waiver:
            w_col1, w_col2, w_col3 = st.columns(3)
            with w_col1:
                w_course = st.selectbox("Waiver for Target Course", ["CS301", "CS201", "CS401", "CS402"])
            with w_col2:
                w_prereq = st.selectbox("Waived Prerequisite", ["CS101", "CS201", "CS301"])
            with w_col3:
                w_prof = st.text_input("Approved By", value="Dr. Alan Turing")
            waivers.append(PrerequisiteWaiver(
                course_code=w_course, waived_prerequisite=w_prereq, approved_by=w_prof
            ))

        user_query = st.text_input(
            "Natural Language Inquiry",
            value=f"Am I eligible to register for {target_course}?",
        )

        verify_btn = st.form_submit_button("🔍 Run Formal Z3 Verification", type="primary")

        if verify_btn:
            profile = StudentProfile(
                student_id=st_id,
                name=st_name,
                year=None if year == "Unknown (None)" else int(year),
                cgpa=cgpa,
                current_registered_credits=current_credits,
                completed_courses=completed_courses,
                on_probation=on_probation,
                registration_hold=registration_hold,
                department_credit_approval=dept_approval,
                faculty_waivers=waivers,
            )

            with st.spinner("Retrieving policy passages and executing Z3 solver..."):
                bundle = st.session_state.eligibility_service.verify_registration(
                    profile, target_course, user_query=user_query
                )
                st.session_state.last_verification = bundle
                st.session_state.last_profile = profile
                st.success("Verification complete! Redirecting to Results screen...")
                st.info("Navigate to **Screen 4: Formal Reasoning & Counterexamples** to view the full proof.")


# ---------------------------------------------------------------------------
# SCREEN 4: FORMAL REASONING & COUNTEREXAMPLES
# ---------------------------------------------------------------------------
elif screen == "4. Formal Reasoning & Counterexamples":
    st.header("🔬 Screen 4: Symbolic Reasoning Results & Explanations")

    bundle = getattr(st.session_state, "last_verification", None)

    if not bundle:
        st.warning("No registration query has been executed yet. Go to **Screen 3: Student Eligibility Checker** to submit a query.")
    else:
        # 1. Top Decision Badge
        dec = bundle.decision
        if dec == DecisionEnum.ELIGIBLE:
            st.success(f"### 🟢 OFFICIAL VERDICT: {dec.value}")
        elif dec == DecisionEnum.INELIGIBLE:
            st.error(f"### 🔴 OFFICIAL VERDICT: {dec.value}")
        elif dec == DecisionEnum.UNDETERMINED:
            st.warning(f"### 🟠 OFFICIAL VERDICT: {dec.value}")
        else:
            st.error(f"### 🟣 OFFICIAL VERDICT: {dec.value}")

        st.markdown(f"**Student:** `{bundle.student_name} ({bundle.student_id})` | **Target Course:** `{bundle.target_course}`")
        st.info(f"**Formal Explanation:**\n\n{bundle.explanation}")

        # 2. Failing Policies (UNSAT Core)
        if dec == DecisionEnum.INELIGIBLE and bundle.failing_rule_details:
            st.subheader("❌ Blocking Policies (Identified via Z3 UNSAT Core)")
            st.markdown("The solver proved that registration is mathematically impossible due to the following specific rules:")
            for r in bundle.failing_rule_details:
                with st.expander(f"Rule [{r['rule_id']}]: {r['title']}", expanded=True):
                    st.write(f"**Policy Rule:** {r['description']}")
                    st.caption(f"Citation: `{r['source_doc']}` | Section {r['source_section']} | Passage `{r['passage_id']}`")
                    st.markdown(f"> *\"{r['source_text']}\"*")

        # 3. Missing Information Handler
        if dec == DecisionEnum.UNDETERMINED:
            st.subheader("⚠️ Missing Facts Required to Reach a Decision")
            st.markdown(
                "Because certain student attributes are unknown, both **Eligible** and **Ineligible** assignments exist in the solution space."
            )
            for mf in bundle.missing_facts:
                st.error(f"Missing Field: `{mf}`")

        # 4. Counterexample Model Explorer
        if bundle.counterexample_model:
            st.subheader("🔬 Z3 Counterexample Model")
            st.markdown(
                "The solver constructed a concrete counterexample scenario satisfying domain bounds but violating eligibility:"
            )
            st.json(bundle.counterexample_model)

        # 5. RAG Retrieved Evidence
        st.markdown("---")
        st.subheader("📚 Supporting Policy Passages (Retrieved via RAG)")
        for idx, p in enumerate(bundle.retrieved_passages, 1):
            with st.expander(f"Passage [{p.passage_id}] {p.document} §{p.section} (Similarity Score: {p.score:.3f})"):
                st.write(p.text)
                st.caption(f"Page: {p.page_number} | Course Hints: {p.course_hints} | Rule Hints: {p.rule_hints}")

    # Interactive Counterexample Lab (Independent of current query)
    st.markdown("---")
    st.subheader("🧪 Interactive Counterexample Lab")
    st.markdown("Test general claims (hypotheses) about university policies and let Z3 search for a counterexample:")

    claim_choice = st.selectbox(
        "Choose an Academic Claim to Test",
        [
            "Claim 1: 'All students with CGPA >= 8.5 are eligible for Machine Learning (CS301)'",
            "Claim 2: 'Any student who completed CS101 and CS201 is eligible for CS301'",
            "Claim 3: 'Any student with department approval can register for 28 credits in CS401'",
            "Claim 4: 'A faculty waiver granted for CS201 also satisfies prerequisites for CS301'",
            "Claim 5: 'Any student with CGPA >= 8.0 and not on probation is eligible for CS401'",
        ],
    )

    if st.button("Search for Counterexample in Z3"):
        with st.spinner("Solving Actual_Policies ∧ Claim ∧ ¬Eligibility..."):
            if "Claim 1" in claim_choice:
                res = st.session_state.ce_gen.check_claim_high_gpa_implies_eligible("CS301", min_gpa=8.5)
            elif "Claim 2" in claim_choice:
                res = st.session_state.ce_gen.check_claim_prerequisites_alone_imply_eligible("CS301")
            elif "Claim 3" in claim_choice:
                res = st.session_state.ce_gen.check_claim_approval_alone_grants_28_credits("CS401")
            elif "Claim 4" in claim_choice:
                res = st.session_state.ce_gen.check_claim_waiver_for_other_course_applies("CS301", "CS201", "CS101")
            else:
                res = st.session_state.ce_gen.check_claim_good_standing_implies_elective_eligible("CS401")

            if res.found:
                st.error("❌ CLAIM FALSIFIED! Z3 found a concrete counterexample:")
                st.markdown(f"**Explanation:** {res.explanation}")
                st.json(res.counterexample)
            else:
                st.success("✅ CLAIM HOLDS! No counterexample exists under the policy set.")
