"""
Academic Policy Reasoning Assistant - NITech
A clean, student-centric application that verifies registration eligibility
using RAG (Retrieval-Augmented Generation) and the Z3 Formal Logic Solver.
"""

import streamlit as st
import json
from pathlib import Path

from services.eligibility_service import EligibilityService
from reasoning.rule_schema import StudentProfile, PrerequisiteWaiver, DecisionEnum
from reasoning.counterexample import CounterexampleGenerator

# Page settings
st.set_page_config(
    page_title="Counterexample-Guided Academic Policy Reasoning with LLMs and Z3",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Initialize services in session state
if "eligibility_service" not in st.session_state:
    st.session_state.eligibility_service = EligibilityService()
if "ce_gen" not in st.session_state:
    st.session_state.ce_gen = CounterexampleGenerator()

DATA_DIR = Path(__file__).parent / "data"

# Custom Styling for clean modern card UI
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #6B7280;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .eligible-card {
        background-color: #ECFDF5;
        border: 2px solid #10B981;
        color: #065F46;
    }
    .ineligible-card {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        color: #991B1B;
    }
    .undetermined-card {
        background-color: #FFFBEB;
        border: 2px solid #F59E0B;
        color: #92400E;
    }
    .badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.9rem;
        margin-bottom: 0.8rem;
    }
    .badge-eligible { background-color: #10B981; color: white; }
    .badge-ineligible { background-color: #EF4444; color: white; }
    .badge-undetermined { background-color: #F59E0B; color: white; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎓 Counterexample-Guided Academic Policy Reasoning with LLMs and Z3</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Formal Verification of Academic Eligibility via LLMs, RAG, and the Z3 SMT Solver</div>', unsafe_allow_html=True)

# Main Navigation Tabs
tab_advisor, tab_handbook, tab_counterexamples = st.tabs([
    "💬 Check Course Eligibility",
    "📖 Policy Handbook",
    "🔬 Counterexample Lab",
])

# ---------------------------------------------------------------------------
# TAB 1: STUDENT ELIGIBILITY ADVISOR
# ---------------------------------------------------------------------------
with tab_advisor:
    # Load Preset Students
    with open(DATA_DIR / "students.json", "r", encoding="utf-8") as f:
        preset_students = json.load(f)["students"]
    preset_dict = {s["student_id"]: s for s in preset_students}

    # Friendly descriptions for students
    student_labels = {
        "S001": "Alice Johnson (Year 2, CGPA 8.5 — Eligible for CS301)",
        "S002": "Bob Smith (Year 2, CGPA 8.0 — Missing CS201 prerequisite)",
        "S003": "Charlie Davis (Year 2 — Missing CS201, but HAS Faculty Waiver)",
        "S004": "David Miller (Year 2 — Waiver for CS201, but applying for CS301)",
        "S005": "Eva Green (Year 2 — CGPA 6.4 < 7.0 required for CS301)",
        "S006": "Frank White (Year 1 — Standing year restricted for CS301)",
        "S007": "Grace Lee (Year 3 — 26 credits exceeds 24 without approval)",
        "S008": "Hannah Brown (Year 4, CGPA 8.9 — Approved 26-credit overload)",
        "S009": "Ian Clark (Year 4 — 30 credits exceeds absolute 28 cap)",
        "S010": "Julia Roberts (Year 3 — Academic probation blocks CS401)",
        "S011": "Kevin Patel (Year 3 — Registration hold blocks enrollment)",
        "S012": "Laura Scott (Year 2 — Cannot repeat completed CS101)",
        "S013": "Marcus Vance (Year 2 — Missing CGPA record)",
    }

    st.subheader("1. Select or Enter Student Information")
    col_preset, col_course = st.columns([1.4, 1])

    with col_preset:
        preset_choice = st.selectbox(
            "Choose a Student Scenario",
            ["-- Choose from preset students --"] + [f"{sid}: {student_labels.get(sid, s['name'])}" for sid, s in preset_dict.items() if sid in student_labels],
        )

    with col_course:
        target_course = st.selectbox(
            "Target Course to Register",
            [
                "CS301 - Machine Learning (4 credits)",
                "CS401 - Deep Learning (4 credits)",
                "CS201 - Data Structures (4 credits)",
                "CS402 - Natural Language Processing (3 credits)",
                "CS101 - Python Programming (4 credits)",
            ],
            index=0,
        )
        course_code = target_course.split(" - ")[0].strip()

    # Preload facts
    preset_data = {}
    if preset_choice != "-- Choose from preset students --":
        pid = preset_choice.split(":")[0].strip()
        preset_data = preset_dict.get(pid, {})

    with st.expander("👤 View / Edit Student Details", expanded=(preset_choice == "-- Choose from preset students --")):
        col1, col2 = st.columns(2)
        with col1:
            st_name = st.text_input("Student Name", value=preset_data.get("name", "Student"))
            year_val = preset_data.get("year", 2)
            year = st.selectbox(
                "Academic Year",
                [1, 2, 3, 4, "Unknown"],
                index=([1, 2, 3, 4, None].index(year_val) if year_val in [1, 2, 3, 4] else 1),
            )
            cgpa_val = preset_data.get("cgpa", 8.0)
            cgpa_unknown = st.checkbox("CGPA is unknown / not recorded", value=(cgpa_val is None))
            cgpa = None if cgpa_unknown else st.number_input(
                "CGPA (out of 10.0)", min_value=0.0, max_value=10.0, value=float(cgpa_val or 7.5), step=0.1
            )

        with col2:
            current_credits = st.number_input(
                "Currently Registered Credits",
                min_value=0, max_value=32, value=int(preset_data.get("current_registered_credits", 16))
            )
            default_completed = preset_data.get("completed_courses", ["CS101"])
            completed_courses = st.multiselect(
                "Completed Passed Courses",
                ["CS101", "CS201", "CS301", "CS401", "CS402"],
                default=default_completed if default_completed is not None else [],
            )

        # Status Checkboxes
        st.markdown("**Academic Status & Approvals:**")
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            on_probation = st.checkbox("Academic Probation", value=bool(preset_data.get("on_probation", False)))
        with col_c2:
            registration_hold = st.checkbox("Registration Hold", value=bool(preset_data.get("registration_hold", False)))
        with col_c3:
            dept_approval = st.checkbox("Dept Overload Approval (>24 cr)", value=bool(preset_data.get("department_credit_approval", False)))

        # Prerequisite Waiver
        has_waiver = st.checkbox("Has a Faculty Prerequisite Waiver", value=bool(preset_data.get("faculty_waivers")))
        waivers = []
        if has_waiver:
            w_col1, w_col2 = st.columns(2)
            with w_col1:
                w_course = st.selectbox("Waiver for Course", ["CS301", "CS201", "CS401", "CS402"])
            with w_col2:
                w_prereq = st.selectbox("Waived Prerequisite", ["CS201", "CS101", "CS301"])
            waivers.append(PrerequisiteWaiver(course_code=w_course, waived_prerequisite=w_prereq, approved_by="Faculty Instructor"))

    st.subheader("2. Ask Your Question")
    user_query = st.text_input(
        "Question",
        value=f"Am I eligible to register for {course_code}?",
        help="You can ask in natural language. The system retrieves the exact policy rules and mathematically checks your facts.",
    )

    if st.button("Check Registration Eligibility", type="primary", use_container_width=True):
        student_obj = StudentProfile(
            student_id=preset_data.get("student_id", "S_STUDENT"),
            name=st_name,
            year=None if year == "Unknown" else int(year),
            cgpa=cgpa,
            current_registered_credits=current_credits,
            completed_courses=completed_courses,
            on_probation=on_probation,
            registration_hold=registration_hold,
            department_credit_approval=dept_approval,
            faculty_waivers=waivers,
        )

        with st.spinner("Checking official academic regulations and verifying with Z3 formal solver..."):
            bundle = st.session_state.eligibility_service.verify_registration(
                student_obj, course_code, user_query=user_query
            )
            st.session_state.current_result = bundle

    # Display Clean Result
    if "current_result" in st.session_state:
        b = st.session_state.current_result

        st.markdown("---")
        st.subheader("Decision & Explanation")

        if b.decision == DecisionEnum.ELIGIBLE:
            st.markdown(
                f"""
                <div class="result-card eligible-card">
                    <span class="badge badge-eligible">ELIGIBLE TO REGISTER</span>
                    <h3>Yes, you are eligible to register for {b.target_course}!</h3>
                    <p style="font-size: 1.05rem; margin-top: 0.5rem;">
                        All required prerequisites, minimum CGPA thresholds, semester credit limits, and academic standing requirements are fully satisfied.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        elif b.decision == DecisionEnum.INELIGIBLE:
            st.markdown(
                f"""
                <div class="result-card ineligible-card">
                    <span class="badge badge-ineligible">NOT ELIGIBLE (REGISTRATION BLOCKED)</span>
                    <h3>You cannot register for {b.target_course} at this time.</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.error("#### ❌ What is preventing your registration:")
            for r in b.failing_rule_details:
                st.markdown(f"**Policy Rule `{r['rule_id']}` ({r['title']})**")
                st.markdown(f"> *{r['description']}*")
                st.caption(f"Official Citation: Section {r['source_section']}, Passage {r['passage_id']} in `{r['source_doc']}`")

        elif b.decision == DecisionEnum.UNDETERMINED:
            st.markdown(
                f"""
                <div class="result-card undetermined-card">
                    <span class="badge badge-undetermined">MORE INFORMATION NEEDED</span>
                    <h3>Eligibility is currently undecided.</h3>
                    <p style="font-size: 1.05rem;">
                        Your registration could be either approved or denied depending on missing student information:
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            for f in b.missing_facts:
                st.warning(f"⚠️ **Missing Field:** Please provide your **`{f}`** to determine eligibility.")

        # Supporting Policy Citations (Clean, plain English)
        if b.retrieved_passages:
            with st.expander("📖 View Supporting University Policy Rules", expanded=True):
                for p in b.retrieved_passages[:2]:
                    st.markdown(f"**Rule Passage [{p.passage_id}] (Section {p.section}):**")
                    st.info(f"*\"{p.text}\"*")

        # Collapsible technical proof for professor / grading review
        with st.expander("🛠️ Advanced: View Formal Z3 Proof Details (For Graders/Technical Review)"):
            st.write(f"**Solver Verdict:** `{b.decision.value}`")
            st.write(f"**Failing Rule IDs (UNSAT Core):** `{b.failing_rules}`")
            if b.counterexample_model:
                st.write("**Z3 Satisfying Counterexample Assignment:**")
                st.json(b.counterexample_model)


# ---------------------------------------------------------------------------
# TAB 2: UNIVERSITY POLICY HANDBOOK
# ---------------------------------------------------------------------------
with tab_handbook:
    st.subheader("Official Course Catalog & Regulation Rules")
    st.markdown("Northbridge Institute of Technology (NITech) — Academic Regulations (2026-2027)")

    col_c1, col_c2 = st.columns([1, 1.2])

    with col_c1:
        st.markdown("#### Course Catalog")
        courses_data = [
            {"Code": "CS101", "Course Name": "Python Programming", "Credits": 4, "Prerequisite": "None"},
            {"Code": "CS201", "Course Name": "Data Structures", "Credits": 4, "Prerequisite": "CS101"},
            {"Code": "CS301", "Course Name": "Machine Learning", "Credits": 4, "Prerequisite": "CS101 and CS201"},
            {"Code": "CS401", "Course Name": "Deep Learning", "Credits": 4, "Prerequisite": "CS301"},
            {"Code": "CS402", "Course Name": "Natural Language Processing", "Credits": 3, "Prerequisite": "CS301"},
        ]
        st.table(courses_data)

    with col_c2:
        st.markdown("#### Core Academic Regulations")
        st.markdown("""
        * **Rule R007–R010 (CGPA Requirements):**
          * CS201 requires minimum **6.0 CGPA**.
          * CS301 requires minimum **7.0 CGPA**.
          * CS401 requires minimum **7.5 CGPA**.
          * CS402 requires minimum **7.0 CGPA**.
        * **Rule R011 (Probation):** Students on academic probation cannot register for 400-level electives (CS401, CS402).
        * **Rule R012–R014 (Credit Limits):**
          * Standard semester limit is **24 credits**.
          * Final-year students with **CGPA $\ge$ 8.0** and **Department Approval** may take up to **28 credits**.
          * Absolute ceiling: No student may register for more than 28 credits.
        * **Rule R006 & R021 (Faculty Waivers):**
          * A prerequisite can be waived with explicit faculty instructor approval.
          * Waivers are strictly course-specific and do not carry over to other courses.
        """)


# ---------------------------------------------------------------------------
# TAB 3: COUNTEREXAMPLE LAB
# ---------------------------------------------------------------------------
with tab_counterexamples:
    st.subheader("🔬 Counterexample Verification Lab")
    st.markdown(
        "Test general academic claims (common LLM generalizations) and see if the **Z3 Solver** "
        "can find a concrete counterexample that falsifies the claim:"
    )

    claim_choice = st.selectbox(
        "Select a Policy Claim to Test",
        [
            "Claim 1: 'All students with high CGPA (>= 8.5) are eligible for Machine Learning (CS301)'",
            "Claim 2: 'Any student who completed CS101 and CS201 is eligible for CS301'",
            "Claim 3: 'Any student with department approval can take 28 credits in CS401'",
            "Claim 4: 'A faculty waiver granted for CS201 also satisfies prerequisites for CS301'",
            "Claim 5: 'Any student not on probation with CGPA >= 8.0 is eligible for CS401'",
        ],
    )

    if st.button("Falsify Claim with Z3 Solver", type="primary"):
        with st.spinner("Searching for a satisfying assignment that violates the claim..."):
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
                st.error("❌ **CLAIM FALSIFIED BY COUNTEREXAMPLE!**")
                st.markdown(f"**Explanation:** {res.explanation}")
                if res.counterexample:
                    ce = res.counterexample
                    st.info(
                        f"**Concrete Counterexample Student Found:**\n"
                        f"- Year: `{ce.get('year', 2)}`\n"
                        f"- CGPA: `{ce.get('cgpa', 'N/A')}`\n"
                        f"- Completed Courses: CS101={ce.get('completed_CS101', False)}, CS201={ce.get('completed_CS201', False)}\n"
                        f"- Result under university policy: **Ineligible!**"
                    )
            else:
                st.success("✅ **CLAIM HOLDS:** No counterexample exists under the university policies.")
