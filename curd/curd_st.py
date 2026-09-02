import streamlit as st
import pandas as pd
import numpy as np
from curd.mongo_curd1 import EMIMongoDBManager, OperationError
from curd.config import DATASET_COLUMNS, CATEGORICAL_FIELDS, STATUS_VALUES


def curd_op(db):
    st.header("📝 Applicant Management")
    
    crud_op = st.radio("Operation", [
        "📖 View", "➕ Create", "✏️ Update", 
        "🔍 Search", "🗑️ Delete"
    ], horizontal=True)
    
    if crud_op == "📖 View":
        col1, col2 = st.columns([1, 3])
        with col1:
            status_f = st.selectbox("Status", ["All"] + STATUS_VALUES)
            elig_f = st.selectbox("Eligibility", ["All", "Eligible", "Not Eligible"])
            limit = st.selectbox("Limit", [10, 25, 50, 100])
            sort = st.selectbox("Sort By", ["created_at", "monthly_salary", "credit_score", "foir_percentage"])
        
        with col2:
            query = {"deleted": {"$ne": True}}
            if status_f != "All":
                query["status"] = status_f
            if elig_f == "Eligible":
                query["emi_eligibility"] = 1
            elif elig_f == "Not Eligible":
                query["emi_eligibility"] = 0
            
            results = db.run_aggregation([
                {"$match": query},
                {"$sort": {sort: -1}},
                {"$limit": limit}
            ])
            
            if results:
                df = pd.DataFrame(results)
                display_cols = ['applicant_id', 'age', 'gender', 'education', 'employment_type',
                               'company_type', 'monthly_salary', 'credit_score', 'foir_percentage',
                               'emi_eligibility', 'status']
                display_cols = [c for c in display_cols if c in df.columns]
                st.dataframe(df[display_cols].style.format({'monthly_salary': '₹{:,.0f}'}), 
                            use_container_width=True, height=400)
                
                # Detail view
                st.markdown("---")
                selected_id = st.number_input("View Details - Applicant ID", 1, 999999, 1)
                if st.button("📋 View Full Details"):
                    detail = db.get_applicant_by_id(selected_id)
                    if detail:
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown("### Demographic & Employment")
                            st.json({k: detail[k] for k in ['age', 'gender', 'marital_status', 'education',
                                                              'family_size', 'dependents', 'employment_type',
                                                              'company_type', 'years_of_employment', 'monthly_salary']})
                        with col_b:
                            st.markdown("### Financial Details")
                            st.json({k: detail[k] for k in ['credit_score', 'bank_balance', 'emergency_fund',
                                                              'existing_loans', 'current_emi_amount', 'foir_percentage',
                                                              'debt_to_income_ratio', 'net_monthly_income']})
                        st.markdown("### Expenses")
                        st.json({k: detail[k] for k in ['monthly_rent', 'school_fees', 'college_fees',
                                                          'travel_expenses', 'groceries_utilities', 
                                                          'other_monthly_expenses', 'total_monthly_expenses']})
                    else:
                        st.warning("Not found")
            else:
                st.warning("No records")
    
    elif crud_op == "➕ Create":
        st.subheader("Create New Applicant")
        
        with st.form("create_form"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("#### 👤 Personal")
                age = st.number_input("Age*", 18, 100, 30)
                gender = st.selectbox("Gender*", CATEGORICAL_FIELDS["gender"])
                marital_status = st.selectbox("Marital Status*", CATEGORICAL_FIELDS["marital_status"])
                education = st.selectbox("Education*", CATEGORICAL_FIELDS["education"])
                family_size = st.number_input("Family Size", 0, 20, 2)
                dependents = st.number_input("Dependents", 0, 20, 0)
            
            with col2:
                st.markdown("#### 💼 Employment")
                employment_type = st.selectbox("Employment*", CATEGORICAL_FIELDS["employment_type"])
                company_type = st.selectbox("Company Type", CATEGORICAL_FIELDS["company_type"])
                years_emp = st.number_input("Years of Employment", 0, 50, 3)
                monthly_salary = st.number_input("Monthly Salary (₹)*", 1, 10000000, 50000, step=1000)
            
            with col3:
                st.markdown("#### 🏠 Housing & Expenses")
                house_type = st.selectbox("House Type", CATEGORICAL_FIELDS["house_type"])
                monthly_rent = st.number_input("Monthly Rent (₹)", 0, 500000, 0, step=1000)
                school_fees = st.number_input("School Fees (₹)", 0, 200000, 0, step=1000)
                college_fees = st.number_input("College Fees (₹)", 0, 200000, 0, step=1000)
                travel = st.number_input("Travel (₹)", 0, 100000, 0, step=1000)
                groceries = st.number_input("Groceries & Utilities (₹)", 0, 100000, 0, step=1000)
                other_exp = st.number_input("Other Expenses (₹)", 0, 100000, 0, step=1000)
            
            with col4:
                st.markdown("#### 💰 Financial & Loan")
                existing_loans = st.number_input("Existing Loans", 0, 20, 0)
                current_emi = st.number_input("Current EMI (₹)", 0, 500000, 0, step=1000)
                credit_score = st.slider("Credit Score*", 300, 900, 700)
                bank_balance = st.number_input("Bank Balance (₹)", 0, 100000000, 0, step=10000)
                emergency_fund = st.number_input("Emergency Fund (₹)", 0, 100000000, 0, step=10000)
                emi_scenario = st.selectbox("EMI Scenario", CATEGORICAL_FIELDS["emi_scenario"])
                req_amount = st.number_input("Requested Amount (₹)*", 1, 50000000, 500000, step=50000)
                req_tenure = st.selectbox("Tenure (Months)*", [12,24,36,48,60,84,120,180,240,360])
            
            submitted = st.form_submit_button("✅ Create Applicant", type="primary", use_container_width=True)
            
            if submitted:
                try:
                    result = db.create_applicant({
                        "age": age, "gender": gender, "marital_status": marital_status,
                        "education": education, "family_size": family_size, "dependents": dependents,
                        "employment_type": employment_type, "company_type": company_type,
                        "years_of_employment": years_emp, "monthly_salary": monthly_salary,
                        "house_type": house_type, "monthly_rent": monthly_rent,
                        "school_fees": school_fees, "college_fees": college_fees,
                        "travel_expenses": travel, "groceries_utilities": groceries,
                        "other_monthly_expenses": other_exp,
                        "existing_loans": existing_loans, "current_emi_amount": current_emi,
                        "credit_score": credit_score, "bank_balance": bank_balance,
                        "emergency_fund": emergency_fund, "emi_scenario": emi_scenario,
                        "requested_amount": req_amount, "requested_tenure": req_tenure
                    })
                    st.success(f"✅ Created Applicant ID: {result['applicant_id']}")
                    st.markdown("### Derived Fields Calculated:")
                    st.json({
                        "Total Expenses": f"₹{result['total_monthly_expenses']:,.0f}",
                        "FOIR": f"{result['foir_percentage']}%",
                        "DTI Ratio": result['debt_to_income_ratio'],
                        "Net Income": f"₹{result['net_monthly_income']:,.0f}",
                        "Estimated EMI": f"₹{result['estimated_emi']:,.0f}",
                        "Proposed FOIR": f"{result['proposed_foir_percentage']}%"
                    })
                    st.balloons()
                except ValueError as e:
                    st.error(f"Validation: {e}")
                except OperationError as e:
                    st.error(f"Database: {e}")
    
    elif crud_op == "✏️ Update":
        st.subheader("Update Applicant")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            update_id = st.number_input("Applicant ID", 1, 999999, 1)
            if st.button("🔍 Fetch"):
                st.session_state['update_id'] = update_id
        
        with col2:
            if 'update_id' in st.session_state or update_id:
                aid = st.session_state.get('update_id', update_id)
                applicant = db.get_applicant_by_id(aid)
                
                if applicant:
                    st.success(f"Found: {aid}")
                    with st.form("update_form"):
                        ucol1, ucol2, ucol3 = st.columns(3)
                        
                        with ucol1:
                            st.markdown("#### Personal")
                            u_age = st.number_input("Age", 18, 100, applicant['age'], key='u_age')
                            u_gender = st.selectbox("Gender", CATEGORICAL_FIELDS["gender"], 
                                                   index=CATEGORICAL_FIELDS["gender"].index(applicant['gender']), key='u_gender')
                            u_education = st.selectbox("Education", CATEGORICAL_FIELDS["education"],
                                                      index=CATEGORICAL_FIELDS["education"].index(applicant['education']), key='u_edu')
                        
                        with ucol2:
                            st.markdown("#### Financial")
                            u_salary = st.number_input("Salary", 1, 10000000, 
                                                       int(applicant['monthly_salary']), step=1000, key='u_sal')
                            u_credit = st.slider("Credit Score", 300, 900, applicant['credit_score'], key='u_cred')
                            u_emi = st.number_input("Current EMI", 0, 500000,
                                                    int(applicant.get('current_emi_amount', 0)), step=1000, key='u_emi')
                        
                        with ucol3:
                            st.markdown("#### Status")
                            u_status = st.selectbox("Status", STATUS_VALUES,
                                                    index=STATUS_VALUES.index(applicant['status']), key='u_stat')
                            u_notes = st.text_area("Review Notes", applicant.get('review_notes', ''), key='u_notes')
                        
                        if st.form_submit_button("💾 Update", type="primary", use_container_width=True):
                            if db.update_applicant(aid, {
                                "age": u_age, "gender": u_gender, "education": u_education,
                                "monthly_salary": u_salary, "credit_score": u_credit,
                                "current_emi_amount": u_emi, "status": u_status,
                                "review_notes": u_notes or None, "reviewed_by": "Web User"
                            }):
                                st.success("✅ Updated!")
                                st.rerun()
                else:
                    st.error("Not found")
    
    elif crud_op == "🔍 Search":
        st.subheader("Search Applicants")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            search_term = st.text_input("Search Term")
            search_fields = st.multiselect("Fields", 
                ['education', 'employment_type', 'company_type', 'house_type', 'emi_scenario'],
                default=['company_type', 'employment_type'])
            min_credit = st.slider("Min Credit", 300, 900, 300)
            max_credit = st.slider("Max Credit", 300, 900, 900)
            min_salary = st.number_input("Min Salary", 0, 10000000, 0, step=10000)
            max_salary = st.number_input("Max Salary", 0, 10000000, 10000000, step=10000)
            
            if st.button("🔍 Search", type="primary", use_container_width=True):
                # FIX: Use proper MongoDB range criteria and operators
                criteria = {
                    'credit_score': (min_credit, max_credit),
                    'monthly_salary': (min_salary, max_salary)
                }
                operators = {
                    'credit_score': 'between',  # Custom handling for ranges
                    'monthly_salary': 'between'
                }
                
                # Since our backend doesn't have 'between' built-in natively yet, 
                # it's cleaner to build the query dict directly here or adjust the backend.
                # The easiest fix without changing backend logic further:
                
                range_query = {
                    "credit_score": {"$gte": min_credit, "$lte": max_credit},
                    "monthly_salary": {"$gte": min_salary, "$lte": max_salary}
                }
                
                # Directly use run_aggregation or a modified filter
                # For simplicity, let's query using the existing filter_by_criteria for text, 
                # and apply the range directly:
                
                base_query = {"deleted": {"$ne": True}, **range_query}
                
                results = list(db.applicants.find(base_query).sort("created_at", -1).limit(100))
                st.session_state['search_results'] = [db._serialize(r) for r in results]
                
                if search_term:
                    text_results = db.search_applicants(search_term, search_fields)
                    existing_ids = {r['applicant_id'] for r in st.session_state['search_results']}
                    for r in text_results:
                        if r['applicant_id'] not in existing_ids:
                            # Only append if it also matches the numeric filters
                            if min_credit <= r.get('credit_score', 0) <= max_credit and \
                               min_salary <= r.get('monthly_salary', 0) <= max_salary:
                                st.session_state['search_results'].append(r)
        
        with col2:
            if 'search_results' in st.session_state:
                results = st.session_state['search_results']
                if results:
                    st.success(f"Found {len(results)} results")
                    df = pd.DataFrame(results)
                    display_cols = ['applicant_id', 'age', 'gender', 'education', 'employment_type',
                                   'company_type', 'monthly_salary', 'credit_score', 'foir_percentage', 'status']
                    display_cols = [c for c in display_cols if c in df.columns]
                    st.dataframe(df[display_cols].style.format({'monthly_salary': '₹{:,.0f}'}), 
                                use_container_width=True)
                else:
                    st.warning("No results")
            else:
                st.info("Enter search criteria and click Search")
    
    elif crud_op == "🗑️ Delete":
        st.subheader("Delete Applicant")
        
        st.warning("⚠️ Soft delete - records retained for audit")
        
        col1, col2 = st.columns(2)
        
        with col1:
            del_id = st.number_input("Applicant ID", 1, 999999, 1)
            if st.button("🔍 Preview"):
                st.session_state['delete_preview'] = db.get_applicant_by_id(del_id)
        
        with col2:
            if 'delete_preview' in st.session_state and st.session_state['delete_preview']:
                app = st.session_state['delete_preview']
                st.error("⚠️ To Delete:")
                st.json({
                    "ID": app['applicant_id'],
                    "Name": f"{app['employment_type']}@{app['company_type']}",
                    "Salary": f"₹{app['monthly_salary']:,}",
                    "Credit": app['credit_score'],
                    "Status": app['status']
                })
                
                if st.button("🗑️ Confirm Delete", type="secondary"):
                    if db.delete_applicant(del_id):
                        st.success(f"✅ Deleted {del_id}")
                        del st.session_state['delete_preview']
                        st.rerun()
            else:
                st.info("Enter ID and Preview")
        
        st.markdown("---")
        st.subheader("Purge Old Deleted Records")
        days = st.number_input("Days old", 1, 365, 30)
        if st.button("🧹 Purge"):
            result = db.purge_deleted(days)
            st.success(f"Purged {result['purged_count']} records")