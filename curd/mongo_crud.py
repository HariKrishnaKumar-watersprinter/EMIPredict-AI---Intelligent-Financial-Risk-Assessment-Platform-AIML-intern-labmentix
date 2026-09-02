"""
EMIPredict AI - MongoDB CRUD Operations
Customized for your dataset with 27 columns
"""

from pymongo import MongoClient
from pymongo.errors import (
    ConnectionFailure, OperationFailure, ServerSelectionTimeoutError
)
from bson.errors import InvalidId 
from pymongo.server_api import ServerApi
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from bson import ObjectId
import pandas as pd
import numpy as np
import json
import os
import logging
from functools import wraps

from curd.config import (
    MongoDBConfig, DATASET_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMNS,
    CATEGORICAL_FIELDS, STATUS_VALUES, FIELD_VALIDATION, REQUIRED_FIELDS,
    EXPENSE_COLUMNS
)

# Logging setup
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/mongo_crud.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def handle_mongo_errors(func):
    """Decorator for MongoDB error handling"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ConnectionFailure as e:
            logger.error(f"Connection Error: {e}")
            raise ConnectionError(f"MongoDB connection failed: {e}")
        except ServerSelectionTimeoutError as e:
            logger.error(f"Server Timeout: {e}")
            raise ConnectionError(f"MongoDB timeout: {e}")
        except OperationFailure as e:
            logger.error(f"Operation Failed: {e.details}")
            raise OperationError(f"Database operation failed: {e.details}")
        except InvalidId as e:
            logger.error(f"Invalid ID: {e}")
            raise ValueError(f"Invalid ID: {e}")
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            raise
    return wrapper


class OperationError(Exception):
    """Custom operation error"""
    pass


class EMIMongoDBManager:
    """
    MongoDB Manager for EMIPredict AI
    Customized for your 27-column dataset
    """
    
    def __init__(self, connection_uri: str = None, db_name: str = None):
        self.uri = connection_uri or MongoDBConfig.MONGO_URI
        self.db_name = db_name or MongoDBConfig.DATABASE_NAME
        self.client = None
        self.db = None
        self._connect()
        self._setup_collections()
        self._create_indexes()
        logger.info(f"MongoDB connected: {self.db_name}")
    
    def _connect(self):
        """Establish connection with pooling"""
        try:
            self.client = MongoClient(
                self.uri,
                server_api=ServerApi('1'),
                maxPoolSize=MongoDBConfig.MAX_POOL_SIZE,
                minPoolSize=MongoDBConfig.MIN_POOL_SIZE,
                serverSelectionTimeoutMS=MongoDBConfig.SERVER_SELECTION_TIMEOUT_MS,
                retryWrites=True,
                w="majority"
            )
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
        except Exception as e:
            logger.critical(f"MongoDB connection failed: {e}")
            raise ConnectionError(f"Connection failed: {e}")
    
    def _setup_collections(self):
        """Setup collections"""
        self.applicants = self.db[MongoDBConfig.APPLICANTS_COLLECTION]
        self.audit_logs = self.db[MongoDBConfig.AUDIT_LOG_COLLECTION]
        self.prediction_history = self.db[MongoDBConfig.PREDICTION_HISTORY_COLLECTION]
        
        # Initialize counter for auto-increment ID
        if self.db.counters.find_one({"_id": "applicant_id"}) is None:
            self.db.counters.insert_one({"_id": "applicant_id", "seq": 0})
    
    def _create_indexes(self):
        """Create indexes optimized for your dataset queries"""
        logger.info("Creating indexes...")
        
        # Single field indexes
        indexes = [
            ("applicant_id", True),           # Unique
            ("credit_score", False),          # Range queries
            ("monthly_salary", False),        # Range queries
            ("employment_type", False),       # Filter
            ("company_type", False),          # Filter
            ("education", False),             # Filter
            ("emi_eligibility", False),       # Filter
            ("status", False),                # Filter
            ("created_at", -1),               # Sort
            ("emi_scenario", False),          # Filter
            ("house_type", False),            # Filter
        ]
        
        for field, unique in indexes:
            try:
                self.applicants.create_index([(field, 1)], unique=unique, background=True)
            except OperationFailure:
                pass
        
        # Compound indexes
        compound_indexes = [
            ([("status", 1), ("created_at", -1)], False),
            ([("credit_score", 1), ("monthly_salary", 1)], False),
            ([("emi_eligibility", 1), ("credit_score", -1)], False),
            ([("employment_type", 1), ("company_type", 1)], False),
        ]
        
        for fields, unique in compound_indexes:
            try:
                self.applicants.create_index(fields, unique=unique, background=True)
            except OperationFailure:
                pass
        
        # Text index for search
        try:
            self.applicants.create_index(
                [("education", "text"), ("employment_type", "text"), ("company_type", "text")],
                background=True
            )
        except OperationFailure:
            pass
        
        # Audit log indexes with TTL
        self.audit_logs.create_index([("timestamp", -1)], background=True)
        self.audit_logs.create_index([("applicant_id", 1), ("timestamp", -1)], background=True)
        self.audit_logs.create_index([("operation", 1)], background=True)
        
        try:
            self.audit_logs.create_index(
                [("timestamp", 1)],
                expireAfterSeconds=MongoDBConfig.TTL_AUDIT_DAYS * 24 * 60 * 60,
                background=True
            )
        except OperationFailure:
            pass
        
        # Prediction history indexes
        self.prediction_history.create_index([("applicant_id", 1), ("timestamp", -1)], background=True)
        
        logger.info("Indexes created")
    
    def _get_next_id(self) -> int:
        """Get auto-incrementing applicant ID"""
        result = self.db.counters.find_one_and_update(
            {"_id": "applicant_id"},
            {"$inc": {"seq": 1}},
            return_document=True
        )
        return result['seq']
    
    def _validate_data(self, data: Dict) -> Tuple[bool, str]:
        """Validate applicant data against schema"""
        # Check required fields
        for field in REQUIRED_FIELDS:
            if field not in data or data[field] is None:
                return False, f"Missing required field: {field}"
        
        # Validate each field
        for field, rules in FIELD_VALIDATION.items():
            if field not in data or data[field] is None:
                continue
            
            value = data[field]
            
            # Type check
            expected_type = rules.get("type")
            if expected_type and not isinstance(value, expected_type):
                if isinstance(expected_type, tuple):
                    if not any(isinstance(value, t) for t in expected_type):
                        return False, f"{field} has invalid type"
                else:
                    return False, f"{field} has invalid type"
            
            # Allowed values check
            if "allowed" in rules and value not in rules["allowed"]:
                return False, f"{field}: '{value}' not allowed. Options: {rules['allowed']}"
            
            # Min check
            if "min" in rules and value < rules["min"]:
                return False, f"{field} must be >= {rules['min']}"
            
            # Max check
            if "max" in rules and value > rules["max"]:
                return False, f"{field} must be <= {rules['max']}"
        
        return True, "Valid"
    
    def _calculate_derived_fields(self, data: Dict) -> Dict:
        """Calculate derived fields from your dataset"""
        monthly_salary = data.get('monthly_salary', 0)
        
        # Total monthly expenses
        total_expenses = (
            data.get('monthly_rent', 0) +
            data.get('school_fees', 0) +
            data.get('college_fees', 0) +
            data.get('travel_expenses', 0) +
            data.get('groceries_utilities', 0) +
            data.get('other_monthly_expenses', 0)
        )
        
        # Debt to income ratio
        current_emi = data.get('current_emi_amount', 0)
        debt_to_income = round(current_emi / monthly_salary, 4) if monthly_salary > 0 else 0
        
        # Expense to income ratio
        expense_to_income = round(total_expenses / monthly_salary, 4) if monthly_salary > 0 else 0
        
        # Total obligations
        total_obligations = total_expenses + current_emi
        
        # Net monthly income
        net_income = monthly_salary - total_obligations
        
        # FOIR (Fixed Obligation to Income Ratio) - used in lending
        foir = round(total_obligations / monthly_salary * 100, 2) if monthly_salary > 0 else 0
        
        # Savings potential
        savings_potential = max(0, net_income)
        
        # Financial cushion (bank balance + emergency fund)
        financial_cushion = data.get('bank_balance', 0) + data.get('emergency_fund', 0)
        
        # Months of emergency fund
        months_emergency = round(
            data.get('emergency_fund', 0) / total_obligations, 2
        ) if total_obligations > 0 else 0
        
        # EMI burden (if approved)
        requested_amount = data.get('requested_amount', 0)
        requested_tenure = data.get('requested_tenure', 1)
        estimated_emi = round(requested_amount / requested_tenure, 2) if requested_tenure > 0 else 0
        
        # Proposed FOIR after new EMI
        proposed_foir = round((total_obligations + estimated_emi) / monthly_salary * 100, 2) if monthly_salary > 0 else 0
        
        return {
            "total_monthly_expenses": total_expenses,
            "debt_to_income_ratio": debt_to_income,
            "expense_to_income_ratio": expense_to_income,
            "total_obligations": total_obligations,
            "net_monthly_income": net_income,
            "foir_percentage": foir,
            "savings_potential": savings_potential,
            "financial_cushion": financial_cushion,
            "months_of_emergency_fund": months_emergency,
            "estimated_emi": estimated_emi,
            "proposed_foir_percentage": proposed_foir
        }
    
    def _log_operation(self, applicant_id: int, operation: str,
                       old_values: Dict = None, new_values: Dict = None,
                       performed_by: str = 'System'):
        """Log to audit collection"""
        try:
            self.audit_logs.insert_one({
                "applicant_id": applicant_id,
                "operation": operation,
                "old_values": old_values,
                "new_values": new_values,
                "performed_by": performed_by,
                "timestamp": datetime.utcnow()
            })
        except Exception as e:
            logger.error(f"Audit log failed: {e}")
    
    def _serialize(self, doc: Dict) -> Dict:
        """Convert MongoDB document to JSON-serializable"""
        if doc is None:
            return None
        
        serialized = {}
        for key, value in doc.items():
            if isinstance(value, ObjectId):
                serialized[key] = str(value)
            elif isinstance(value, datetime):
                serialized[key] = value.isoformat()
            elif isinstance(value, (np.int64, np.int32)):
                serialized[key] = int(value)
            elif isinstance(value, (np.float64, np.float32)):
                serialized[key] = float(value)
            else:
                serialized[key] = value
        return serialized
    
    # =========================================================================
    # CREATE OPERATIONS
    # =========================================================================
    
    @handle_mongo_errors
    def create_applicant(self, applicant_data: Dict,
                         performed_by: str = 'System') -> Dict:
        """
        Create new applicant with your 27-column schema
        
        Required: age, gender, marital_status, education, monthly_salary,
                  employment_type, credit_score, requested_amount, requested_tenure
                  
        Optional: years_of_employment, company_type, house_type, monthly_rent,
                  family_size, dependents, school_fees, college_fees,
                  travel_expenses, groceries_utilities, other_monthly_expenses,
                  existing_loans, current_emi_amount, bank_balance, emergency_fund,
                  emi_scenario, emi_eligibility, max_monthly_emi
        """
        for key, value in applicant_data.items():
            if hasattr(value, 'item'):  # Checks if it's a Numpy type
                applicant_data[key] = value.item()
        # Validate
        is_valid, message = self._validate_data(applicant_data)
        if not is_valid:
            raise ValueError(message)
        
        # Calculate derived fields
        derived = self._calculate_derived_fields(applicant_data)
        
        # Get auto-increment ID
        applicant_id = self._get_next_id()
        now = datetime.utcnow()
        
        # Build document with all 27 columns + derived fields + metadata
        document = {
            # Auto-generated
            "applicant_id": applicant_id,
            "created_at": now,
            "updated_at": now,
            "version": 1,
            
            # === YOUR 27 DATASET COLUMNS ===
            # Demographic (6)
            "age": int(applicant_data['age']),
            "gender": applicant_data['gender'],
            "marital_status": applicant_data['marital_status'],
            "education": applicant_data['education'],
            "family_size": int(applicant_data.get('family_size', 0)),
            "dependents": int(applicant_data.get('dependents', 0)),
            
            # Employment (4)
            "monthly_salary": float(applicant_data['monthly_salary']),
            "employment_type": applicant_data['employment_type'],
            "years_of_employment": float(applicant_data.get('years_of_employment', 0)),
            "company_type": applicant_data.get('company_type'),
            
            # Housing (2)
            "house_type": applicant_data.get('house_type'),
            "monthly_rent": float(applicant_data.get('monthly_rent', 0)),
            
            # Expenses (5)
            "school_fees": float(applicant_data.get('school_fees', 0)),
            "college_fees": float(applicant_data.get('college_fees', 0)),
            "travel_expenses": float(applicant_data.get('travel_expenses', 0)),
            "groceries_utilities": float(applicant_data.get('groceries_utilities', 0)),
            "other_monthly_expenses": float(applicant_data.get('other_monthly_expenses', 0)),
            
            # Financial (5)
            "existing_loans": applicant_data.get('existing_loans'),
            "current_emi_amount": float(applicant_data.get('current_emi_amount', 0)),
            "credit_score": int(applicant_data['credit_score']),
            "bank_balance": float(applicant_data.get('bank_balance', 0)),
            "emergency_fund": float(applicant_data.get('emergency_fund', 0)),
            
            # Loan Request (3)
            "emi_scenario": applicant_data.get('emi_scenario'),
            "requested_amount": float(applicant_data['requested_amount']),
            "requested_tenure": int(applicant_data['requested_tenure']),
            
            # Targets (2)
            "emi_eligibility": applicant_data.get('emi_eligibility'),
            "max_monthly_emi": float(applicant_data['max_monthly_emi']) if applicant_data.get('max_monthly_emi') else None,
            
            # === DERIVED FIELDS ===
            **derived,
            
            # === ML PREDICTION FIELDS ===
            "predicted_eligibility": applicant_data.get('predicted_eligibility'),
            "predicted_probability": applicant_data.get('predicted_probability'),
            "predicted_max_emi": float(applicant_data['max_monthly_emi']) if applicant_data.get('max_monthly_emi') else None,
            "risk_category": None,
            "prediction_timestamp": None,
            
            # === WORKFLOW FIELDS ===
            "status": applicant_data.get('status', 'Pending'),
            "review_notes": None,
            "reviewed_by": None,
            "review_timestamp": None,
            
            # === FLAGS ===
            "deleted": False
        }
        
        # Insert
        result = self.applicants.insert_one(document)
        document['_id'] = result.inserted_id
        
        # Audit log
        self._log_operation(
            applicant_id=applicant_id,
            operation="CREATE",
            new_values=self._serialize(document),
            performed_by=performed_by
        )
        
        logger.info(f"Created applicant {applicant_id}")
        return self._serialize(document)
    
    @handle_mongo_errors
    def create_bulk_applicants(self, applicants_data: List[Dict],
                                performed_by: str = 'System') -> Dict:
        """Create multiple applicants"""
        successful = []
        failed = []
        
        for data in applicants_data:
            try:
                result = self.create_applicant(data, performed_by)
                successful.append(result['applicant_id'])
            except Exception as e:
                failed.append({"error": str(e), "data": {k: v for k, v in data.items() if k in DATASET_COLUMNS}})
        
        logger.info(f"Bulk create: {len(successful)} success, {len(failed)} failed")
        return {
            "successful_count": len(successful),
            "failed_count": len(failed),
            "successful_ids": successful,
            "failed_details": failed
        }
    
    @handle_mongo_errors
    def import_from_csv(self, csv_path: str, performed_by: str = 'System',
                         batch_size: int = 1000) -> Dict:
        """
        Import your CSV dataset into MongoDB
        
        Handles all 27 columns from your CSV automatically
        """
        df = pd.read_csv(csv_path)
        
        # Validate columns
        missing_cols = set(DATASET_COLUMNS) - set(df.columns)
        if missing_cols:
            logger.warning(f"Missing columns in CSV: {missing_cols}")
        
        # Convert DataFrame to list of dicts
        applicants_list = []
        skipped = 0
        
        for idx, row in df.iterrows():
            data = {}
            for col in DATASET_COLUMNS:
                if col in row.index and pd.notna(row[col]):
                    data[col] = row[col]
                else:
                    # Set defaults
                    if col in EXPENSE_COLUMNS + ["monthly_rent", "current_emi_amount", 
                                                  "bank_balance", "emergency_fund"]:
                        data[col] = 0
                    elif col in ["family_size", "dependents", "existing_loans"]:
                        data[col] = 0
                    elif col in ["years_of_employment"]:
                        data[col] = 0
                    elif col == "emi_eligibility":
                        data[col] = None
                    elif col == "max_monthly_emi":
                        data[col] = None
                    else:
                        data[col] = None
            
            if len([v for v in data.values() if v is not None]) >= 5:
                applicants_list.append(data)
            else:
                skipped += 1
        
        # Process in batches
        all_successful = []
        all_failed = []
        
        for i in range(0, len(applicants_list), batch_size):
            batch = applicants_list[i:i+batch_size]
            result = self.create_bulk_applicants(batch, performed_by)
            all_successful.extend(result['successful_ids'])
            all_failed.extend(result['failed_details'])
            logger.info(f"Batch {i//batch_size + 1}: {len(result['successful_ids'])} imported")
        
        return {
            "total_in_csv": len(df),
            "successful_count": len(all_successful),
            "failed_count": len(all_failed) + skipped,
            "successful_ids": all_successful,
            "failed_details": all_failed[:10],  # First 10 failures
            "skipped_rows": skipped
        }
    
    # =========================================================================
    # READ OPERATIONS
    # =========================================================================
    
    @handle_mongo_errors
    def get_applicant_by_id(self, applicant_id: int) -> Optional[Dict]:
        """Get applicant by ID"""
        doc = self.applicants.find_one({
            "applicant_id": applicant_id,
            "deleted": {"$ne": True}
        })
        return self._serialize(doc)
    
    @handle_mongo_errors
    def get_all_applicants(self, limit: int = 100, offset: int = 0,
                           status_filter: str = None,
                           eligibility_filter: int = None,
                           sort_by: str = "created_at",
                           sort_order: int = -1) -> List[Dict]:
        """Get applicants with filtering and pagination"""
        query = {"deleted": {"$ne": True}}
        
        if status_filter:
            query["status"] = status_filter
        if eligibility_filter is not None:
            query["emi_eligibility"] = eligibility_filter
        
        cursor = self.applicants.find(query).sort(sort_by, sort_order).skip(offset).limit(limit)
        return [self._serialize(doc) for doc in cursor]
    
    @handle_mongo_errors
    def get_applicants_dataframe(self, status_filter: str = None,
                                  query: Dict = None,
                                  include_deleted: bool = False) -> pd.DataFrame:
        """Get applicants as DataFrame"""
        if query is None:
            query = {}
        if not include_deleted:
            query["deleted"] = {"$ne": True}
        if status_filter:
            query["status"] = status_filter
        
        cursor = self.applicants.find(query).sort("created_at", -1)
        df = pd.DataFrame(list(cursor))
        
        if '_id' in df.columns:
            df['_id'] = df['_id'].astype(str)
        
        return df
    
    @handle_mongo_errors
    def search_applicants(self, search_term: str, 
                          fields: List[str] = None,
                          limit: int = 50) -> List[Dict]:
        """Search across specified fields using regex"""
        if fields is None:
            fields = ['education', 'employment_type', 'company_type', 'house_type', 'emi_scenario']
        
        regex = {"$regex": search_term, "$options": "i"}
        query = {"$or": [{field: regex} for field in fields], "deleted": {"$ne": True}}
        
        cursor = self.applicants.find(query).limit(limit)
        return [self._serialize(doc) for doc in cursor]
    
    @handle_mongo_errors
    def filter_by_criteria(self, criteria: Dict,
                            operators: Dict = None,
                            limit: int = 100) -> List[Dict]:
        """
        Filter applicants with custom operators
        
        Example:
            criteria = {'credit_score': 700, 'monthly_salary': 50000}
            operators = {'credit_score': '>=', 'monthly_salary': '>='}
        """
        if operators is None:
            operators = {}
        
        mongo_query = {"deleted": {"$ne": True}}
        
        for field, value in criteria.items():
            op = operators.get(field, '=')
            if op == '=':
                mongo_query[field] = value
            elif op == 'in':
                mongo_query[field] = {"$in": value}
            elif op == 'nin':
                mongo_query[field] = {"$nin": value}
            elif op == 'ne':
                mongo_query[field] = {"$ne": value}
            elif op == 'exists':
                mongo_query[field] = {"$exists": value}
            else:
                mongo_query[field] = {f"${op}": value}
        
        cursor = self.applicants.find(mongo_query).sort("created_at", -1).limit(limit)
        return [self._serialize(doc) for doc in cursor]
    
    @handle_mongo_errors
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics using aggregation"""
        pipeline = [
            {"$match": {"deleted": {"$ne": True}}},
            {"$facet": {
                "total_count": [{"$count": "count"}],
                "status_dist": [
                    {"$group": {"_id": "$status", "count": {"$sum": 1}}}
                ],
                "eligibility_dist": [
                    {"$group": {"_id": "$emi_eligibility", "count": {"$sum": 1}}}
                ],
                "averages": [
                    {"$group": {
                        "_id": None,
                        "avg_salary": {"$avg": "$monthly_salary"},
                        "avg_credit": {"$avg": "$credit_score"},
                        "avg_requested": {"$avg": "$requested_amount"},
                        "avg_max_emi": {"$avg": "$max_monthly_emi"},
                        "avg_foir": {"$avg": "$foir_percentage"},
                        "avg_dti": {"$avg": "$debt_to_income_ratio"},
                        "total_requested": {"$sum": "$requested_amount"},
                        "total_bank_balance": {"$sum": "$bank_balance"},
                        "total_emergency_fund": {"$sum": "$emergency_fund"}
                    }}
                ],
                "employment_dist": [
                    {"$group": {"_id": "$employment_type", "count": {"$sum": 1},
                               "avg_salary": {"$avg": "$monthly_salary"}}}
                ],
                "company_dist": [
                    {"$group": {"_id": "$company_type", "count": {"$sum": 1}}}
                ],
                "education_dist": [
                    {"$group": {"_id": "$education", "count": {"$sum": 1}}}
                ],
                "credit_buckets": [
                    {"$bucket": {
                        "groupBy": "$credit_score",
                        "boundaries": [300, 550, 650, 750, 850, 901],
                        "default": "Other",
                        "output": {"count": {"$sum": 1}}
                    }}
                ],
                "salary_ranges": [
                    {"$bucket": {
                        "groupBy": "$monthly_salary",
                        "boundaries": [0, 25000, 50000, 75000, 100000, 200000, float('inf')],
                        "default": "Other",
                        "output": {"count": {"$sum": 1}}
                    }}
                ],
                "scenario_dist": [
                    {"$group": {"_id": "$emi_scenario", "count": {"$sum": 1}}}
                ],
                "house_dist": [
                    {"$group": {"_id": "$house_type", "count": {"$sum": 1}}}
                ],
                "risk_dist": [
                    {"$match": {"risk_category": {"$ne": None}}},
                    {"$group": {"_id": "$risk_category", "count": {"$sum": 1}}}
                ]
            }}
        ]
        
        results = list(self.applicants.aggregate(pipeline))[0]
        
        stats = {
            "total_applicants": results["total_count"][0]["count"] if results["total_count"] else 0,
            "status_distribution": {r["_id"]: r["count"] for r in results["status_dist"]},
            "eligibility_distribution": {
                "Eligible" if r["_id"] == 1 else "Not Eligible": r["count"]
                for r in results["eligibility_dist"] if r["_id"] is not None
            },
            "employment_distribution": {r["_id"]: r["count"] for r in results["employment_dist"]},
            "company_distribution": {r["_id"]: r["count"] for r in results["company_dist"]},
            "education_distribution": {r["_id"]: r["count"] for r in results["education_dist"]},
            "scenario_distribution": {r["_id"]: r["count"] for r in results["scenario_dist"] if r["_id"]},
            "house_distribution": {r["_id"]: r["count"] for r in results["house_dist"] if r["_id"]},
            "risk_distribution": {r["_id"]: r["count"] for r in results["risk_dist"]},
            "credit_score_buckets": {
                self._format_credit_bucket(r["_id"]): r["count"]
                for r in results["credit_buckets"]
            },
            "salary_ranges": {
                self._format_salary_range(r["_id"]): r["count"]
                for r in results["salary_ranges"]
            }
        }
        
        if results["averages"]:
            avg = results["averages"][0]
            stats["averages"] = {
                "avg_salary": round(avg.get("avg_salary", 0), 2),
                "avg_credit_score": round(avg.get("avg_credit", 0), 2),
                "avg_requested_amount": round(avg.get("avg_requested", 0), 2),
                "avg_max_emi": round(avg.get("avg_max_emi", 0), 2),
                "avg_foir": round(avg.get("avg_foir", 0), 2),
                "avg_dti": round(avg.get("avg_dti", 0), 4),
                "total_requested_amount": round(avg.get("total_requested", 0), 2),
                "total_bank_balance": round(avg.get("total_bank_balance", 0), 2),
                "total_emergency_fund": round(avg.get("total_emergency_fund", 0), 2)
            }
        
        # Calculate eligibility rate
        eligible = stats["eligibility_distribution"].get("Eligible", 0)
        not_eligible = stats["eligibility_distribution"].get("Not Eligible", 0)
        total = eligible + not_eligible
        stats["eligibility_rate"] = round(eligible / total * 100, 2) if total > 0 else 0
        
        return stats
    
    def _format_credit_bucket(self, value):
        buckets = {300: "300-549", 550: "550-649", 650: "650-749", 750: "750-850", "Other": "Other"}
        return buckets.get(value, str(value))
    
    def _format_salary_range(self, value):
        ranges = {
            0: "<25K", 25000: "25K-50K", 50000: "50K-75K",
            75000: "75K-1L", 100000: "1L-2L", 200000: ">2L", "Other": "Other"
        }
        return ranges.get(value, str(value))
    
    @handle_mongo_errors
    def get_applicant_count(self, query: Dict = None) -> int:
        """Get count"""
        if query is None:
            query = {"deleted": {"$ne": True}}
        return self.applicants.count_documents(query)
    
    @handle_mongo_errors
    def get_audit_log(self, applicant_id: int = None,
                      operation: str = None, limit: int = 100) -> List[Dict]:
        """Get audit log"""
        query = {}
        if applicant_id:
            query["applicant_id"] = applicant_id
        if operation:
            query["operation"] = operation
        
        cursor = self.audit_logs.find(query).sort("timestamp", -1).limit(limit)
        return [self._serialize(doc) for doc in cursor]
    
    @handle_mongo_errors
    def get_prediction_history(self, applicant_id: int = None,
                                limit: int = 100) -> List[Dict]:
        """Get prediction history"""
        query = {}
        if applicant_id:
            query["applicant_id"] = applicant_id
        
        cursor = self.prediction_history.find(query).sort("timestamp", -1).limit(limit)
        return [self._serialize(doc) for doc in cursor]
    
    @handle_mongo_errors
    def run_aggregation(self, pipeline: List[Dict]) -> List[Dict]:
        """Run custom aggregation pipeline"""
        results = list(self.applicants.aggregate(pipeline))
        return [self._serialize(doc) for doc in results]
    
    @handle_mongo_errors
    def get_distinct_values(self, field: str) -> List:
        """Get distinct values for a field"""
        return self.applicants.distinct(field, {"deleted": {"$ne": True}})
    
    # =========================================================================
    # UPDATE OPERATIONS
    # =========================================================================
    
    @handle_mongo_errors
    def update_applicant(self, applicant_id: int, update_data: Dict,
                         performed_by: str = 'System') -> bool:
        """Update applicant fields"""
        old_doc = self.applicants.find_one({"applicant_id": applicant_id, "deleted": {"$ne": True}})
        if not old_doc:
            return False
        
        # Recalculate derived fields if financial data changed
        financial_fields = ['monthly_salary', 'monthly_rent', 'school_fees', 'college_fees',
                           'travel_expenses', 'groceries_utilities', 'other_monthly_expenses',
                           'current_emi_amount', 'requested_amount', 'requested_tenure']
        
        if any(f in update_data for f in financial_fields):
            merged = {**old_doc, **update_data}
            derived = self._calculate_derived_fields(merged)
            update_data.update(derived)
        
        # Protected fields
        for field in ['_id', 'applicant_id', 'created_at', 'version']:
            update_data.pop(field, None)
        
        update_data['updated_at'] = datetime.utcnow()
        update_data['version'] = old_doc.get('version', 0) + 1
        
        result = self.applicants.update_one(
            {"applicant_id": applicant_id},
            {"$set": update_data}
        )
        
        if result.modified_count > 0:
            self._log_operation(
                applicant_id=applicant_id,
                operation="UPDATE",
                old_values=self._serialize({k: old_doc.get(k) for k in update_data.keys()}),
                new_values=self._serialize(update_data),
                performed_by=performed_by
            )
            logger.info(f"Updated applicant {applicant_id}")
            return True
        
        return False
    
    @handle_mongo_errors
    def update_prediction(self, applicant_id: int, eligibility: int,
                          probability: float, max_emi: float,
                          model_name: str, prediction_time_ms: float,
                          performed_by: str = 'System') -> bool:
        """Update prediction results"""
        # Risk categorization
        if eligibility == 1:
            if probability > 0.8:
                risk = 'Low Risk'
            elif probability > 0.6:
                risk = 'Medium Risk'
            else:
                risk = 'High Risk'
        else:
            risk = 'Very High Risk'
        
        now = datetime.utcnow()
        
        update_data = {
            "predicted_eligibility": eligibility,
            "predicted_probability": round(probability, 4),
            "predicted_max_emi": round(max_emi, 2),
            "risk_category": risk,
            "prediction_timestamp": now
        }
        
        success = self.update_applicant(applicant_id, update_data, performed_by)
        
        if success:
            self.prediction_history.insert_one({
                "applicant_id": applicant_id,
                "model_used": model_name,
                "eligibility_prediction": eligibility,
                "eligibility_probability": round(probability, 4),
                "max_emi_prediction": round(max_emi, 2),
                "risk_category": risk,
                "prediction_time_ms": prediction_time_ms,
                "timestamp": now
            })
        
        return success
    
    @handle_mongo_errors
    def update_status(self, applicant_id: int, status: str,
                      review_notes: str = None, reviewed_by: str = 'System') -> bool:
        """Update application status"""
        return self.update_applicant(applicant_id, {
            "status": status,
            "review_notes": review_notes,
            "reviewed_by": reviewed_by,
            "review_timestamp": datetime.utcnow()
        }, reviewed_by)
    
    @handle_mongo_errors
    def bulk_update_status(self, applicant_ids: List[int], status: str,
                           reviewed_by: str = 'System') -> Dict:
        """Bulk status update"""
        now = datetime.utcnow()
        
        result = self.applicants.update_many(
            {"applicant_id": {"$in": applicant_ids}, "deleted": {"$ne": True}},
            {"$set": {
                "status": status,
                "reviewed_by": reviewed_by,
                "review_timestamp": now,
                "updated_at": now
            }}
        )
        
        for aid in applicant_ids:
            self._log_operation(aid, "BULK_STATUS_UPDATE", 
                               new_values={"status": status}, performed_by=reviewed_by)
        
        return {"matched_count": result.matched_count, "modified_count": result.modified_count}
    
    # =========================================================================
    # DELETE OPERATIONS
    # =========================================================================
    
    @handle_mongo_errors
    def delete_applicant(self, applicant_id: int,
                         performed_by: str = 'System') -> bool:
        """Soft delete applicant"""
        old_doc = self.applicants.find_one({"applicant_id": applicant_id, "deleted": {"$ne": True}})
        if not old_doc:
            return False
        
        result = self.applicants.update_one(
            {"applicant_id": applicant_id},
            {"$set": {
                "status": "Cancelled",
                "deleted": True,
                "deleted_at": datetime.utcnow(),
                "deleted_by": performed_by,
                "updated_at": datetime.utcnow()
            }}
        )
        
        if result.modified_count > 0:
            self._log_operation(applicant_id, "DELETE", 
                               old_values=self._serialize(old_doc), performed_by=performed_by)
            logger.info(f"Soft deleted applicant {applicant_id}")
            return True
        
        return False
    
    @handle_mongo_errors
    def hard_delete_applicant(self, applicant_id: int,
                               performed_by: str = 'System') -> bool:
        """Permanently delete"""
        self.prediction_history.delete_many({"applicant_id": applicant_id})
        self.audit_logs.delete_many({"applicant_id": applicant_id})
        
        result = self.applicants.delete_one({"applicant_id": applicant_id})
        if result.deleted_count > 0:
            logger.warning(f"Hard deleted applicant {applicant_id}")
            return True
        return False
    
    @handle_mongo_errors
    def delete_by_criteria(self, criteria: Dict,
                           operators: Dict = None,
                           performed_by: str = 'System') -> Dict:
        """Soft delete by criteria"""
        if operators is None:
            operators = {}
        
        mongo_query = {"deleted": {"$ne": True}}
        for field, value in criteria.items():
            op = operators.get(field, '=')
            if op == '=':
                mongo_query[field] = value
            else:
                mongo_query[field] = {f"${op}": value}
        
        now = datetime.utcnow()
        result = self.applicants.update_many(
            mongo_query,
            {"$set": {
                "status": "Cancelled",
                "deleted": True,
                "deleted_at": now,
                "deleted_by": performed_by,
                "updated_at": now
            }}
        )
        
        logger.warning(f"Bulk deleted {result.modified_count} by criteria")
        return {"modified_count": result.modified_count}
    
    @handle_mongo_errors
    def purge_deleted(self, days_old: int = 30) -> Dict:
        """Purge soft-deleted records older than X days"""
        cutoff = datetime.utcnow() - timedelta(days=days_old)
        query = {"deleted": True, "deleted_at": {"$lt": cutoff}}
        
        ids = [d['applicant_id'] for d in self.applicants.find(query, {"applicant_id": 1})]
        self.prediction_history.delete_many({"applicant_id": {"$in": ids}})
        self.audit_logs.delete_many({"applicant_id": {"$in": ids}})
        
        result = self.applicants.delete_many(query)
        logger.info(f"Purged {result.deleted_count} records")
        return {"purged_count": result.deleted_count}
    
    # =========================================================================
    # EXPORT
    # =========================================================================
    
    @handle_mongo_errors
    def export_to_csv(self, output_path: str, query: Dict = None,
                      status_filter: str = None) -> int:
        """Export to CSV with your 27 columns"""
        df = self.get_applicants_dataframe(status_filter, query)
        
        # Export only original 27 columns
        export_cols = [c for c in DATASET_COLUMNS if c in df.columns]
        df[export_cols].to_csv(output_path, index=False)
        
        logger.info(f"Exported {len(df)} records to {output_path}")
        return len(df)
    
    @handle_mongo_errors
    def export_to_json(self, output_path: str, query: Dict = None,
                       status_filter: str = None) -> int:
        """Export to JSON"""
        df = self.get_applicants_dataframe(status_filter, query)
        export_cols = [c for c in DATASET_COLUMNS if c in df.columns]
        df[export_cols].to_json(output_path, orient='records', indent=2, default_handler=str)
        
        logger.info(f"Exported {len(df)} records to {output_path}")
        return len(df)
    
    # =========================================================================
    # UTILITY
    # =========================================================================
    
    def applicant_exists(self, applicant_id: int) -> bool:
        return self.applicants.count_documents({"applicant_id": applicant_id, "deleted": {"$ne": True}}) > 0
    
    def health_check(self) -> Dict:
        try:
            start = time.time()
            self.client.admin.command('ping')
            ping_ms = (time.time() - start) * 1000
            
            db_stats = self.db.command("dbstats")
            indexes = list(self.applicants.list_indexes())
            
            return {
                "status": "healthy",
                "ping_time_ms": round(ping_ms, 2),
                "database": self.db_name,
                "collections": db_stats.get("collections", 0),
                "data_size_mb": round(db_stats.get("dataSize", 0) / (1024*1024), 2),
                "storage_size_mb": round(db_stats.get("storageSize", 0) / (1024*1024), 2),
                "indexes_count": len(indexes),
                "applicant_count": self.get_applicant_count()
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    def close(self):
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


import time

# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_crud():
    """Demonstrate CRUD with your dataset"""
    
    print("="*80)
    print("EMIPredict AI - MongoDB CRUD with Your Dataset")
    print(f"Dataset Columns: {len(DATASET_COLUMNS)}")
    print("="*80)
    
    db = EMIMongoDBManager()
    
    # Health check
    print("\n--- Health Check ---")
    health = db.health_check()
    print(f"Status: {health['status']}")
    print(f"Applicants: {health['applicant_count']}")
    print(f"DB Size: {health['data_size_mb']} MB")
    
    # =========================================================================
    # CREATE
    # =========================================================================
    print("\n" + "-"*60)
    print("1. CREATE")
    print("-"*60)
    
    # Create single applicant with all your columns
    print("\n>>> Creating applicant with 27 columns...")
    new_applicant = {
        # Demographic (6)
        "age": 32,
        "gender": "Male",
        "marital_status": "Married",
        "education": "Master",
        "family_size": 4,
        "dependents": 2,
        
        # Employment (4)
        "monthly_salary": 85000,
        "employment_type": "Salaried",
        "years_of_employment": 7,
        "company_type": "MNC",
        
        # Housing (2)
        "house_type": "Rented",
        "monthly_rent": 15000,
        
        # Expenses (5)
        "school_fees": 8000,
        "college_fees": 0,
        "travel_expenses": 5000,
        "groceries_utilities": 12000,
        "other_monthly_expenses": 5000,
        
        # Financial (5)
        "existing_loans": 2,
        "current_emi_amount": 18000,
        "credit_score": 745,
        "bank_balance": 250000,
        "emergency_fund": 150000,
        
        # Loan Request (3)
        "emi_scenario": "New Loan",
        "requested_amount": 750000,
        "requested_tenure": 60,
        
        # Targets (2)
        "emi_eligibility": 1,
        "max_monthly_emi": 32000
    }
    
    created = db.create_applicant(new_applicant)
    app_id = created['applicant_id']
    print(f"✅ Created applicant ID: {app_id}")
    print(f"   Derived fields calculated:")
    print(f"   - Total Expenses: ₹{created['total_monthly_expenses']:,.0f}")
    print(f"   - FOIR: {created['foir_percentage']}%")
    print(f"   - DTI Ratio: {created['debt_to_income_ratio']}")
    print(f"   - Net Income: ₹{created['net_monthly_income']:,.0f}")
    print(f"   - Estimated EMI: ₹{created['estimated_emi']:,.0f}")
    print(f"   - Proposed FOIR: {created['proposed_foir_percentage']}%")
    
    # Create more applicants
    print("\n>>> Creating bulk applicants...")
    bulk_data = [
        {
            "age": 28, "gender": "Female", "marital_status": "Single", "education": "Bachelor",
            "family_size": 1, "dependents": 0, "monthly_salary": 55000,
            "employment_type": "Salaried", "years_of_employment": 3, "company_type": "Startup",
            "house_type": "Rented", "monthly_rent": 8000,
            "school_fees": 0, "college_fees": 5000, "travel_expenses": 3000,
            "groceries_utilities": 8000, "other_monthly_expenses": 4000,
            "existing_loans": 1, "current_emi_amount": 8000,
            "credit_score": 680, "bank_balance": 100000, "emergency_fund": 50000,
            "emi_scenario": "New Loan", "requested_amount": 300000, "requested_tenure": 36,
            "emi_eligibility": 1, "max_monthly_emi": 18000
        },
        {
            "age": 45, "gender": "Male", "marital_status": "Married", "education": "PhD",
            "family_size": 5, "dependents": 3, "monthly_salary": 200000,
            "employment_type": "Business", "years_of_employment": 18, "company_type": "Private",
            "house_type": "Owned", "monthly_rent": 0,
            "school_fees": 25000, "college_fees": 15000, "travel_expenses": 10000,
            "groceries_utilities": 20000, "other_monthly_expenses": 15000,
            "existing_loans": 4, "current_emi_amount": 55000,
            "credit_score": 810, "bank_balance": 5000000, "emergency_fund": 1000000,
            "emi_scenario": "Top-Up", "requested_amount": 2000000, "requested_tenure": 120,
            "emi_eligibility": 1, "max_monthly_emi": 65000
        },
        {
            "age": 24, "gender": "Male", "marital_status": "Single", "education": "High School",
            "family_size": 3, "dependents": 1, "monthly_salary": 22000,
            "employment_type": "Freelancer", "years_of_employment": 1, "company_type": "Other",
            "house_type": "Family-Owned", "monthly_rent": 0,
            "school_fees": 0, "college_fees": 0, "travel_expenses": 2000,
            "groceries_utilities": 6000, "other_monthly_expenses": 3000,
            "existing_loans": 0, "current_emi_amount": 0,
            "credit_score": 510, "bank_balance": 15000, "emergency_fund": 5000,
            "emi_scenario": "New Loan", "requested_amount": 100000, "requested_tenure": 24,
            "emi_eligibility": 0, "max_monthly_emi": 0
        }
    ]
    
    bulk_result = db.create_bulk_applicants(bulk_data)
    print(f"✅ Bulk: {bulk_result['successful_count']} created, {bulk_result['failed_count']} failed")
    
    # =========================================================================
    # READ
    # =========================================================================
    print("\n" + "-"*60)
    print("2. READ")
    print("-"*60)
    
    # Read by ID
    print("\n>>> Read by ID...")
    applicant = db.get_applicant_by_id(app_id)
    print(f"✅ Found: Age={applicant['age']}, Salary=₹{applicant['monthly_salary']:,}, "
          f"Credit={applicant['credit_score']}, Company={applicant['company_type']}")
    print(f"   Expenses: Rent=₹{applicant['monthly_rent']:,}, School=₹{applicant['school_fees']:,}, "
          f"Travel=₹{applicant['travel_expenses']:,}")
    
    # Read all
    print("\n>>> Read all (limit 5)...")
    all_apps = db.get_all_applicants(limit=5)
    for a in all_apps:
        print(f"   ID:{a['applicant_id']} | {a['employment_type']}@{a['company_type']} | "
              f"₹{a['monthly_salary']:,} | Credit:{a['credit_score']} | {a['status']}")
    
    # Filter by criteria
    print("\n>>> Filter: Credit >= 700 AND Salary >= 50000...")
    filtered = db.filter_by_criteria(
        {'credit_score': 700, 'monthly_salary': 50000},
        {'credit_score': '>=', 'monthly_salary': '>='}
    )
    print(f"✅ Found {len(filtered)} applicants")
    
    # Search
    print("\n>>> Search: 'MNC'...")
    search_results = db.search_applicants('MNC', ['company_type'])
    print(f"✅ Found {len(search_results)} at MNC companies")
    
    # Get statistics
    print("\n>>> Statistics (Aggregation Pipeline)...")
    stats = db.get_statistics()
    print(f"   Total: {stats['total_applicants']}")
    print(f"   Eligibility Rate: {stats['eligibility_rate']}%")
    print(f"   Status: {stats['status_distribution']}")
    print(f"   Employment: {stats['employment_distribution']}")
    print(f"   Company Types: {stats['company_distribution']}")
    print(f"   Education: {stats['education_distribution']}")
    print(f"   EMI Scenarios: {stats['scenario_distribution']}")
    print(f"   House Types: {stats['house_distribution']}")
    print(f"   Credit Buckets: {stats['credit_score_buckets']}")
    print(f"   Salary Ranges: {stats['salary_ranges']}")
    if stats.get('averages'):
        print(f"   Avg Salary: ₹{stats['averages']['avg_salary']:,.0f}")
        print(f"   Avg Credit: {stats['averages']['avg_credit_score']:.0f}")
        print(f"   Avg FOIR: {stats['averages']['avg_foir']}%")
        print(f"   Avg DTI: {stats['averages']['avg_dti']}")
    
    # Custom aggregation
    print("\n>>> Custom Aggregation: Avg salary by company type...")
    pipeline = [
        {"$match": {"deleted": {"$ne": True}, "company_type": {"$ne": None}}},
        {"$group": {
            "_id": "$company_type",
            "count": {"$sum": 1},
            "avg_salary": {"$avg": "$monthly_salary"},
            "avg_credit": {"$avg": "$credit_score"},
            "avg_foir": {"$avg": "$foir_percentage"}
        }},
        {"$sort": {"avg_salary": -1}}
    ]
    agg = db.run_aggregation(pipeline)
    for r in agg:
        print(f"   {r['_id']}: Count={r['count']}, Avg Salary=₹{r['avg_salary']:,.0f}, "
              f"Avg Credit={r['avg_credit']:.0f}, Avg FOIR={r['avg_foir']}%")
    
    # =========================================================================
    # UPDATE
    # =========================================================================
    print("\n" + "-"*60)
    print("3. UPDATE")
    print("-"*60)
    
    # Update applicant
    print("\n>>> Update applicant...")
    success = db.update_applicant(app_id, {
        "monthly_salary": 95000,
        "credit_score": 760,
        "company_type": "Mid-Size",
        "monthly_rent": 18000
    })
    print(f"✅ Update {'successful' if success else 'failed'}")
    
    # Verify derived fields recalculated
    updated = db.get_applicant_by_id(app_id)
    print(f"   New Salary: ₹{updated['monthly_salary']:,}")
    print(f"   New FOIR: {updated['foir_percentage']}% (recalculated)")
    print(f"   Version: {updated['version']}")
    
    # Update prediction
    print("\n>>> Update prediction...")
    db.update_prediction(app_id, 1, 0.94, 35000, 'XGBoost', 12.5)
    updated = db.get_applicant_by_id(app_id)
    print(f"✅ Predicted: Eligible={updated['predicted_eligibility']}, "
          f"Prob={updated['predicted_probability']}, Risk={updated['risk_category']}")
    
    # Update status
    print("\n>>> Update status...")
    db.update_status(app_id, 'Approved', 'Good profile', 'Officer A')
    print(f"✅ Status updated")
    
    # Bulk status update
    print("\n>>> Bulk status update...")
    pending = db.get_all_applicants(limit=10, status_filter='Pending')
    if pending:
        ids = [a['applicant_id'] for a in pending[:3]]
        result = db.bulk_update_status(ids, 'Under Review')
        print(f"✅ Bulk updated: {result['modified_count']} records")
    
    # =========================================================================
    # DELETE
    # =========================================================================
    print("\n" + "-"*60)
    print("4. DELETE")
    print("-"*60)
    
    # Soft delete
    temp = db.create_applicant({
        "age": 20, "gender": "Male", "marital_status": "Single",
        "education": "High School", "monthly_salary": 10000,
        "employment_type": "Unemployed", "credit_score": 300,
        "requested_amount": 50000, "requested_tenure": 12
    })
    temp_id = temp['applicant_id']
    
    print(f"\n>>> Soft delete applicant {temp_id}...")
    db.delete_applicant(temp_id)
    print(f"✅ Soft deleted")
    print(f"   Exists in DB: {db.applicant_exists(temp_id)} (should be False)")
    
    # =========================================================================
    # EXPORT
    # =========================================================================
    print("\n" + "-"*60)
    print("5. EXPORT")
    print("-"*60)
    
    csv_count = db.export_to_csv('data/exported_applicants.csv')
    print(f"✅ Exported {csv_count} records to CSV (27 columns)")
    
    json_count = db.export_to_json('data/exported_applicants.json')
    print(f"✅ Exported {json_count} records to JSON")
    
    # =========================================================================
    # AUDIT
    # =========================================================================
    print("\n" + "-"*60)
    print("6. AUDIT LOG")
    print("-"*60)
    
    logs = db.get_audit_log(applicant_id=app_id)
    print(f"✅ {len(logs)} log entries for applicant {app_id}")
    for log in logs:
        print(f"   [{log['timestamp']}] {log['operation']} by {log['performed_by']}")
    
    # =========================================================================
    # FINAL
    # =========================================================================
    print("\n" + "-"*60)
    print("FINAL HEALTH CHECK")
    print("-"*60)
    health = db.health_check()
    for k, v in health.items():
        print(f"   {k}: {v}")
    
    db.close()
    
    print("\n" + "="*80)
    print("CRUD DEMONSTRATION COMPLETED")
    print("="*80)


if __name__ == "__main__":
    demonstrate_crud()