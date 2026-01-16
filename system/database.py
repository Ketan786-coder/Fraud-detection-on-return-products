"""
Database Models & Integration
SQLite-based persistence for returns and fraud analysis
"""

import sqlite3
import json
from datetime import datetime
from typing import Optional, List, Dict
from dataclasses import dataclass, asdict


@dataclass
class ReturnRecord:
    """Return database record"""
    return_id: str
    product_sku: str
    product_name: str
    customer_id: str
    delivery_date: str
    return_date: str
    product_value: float
    delivery_images: List[str]
    return_images: List[str]
    expected_accessories: List[str]
    fraud_score: Optional[float] = None
    risk_level: Optional[str] = None
    recommendation: Optional[str] = None
    fraud_type: Optional[str] = None
    component_scores: Optional[Dict] = None
    confidence: Optional[float] = None
    analysis_timestamp: Optional[str] = None
    status: str = "PENDING"  # PENDING, APPROVED, DENIED, MANUAL_REVIEW
    manual_review_notes: Optional[str] = None
    created_at: Optional[str] = None


@dataclass
class AngleValidationRecord:
    """Angle validation record"""
    return_id: str
    delivery_angle_1: Optional[float] = None
    delivery_angle_2: Optional[float] = None
    delivery_angle_3: Optional[float] = None
    delivery_angle_4: Optional[float] = None
    delivery_angle_5: Optional[float] = None
    delivery_angle_6: Optional[float] = None
    return_angle_1: Optional[float] = None
    return_angle_2: Optional[float] = None
    return_angle_3: Optional[float] = None
    return_angle_4: Optional[float] = None
    return_angle_5: Optional[float] = None
    return_angle_6: Optional[float] = None
    angles_match: bool = False
    angle_match_confidence: float = 0.0
    created_at: Optional[str] = None


class Database:
    """SQLite database for fraud detection system"""

    def __init__(self, db_path: str = 'fraud_detection.db'):
        self.db_path = db_path
        self.init_database()

    def get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_database(self):
        """Initialize database schema"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Returns table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS returns (
                return_id TEXT PRIMARY KEY,
                product_sku TEXT NOT NULL,
                product_name TEXT NOT NULL,
                customer_id TEXT NOT NULL,
                delivery_date TEXT NOT NULL,
                return_date TEXT NOT NULL,
                product_value REAL NOT NULL,
                delivery_images TEXT NOT NULL,
                return_images TEXT NOT NULL,
                expected_accessories TEXT NOT NULL,
                fraud_score REAL,
                risk_level TEXT,
                recommendation TEXT,
                fraud_type TEXT,
                component_scores TEXT,
                confidence REAL,
                analysis_timestamp TEXT,
                status TEXT DEFAULT 'PENDING',
                manual_review_notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Angle validation table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS angle_validations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                return_id TEXT NOT NULL,
                delivery_angle_1 REAL,
                delivery_angle_2 REAL,
                delivery_angle_3 REAL,
                delivery_angle_4 REAL,
                delivery_angle_5 REAL,
                delivery_angle_6 REAL,
                return_angle_1 REAL,
                return_angle_2 REAL,
                return_angle_3 REAL,
                return_angle_4 REAL,
                return_angle_5 REAL,
                return_angle_6 REAL,
                angles_match BOOLEAN DEFAULT 0,
                angle_match_confidence REAL DEFAULT 0.0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (return_id) REFERENCES returns(return_id)
            )
        """)

        # Analysis history table (for continuous improvement)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                return_id TEXT NOT NULL,
                fraud_score REAL NOT NULL,
                actual_fraud BOOLEAN,
                component_scores TEXT,
                product_type TEXT,
                image_quality TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (return_id) REFERENCES returns(return_id)
            )
        """)

        # System settings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

        # Initialize default settings
        self.set_setting('auto_approve_threshold', '20')
        self.set_setting('auto_deny_threshold', '80')
        self.set_setting('manual_review_min', '20')
        self.set_setting('manual_review_max', '80')

    def set_setting(self, key: str, value: str):
        """Set system setting"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO system_settings (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        """, (key, value))
        
        conn.commit()
        conn.close()

    def get_setting(self, key: str, default: str = None) -> str:
        """Get system setting"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT value FROM system_settings WHERE key = ?", (key,))
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else default

    def create_return(self, record: ReturnRecord) -> bool:
        """Create new return record"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO returns (
                    return_id, product_sku, product_name, customer_id,
                    delivery_date, return_date, product_value,
                    delivery_images, return_images, expected_accessories,
                    status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.return_id,
                record.product_sku,
                record.product_name,
                record.customer_id,
                record.delivery_date,
                record.return_date,
                record.product_value,
                json.dumps(record.delivery_images),
                json.dumps(record.return_images),
                json.dumps(record.expected_accessories),
                record.status,
                datetime.now().isoformat()
            ))

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error creating return: {e}")
            return False

    def get_return(self, return_id: str) -> Optional[ReturnRecord]:
        """Get return record"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM returns WHERE return_id = ?", (return_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return self._row_to_return(row)

    def update_fraud_analysis(self, return_id: str, fraud_score: float, 
                            risk_level: str, recommendation: str,
                            component_scores: Dict, fraud_type: Optional[str],
                            confidence: float):
        """Update return with fraud analysis results"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE returns SET
                    fraud_score = ?,
                    risk_level = ?,
                    recommendation = ?,
                    fraud_type = ?,
                    component_scores = ?,
                    confidence = ?,
                    analysis_timestamp = ?,
                    status = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE return_id = ?
            """, (
                fraud_score,
                risk_level,
                recommendation,
                fraud_type,
                json.dumps(component_scores),
                confidence,
                datetime.now().isoformat(),
                'MANUAL_REVIEW' if 20 <= fraud_score <= 80 else ('APPROVED' if fraud_score < 20 else 'DENIED'),
                return_id
            ))

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error updating fraud analysis: {e}")
            return False

    def update_angle_validation(self, return_id: str, angles: Dict):
        """Update angle validation results"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO angle_validations (
                    return_id,
                    delivery_angle_1, delivery_angle_2, delivery_angle_3,
                    delivery_angle_4, delivery_angle_5, delivery_angle_6,
                    return_angle_1, return_angle_2, return_angle_3,
                    return_angle_4, return_angle_5, return_angle_6,
                    angles_match, angle_match_confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                return_id,
                angles.get('delivery_angle_1'),
                angles.get('delivery_angle_2'),
                angles.get('delivery_angle_3'),
                angles.get('delivery_angle_4'),
                angles.get('delivery_angle_5'),
                angles.get('delivery_angle_6'),
                angles.get('return_angle_1'),
                angles.get('return_angle_2'),
                angles.get('return_angle_3'),
                angles.get('return_angle_4'),
                angles.get('return_angle_5'),
                angles.get('return_angle_6'),
                angles.get('angles_match', False),
                angles.get('angle_match_confidence', 0.0)
            ))

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error updating angle validation: {e}")
            return False

    def update_return_status(self, return_id: str, status: str, 
                            notes: Optional[str] = None) -> bool:
        """Update return status"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE returns SET
                    status = ?,
                    manual_review_notes = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE return_id = ?
            """, (status, notes, return_id))

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error updating return status: {e}")
            return False

    def record_analysis_history(self, return_id: str, fraud_score: float,
                               component_scores: Dict, actual_fraud: Optional[bool] = None):
        """Record analysis for continuous improvement"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO analysis_history (
                    return_id, fraud_score, component_scores, actual_fraud
                ) VALUES (?, ?, ?, ?)
            """, (return_id, fraud_score, json.dumps(component_scores), actual_fraud))

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error recording analysis history: {e}")
            return False

    def get_pending_returns(self, limit: int = 100) -> List[ReturnRecord]:
        """Get pending returns"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM returns 
            WHERE status = 'PENDING' 
            ORDER BY created_at DESC 
            LIMIT ?
        """, (limit,))

        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_return(row) for row in rows]

    def get_manual_review_returns(self, limit: int = 100) -> List[ReturnRecord]:
        """Get returns pending manual review"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM returns 
            WHERE status = 'MANUAL_REVIEW' 
            ORDER BY fraud_score DESC 
            LIMIT ?
        """, (limit,))

        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_return(row) for row in rows]

    def get_statistics(self) -> Dict:
        """Get system statistics"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Total returns
        cursor.execute("SELECT COUNT(*) FROM returns")
        total = cursor.fetchone()[0]

        # By status
        cursor.execute("SELECT status, COUNT(*) FROM returns GROUP BY status")
        by_status = {row[0]: row[1] for row in cursor.fetchall()}

        # Average fraud score
        cursor.execute("SELECT AVG(fraud_score) FROM returns WHERE fraud_score IS NOT NULL")
        avg_fraud_score = cursor.fetchone()[0] or 0

        # Approved vs Denied
        cursor.execute("SELECT COUNT(*) FROM returns WHERE status = 'APPROVED'")
        approved = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM returns WHERE status = 'DENIED'")
        denied = cursor.fetchone()[0]

        conn.close()

        return {
            'total_returns': total,
            'by_status': by_status,
            'average_fraud_score': avg_fraud_score,
            'approved': approved,
            'denied': denied,
            'manual_review': by_status.get('MANUAL_REVIEW', 0)
        }

    def _row_to_return(self, row) -> ReturnRecord:
        """Convert database row to ReturnRecord"""
        return ReturnRecord(
            return_id=row['return_id'],
            product_sku=row['product_sku'],
            product_name=row['product_name'],
            customer_id=row['customer_id'],
            delivery_date=row['delivery_date'],
            return_date=row['return_date'],
            product_value=row['product_value'],
            delivery_images=json.loads(row['delivery_images']),
            return_images=json.loads(row['return_images']),
            expected_accessories=json.loads(row['expected_accessories']),
            fraud_score=row['fraud_score'],
            risk_level=row['risk_level'],
            recommendation=row['recommendation'],
            fraud_type=row['fraud_type'],
            component_scores=json.loads(row['component_scores']) if row['component_scores'] else None,
            confidence=row['confidence'],
            analysis_timestamp=row['analysis_timestamp'],
            status=row['status'],
            manual_review_notes=row['manual_review_notes'],
            created_at=row['created_at']
        )


# Initialize default database
if __name__ == "__main__":
    db = Database()
    print("✓ Database initialized")
    print("Location: fraud_detection.db")
    
    # Show statistics
    stats = db.get_statistics()
    print(f"\n📊 Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
