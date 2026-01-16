# Backend Integration Guide

## System Architecture

```
Frontend (HTML/JS)
       ↓
Flask REST API
       ↓
Core Engines:
  ├── Fraud Detection Engine
  ├── Image Validator
  ├── Angle Validator (4 methods)
       ↓
Database (SQLite)
       ↓
Workflow Integration Points
```

---

## REST API Endpoints

### 1. Web Interface
```
GET /
  Returns: HTML interface for image capture and analysis
```

### 2. Fraud Analysis
```
POST /api/analyze-return
  Required files:
    - delivery_images (up to 6)
    - return_images (up to 6)
  
  Optional fields:
    - return_id
    - product_sku
    - product_name
    - customer_id
    - product_value
  
  Response:
    {
      "success": true,
      "analysis": {
        "fraud_risk_score": 25,
        "risk_level": "MEDIUM-LOW",
        "recommendation": "LIKELY APPROVE",
        "component_scores": {...},
        "confidence": 0.85
      },
      "angle_validation": {...}
    }
```

### 3. Return Management
```
GET /api/returns/pending
  Returns: List of pending returns
  Query params: limit=100

GET /api/returns/manual-review
  Returns: Returns flagged for manual review
  Query params: limit=100

GET /api/returns/<return_id>
  Returns: Specific return details

POST /api/returns/<return_id>/approve
  Approves a return
  Body: {"notes": "approved by manager"}

POST /api/returns/<return_id>/deny
  Denies a return
  Body: {"notes": "suspected fraud"}

POST /api/returns/<return_id>/manual-review
  Flags for manual review
  Body: {"notes": "borderline case"}
```

### 4. Dashboard & Statistics
```
GET /api/dashboard
  Returns: Statistics and recent returns

GET /api/system-stats
  Returns: System capabilities and configuration
```

---

## Database Schema

### returns table
```sql
CREATE TABLE returns (
  return_id TEXT PRIMARY KEY,
  product_sku TEXT NOT NULL,
  product_name TEXT,
  customer_id TEXT,
  delivery_date TEXT,
  return_date TEXT,
  product_value REAL,
  delivery_images TEXT (JSON list),
  return_images TEXT (JSON list),
  expected_accessories TEXT (JSON list),
  fraud_score REAL,
  risk_level TEXT,
  recommendation TEXT,
  fraud_type TEXT,
  component_scores TEXT (JSON),
  confidence REAL,
  analysis_timestamp TEXT,
  status TEXT,
  manual_review_notes TEXT,
  created_at TEXT,
  updated_at TEXT
)
```

### angle_validations table
```sql
CREATE TABLE angle_validations (
  id INTEGER PRIMARY KEY,
  return_id TEXT,
  delivery_angle_1 REAL,
  delivery_angle_2 REAL,
  ... (6 angles each)
  return_angle_1 REAL,
  ... (6 angles each)
  angles_match BOOLEAN,
  angle_match_confidence REAL,
  created_at TEXT,
  FOREIGN KEY (return_id) REFERENCES returns(return_id)
)
```

### analysis_history table
```sql
CREATE TABLE analysis_history (
  id INTEGER PRIMARY KEY,
  return_id TEXT,
  fraud_score REAL,
  actual_fraud BOOLEAN,
  component_scores TEXT (JSON),
  product_type TEXT,
  image_quality TEXT,
  created_at TEXT,
  FOREIGN KEY (return_id) REFERENCES returns(return_id)
)
```

---

## Integration Examples

### Example 1: Basic Integration (Python)
```python
import requests
import cv2

# Load images
delivery_img = cv2.imread('delivery.jpg')
return_img = cv2.imread('return.jpg')

# Prepare request
files = {
    'delivery_images': open('delivery.jpg', 'rb'),
    'return_images': open('return.jpg', 'rb'),
}
data = {
    'return_id': 'RET_001',
    'product_sku': 'PHONE_123',
    'product_name': 'iPhone 13',
    'customer_id': 'CUST_456',
    'product_value': 999.99
}

# Send to API
response = requests.post('http://localhost:5000/api/analyze-return',
                        files=files, data=data)

result = response.json()

# Handle result
if result['success']:
    score = result['analysis']['fraud_risk_score']
    recommendation = result['analysis']['recommendation']
    
    print(f"Fraud Score: {score}")
    print(f"Recommendation: {recommendation}")
```

### Example 2: Workflow Integration (Node.js)
```javascript
async function analyzeReturn(returnId, productSku, imagePaths) {
    const formData = new FormData();
    
    formData.append('return_id', returnId);
    formData.append('product_sku', productSku);
    
    // Add images
    for (const path of imagePaths.delivery) {
        const file = await fetch(path).then(r => r.blob());
        formData.append('delivery_images', file);
    }
    
    for (const path of imagePaths.return) {
        const file = await fetch(path).then(r => r.blob());
        formData.append('return_images', file);
    }
    
    // Send request
    const response = await fetch('/api/analyze-return', {
        method: 'POST',
        body: formData
    });
    
    return response.json();
}

// Usage
analyzeReturn('RET_001', 'PHONE_123', {
    delivery: ['delivery_front.jpg', 'delivery_back.jpg'],
    return: ['return_front.jpg', 'return_back.jpg']
}).then(result => {
    console.log(`Fraud Score: ${result.analysis.fraud_risk_score}`);
    console.log(`Recommendation: ${result.analysis.recommendation}`);
});
```

### Example 3: Automatic Workflow (SQL)
```sql
-- Get pending approvals
SELECT * FROM returns 
WHERE fraud_score < 20 
AND status = 'PENDING'
ORDER BY created_at DESC
LIMIT 100;

-- Get fraud cases for investigation
SELECT * FROM returns 
WHERE fraud_score > 75 
AND status = 'PENDING'
ORDER BY fraud_score DESC;

-- Get uncertain cases for manual review
SELECT * FROM returns 
WHERE fraud_score BETWEEN 40 AND 70
AND status = 'MANUAL_REVIEW'
ORDER BY fraud_score DESC;

-- Fraud statistics
SELECT 
  risk_level,
  COUNT(*) as count,
  AVG(fraud_score) as avg_score
FROM returns
GROUP BY risk_level;
```

---

## Workflow Integration Points

### Point 1: Return Initiation
```
Customer submits return
  ↓
System prompts for image capture
  ↓
Returns 6 images (delivery reference + return)
  ↓
Automatic fraud analysis
```

### Point 2: Automatic Processing
```
Fraud Score < 20: AUTO-APPROVE
  → Add to approval queue
  → Notify warehouse

Fraud Score > 80: AUTO-DENY
  → Add to denied queue
  → Send notification to customer

40 < Score < 70: MANUAL REVIEW
  → Add to review queue
  → Notify fraud team
```

### Point 3: Manual Review
```
Fraud team reviews analysis + images
  ↓
Makes decision (APPROVE/DENY)
  ↓
Updates status in database
  ↓
System records actual fraud outcome
  ↓
Feedback loop improves future accuracy
```

### Point 4: Continuous Improvement
```
Each return analyzed → Recorded in analysis_history
  ↓
Track accuracy metrics:
  - True positive rate
  - False positive rate
  - By product type
  
Quarterly review:
  - Adjust fraud score weights
  - Fine-tune thresholds
  - Re-calibrate models
  
No ML retraining needed
  (Rule-based system adapts via thresholds)
```

---

## Configuration

### System Settings (stored in database)
```python
# Default thresholds
db.set_setting('auto_approve_threshold', '20')
db.set_setting('auto_deny_threshold', '80')
db.set_setting('manual_review_min', '20')
db.set_setting('manual_review_max', '80')

# Get settings
auto_approve = int(db.get_setting('auto_approve_threshold', '20'))
```

### Component Weights (in fraud_detection_engine.py)
```python
WEIGHTS = {
    'ocr': 0.25,        # Serial number verification
    'accessory': 0.20,  # Missing accessories
    'damage': 0.30,     # Physical damage
    'swap': 0.15,       # Product swap detection
    'wear': 0.10        # Wear/usage patterns
}
```

Adjust weights by product category:
```python
# For electronics (emphasize damage)
WEIGHTS = {'ocr': 0.25, 'accessory': 0.15, 'damage': 0.40, 'swap': 0.15, 'wear': 0.05}

# For fashion (emphasize wear)
WEIGHTS = {'ocr': 0.10, 'accessory': 0.20, 'damage': 0.20, 'swap': 0.10, 'wear': 0.40}
```

---

## Deployment Considerations

### 1. Database Backup
```bash
# Regular backup of SQLite database
cp fraud_detection.db fraud_detection_backup_$(date +%Y%m%d).db
```

### 2. Image Storage
Current implementation: Images loaded in memory, not persisted
Options for production:
- Store images in cloud storage (S3, GCS)
- Keep temporary cache, delete after analysis
- Archive to cold storage after 30 days

### 3. Scalability
Current: Single-server deployment
For scale:
- Use load balancer (Nginx, HAProxy)
- Multi-worker Flask (Gunicorn)
- Database replication
- Image processing queue (Celery)
- Cache layer (Redis)

### 4. Monitoring
Track key metrics:
```python
# Average processing time
# Fraud detection accuracy
# False positive/negative rates
# User engagement (returns submitted)
# System uptime
```

---

## Testing Integration

### Test 1: End-to-end API test
```bash
curl -X POST http://localhost:5000/api/analyze-return \
  -F "delivery_images=@delivery.jpg" \
  -F "return_images=@return.jpg" \
  -F "return_id=TEST_001" \
  -F "product_sku=TEST_SKU"
```

### Test 2: Database query
```python
from database import Database

db = Database()

# Create test return
from database import ReturnRecord
record = ReturnRecord(...)
db.create_return(record)

# Query it back
result = db.get_return('TEST_001')
print(result)
```

### Test 3: Angle validation
```python
from angle_validator import AngleValidationSystem
import cv2

validator = AngleValidationSystem()

delivery_imgs = [cv2.imread(f) for f in delivery_files]
return_imgs = [cv2.imread(f) for f in return_files]

result = validator.validate_with_automatic_detection(delivery_imgs, return_imgs)
print(result)
```

---

## Next Steps

1. ✓ **API & Database set up** - Done
2. ✓ **Web interface created** - Done
3. ✓ **Angle validation implemented** - Done
4. → **Deploy to production** - Choose hosting
5. → **Integrate with your return system** - Use API endpoints
6. → **Set up monitoring** - Track accuracy
7. → **Implement feedback loop** - Continuous improvement
8. → **Scale as needed** - Multi-server setup

---

## Support

**API Documentation**: Use swagger/OpenAPI (can be added)
**Database Schema**: See above or check `database.py`
**Error Handling**: All endpoints return JSON with error details
**Logs**: Check Flask console output or add logging

---

**Status**: Production Ready with Full Integration
**Version**: 1.0
**Last Updated**: 2024
