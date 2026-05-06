# Training Results & Analysis

## Model Performance

| Metric | Training (mAP50) | Real-World Tests |
|--------|------------------|------------------|
| Spaghetti | 87.0% | 88.9% |
| Layer Shift | 93.8% | 61.1% |
| **Overall** | **90.8%** | **75.0%** |

The gap between training and real-world performance for layer shift is explained by the 50% confidence threshold and the inclusion of small (~2mm) displacements that are difficult to detect at camera distance.

---

## Integration Testing Results

### Spaghetti Detection

| Test | Scenario | Attempt 1 | Attempt 2 | Success |
|------|----------|-----------|-----------|---------|
| 1 | Small strand | ✅ 67.9% / 4s | ✅ 54% / 20s | 100% |
| 2 | Medium | ✅ 85.9% / 3s | ✅ 59.1% / 3s | 100% |
| 3 | Large | ✅ 61.1% / 45s | ✅ 66.7% / 3s | 100% |
| 4 | On partial print | ✅ 57.2% / 10s | ✅ 55.5% / 125s | 100% |
| 5 | Various shapes | ❌ | ✅ 50.6% / 13s | 66% |
| 6 | Normal (false positive) | ✅ x / 30s | ✅ x / 30s | 66% |

**True Positive Rate: 88.9%**

---

### Layer Shift Detection

| Test | Scenario | Attempt 1 | Attempt 2 | Attempt 3 | Success |
|------|----------|-----------|-----------|-----------|---------|
| 7 | Small (~2mm) | ❌ 35.4% | ❌ 37.0% | ✅ 50% / 10s | 33% |
| 8 | Medium (~5mm) | ❌ 31.9% | ✅ 56.7% / 16s | ✅ 50.6% / 36s | 66% |
| 9 | Large (~10mm) | ✅ 53.9% / 65s | ✅ 51.5% / 10s | ✅ 60.1% / 14s | 100% |
| 10 | Benchy | ❌ 27.7% | ✅ 63.9% / 22s | ✅ 55.1% / 24s | 66% |
| 11 | Cube | ✅ 61.9% / 4s | ✅ 62.4% / 7s | ✅ 64.6% / 5s | 100% |
| 12 | Normal (false positive) | ❌ 65.2% / 15s | ✅ x | ✅ x | 66% |

**True Positive Rate: 61.1%**

#### Detection by Magnitude

| Magnitude | Success Rate | Avg Confidence | Notes |
|-----------|--------------|----------------|-------|
| Small (~2mm) | 33% (1/3) | 35-37% | Below threshold |
| Medium (~5mm) | 66% (2/3) | 48-52% | Marginal |
| Large (~10mm) | 100% (3/3) | 65-78% | Consistent |

---

## Cost Analysis

### Hardware Components

| Component | Cost (USD) | Notes |
|-----------|------------|-------|
| Raspberry Pi 4 (2GB) | $63.25 | Includes power supply, microSD, cable |
| Camera (iPhone SE) | $0 | Existing device |
| Tripod | $10-15 | Generic |
| **Hardware Subtotal** | **~$78.25** | |
| Roboflow, Colab, OctoPrint, Twilio | $0 | Free tiers |
| **Total** | **~$78.25** | |

### 3-Year Cost Comparison

| Item | This System | Obico Pro |
|------|-------------|-----------|
| Hardware | ~$78.25 | $0 |
| Year 1 Subscription | $0 | $6/month = $72 |
| Year 2 | $0 | $72 |
| Year 3 | $0 | $72 |
| **Total** | **~$78.25** | **$216** |
| **Savings** | **~$137.75 (63.7%)** | — |

---

## System Specificity

| Scenario | Result | Notes |
|----------|--------|-------|
| Empty bed | ✅ 3/3 | No false positives |
| Print start | ✅ 3/3 | No false positives |
| Mid-print | ⚠️ 2/3 | 1 FP at 53.3% confidence |
| Final print | ✅ 3/3 | No false positives |
| **True Negative Rate** | **91.7%** | |

---

## Key Findings

1. **Spaghetti detection is highly reliable** — 88.9% TPR with 100% success in scenarios 1-4
2. **Layer shift detection correlates with magnitude** — small displacements (<5mm) are difficult to detect at camera distance
3. **No inter-class confusion** — the model never mistakes spaghetti for layer shift or vice versa
4. **Low false positive rate** — 91.7% specificity means the system rarely interrupts successful prints
5. **Detection time varies** — from 3s (optimal) to 125s (challenging partial print scenario)
