# JSON to CSV Conversion Logic
## PlungePalz Biometric Data — Heart Rate & Temperature

---

## Heart Rate CSV Logic

The key difference is **sampling interval based on activity type**:

| Activity Type | `Array_HR` Interval | Timestamp Formula |
|---|---|---|
| Cold Plunge | Every **1 second** | `index i → timestamp i * 1` |
| Sauna / Cold Shower | Every **5 seconds** | `index i → timestamp i * 5` |

### Cold Plunge Example (297 HR values)

```
Array_HR[0]   = 63  → timestamp: 0
Array_HR[1]   = 63  → timestamp: 1
Array_HR[2]   = 63  → timestamp: 2
...
Array_HR[296] = 92  → timestamp: 296
```

Output CSV is already at 1-second resolution — **no interpolation needed**.

### Sauna Example (71 HR values)

```
Array_HR[0]  = 81  → timestamp: 0
Array_HR[1]  = 115 → timestamp: 5
Array_HR[2]  = 111 → timestamp: 10
...
Array_HR[70] = 67  → timestamp: 350
```

Output CSV is at 5-second resolution — **needs interpolation to 1-second** before passing to the overlay system (same PCHIP/CubicSpline approach used for temperature).

---

## Temperature CSV Logic

**All activity types record temperature every 5 seconds** — same rule regardless of Cold Plunge, Sauna, or Cold Shower.

| Activity Type | `SW_Temp_Array_F` Interval |
|---|---|
| Cold Plunge | Every **5 seconds** |
| Sauna / Cold Shower | Every **5 seconds** |

### Cold Plunge Example (58 temp values)

```
SW_Temp_Array_F[0]  = "76.2" → timestamp: 0
SW_Temp_Array_F[1]  = "76.2" → timestamp: 5
SW_Temp_Array_F[2]  = "76.5" → timestamp: 10
...
SW_Temp_Array_F[57] = "79.5" → timestamp: 285
```

### Sauna Example (71 temp values)

```
SW_Temp_Array_F[0]  = "75.0" → timestamp: 0
SW_Temp_Array_F[1]  = "75.0" → timestamp: 5
...
SW_Temp_Array_F[70] = "77.9" → timestamp: 350
```

Both need **PCHIP interpolation to 1-second intervals** before use in the video overlay.

---

## Matching `s_length` to Array Lengths

`s_length` is the session duration in seconds and serves as a useful sanity check against array lengths:

- **Cold Plunge**: `s_length = "294"`, `Array_HR` has 297 values at 1s each ✓ *(close match, slight overtime)*
- **Sauna**: `s_length = "356"`, `Array_HR` has 71 values × 5s = 355s ✓

---

## DynamoDB JSON Parsing Notes

Depending on the source of the JSON, array elements may be in DynamoDB typed format or plain format:

| Field | DynamoDB Format | Normalized Format |
|---|---|---|
| `Array_HR` | `{"N": "63"}` | `63` |
| `SW_Temp_Array_F` | `{"S": "76.2"}` | `"76.2"` |

Your parsing logic should handle both cases.

---

## Summary: What the Conversion Script Needs to Do

Given a JSON input, the script should:

1. **Read `activityType`** to determine HR sampling interval
   - `Cold Plunge` → 1 second per index
   - `Sauna` / `Cold Shower` → 5 seconds per index

2. **Parse `Array_HR`** — handle both `{"N": "value"}` (DynamoDB) and plain number formats

3. **Build HR timestamps**: `timestamp = index * interval`

4. **Parse `SW_Temp_Array_F`** — handle both `{"S": "value"}` (DynamoDB) and plain string/number formats — always 5s intervals

5. **Build temp timestamps**: `timestamp = index * 5`

6. **Output two CSVs**:

| File | Columns | Notes |
|---|---|---|
| `heartrate.csv` | `timestamp, heart_rate` | Interpolate to 1s if Sauna/Cold Shower |
| `temperature_data.csv` | `timestamp, temp_data` | Raw 5s intervals — interpolation handled by overlay script |
