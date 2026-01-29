# BD-NSCA Prompt Templates (Bilingual)

This document provides concise, safe prompt templates for common scenarios. Use `{{lang}}` param to select language (`en` or `vi`). Replace placeholders like `{{context}}`, `{{agent_state}}`, and `{{instruction}}`.

---

## Template structure
- Input: Context + Agent state + Instruction
- Output: Short intent label + Ordered `actions` array with `action_type` and `params`.

Example input (EN):

```
Lang: en
Context: {{context}}
AgentState: {{agent_state}}
Instruction: {{instruction}}

Return JSON with keys: intent (string), actions (array of {action_type, params}). Keep actions high-level and safe.
```

Example input (VI):

```
Lang: vi
Bối cảnh: {{context}}
Trạng thái tác nhân: {{agent_state}}
Hướng dẫn: {{instruction}}

Trả về JSON gồm: intent (chuỗi), actions (mảng gồm {action_type, params}). Hành động ở mức cao, an toàn.
```

---

## Scenario-specific examples

### Combat (EN)
- Instruction: "Neutralize immediate threats, minimize collateral damage, report status."
- Example expected action_types: `['assess', 'engage', 'take_cover', 'report']`

### Combat (VI)
- Hướng dẫn: "Hạ các mối đe dọa gần nhất, giảm thiểu ảnh hưởng tới dân thường, báo cáo trạng thái."
- Ví dụ action_types: `['assess', 'engage', 'take_cover', 'report']`

### Patrol (EN)
- Instruction: "Patrol the sector, investigate anomalies, and escort civilians to safety if found."
- Example action_types: `['move', 'scan', 'investigate', 'escort']`

### Patrol (VI)
- Hướng dẫn: "Tuần tra khu vực, điều tra sự bất thường, hộ tống dân thường đến nơi an toàn nếu tìm thấy."
- Ví dụ action_types: `['move', 'scan', 'investigate', 'escort']`

### Escort (EN/VI)
- EN Instruction: "Safely escort VIP to designated coordinates, avoid conflict, use alternate routes if danger detected."
- VI Hướng dẫn: "Hộ tống an toàn VIP tới tọa độ chỉ định, tránh xung đột, chuyển đường khác nếu phát hiện nguy hiểm."

### Shopkeeper & Investigation
- Shopkeeper: prioritize peaceful interactions (`interact`, `inventory_check`, `trade`).
- Investigation: prioritize information gathering (`scan`, `collect_evidence`, `report`).

---

💡 Tip: Keep prompts explicit about safety and acceptable levels of detail. Annotators should prefer high-level actions rather than low-level weapon-level instructions.
