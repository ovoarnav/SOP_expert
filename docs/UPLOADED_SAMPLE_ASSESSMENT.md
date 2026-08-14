# Uploaded sample assessment

## What the file is

The uploaded 13-page PDF is an image-only scan and should be treated as a **bundle of multiple source artifacts**, not one uniform standing-order policy.

Observed page map:

| Pages | Apparent artifact | Prototype handling |
|---|---|---|
| 1-2 | Archbold Living-Cairo standing orders for admission or PRN, with medication, lab, and referenced protocol/form sections | Keep together conceptually; verify dates, signatures, and facility scope manually. |
| 3 | Bowel-management physician order template | Classify as an order-set template; do not present it as proof of a resident-specific active order. |
| 4 | Foley removal/retention physician order template | Classify as an order-set template; preserve the conditional sequence exactly. |
| 5-8 | PRN implementation/notification forms | Usually useful as form locators and assessment prompts; decide page by page whether they belong in nurse policy search. |
| 9 | Skin tear treatment orders | High-risk treatment text; manually verify all product names, sequence, timing, and escalation language. |
| 10 | Insulin correction scale and hypoglycemia protocol | Highest OCR-risk page because dose values and actions are arranged in a table. Require row-by-row verification before search. |
| 11 | Pressure ulcer and chronic wound dressing orders | High OCR-risk table. Require cell-by-cell verification and do not let OCR move values between stages. |
| 12 | Adult pressure ulcer prevention and treatment orders | Numbered protocol page; preserve each numbered instruction and printed form/date metadata. |
| 13 | Blank resident care fax form | Exclude from clinical answer search by default; it may be retained only as an administrative form locator. |

## Product changes caused by this file

1. OCR is required in the prototype; rejecting scans would make the initial corpus unusable.
2. OCR output cannot be trusted automatically. The search gate must operate at the page or logical-source-item level, not only at the PDF level.
3. Printed dates and facility scope vary by page, so those fields belong on page/source-item metadata as well as document metadata.
4. Blank forms and protocols must not be blended into one generated answer.
5. Tables require a stricter review mode than ordinary paragraphs.
6. Nurse search should begin as extractive evidence retrieval. Generative summarization comes later, after citation and numeric validation.
