# LAB05-RAG System Development II

This project demonstrates 9 common problems in LLM and RAG (Retrieval-Augmented Generation) systems. All simulations use the same real dataset, privacy.txt, which is a Thai question-answer knowledge base about online privacy and personal data protection (account security, phishing/malware threats, privacy-protection technology, PDPA law, and social media/digital-footprint safety). This allows each problem to be tested with real data instead of isolated sample data.

# Structure:

```text
RAG-Project/
├── data_loader.py                 # Shared parser: privacy.txt -> list of dict
├── main.py                        # Main menu for running each problem
├── privacy.txt                    # Raw data / RAG Knowledge Base
├── problem01_hallucination.py     # Hallucination / answer without supporting context
├── problem02_transformer.py       # Vocabulary Mismatch + Token Position
├── problem03_data_quality.py      # Duplicate / Noise / Normalization
├── problem04_chunking.py          # Chunk Size / Overlap
├── problem05_metadata.py          # Metadata Filtering
├── problem06_reranking.py         # Top-k / Re-ranking
├── problem07_generation.py        # Correct Retrieval but incorrect Generation
├── problem08_config.py            # RAG Configuration
└── problem09_evaluation.py        # Chunk & Retrieval Evaluation
```

# Dataset:
The project uses one shared Knowledge Base containing Thai questions and answers about online privacy and personal data protection: authentication and account security basics, online scam/threat techniques (phishing, ransomware, spyware), privacy-protection technology, PDPA law and data-subject rights, and social media/digital-footprint safety.

The dataset contains 56 entries, covering 5 main categories and 3 language styles: formal (ทางการ), casual (กันเอง), and slang (แสลง).

Each entry has three lines:
```text

[หมวด: <category>]
or
[หมวด: <category> | ภาษา: <variant>]

Q: <question>
A: <answer>

```
# Summary:

| # | Problem | Main Idea |
|---|---------|-----------|
| 1 | Hallucination | The LLM answers without supporting context. |
| 2 | Vocabulary Mismatch / Position | BoW cannot handle different wording or word order well. |
| 3 | Data Quality | Duplicate and noisy data reduce data quality. |
| 4 | Chunking | Poor chunk size or overlap can lose context. |
| 5 | Metadata Filtering | Similar content may have the wrong metadata. |
| 6 | Re-ranking | First-stage retrieval may rank the best document too low. |
| 7 | Faithfulness | Retrieval is correct, but generation changes important information. |
| 8 | RAG Configuration | Configuration controls which RAG components are active. |
| 9 | Evaluation | Measure chunking and retrieval with numerical metrics. |

All nine simulations use the same real Knowledge Base through `data_loader.py`. 
This allows different LLM and RAG problems to be tested using the same dataset and pipeline.

---

# ปัญหาและการแก้ไขปัญหาของ RAG System

เอกสารส่วนนี้อธิบายปัญหาที่อาจเกิดขึ้น (หรือถูกจำลองให้เกิดขึ้น) ในแต่ละขั้นตอนการทำงานของระบบ RAG ที่พัฒนาขึ้น โดยอ้างอิงจาก Source Code จริงในโปรเจกต์นี้ ครอบคลุมตั้งแต่ขั้นตอนเตรียมข้อมูล (Data Preparation) → การแบ่งข้อมูล (Chunking) → การค้นคืนเอกสาร (Retrieval) → การกรอง/จัดอันดับ (Metadata Filtering & Re-ranking) → การสร้างคำตอบ (Generation) → การตั้งค่าระบบ (Configuration) → การประเมินผล (Evaluation) โดยแต่ละหัวข้อจะอธิบาย **สาเหตุของปัญหา**, **วิธีการตรวจสอบ**, และ **แนวทางการแก้ไข** ให้สอดคล้องกับโค้ดจริงในไฟล์นั้น ๆ

## Pipeline โดยรวมของระบบ

```
privacy.txt --(data_loader.load_qa)--> Q&A entries (category, lang, question, answer)
        │
        ▼
   [1] Data Quality (dedup/normalize)
        │
        ▼
   [2] Vocabulary/Position handling (BoW vs Transformer)
        │
        ▼
   [3] Chunking (size/overlap)
        │
        ▼
   [4] Retrieval + Metadata Filtering (lang/category)
        │
        ▼
   [5] Re-ranking (specific-term boosting)
        │
        ▼
   [6/7] Generation (grounded vs hallucination/unfaithful)
        │
        ▼
   [8] Configuration (เปิด/ปิด component)
        │
        ▼
   [9] Evaluation (chunk range, Top-k retrieval check)
```

---

## 1) Data Quality — ข้อมูลซ้ำซ้อนและสัญญาณรบกวน
**ไฟล์:** `problem03_data_quality.py`

**สาเหตุของปัญหา**
ข้อมูลดิบที่จะนำเข้า Knowledge Base มักมาจากหลายแหล่งหรือผ่านการ scrape ทำให้เกิดคำถามซ้ำ (`q`, `q` ซ้ำ), ช่องว่างส่วนเกิน (`"   " + q + "   "`), และสัญลักษณ์แทรกปน (`q.replace(" ", "_") + "!!!"`) รวมถึงแถวว่างจากการดึงข้อมูลผิดพลาด ข้อมูลลักษณะนี้ทำให้ Retrieval มองว่าข้อความที่ความหมายเดียวกันเป็นคนละเอกสาร (แยก Embedding กัน) และเปลืองพื้นที่ Index โดยไม่จำเป็น

**วิธีการตรวจสอบ**
ฟังก์ชัน `normalize(text)` แปลงข้อความเป็นตัวพิมพ์เล็ก ลบสัญลักษณ์ `_ @ ! -` และรวบช่องว่างซ้ำให้เหลือช่องเดียว จากนั้นเปรียบเทียบจำนวนก่อน/หลัง deduplicate ด้วย `dict.fromkeys(normalized)` และยังตรวจสอบกับข้อมูลจริงทั้งไฟล์ `privacy.txt` (56 entries) โดยนับจำนวนคำถามที่ซ้ำกันหลัง normalize (`dup_count`)

**แนวทางการแก้ไข**
- ทำ Normalization (lowercase, ตัดสัญลักษณ์, รวบช่องว่าง) ทุกครั้งก่อนสร้าง Embedding/Index
- Deduplicate ด้วยข้อความที่ normalize แล้ว (ไม่ใช่ข้อความดิบ) ก่อนบันทึกลง Vector Store
- กรองแถวว่าง/แถวไม่สมบูรณ์ทิ้งตั้งแต่ขั้นตอน parsing (ซึ่ง `data_loader.load_qa()` ก็ทำอยู่แล้ว โดยข้าม block ที่ไม่มี header/Q/A ครบ)

---

## 2) Vocabulary Mismatch / Position — BoW เทียบกับ Transformer
**ไฟล์:** `problem02_transformer.py`

**สาเหตุของปัญหา**
คำถามที่ถามเรื่องเดียวกันแต่ใช้ระดับภาษาต่างกัน (formal vs แสลง) มีคำศัพท์คนละชุด ทำให้ Bag-of-Words (`bow()`) นับคำตรงกันได้น้อยมากหรือไม่ตรงเลย (Vocabulary Mismatch) นอกจากนี้ BoW ยังไม่สนใจลำดับคำ (Position) เพราะเป็นเพียง dictionary นับความถี่คำ ทำให้ข้อความที่สลับลำดับคำจนความหมายเปลี่ยนไป (`b = " ".join(reversed(a.split()))`) กลับมี BoW เหมือนเดิมทุกประการ

**วิธีการตรวจสอบ**
เปรียบเทียบ `bow(formal["question"])` กับ `bow(slang["question"])` ของคำถาม VPN คู่เดียวกัน แล้วดูค่า intersection ของคำ (`set(bow(formal)) & set(bow(slang))`) ว่าคำที่ตรงกันมีน้อยทั้งที่ถามเรื่องเดียวกัน และทดสอบ `bow(a) == bow(b)` ระหว่างข้อความต้นฉบับกับข้อความที่สลับลำดับคำ เพื่อยืนยันว่า BoW แยกความแตกต่างของลำดับคำไม่ได้

**แนวทางการแก้ไข**
- เปลี่ยนจาก BoW/Keyword matching ไปใช้ Sentence/Dense Embedding (Transformer-based) ที่เข้าใจความหมายเชิงบริบท (semantic) ไม่ใช่แค่คำตรงตัว
- ใช้ Positional Encoding + Self-Attention (จุดเด่นของ Transformer) เพื่อให้โมเดลรับรู้ลำดับคำและความสัมพันธ์ระหว่างคำในประโยค
- เมื่อทำ Retrieval ควร embed ทั้งคำถามแบบทางการและแบบแสลงให้อยู่ใน vector space เดียวกัน เพื่อให้ค้นเจอเอกสารที่ถูกต้องไม่ว่าใช้ระดับภาษาใด

---

## 3) Chunking — ขนาด Chunk และ Overlap
**ไฟล์:** `problem04_chunking.py`

**สาเหตุของปัญหา**
ฟังก์ชัน `chunk(words, size, overlap=0)` แบ่งข้อความยาว (คำตอบรวมของหมวดหนึ่ง ๆ) ด้วยขนาดต่าง ๆ กัน หาก `size` ใหญ่เกินไป (เช่น 200 คำ) หลาย Q&A จะถูกยัดรวมใน Chunk เดียว ทำให้ Embedding ของ Chunk นั้น "เจือจาง" ความหมายเฉพาะของแต่ละคำตอบ ในทางกลับกัน หาก `size` เล็กเกินไป (เช่น 15 คำ) คำอธิบายอาจถูกตัดกลางประโยค ทำให้เสีย Context ที่จำเป็นต่อความเข้าใจ

**วิธีการตรวจสอบ**
รันการแบ่ง Chunk ด้วย 3 ค่าพารามิเตอร์บนข้อมูลจริงจากหมวดเดียวกัน (`size=200`, `size=15`, และ `size=60, overlap=15`) แล้วพิมพ์ตัวอย่าง Chunk ออกมาดูว่าข้อความสมบูรณ์หรือถูกตัดกลางคัน และเปรียบเทียบว่า Chunk ที่มี Overlap ยังคง Context ตรงรอยต่อไว้ได้หรือไม่

**แนวทางการแก้ไข**
- เลือกขนาด Chunk ที่ใกล้เคียงความยาวคำตอบ 1 รายการ (แทนที่จะรวมหลาย Q&A เข้าด้วยกัน) เพื่อไม่ให้ Embedding เจือจาง
- กำหนด Overlap (เช่น 15–20% ของ chunk size) เพื่อรักษาบริบทตรงรอยต่อระหว่าง Chunk ที่ติดกัน
- ในระบบจริงควรแบ่ง Chunk ตามหน่วยความหมาย (เช่น ทีละคู่ Q&A) แทนการตัดตามจำนวนคำล้วน ๆ เพื่อไม่ให้ตัดกลางประโยคของคำตอบเดียวกัน

---

## 4) Metadata Filtering — Similarity สูงแต่ Metadata ผิด
**ไฟล์:** `problem05_metadata.py`

**สาเหตุของปัญหา**
ฟังก์ชัน `score()` นับจำนวนคำในคำถามที่ปรากฏใน question+answer เพียงอย่างเดียว เมื่อค้นด้วยคำถาม `"VPN ปลอดภัยขึ้นจริงเหรอ"` โดยไม่กรอง `lang` ระบบอาจเลือกเอกสารที่มีคำตรงกันมากที่สุดแต่เป็นภาษาแสลง/กันเอง (`bad = search(data, QUERY)`) ทั้งที่ระบบต้องการคำตอบโทนทางการสำหรับแชทบอทด้านความปลอดภัย (คลินิก) การจับคู่ด้วย keyword similarity เพียงอย่างเดียวจึงไม่รับประกันว่า Metadata (เช่น ระดับภาษา/หมวดหมู่) จะตรงกับความต้องการของระบบ

**วิธีการตรวจสอบ**
เปรียบเทียบผลลัพธ์การค้นแบบไม่กรอง (`search(data, QUERY)`) กับแบบกรอง `lang="ทางการ"` (`search(data, QUERY, lang="ทางการ")`) แล้วตรวจดูว่า `bad['lang']` ต่างจากโทนที่ต้องการหรือไม่ ทั้งที่ query เดียวกัน

**แนวทางการแก้ไข**
- เพิ่มขั้นตอน Metadata Filtering (filter ตาม `lang`, `category` หรือ field อื่นที่เกี่ยวข้อง) ก่อนหรือควบคู่กับ Similarity Search แทนที่จะพึ่ง keyword score เพียงอย่างเดียว
- จัดเก็บ Metadata ที่จำเป็น (register/tone, category, source, date) แนบไปกับทุก Chunk ตั้งแต่ตอนสร้าง Index เพื่อให้ Retrieval Layer ใช้กรองได้ทันที
- กรณีระบบต้องคุยหลายโทนภาษา ควรให้ผู้ใช้/ระบบระบุ Metadata ที่ต้องการ (เช่น โทนทางการ) เป็นเงื่อนไขบังคับ ไม่ใช่ทางเลือกเสริม

---

## 5) Re-ranking — First-stage Retrieval จัดอันดับผิด
**ไฟล์:** `problem06_reranking.py`

**สาเหตุของปัญหา**
`first_stage()` ให้คะแนนจากคำทั่วไป (`GENERIC_TERMS = ["ข้อมูล", "ป้องกัน"]`) ซึ่งปรากฏในหลายเอกสาร ทำให้เอกสารที่เจาะจงและตรงกับคำถามจริง ๆ (มีคำเฉพาะ เช่น `"OTP"`, `"Password Manager"`) อาจได้คะแนนเท่ากับหรือต่ำกว่าเอกสารทั่วไปอื่น ๆ และหลุดไปอยู่อันดับท้าย ๆ ใน Top-k แรก

**วิธีการตรวจสอบ**
คัด Top 6 จาก `first_stage()` แล้วพิมพ์คะแนนของแต่ละเอกสารเทียบกับ `rerank()` (ซึ่งให้น้ำหนักเพิ่ม +3 ต่อคำเฉพาะที่พบ) เพื่อดูว่าลำดับก่อน/หลัง Re-ranking เปลี่ยนไปอย่างไร และเอกสารที่ตรงประเด็นที่สุดถูกดันขึ้นมาอยู่บนสุดหรือไม่

**แนวทางการแก้ไข**
- เพิ่มขั้นตอน Re-ranking (Cross-encoder หรือการให้น้ำหนักคำเฉพาะ/Entity สำคัญ) หลังจาก First-stage Retrieval เพื่อจัดลำดับใหม่ด้วยสัญญาณที่ละเอียดกว่า
- ให้ First-stage คืนค่าจำนวนผู้สมัคร (candidates) มากพอ (เช่น Top-20/50) ก่อนส่งต่อให้ Re-ranker เลือก Top-k สุดท้าย เพื่อไม่ให้เอกสารที่ถูกจัดอันดับต่ำเกินไปตกหล่นไปตั้งแต่ First-stage
- ปรับน้ำหนักคำเฉพาะ (specific/technical terms) ให้สูงกว่าคำทั่วไปตั้งแต่การออกแบบ Scoring Function

---

## 6) Hallucination — ตอบทั้งที่ไม่มี Context รองรับ
**ไฟล์:** `problem01_hallucination.py`

**สาเหตุของปัญหา**
เมื่อคำถามอยู่นอกขอบเขต Knowledge Base (เช่น `"ข้าวมันไก่ราคาเท่าไร"`) ฟังก์ชัน `retrieve()` จะไม่พบเอกสารที่เกี่ยวข้อง (`context` ว่าง) แต่ `bad_generate()` ยังคงพยายามตอบโดยไม่มีหลักฐานรองรับ ("ถ้าข้อมูลไม่สำคัญ ถือว่าไม่อันตราย...") ซึ่งเป็นคำตอบที่ระบบ "กุขึ้นเอง" (Hallucination) ไม่ได้อ้างอิงจาก Context จริง

**วิธีการตรวจสอบ**
ทดสอบ 2 คำถามคู่กัน คือคำถามนอกขอบเขต (`q_out_of_kb`) และคำถามในขอบเขต (`q_in_kb`) แล้วดูว่า `retrieve()` คืนรายการเอกสารว่างหรือไม่ (`ctx` เป็น list ว่างสำหรับคำถามนอกขอบเขต) จากนั้นเปรียบเทียบพฤติกรรมของ `bad_generate()` กับ `grounded_generate()` เมื่อไม่มี Context

**แนวทางการแก้ไข**
- บังคับให้ Generator ตรวจสอบว่ามี Context หรือไม่ก่อนตอบเสมอ หากไม่มีให้ตอบตรง ๆ ว่า "ไม่พบข้อมูลที่สนับสนุนคำตอบใน Knowledge Base" (ตามที่ `grounded_generate()` ทำ) แทนการเดาคำตอบ
- ตั้ง Threshold ของ Retrieval Score ขั้นต่ำ หากคะแนนต่ำกว่าค่านี้ให้ถือว่า "ไม่พบ" แม้จะมีเอกสารคืนมาบ้างก็ตาม
- ใน Prompt ของ LLM ควรระบุชัดเจนว่าให้ตอบจาก Context ที่ให้มาเท่านั้น (Grounded Generation) และปฏิเสธหากข้อมูลไม่เพียงพอ

---

## 7) Faithfulness — Retrieval ถูกแต่ Generation ผิด
**ไฟล์:** `problem07_generation.py`

**สาเหตุของปัญหา**
แม้ Retrieval จะดึง Context ที่ถูกต้องมาแล้ว (`entry["answer"]`) แต่ตัว Generator เองอาจ "แก้ไข" รายละเอียดสำคัญโดยไม่ตั้งใจ เช่น `bad_generator()` เปลี่ยนความยาวรหัสผ่านขั้นต่ำจาก 12 ตัวอักษรเป็น 6 ตัวอักษร และกลับเงื่อนไขความปลอดภัยจาก "ห้ามใช้ข้อมูลส่วนตัวที่เดาง่าย" เป็น "ให้ใช้ข้อมูลส่วนตัวที่จำง่าย" ซึ่งในโดเมนความปลอดภัย ความผิดพลาดลักษณะนี้อันตรายมาก เพราะดูเหมือนคำตอบปกติแต่เนื้อหากลับผิด/เป็นอันตราย

**วิธีการตรวจสอบ**
เปรียบเทียบข้อความ Context ต้นฉบับ (`context`) กับผลลัพธ์ของ `bad_generator(context)` และ `grounded_generator(context)` แบบ line-by-line เพื่อดูว่ามีตัวเลข/เงื่อนไขใดถูกเปลี่ยนไปจากต้นฉบับหรือไม่ — นี่คือหลักการของ Faithfulness Evaluation (เทียบคำตอบกับ Context ต้นทาง)

**แนวทางการแก้ไข**
- ออกแบบ Prompt ให้ LLM ตอบโดยอ้างอิง Context ที่ให้มาเท่านั้น ห้ามดัดแปลงตัวเลข/เงื่อนไขสำคัญ (grounded generation เช่นเดียวกับ `grounded_generator()` ที่คืน context ตรง ๆ)
- เพิ่มขั้นตอน Faithfulness / Consistency Check หลังสร้างคำตอบ เช่น เทียบตัวเลข/entity สำคัญระหว่างคำตอบกับ Context ก่อนส่งให้ผู้ใช้ หากไม่ตรงให้ Reject หรือ Regenerate
- สำหรับโดเมนที่มีความเสี่ยงสูง (เช่น ความปลอดภัย/สุขภาพ/กฎหมาย) ควรมี Human-in-the-loop หรือ Rule-based Validator ตรวจสอบตัวเลข/เงื่อนไขสำคัญเพิ่มเติมก่อน deploy

---

## 8) RAG Configuration — พฤติกรรมระบบเปลี่ยนตาม Config
**ไฟล์:** `problem08_config.py`

**สาเหตุของปัญหา**
พฤติกรรมของ Pipeline ทั้งหมด (ใช้ Memory หรือไม่, ใช้ Hybrid Retrieval (`USE_HYBRID`) หรือ Dense อย่างเดียว, เปิด/ปิด Re-ranking (`USE_RERANK`), เปิด/ปิด LLM Generation (`USE_LLM`), แสดง/ไม่แสดงแหล่งอ้างอิง (`SHOW_SOURCES`)) ถูกควบคุมด้วย dictionary `CONFIG` เพียงชุดเดียว หาก Config ถูกตั้งค่าไม่เหมาะสม (เช่นปิด Re-ranking ทั้งที่ข้อมูลมีความกำกวมสูง หรือปิด `SHOW_SOURCES` ในระบบที่ต้องการความโปร่งใส) จะทำให้คุณภาพหรือความน่าเชื่อถือของคำตอบลดลงโดยที่ไม่ใช่ปัญหาที่ตัวโมเดลหรือข้อมูล แต่มาจากการตั้งค่า

**วิธีการตรวจสอบ**
พิมพ์ค่า `CONFIG` และผล pipeline ที่ประกอบขึ้นจริงตามค่านั้น (`run()` จะพิมพ์ทีละขั้นว่าใช้ Component ใดบ้างตามค่า Boolean ใน CONFIG) ทำให้เห็นชัดว่าการเปลี่ยนค่าใน dict เพียงจุดเดียวส่งผลต่อพฤติกรรม Pipeline ทั้งสาย

**แนวทางการแก้ไข**
- แยก Configuration ออกจาก Logic (เช่นใช้ config file/env var) เพื่อให้ปรับพฤติกรรมระบบได้โดยไม่ต้องแก้โค้ด และทดสอบ config แต่ละชุดก่อน deploy จริง
- ตั้งค่า Default ที่ปลอดภัย (เช่นเปิด `USE_RERANK` และ `SHOW_SOURCES` เป็นค่าเริ่มต้นสำหรับระบบด้านความปลอดภัย/ความน่าเชื่อถือสูง) แทนการปิดไว้เป็นค่าเริ่มต้น
- เพิ่ม Validation/Sanity check ของ config ตอน start up เช่น หาก `USE_LLM=False` ระบบควรแจ้งเตือนว่าจะคืนคำตอบแบบ Retrieval-only เท่านั้น

---

## 9) Evaluation — วัดผล Chunking และ Retrieval ด้วยตัวเลข
**ไฟล์:** `problem09_evaluation.py`

**สาเหตุของปัญหา**
หากไม่มีการวัดผลเชิงตัวเลข จะไม่ทราบว่าการตั้งค่า Chunking (`CHUNK_SIZE=400`, `CHUNK_OVERLAP=50`) และ Retrieval เหมาะสมกับข้อมูลจริงหรือไม่ เช่น Chunk อาจถูกแบ่งมากเกินความจำเป็นหรือ Retrieval อาจหาเอกสารที่ถูกต้องเจอแต่จัดอันดับไว้ต่ำเกินไป (เช่นตัวอย่างที่เอกสารที่ถูกต้องอยู่อันดับ 8 ซึ่ง Top-5 หาไม่เจอ แต่ Top-10 เจอ)

**วิธีการตรวจสอบ**
ฟังก์ชัน `ranges()` คำนวณจำนวนและขอบเขตของ Chunk จริงจากความยาวข้อความรวมทั้งไฟล์ (`total_chars` จาก `privacy.txt`) และตรวจสอบขนาด Overlap จริงระหว่าง Chunk ที่ 1 และ 2 (`r[0][1] - r[1][0]`) ส่วนฝั่ง Retrieval ใช้ Golden Set คือทุกคำถามในชุดข้อมูล (56 ข้อ) แล้วตรวจสอบว่าเอกสารที่ถูกต้องอยู่ใน Top-k หรือไม่ ที่ค่า `k = [1, 3, 5, 10]` (Recall@k / Hit-rate@k)

**แนวทางการแก้ไข**
- ใช้ Golden Set (คำถาม-คำตอบที่รู้คำตอบจริง) วัด Recall@k / MRR / nDCG เป็นระยะ เพื่อเทียบผลก่อน-หลังปรับ Chunking หรือ Retrieval
- หากพบว่าเอกสารถูกต้องอยู่นอก Top-k ที่ใช้งานจริงบ่อยครั้ง (เช่นอยู่อันดับ 8 แต่ระบบใช้ Top-5) ให้พิจารณาเพิ่ม k, ปรับ Chunking ให้ตรงประเด็นมากขึ้น หรือเพิ่ม Re-ranking (ดูข้อ 5)
- ทำ Evaluation แบบ Automate (รันซ้ำได้ทุกครั้งที่เปลี่ยน Config/ปรับ pipeline) เพื่อเปรียบเทียบผลกระทบของการเปลี่ยนแปลงแต่ละครั้งอย่างเป็นระบบ

---

## สรุปภาพรวม

| ขั้นตอน | ปัญหาหลักที่พบ | สาเหตุ | แนวทางแก้ไขหลัก |
|---|---|---|---|
| Data Preparation | ข้อมูลซ้ำ/สัญญาณรบกวน | ที่มาข้อมูลหลายแหล่ง/ไม่สะอาด | Normalize + Deduplicate ก่อน Index |
| Representation | Vocabulary Mismatch, ไม่รับรู้ลำดับคำ | ใช้ BoW/keyword matching | เปลี่ยนไปใช้ Transformer/Dense Embedding |
| Chunking | Chunk ใหญ่/เล็กเกินไป | ตัดตามจำนวนคำโดยไม่สนความหมาย | ปรับขนาดให้พอดีหน่วยความหมาย + ใช้ Overlap |
| Retrieval | Similarity สูงแต่ Metadata ผิด | ไม่กรอง Metadata (lang/category) | เพิ่ม Metadata Filtering ร่วมกับ Similarity |
| Ranking | เอกสารที่ตรงประเด็นอยู่อันดับท้าย | First-stage ให้น้ำหนักคำทั่วไปเท่ากันหมด | เพิ่มขั้นตอน Re-ranking |
| Generation | ตอบทั้งที่ไม่มี Context (Hallucination) | Generator ไม่ตรวจสอบ Context ก่อนตอบ | บังคับ Grounded Generation + ปฏิเสธเมื่อไม่พบข้อมูล |
| Generation | Retrieval ถูกแต่คำตอบผิด (Unfaithful) | Generator ดัดแปลงตัวเลข/เงื่อนไข | จำกัด Prompt ให้ตอบจาก Context + Faithfulness Check |
| Configuration | พฤติกรรมระบบไม่เหมาะกับ Use case | ตั้งค่า Component ผิด/ปิดส่วนสำคัญ | แยก Config ออกจาก Logic + Default ที่ปลอดภัย |
| Evaluation | ไม่ทราบคุณภาพ Chunking/Retrieval จริง | ไม่มีการวัดผลเชิงตัวเลข | วัด Recall@k บน Golden Set อย่างสม่ำเสมอ |

การจำลองทั้ง 9 ปัญหาข้างต้นใช้ Knowledge Base จริงจาก `privacy.txt` ผ่าน `data_loader.py` ร่วมกันทั้งหมด ทำให้ผลลัพธ์ที่แสดงในแต่ละไฟล์ `problemXX_*.py` สะท้อนพฤติกรรมจริงของระบบ RAG ที่พัฒนาขึ้น ไม่ใช่ตัวอย่างสมมติที่แยกออกจากกัน และสามารถใช้เป็นแนวทางตรวจสอบ/ปรับปรุงระบบ RAG ในโดเมนอื่น ๆ ได้ในลักษณะเดียวกัน