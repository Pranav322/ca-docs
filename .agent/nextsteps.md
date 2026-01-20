# 🚀 CA Final EdTech: Product Roadmap (2026 Scheme)

## 🎯 Phase 1: Core Subject Ingestion (Current Focus)
**Goal:** Build a high-precision RAG for the 5 main papers (FR, AFM, Audit, DT, IDT).
- [ ] **Database:** Finalize Postgres/Neon with `pgvector`.
- [ ] **Schema:** Add `content_type`, `source_type`, and `applicable_attempts` to the documents table.
- [ ] **Ingestor:** - [ ] Auto-extract Section Numbers and Ind AS keywords.
    - [ ] Classify chunks using a cheap model (GPT-4o-mini).

## 📌 Phase 2: Exam-Ready Features (Next)
**Goal:** Add tools for the 30% MCQ mandate and memorization.
- [ ] **MCQ Bank:** Pre-generate quiz seeds per unit in a `question_bank` table.
- [ ] **Memory Palace:** Create a "Flashcard" generator for extracted section numbers and mnemonics.
- [ ] **Practice Tracker:** Ingest RTPs/MTPs to give students "Exam-specific" context.

## 📌 Phase 3: The "Special" Modules
**Goal:** Ingest mandatory SPOM sets and the multidisciplinary Paper 6.
- [ ] **SPOM Mandatory:** Ingest Set A (Law) and Set B (Costing).
- [ ] **Paper 6 (The Mixer):** Ingest the "Case Study Digest" and allow the AI to search across all subject folders at once.

## 📌 Phase 4: Articleship & Updates
**Goal:** Real-world utility for working students.
- [ ] **Amendment Engine:** Create a system to flag/update chunks when new Finance Acts are released.
- [ ] **Office Assistant:** Ingest practical guides for GST filing and Excel for CAs.